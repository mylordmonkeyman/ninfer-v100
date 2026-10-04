#!/usr/bin/env python3
"""Qualify the opt-in SV2 device route matrix against the legacy host reduction."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

from v100_expert_profile import atomic_json
from v100_sv0_report import read_records
from v100_sv1_residency import run, read_logits


MODES = ('legacy', 'device')
ARMS = ('off', 'lru')


def invoke(executable, output, name, workload, arm, slots, telemetry=False):
    old = os.environ.get('NINFER_V100_DEVICE_ROUTE_COMBINE')
    try:
        os.environ['NINFER_V100_DEVICE_ROUTE_COMBINE'] = '1' if name.startswith('device-') else '0'
        return run(executable, output, name, workload, arm, slots,
                   telemetry=telemetry, discover=slots == 512)
    finally:
        if old is None:
            os.environ.pop('NINFER_V100_DEVICE_ROUTE_COMBINE', None)
        else:
            os.environ['NINFER_V100_DEVICE_ROUTE_COMBINE'] = old


def validate_transfers(path, device):
    rows = [row for row in read_records(path) if row.get('kind') == 'expert_layer']
    if not rows:
        raise ValueError(f'{path}: no expert-layer transfer evidence')
    hits = sum(row['gpu_hit_routes'] for row in rows)
    misses = sum(row['cpu_miss_routes'] for row in rows)
    routes = sum(row['routes'] for row in rows)
    if hits + misses != routes:
        raise ValueError(f'{path}: incomplete route accounting')
    if device:
        if any(row['cache_result_d2h_bytes'] or row['routed_sum_h2d_bytes'] for row in rows):
            raise ValueError(f'{path}: device route path retained legacy full transfers')
        expected = sum(row['cpu_miss_routes']*2560*4 for row in rows)
        actual = sum(row['cpu_miss_h2d_bytes'] for row in rows)
        if actual != expected:
            raise ValueError(f'{path}: miss-only H2D byte accounting mismatch')
    return dict(routes=routes, hits=hits, misses=misses,
                hit_fraction=hits/routes, cpu_miss_h2d_bytes=sum(
                    row.get('cpu_miss_h2d_bytes', 0) for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('at least three fresh-process observations required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(os.environ['NINFER_FLASH_NEXT_ORACLE_MANIFEST'])
    positions = json.loads(manifest.read_text())['positions']
    if len(positions) < 2432:
        raise ValueError('need held-out corpus through position 2431')
    workload = output/'held-out.json'
    atomic_json(workload, dict(name='sv2-held-out', source_manifest=str(manifest),
                source_start=2048, source_end=2432, prefill_tokens=128,
                tokens=[row['token_id'] for row in positions[2048:2432]]))

    diagnostics = {}
    # Discover once on the legacy LRU plan, then hold the exact cache capacity fixed.
    legacy_lru = invoke(args.executable, output, 'legacy-diagnostic-lru',
                        workload, 'lru', 512, telemetry=True)
    slots = legacy_lru['startup']['slots_per_layer']
    diagnostics['legacy-lru'] = legacy_lru
    for mode, arm in ((m, a) for m in MODES for a in ARMS if (m, a) != ('legacy', 'lru')):
        diagnostics[f'{mode}-{arm}'] = invoke(args.executable, output,
            f'{mode}-diagnostic-{arm}', workload, arm, slots, telemetry=True)
    cached_startups = [diagnostics[f'{mode}-lru']['startup'] for mode in MODES]
    if any(row['slots_per_layer'] != slots for row in cached_startups):
        raise ValueError('legacy and device paths did not use identical cache capacity')
    for key in ('cache_bytes', 'transfer_budget_bytes'):
        if len({row[key] for row in cached_startups}) != 1:
            raise ValueError(f'legacy and device cache allocation differs: {key}')

    transfers = {}
    for mode in MODES:
        for arm in ARMS:
            name = f'{mode}-diagnostic-{arm}'
            transfers[f'{mode}-{arm}'] = validate_transfers(
                output/f'{name}.log', mode == 'device')
    if transfers['device-off']['hits'] != 0:
        raise ValueError('cache-off control unexpectedly recorded GPU hits')
    if not (0 < transfers['device-lru']['hits'] < transfers['device-lru']['routes']):
        raise ValueError('LRU device route path did not exercise mixed hits and misses')

    observations = []
    order = tuple((mode, arm) for mode in MODES for arm in ARMS)
    for repeat in range(args.repeats):
        for mode, arm in (order if repeat % 2 == 0 else tuple(reversed(order))):
            result = invoke(args.executable, output, f'{mode}-timing-{repeat}-{arm}',
                            workload, arm, slots)
            observations.append(result)
            atomic_json(output/'timing-observations.json', observations)

    numerical = []
    for arm in ARMS:
        reference = diagnostics[f'legacy-{arm}']
        expected = read_logits(output/f'legacy-diagnostic-{arm}.bf16',
                               reference['complete']['final_logit_words'])
        actual_row = diagnostics[f'device-{arm}']
        actual = read_logits(output/f'device-diagnostic-{arm}.bf16',
                             actual_row['complete']['final_logit_words'])
        exact = expected == actual
        top1_exact = reference['complete']['decode_top1'] == actual_row['complete']['decode_top1']
        if not exact or not top1_exact:
            raise ValueError(f'{arm}: device combine changed held-out output')
        numerical.append(dict(arm=arm, final_logits_bitwise_equal=exact,
                              decode_top1_exact=top1_exact))
        for repeat in range(args.repeats):
            for mode in MODES:
                row = next(item for item in observations
                           if item['name'] == f'{mode}-timing-{repeat}-{arm}')
                observed = read_logits(output/f'{row["name"]}.bf16',
                                       row['complete']['final_logit_words'])
                if observed != expected or row['complete']['decode_top1'] != reference['complete']['decode_top1']:
                    raise ValueError(f'{row["name"]}: repeated output changed legacy arithmetic')

    throughput = []
    for mode in MODES:
        for arm in ARMS:
            rows = [row for row in observations if row['name'].startswith(f'{mode}-') and row['arm'] == arm]
            values = [next(m['tokens_per_s'] for m in row['measurements']
                           if m['shape'] == 'decode') for row in rows]
            throughput.append(dict(mode=mode, arm=arm, observations=len(values),
                median=statistics.median(values), minimum=min(values), maximum=max(values)))
    report = dict(schema=1, milestone='SV2', qualified=False,
        scope='held_out_device_route_transfer_and_arithmetic_screen',
        candidate_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        workload_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(),
        cache_slots_per_layer=slots, transfers=transfers, numerical=numerical,
        throughput=throughput,
        limitations=['opt-in path; default unchanged',
                     'single held-out 128-prefill plus 256-decode workload',
                     'MTP, continuation and production server qualification pending'])
    atomic_json(output/'runtime-report.json', report)
    lines = ['SV2 opt-in device route screen; no default promotion.', '',
             '| Mode | Cache | Decode t/s median (range) |', '|---|---|---:|']
    for row in throughput:
        lines.append(f"| {row['mode']} | {row['arm']} | {row['median']:.3f} ({row['minimum']:.3f}–{row['maximum']:.3f}) |")
    lines.extend(['', 'Held-out final BF16 logits and decode top-1 are exact for each cache arm.',
                  'Device mode reports zero cache-result D2H, zero routed-sum H2D, and exact miss-only H2D bytes.'])
    (output/'runtime-report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
