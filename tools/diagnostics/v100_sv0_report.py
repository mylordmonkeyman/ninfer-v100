"""Summarize opt-in SV0 JSONL diagnostics; missing measurements never qualify a candidate."""
import argparse
import json
import math
from pathlib import Path
import statistics

BASELINE_SHA = "e184de12772999f8d336ea395cedcc0cf0543c52"


def numeric(value, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected a number")
    if not math.isfinite(value) or value < 0 or (integer and not isinstance(value, int)):
        raise ValueError(f"{name}: expected a finite nonnegative {'integer' if integer else 'number'}")
    return value


def read_records(path):
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.startswith('{'):
            continue
        row = json.loads(line)
        if row.get('sv') != 0:
            continue
        if row.get('schema') != 1:
            raise ValueError(f'{path}:{line_number}: unsupported SV0 schema')
        records.append(row)
    if not records:
        raise ValueError(f'{path}: no SV0 measurements (set NINFER_V100_TELEMETRY=1)')
    return records


def distribution(values):
    return dict(observations=len(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values)) if values else None


def summarize(records):
    # A log is a caller-defined interval. Never invent request boundaries from layer numbers:
    # MTP, continuation, and concurrent requests make that inference invalid.
    frequency = [[0]*512 for _ in range(48)]
    layers = {}
    ple = []
    stages = []
    for row in records:
        kind = row['kind']
        if kind == 'ple_gather':
            for key in ('tokens', 'payload_bytes', 'gather_us'):
                numeric(row[key], key, integer=key != 'gather_us')
            ple.append(row)
            continue
        if kind == 'prefill_stage_ledger':
            numeric(row['total_chunk_ms'], 'total_chunk_ms')
            stages.append(row)
            continue
        if kind != 'expert_layer':
            raise ValueError(f'unknown SV0 record kind: {kind}')
        layer = numeric(row['layer'], 'layer', integer=True)
        if layer >= 48 or len(row['frequency']) != 512:
            raise ValueError('wrong layer/expert geometry')
        counts = [numeric(v, 'frequency', integer=True) for v in row['frequency']]
        routes = numeric(row['routes'], 'routes', integer=True)
        tokens = numeric(row['tokens'], 'tokens', integer=True)
        if not tokens or not isinstance(row['prefill'], bool):
            raise ValueError('invalid token count or execution phase')
        hits = numeric(row['gpu_hit_routes'], 'gpu_hit_routes', integer=True)
        misses = numeric(row['cpu_miss_routes'], 'cpu_miss_routes', integer=True)
        if routes != sum(counts) or routes != tokens*10 or hits+misses != routes:
            raise ValueError('routing histogram/branch counts disagree')
        distinct = sum(v > 0 for v in counts)
        if row['distinct_experts'] != distinct:
            raise ValueError('distinct expert count disagrees with histogram')
        # Captures baseline inefficiency: a partial-hit layer still downloads every route slot.
        if row['cache_result_d2h_bytes'] != (routes*2560*4 if hits else 0):
            raise ValueError('SV0 baseline cache-result D2H bytes inconsistent')
        if row['routed_sum_h2d_bytes'] != tokens*2560*4:
            raise ValueError('SV0 baseline routed-sum H2D bytes inconsistent')
        for expert, count in enumerate(counts):
            frequency[layer][expert] += count
        cell = layers.setdefault((layer, row['prefill']), [])
        cell.append(row)
    layer_summary = []
    for (layer, prefill), rows in sorted(layers.items()):
        hits = sum(r['gpu_hit_routes'] for r in rows)
        misses = sum(r['cpu_miss_routes'] for r in rows)
        result = dict(layer=layer, prefill=prefill, layer_calls=len(rows),
                      hits=hits, misses=misses, hit_fraction=hits/(hits+misses))
        for key in ('cache_result_d2h_bytes', 'routed_sum_h2d_bytes', 'route_input_d2h_bytes'):
            result[key] = sum(numeric(r[key], key, integer=True) for r in rows)
        for key in ('cpu_branch_us', 'gpu_hit_window_us', 'merge_wait_us', 'branch_wall_us',
                    'router_rendezvous_us', 'overlap_lower_bound_us', 'distinct_experts',
                    'routes_per_distinct_expert'):
            result[key] = distribution([numeric(r[key], key) for r in rows if r[key] is not None])
        # Background fills span inference calls. Global cumulative totals are snapshots, not
        # per-layer deltas, and must not be summed across layers or cache resets.
        result['last_cache_snapshot'] = rows[-1]['cache']
        layer_summary.append(result)
    return dict(schema=1, milestone='SV0', scope='caller_supplied_log_interval',
                qualified=False, layer_summary=layer_summary, routing_frequency=frequency,
                ple_gather=ple, prefill_stage_ledger=stages,
                unavailable=['request_identity', 'ple_page_read_time', 'isolated_cache_result_d2h_time',
                             'isolated_routed_sum_h2d_time', 'bf16_time_by_dispatched_implementation',
                             'complete_vram_breakdown', 'request_throughput', 'oracle_acceptance',
                             'disabled_telemetry_throughput_parity'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-sha', required=True)
    parser.add_argument('--baseline-sha', default=BASELINE_SHA)
    parser.add_argument('--provenance', type=Path, help='JSON hardware/build/artifact/flags/cache state')
    args = parser.parse_args()
    report = summarize(read_records(args.log))
    report['baseline_sha'] = args.baseline_sha
    report['candidate_sha'] = args.candidate_sha
    report['provenance'] = json.loads(args.provenance.read_text()) if args.provenance else None
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    hits = sum(r['hits'] for r in report['layer_summary'])
    misses = sum(r['misses'] for r in report['layer_summary'])
    print(f"SV0: {len(report['layer_summary'])} layer/phase cells; {hits} hits, {misses} CPU routes")
    print('Qualification pending: this diagnostic report does not establish correctness or throughput.')


if __name__ == '__main__':
    main()
