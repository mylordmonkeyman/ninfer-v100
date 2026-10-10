"""Bounded structural comparison; different artifacts are not matched quantization."""
import argparse
from collections import Counter, defaultdict
import copy
import gzip
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys

import ab_v100 as ab
from validate_records import read_records

ROOT = Path(__file__).resolve().parents[2]

LONG = ' '.join(f'Record {i}: the solar station stores energy during daylight and supplies the village after sunset.'
                for i in range(360)) + ' Explain how its battery storage works in detail.'
COMMON = dict(NINFER_V100_DEVICE_ROUTE_COMBINE='1', NINFER_V100_CPU_EXPERT_GROUP='1',
    NINFER_V100_PREFILL_EXPERT_POLICY='auto', NINFER_V100_PREFILL_STREAM_MIN_TOKENS='256',
    NINFER_V100_PREFILL_EXPERT_STREAM_MIN_ROUTES='20', NINFER_FLASH_NEXT_CPU_EXPERT_WORKERS='88',
    NINFER_FLASH_NEXT_EXPERT_CACHE_SERIAL='0', NINFER_FLASH_NEXT_EXPERT_CACHE_PREFILL='1',
    NINFER_FLASH_NEXT_EXPERT_CACHE_GROUPED_PREFILL='1',
    NINFER_FLASH_NEXT_EXPERT_CACHE_BATCHED_DECODE='0', NINFER_V100_SV7_FP16_TC='0',
    NINFER_V100_ROUTE_HANDOFF='0', NINFER_V100_PLE_IO='mmap',
    NINFER_FLASH_NEXT_FP32_MOE_ROUTED_INPUT='0', NINFER_FLASH_NEXT_CPU_EXPERT_FP32_INTERMEDIATE='0')
ARMS = [('ninfer', 'static64', dict(COMMON, NINFER_V100_EXPERT_POLICY='static',
                                   NINFER_FLASH_NEXT_EXPERT_CACHE_MAX_SLOTS='64')),
        ('strata', 'native', {}),
        ('ninfer', 'adaptive156', dict(COMMON, NINFER_V100_EXPERT_POLICY='lru',
                                      NINFER_FLASH_NEXT_EXPERT_CACHE_MAX_SLOTS='156'))]


def residency_arms():
    qualified = dict(NINFER_V100_PREFILL_EXPERT_GEMM='fp16',
                     NINFER_V100_PREFILL_RESIDENT_GEMM='1',
                     NINFER_V100_PREFILL_CPU_STREAM_OVERLAP='1')
    return [(engine, arm, dict(flags, **qualified))
            for engine, arm, flags in (ARMS[0], ARMS[2])]


def residency_observation(folder, output_tokens):
    rows = [json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
    if not rows:
        raise ValueError('missing residency requests')
    for row in rows:
        usage = row['response']['usage']
        cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
        if (usage['prompt_tokens'] != 7111 or usage['completion_tokens'] != output_tokens
                or cached != 0 or row['response']['choices'][0]['finish_reason'] != 'length'):
            raise ValueError('residency request did not conserve the fixed frontend workload')
        timing = row.get('native_timing') or {}
        if any(not isinstance(timing.get(k), (int, float)) or not math.isfinite(timing[k])
               or timing[k] <= 0 for k in ('prefill', 'decode')):
            raise ValueError('missing finite native prefill/decode timings')
    log = (folder/'server.log').read_text()
    allocated = {key: int(value) for key, value in re.findall(
        r'^phase13\.cache\.(slots_per_layer|bytes|transfer_bytes|reserve_bytes)=(\d+)$',
        log, re.MULTILINE)}
    if not allocated.get('slots_per_layer') or not allocated.get('bytes'):
        raise ValueError('missing actual residency allocation')
    seeded = re.search(r'^v100\.profile\.seeded=(\d+)$', log, re.MULTILINE)
    if not seeded or int(seeded[1]) != allocated['slots_per_layer']*48:
        raise ValueError('startup profile did not seed the actual cache capacity')
    memory = [r.get('gpu', {}).get('metrics', {}).get('memory_used_bytes')
              for line in (folder/'system.jsonl').read_text().splitlines()
              for r in [json.loads(line)] if r.get('kind') == 'system_sample']
    return dict(actual_cache=allocated, seeded_experts=int(seeded[1]),
                sampled_peak_gpu_bytes=max((v for v in memory if v is not None), default=None))


def inventory(a):
    # Read only the v2 JSON directory. The canonical geometry validator imports
    # torch; this inventory does not decode weights or need a tensor framework.
    with a.artifact.open('rb') as stream:
        magic, size = struct.unpack('<8sQ', stream.read(16))
        if magic != b'NINFER\x00\x02' or size > 16*1024*1024:
            raise ValueError('unexpected artifact framing or directory size')
        directory = json.loads(stream.read(size))
    by_format = Counter()
    for obj in directory['objects']:
        if obj['kind'] == 'tensor': by_format[obj['format']] += obj['bytes']
    out = dict(ninfer=dict(**directory['identity'], file_bytes=a.artifact.stat().st_size,
        bytes_by_format=dict(by_format), objects=directory['objects'],
        validation='directory inventory only; native engine validates represented payloads'))
    cfg = json.loads(a.strata_config.read_text())
    flags = cfg.get('args') or []
    pack = flags[flags.index('--pack')+1] if '--pack' in flags else None
    if pack:
        pack = Path(pack)
        if not pack.is_absolute(): pack = Path(cfg.get('cwd') or a.strata_server.parent.parent)/pack
        metadata = {}
        for name in ('manifest.json', 'native_experts.txt', 'index.txt'):
            path = pack/name
            if path.exists() and path.stat().st_size <= 2*1024*1024:
                metadata[name] = path.read_text()
        out['strata'] = dict(pack=str(pack), metadata=metadata, config=cfg,
            native_rows=[dict(layer=int(p[0]), gu_type=int(p[1]), down_type=int(p[2]),
                              compact_expert_bytes=int(p[4]))
                         for line in metadata.get('native_experts.txt', '').splitlines()
                         if line.strip() and not line.lstrip().startswith('#')
                         for p in [line.split()]])
    else:
        out['strata'] = dict(config=cfg, pack_metadata_unavailable='config has no --pack')
    (a.output/'representation.json').write_text(json.dumps(out, indent=2)+'\n')


def native_work(folder, engine):
    log = folder/('server.log' if engine == 'ninfer' else 'native-engine.log')
    records = read_records(log)
    phases = defaultdict(lambda: dict(rounds=0, layers=0, counters=Counter(), host_spans_us=Counter(), cache_last_by_layer={}))
    for rec in records:
        if rec.get('kind') != 'round': continue
        if rec['status'] != 'ok': raise ValueError('failed native execution telemetry')
        dst = phases[rec['phase']]; dst['rounds'] += 1
        for layer in rec['layers']:
            dst['layers'] += 1
            dst['counters'].update(layer.get('counters', {}))
            # Inclusive scopes overlap: never sum them into wall time.
            dst['host_spans_us'].update(layer.get('host_us', {}))
            if layer.get('cache_windows'):
                end = layer['cache_windows'][-1]['end']
                dst['cache_last_by_layer'][str(layer['layer'])] = {k:v for k,v in end.items() if k != 'resident_ids'}
    ledger, dispatch, stream_timing, resident_gemm, early_cpu = [], [], [], [], []
    for line in log.read_text().splitlines():
        if '"prefill_stage_ledger"' in line:
            ledger.append(json.loads(line))
        elif '"expert_stream_timing"' in line:
            stream_timing.append(json.loads(line))
        elif '"expert_resident_gemm"' in line:
            resident_gemm.append(json.loads(line))
        elif '"early_cpu_stream"' in line:
            early_cpu.append(json.loads(line))
        elif engine == 'strata' and any(s in line.lower() for s in
            ('prefill timing:', 'mmq', 'fp16', 'cache', 'resident', 'slots')):
            dispatch.append(line)
    return dict(phases=dict(phases), ninfer_prefill_chunks=ledger,
        strata_dispatch_and_phase_lines=dispatch, expert_stream_timing=stream_timing,
        expert_resident_gemm=resident_gemm, early_cpu_stream=early_cpu,
        limits=['Host spans are inclusive, not additive critical-path components.',
                'CUDA stage intervals include stream waits and host gaps.',
                'CPU weight-read counters are modeled reads, not hardware DRAM counters.',
                'Diagnostics use native MTP; Strata native HTTP serving requires speculation.',
                'GPU weight reads, complete decode critical path and child RSS remain unobserved.'])


def summarize(cells):
    groups = defaultdict(list)
    for cell in cells:
        if cell['mode'] != 'timing': continue
        folder = Path(cell['folder'])
        rows = [json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
        for row in rows:
            if row['phase'] not in ('cold', 'measured'): continue
            key = cell['engine']+'/'+cell['arm']+'/'+row['phase']
            groups[key].append(dict(wall_seconds=row['wall_seconds'],
                usage=row['response'].get('usage'), native_timing=row.get('native_timing'),
                speculative=row.get('native_speculative'),
                response_timing=row['response'].get('timings'), output_hash=row['output_sha256']))
    return {key: dict(samples=rows, median_wall_seconds=statistics.median(r['wall_seconds'] for r in rows),
                     min_wall_seconds=min(r['wall_seconds'] for r in rows),
                     max_wall_seconds=max(r['wall_seconds'] for r in rows),
                     unique_outputs=len({r['output_hash'] for r in rows})) for key, rows in groups.items()}


def compress_logs(folder):
    for path in folder.glob('*.log'):
        with path.open('rb') as src, gzip.open(str(path)+'.gz', 'wb', compresslevel=1) as dst:
            import shutil
            shutil.copyfileobj(src, dst)
        path.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('ninfer', 'artifact', 'profile', 'strata-python', 'strata-server',
                 'strata-config', 'strata-binary', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--cpu-stream-overlap-ab', action='store_true',
                   help='qualified stream/resident GEMM with delayed versus early CPU misses')
    p.add_argument('--resident-gemm-ab', action='store_true',
                   help='qualified streamed GEMM versus streamed plus resident GEMM')
    p.add_argument('--expert-gemm-ab', action='store_true',
                   help='static64 SIMT versus bounded FP16 expert GEMM, same cold/warmed requests')
    p.add_argument('--stream-attribution-only', action='store_true',
                   help='one static64 diagnostic server, cold and repeated 7K requests only')
    p.add_argument('--qualified-default-smoke', action='store_true',
                   help='one server with unset V100 selectors; require qualified default dispatch and replay')
    p.add_argument('--residency-gemm-ab', action='store_true',
                   help='step 1: static64 versus adaptive156, both with qualified GEMM and CPU overlap')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    inventory(a)
    topology = json.loads(subprocess.check_output(['lscpu', '--json', '--extended=CPU,CORE,SOCKET'], text=True))['cpus']
    sys.path.insert(0, str(ROOT/'tools/diagnostics'))
    from v100_sv2_serve import physical_core_cpus
    cpus = physical_core_cpus(topology, os.sched_getaffinity(0))
    if len(cpus) != 32: raise ValueError(f'expected 32 allowed physical cores, got {len(cpus)}')
    ninfer_placement = ['taskset', '--cpu-list', ','.join(map(str, cpus))]
    a.ninfer_flags = ['--max-context', '8192', '--kv-capacity', '8192', '--max-concurrency', '1',
        '--prefill-chunk', '2048', '--kv-dtype', 'bf16', '--no-prefix-reuse', '--no-thinking',
        '--no-cuda-graph', '--no-qsa-prefill-mma',
        '--spec', 'mtp', '--draft-tokens', '2', '--lm-head-draft']
    a.long_prompt = LONG
    a.max_output_tokens = 128
    a.warmups = 4; a.repeats = 1; a.telemetry_level = 0
    a.request_schedule = [('cold', 'long')]+[('warmup', 'long')]*4+[('measured', 'long')]
    gemm_arms = [('simt', dict(ARMS[0][2], NINFER_V100_PREFILL_EXPERT_GEMM='simt')),
                 ('fp16', dict(ARMS[0][2], NINFER_V100_PREFILL_EXPERT_GEMM='fp16'))]
    if a.resident_gemm_ab:
        gemm_arms = [('fp16', dict(ARMS[0][2], NINFER_V100_PREFILL_EXPERT_GEMM='fp16',
                                  NINFER_V100_PREFILL_RESIDENT_GEMM='0')),
                     ('fp16-resident', dict(ARMS[0][2], NINFER_V100_PREFILL_EXPERT_GEMM='fp16',
                                           NINFER_V100_PREFILL_RESIDENT_GEMM='1'))]
    if a.cpu_stream_overlap_ab:
        gemm_arms = [(backend, dict(ARMS[0][2], NINFER_V100_PREFILL_EXPERT_GEMM='fp16',
                                  NINFER_V100_PREFILL_RESIDENT_GEMM='1',
                                  NINFER_V100_PREFILL_CPU_STREAM_OVERLAP=value))
                     for backend,value in [('fp16-resident','0'),('fp16-resident-overlap','1')]]
    default_arm = ('ninfer', 'v100-default', dict(ARMS[0][2]))
    manifest_arms = ([default_arm] if a.qualified_default_smoke else
                     residency_arms() if a.residency_gemm_ab else
                     [('ninfer', backend, flags) for backend, flags in gemm_arms]
                     if (a.expert_gemm_ab or a.resident_gemm_ab or a.cpu_stream_overlap_ab) else ARMS)
    (a.output/'manifest.json').write_text(json.dumps(dict(prompt=LONG, short_prompt=ab.SHORT,
        output_limit=32 if (a.stream_attribution_only or a.qualified_default_smoke) else 128,
        physical_cpus=cpus,
        arms=[ARMS[0]] if a.stream_attribution_only else manifest_arms,
        fresh_servers_per_arm=1 if (a.stream_attribution_only or a.qualified_default_smoke) else 3,
        stream_attribution_only=a.stream_attribution_only, expert_gemm_ab=a.expert_gemm_ab, resident_gemm_ab=a.resident_gemm_ab, cpu_stream_overlap_ab=a.cpu_stream_overlap_ab,
        qualified_default_smoke=a.qualified_default_smoke,
        residency_gemm_ab=a.residency_gemm_ab,
        timing_telemetry_level=0, diagnostic_telemetry_level=2,
        cache_budget='actual allocations logged; requested maxima are not equal-memory controls',
        cold_definition='first request after fresh server; host file cache preserved',
        cross_quant_comparison='same represented artifact; arithmetic-changing expert route' if (a.expert_gemm_ab or a.resident_gemm_ab or a.cpu_stream_overlap_ab or a.residency_gemm_ab)
            else 'descriptive; source artifacts and frontend token counts differ',
        ninfer_flags=a.ninfer_flags), indent=2)+'\n')
    cells = []
    def run(engine, arm, overrides, tag, mode):
        if ab.gpu_free_mib() < 28000: raise RuntimeError('GPU occupied; no processes were stopped')
        # Preserve each engine's evidenced placement: NInfer physical-core mask,
        # Strata native inherited CPU set. Record both rather than retune Strata.
        a.launch_prefix = ninfer_placement if engine == 'ninfer' else []
        try:
            result = ab.run_case(a, engine, tag, overrides)
            folder = a.output/(engine+'-'+tag)
            cell = dict(engine=engine, arm=arm, mode=mode, folder=str(folder), result=result)
            if a.residency_gemm_ab:
                cell['residency'] = residency_observation(folder, a.max_output_tokens)
            if mode == 'diagnostic':
                cell['work'] = native_work(folder, engine)
                if engine == 'ninfer' and not cell['work']['ninfer_prefill_chunks']:
                    raise ValueError('missing NInfer prefill stage evidence')
                if engine == 'strata' and not any('prefill timing:' in l for l in cell['work']['strata_dispatch_and_phase_lines']):
                    raise ValueError('missing Strata prefill timing evidence')
            cells.append(cell)
            (a.output/'checkpoint.json').write_text(json.dumps(cells, indent=2)+'\n')
            compress_logs(folder)
        finally:
            (a.output/'summary.json').write_text(json.dumps(dict(cells=cells, timing=summarize(cells)), indent=2)+'\n')
        ab.release_gpu()
    if a.residency_gemm_ab:
        arms = residency_arms()
        for repeat in range(3):
            for engine, arm, flags in (arms if repeat != 1 else list(reversed(arms))):
                run(engine, arm, flags, arm+'-gemm-r'+str(repeat), 'timing')
        timing = summarize(cells)
        report = dict(step=1, status='timing_complete', timing=timing,
            comparison='combined cache policy/capacity choice, not isolated policy attribution',
            output_parity='recorded diagnostically; residency can change CPU/GPU arithmetic and MTP work',
            observations=[dict(arm=c['arm'], mode=c['mode'], **c['residency']) for c in cells])
        (a.output/'residency-report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
        # Separate diagnostics cover both possible winners without instrumenting timing.
        # Step 2 uses these observations first; another job is warranted only for
        # a material attribution gap, not another residency sweep.
        a.max_output_tokens=32; a.telemetry_level=2
        a.request_schedule=[('diagnostic','long'),('diagnostic','long')]
        for engine, arm, flags in arms:
            diagnostic=dict(flags, NINFER_FLASH_NEXT_STAGE_LEDGER='1',
                            NINFER_V100_EXPERT_STREAM_TIMING='1')
            run(engine, arm, diagnostic, arm+'-gemm-diagnostic', 'diagnostic')
            work=cells[-1]['work']
            early=work['early_cpu_stream']
            expected=work['phases']['prefill']['counters'].get('cpu_routes',0)
            if not early or sum(row['cpu_routes'] for row in early)!=expected:
                raise ValueError('residency diagnostic does not conserve prefill CPU routes')
            if any(row['overlap_ms'] for row in early if not row['eligible']):
                raise ValueError('ineligible residency scope reported CPU overlap')
            if not work['expert_resident_gemm'] or not work['expert_stream_timing']:
                raise ValueError('qualified resident/stream expert dispatch was not observed')
        report['status']='complete'
        report['diagnostics']=[dict(arm=c['arm'], residency=c['residency'], work=c['work'])
                               for c in cells if c['mode']=='diagnostic']
        (a.output/'residency-report.json').write_text(json.dumps(report, indent=2)+'\n')
        print('Step 1 residency comparison complete; Step 2 diagnostic evidence retained.', flush=True)
        return
    if a.stream_attribution_only:
        a.max_output_tokens = 32; a.telemetry_level = 2
        a.request_schedule = [('diagnostic', 'long'), ('diagnostic', 'long')]
        engine, arm, overrides = ARMS[0]
        diagnostic = dict(overrides, NINFER_FLASH_NEXT_STAGE_LEDGER='1',
                          NINFER_V100_EXPERT_STREAM_TIMING='1')
        run(engine, arm, diagnostic, arm+'-stream-attribution', 'diagnostic')
        work = cells[-1]['work']; rows = work['expert_stream_timing']
        routes = sum(r['routes'] for r in rows)
        expected = sum(p['counters'].get('nonresident_gpu_routes', 0) for p in work['phases'].values())
        slot_bytes = (2_764_808+255)//256*256
        weight_bytes = sum(r['experts'] for r in rows)*slot_bytes
        expected_bytes = sum(p['counters'].get('expert_h2d_bytes', 0) for p in work['phases'].values())
        if not rows or routes != expected or weight_bytes != expected_bytes:
            raise ValueError(f'incomplete stream attribution: routes {routes}/{expected}, bytes {weight_bytes}/{expected_bytes}')
        return
    if a.qualified_default_smoke:
        selectors = ('NINFER_V100_PREFILL_EXPERT_GEMM',
                     'NINFER_V100_PREFILL_RESIDENT_GEMM',
                     'NINFER_V100_PREFILL_CPU_STREAM_OVERLAP')
        inherited = [name for name in selectors if name in os.environ]
        if inherited: raise ValueError(f'default smoke inherited selector overrides: {inherited}')
        a.max_output_tokens=32; a.telemetry_level=2
        a.request_schedule=[('cold','long'),('measured','long')]
        flags=dict(default_arm[2], NINFER_FLASH_NEXT_STAGE_LEDGER='1',
                   NINFER_V100_EXPERT_STREAM_TIMING='1')
        run('ninfer','v100-default',flags,'v100-default-diagnostic','diagnostic')
        cell=cells[-1]; work=cell['work']
        rows=[json.loads(line) for line in
              (Path(cell['folder'])/'requests.jsonl').read_text().splitlines()]
        signatures={json.dumps([row['output_sha256'],row['response']['usage'],
                                row.get('native_speculative')],sort_keys=True) for row in rows}
        if len(rows)!=2 or len(signatures)!=1:
            raise ValueError('default smoke changed output, usage or native MTP work on replay')
        early=work['early_cpu_stream']; resident=work['expert_resident_gemm']
        streamed=work['expert_stream_timing']
        expected=work['phases']['prefill']['counters'].get('cpu_routes',0)
        if not early or sum(row['cpu_routes'] for row in early)!=expected:
            raise ValueError('default CPU overlap dispatch does not conserve prefill CPU routes')
        if any(row['overlap_ms'] for row in early if not row['eligible']):
            raise ValueError('ineligible default CPU overlap scope reported temporal overlap')
        if not resident or not streamed:
            raise ValueError('default V100 resident or streamed expert dispatch was not observed')
        print(json.dumps(dict(default_smoke='pass', requests=len(rows),
            cpu_routes=expected, temporal_overlap_ms=sum(row['overlap_ms'] for row in early),
            resident_gemm_routes=sum(row['routes'] for row in resident),
            streamed_routes=sum(row['routes'] for row in streamed)),indent=2),flush=True)
        return
    if a.expert_gemm_ab or a.resident_gemm_ab or a.cpu_stream_overlap_ab:
        engine, arm, overrides = ARMS[0]
        arms = gemm_arms
        for repeat in range(3):
            for backend, flags in (arms if repeat != 1 else list(reversed(arms))):
                run(engine, backend, flags, backend+'-r'+str(repeat), 'timing')
        if a.cpu_stream_overlap_ab:
            signatures=set()
            for cell in cells:
                if cell['mode']!='timing':continue
                for line in (Path(cell['folder'])/'requests.jsonl').read_text().splitlines():
                    row=json.loads(line)
                    signatures.add(json.dumps([row['output_sha256'],row['response']['usage'],
                                               row.get('native_speculative')],sort_keys=True))
            if len(signatures)!=1:
                raise ValueError('CPU scheduling changed output, emitted tokens or native MTP work')
        a.max_output_tokens=32; a.telemetry_level=2
        a.request_schedule=[('diagnostic','long'),('diagnostic','long')]
        for backend, flags in arms:
            diagnostic=dict(flags, NINFER_FLASH_NEXT_STAGE_LEDGER='1',
                            NINFER_V100_EXPERT_STREAM_TIMING='1')
            run(engine,backend,diagnostic,backend+'-diagnostic','diagnostic')
            if a.cpu_stream_overlap_ab:
                rows=cells[-1]['work']['early_cpu_stream']
                if bool(rows)!=(backend=='fp16-resident-overlap'):
                    raise ValueError('early CPU dispatch does not match selected backend')
                if rows:
                    expected=cells[-1]['work']['phases']['prefill']['counters'].get('cpu_routes',0)
                    if sum(r['cpu_routes'] for r in rows)!=expected:
                        raise ValueError('early CPU ledger does not conserve native prefill routes')
            if a.resident_gemm_ab:
                rows=cells[-1]['work']['expert_resident_gemm']
                if bool(rows) != (backend=='fp16-resident'):
                    raise ValueError('resident GEMM dispatch does not match selected backend')
        print(json.dumps(summarize(cells),indent=2),flush=True)
        return
    for repeat in range(3):
        for engine, arm, overrides in (ARMS if repeat != 1 else list(reversed(ARMS))):
            run(engine, arm, overrides, arm+'-r'+str(repeat), 'timing')
    # One fresh adaptive reference at the end checks drift without another sweep.
    engine, arm, overrides = ARMS[2]
    run(engine, arm, overrides, arm+'-repeat', 'timing')
    # Profiling is deliberately separate from throughput. One cold 7K request
    # and one short request observe prefill versus steady native-MTP work.
    a.max_output_tokens = 32; a.telemetry_level = 2
    a.request_schedule = [('diagnostic', 'long'), ('diagnostic', 'short')]
    for engine, arm, overrides in ARMS:
        diagnostic = copy.deepcopy(overrides)
        if engine == 'ninfer': diagnostic['NINFER_FLASH_NEXT_STAGE_LEDGER'] = '1'
        else: diagnostic['STRATA_PREFILL_TIMING'] = '1'
        run(engine, arm, diagnostic, arm+'-diagnostic', 'diagnostic')
    print(json.dumps(summarize(cells), indent=2), flush=True)


if __name__ == '__main__': main()
