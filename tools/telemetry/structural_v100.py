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


def residency_observation(folder, output_tokens, expected_prompt_tokens=7111):
    rows = [json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
    if not rows:
        raise ValueError('missing residency requests')
    for row in rows:
        usage = row['response']['usage']
        cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
        if ((expected_prompt_tokens is not None and usage['prompt_tokens'] != expected_prompt_tokens)
                or usage['completion_tokens'] != output_tokens
                or cached != 0 or row['response']['choices'][0]['finish_reason'] != 'length'):
            raise ValueError('residency request did not conserve the fixed frontend workload')
        timing = row.get('native_timing') or {}
        if any(not isinstance(timing.get(k), (int, float)) or not math.isfinite(timing[k])
               or timing[k] <= 0 for k in ('prefill', 'decode')):
            raise ValueError('missing finite native prefill/decode timings')
    log = ((folder/'server.log').read_text() if (folder/'server.log').exists()
           else gzip.open(folder/'server.log.gz','rt').read())
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


def reuse_dense_baseline(source, output):
    """Reuse only the fully completed, unchanged QPN control cell."""
    import shutil
    manifest=json.loads((source/'manifest.json').read_text())
    if (manifest.get('prompt')!=LONG or manifest.get('output_limit')!=128 or
        not manifest.get('gdn_prefill_gemm_ab') or manifest.get('timing_telemetry_level')!=0):
        raise ValueError('reference is not the original dense comparison workload')
    prior=json.loads((source/'checkpoint.json').read_text())
    if len(prior)!=1 or (prior[0]['engine'],prior[0]['arm'],prior[0]['mode'])!=('ninfer','gdn-qpn','timing'):
        raise ValueError('reference does not contain exactly the completed first QPN control')
    folder=source/'ninfer-gdn-qpn-r0'
    rows=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
    if [r['phase'] for r in rows]!=['cold']+['warmup']*4+['measured']:
        raise ValueError('reference QPN request schedule incomplete')
    observation=residency_observation(folder,128)
    flags=json.loads((folder/'launch.json').read_text())['overrides']
    if flags.get('NINFER_V100_GDN_PREFILL_GEMM')!='0':
        raise ValueError('reference QPN route was not explicitly selected')
    target=output/'ninfer-dense-qpn-r0'
    shutil.copytree(folder,target)
    return dict(prior[0],arm='dense-qpn',folder=str(target),residency=observation,
                source_run=38098932339,reused=True)


def reuse_pair_expand_baseline(source, output, flags, physical_cpus, ninfer_flags):
    """Reuse the completed uninstrumented control from the phase measurement."""
    import shutil
    manifest=json.loads((source/'manifest.json').read_text())
    if ((source.parent/'ninfer-sha.txt').read_text().strip()!=
        '2fb6d3ee70514de5398e631f2f09c37e28c9d1b8' or
        not manifest.get('moe_work_breakdown') or manifest.get('prompt')!=LONG or
        manifest.get('output_limit')!=128 or manifest.get('timing_telemetry_level')!=0 or
        manifest.get('physical_cpus')!=physical_cpus or manifest.get('ninfer_flags')!=ninfer_flags):
        raise ValueError('pair expansion reference configuration does not match')
    cells=json.loads((source/'checkpoint.json').read_text())
    if not cells or (cells[0]['engine'],cells[0]['arm'],cells[0]['mode'])!=('ninfer','dense-full-chunk','timing'):
        raise ValueError('pair expansion reference is missing the complete uninstrumented control')
    folder=source/'ninfer-dense-full-chunk-reference'
    rows=[json.loads(l) for l in (folder/'requests.jsonl').read_text().splitlines()]
    if [r['phase'] for r in rows]!=['cold']+['warmup']*4+['measured']:
        raise ValueError('pair expansion reference schedule incomplete')
    old=json.loads((folder/'launch.json').read_text())['overrides']
    expected={k:v for k,v in flags.items() if k!='NINFER_V100_EXPERT_PAIR_EXPAND'}
    if old!=expected:raise ValueError('pair expansion reference selectors differ')
    observation=residency_observation(folder,128)
    target=output/'ninfer-expand-scalar-r0';shutil.copytree(folder,target)
    return dict(cells[0],arm='expand-scalar',folder=str(target),residency=observation,
                source_run=38102677923,reused=True)


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
    ledger, dispatch, stream_timing, resident_gemm, early_cpu, dense_gemm, qsa_dispatch, qsa_comparisons = [], [], [], [], [], [], [], []
    for line in log.read_text().splitlines():
        if '"prefill_stage_ledger"' in line:
            ledger.append(json.loads(line))
        elif '"expert_stream_timing"' in line:
            stream_timing.append(json.loads(line))
        elif '"expert_resident_gemm"' in line:
            resident_gemm.append(json.loads(line))
        elif '"dense_prefill_gemm"' in line:
            dense_gemm.append(json.loads(line))
        elif '"early_cpu_stream"' in line:
            early_cpu.append(json.loads(line))
        elif '"qsa_real_comparison"' in line:
            qsa_comparisons.append(json.loads(line))
        elif '"qsa_score_dispatch"' in line:
            qsa_dispatch.append(json.loads(line))
        elif engine == 'strata' and any(s in line.lower() for s in
            ('prefill timing:', 'mmq', 'fp16', 'cache', 'resident', 'slots')):
            dispatch.append(line)
    return dict(phases=dict(phases), ninfer_prefill_chunks=ledger,
        strata_dispatch_and_phase_lines=dispatch, expert_stream_timing=stream_timing,
        expert_resident_gemm=resident_gemm, early_cpu_stream=early_cpu, dense_prefill_gemm=dense_gemm,
        qsa_score_dispatch=qsa_dispatch, qsa_real_comparison=qsa_comparisons,
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


def prefill_breakdown(folder):
    """Attribute each HTTP request using its byte-delimited native log window.

    Consecutive compute-stream event intervals partition the observed timeline;
    they include host submission gaps and stream dependencies, not just kernels.
    Inclusive SV0 host spans and separate stream timings must not be added to it.
    """
    log = folder/'server.log'
    data = log.read_bytes() if log.exists() else gzip.open(str(log)+'.gz', 'rb').read()
    requests = [json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
    reports = []
    for row in requests:
        begin, end = row['native_log_start_offset'], row['native_log_end_offset']
        if not 0 <= begin < end <= len(data):
            raise ValueError('invalid request native log byte window')
        records = []
        for line in data[begin:end].decode().splitlines():
            if line.startswith('{'):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        chunks = [r for r in records if r.get('kind') == 'prefill_stage_ledger'
                  and r.get('context', {}).get('phase') == 'prefill']
        usage = row['response']['usage']
        if not chunks or sum(c['tokens'] for c in chunks) != usage['prompt_tokens']:
            raise ValueError('prefill ledger does not cover this complete request prompt')
        stages = Counter()
        for chunk in chunks:
            for stage in chunk['stages']:
                stages[stage['stage']] += stage['interval_ms']
        total_ms = sum(c['total_chunk_ms'] for c in chunks)
        residual_ms = sum(c['residual_ms'] for c in chunks)
        if abs(sum(stages.values()) + residual_ms - total_ms) > max(1, total_ms*.0001):
            raise ValueError('stage ledger intervals do not reconcile')
        native_ms = row['native_timing']['prefill']*1000
        groups = Counter()
        for name, ms in stages.items():
            if name.startswith('MoE:'):
                group = 'moe_inclusive'
            elif name.startswith('Attention: QSA'):
                group = 'qsa_projection' if 'projection' in name else 'qsa_attention_indexer'
            elif name.startswith('Attention: GDN'):
                group = 'gdn_projection' if 'projection' in name else 'gdn_recurrence_controls'
            elif name == 'PLE injection':
                group = 'ple_inclusive'
            elif name.startswith('Hyper '):
                group = 'hyper_norm_boundaries'
            elif name.startswith('Preamble:'):
                group = 'embedding_staging'
            else:
                group = 'final_norm_head'
            groups[group] += ms
        # Keep this summary request-wide and separate from the stage partition.
        # New submission-owned rows are validated by moe_stream_breakdown.
        stream_rows = [r for r in records if r.get('kind') == 'expert_stream_timing']
        host = Counter()
        for rec in records:
            if rec.get('kind') == 'round' and rec.get('phase') == 'prefill':
                for layer in rec.get('layers', []):
                    host.update(layer.get('host_us', {}))
        reports.append(dict(index=row['index'], phase=row['phase'], usage=usage,
            wall_seconds=row['wall_seconds'], native_timing=row['native_timing'],
            output_hash=row['output_sha256'], speculative=row.get('native_speculative'),
            prefill_chunks=len(chunks), ledger_seconds=total_ms/1000,
            native_minus_ledger_seconds=(native_ms-total_ms)/1000,
            residual_seconds=residual_ms/1000,
            categories_seconds={k:v/1000 for k,v in groups.items()},
            category_pct_of_ledger={k:100*v/total_ms for k,v in groups.items()},
            stages_seconds={k:v/1000 for k,v in stages.items()},
            inclusive_prefill_host_seconds={k:v/1e6 for k,v in host.items()},
            request_wide_nonadditive_stream_ms={k:sum(r.get(k,0) for r in stream_rows)
                for k in ('copy_ms','compute_wait_ms','kernel_interval_ms')},
            limits=['Stage event intervals include waits and launch gaps; not pure GPU kernel time.',
                    'Projection stages are not exclusively BF16; dtype attribution requires dispatch evidence.',
                    'PLE includes gather publication, projections, convolution and injection.',
                    'Native-minus-ledger is reconciliation, not a measured launch-gap category.',
                    'This stream summary is request-wide; phase-owned rows are validated separately when requested.',
                    'Inclusive host and separate-stream intervals overlap; do not sum into latency.']))
    return reports



def architecture_requests(folder, engine):
    """Byte-owned low-overhead NInfer host spans or installed Strata timers.

    These are observations of the instrumented timeline, not kernel counters.
    Host spans and CUDA intervals overlap and are deliberately kept separate.
    """
    path = folder/('server.log' if engine == 'ninfer' else 'native-engine.log')
    data = path.read_bytes() if path.exists() else gzip.open(str(path)+'.gz', 'rb').read()
    requests = [json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
    reports = prefill_breakdown(folder) if engine == 'ninfer' else []
    for ordinal, request in enumerate(requests):
        begin, end = request['native_log_start_offset'], request['native_log_end_offset']
        if not 0 <= begin < end <= len(data):
            raise ValueError('invalid architecture request byte window')
        text = data[begin:end].decode()
        if engine == 'ninfer':
            rows = [json.loads(line) for line in text.splitlines()
                    if line.startswith('{') and '"moe_arch_timing"' in line]
            phases = defaultdict(lambda: dict(layers=0, tokens_by_layer=Counter(),
                host_us=Counter(), routes=Counter()))
            owners = set()
            for row in rows:
                context = row.get('context') or {}
                owner = (context.get('executor'), context.get('transaction'), row['layer'])
                if owner in owners or row.get('heavy_stream_drain'):
                    raise ValueError('duplicate architecture owner or heavyweight stream drain')
                owners.add(owner)
                if row['routes'] != sum(row[k] for k in ('resident_routes','stream_routes','cpu_routes')):
                    raise ValueError('architecture route conservation failed')
                if any(not math.isfinite(v) or v < 0 for v in row['host_us'].values()):
                    raise ValueError('invalid architecture host duration')
                dst = phases[context['phase']]
                dst['layers'] += 1
                dst['tokens_by_layer'][str(row['layer'])] += row['tokens']
                dst['host_us'].update(row['host_us'])
                dst['routes'].update({k:row[k] for k in
                    ('routes','resident_routes','stream_routes','cpu_routes','stream_experts')})
            prefill = phases.get('prefill')
            if (not prefill or len(prefill['tokens_by_layer']) != 48
                    or set(prefill['tokens_by_layer'].values()) != {request['response']['usage']['prompt_tokens']}):
                raise ValueError('architecture prefill does not cover all 48 layers and prompt tokens')
            reports[ordinal]['architecture_phases'] = dict(phases)
        else:
            lines = [line for line in text.splitlines() if 'strata prefill timing:' in line]
            headers = [re.search(r'prefill timing: (\d+) tokens, GPU timeline ([0-9.]+) ms, '
                        r'wall ([0-9.]+) ms, host staging ([0-9.]+) ms:', line) for line in lines]
            headers = [m for m in headers if m]
            if not headers: raise ValueError('missing installed Strata prefill timer')
            phases_ms = Counter()
            for line in lines:
                for name, milliseconds in re.findall(r' ([a-z+ /]+) ([0-9.]+) \([0-9.]+%\)', line):
                    phases_ms[name.strip()] += float(milliseconds)
            decode_lines=[line for line in text.splitlines() if
                'strata decode timing:' in line or 'strata decode GPU stages' in line]
            if not decode_lines: raise ValueError('missing installed Strata decode timer')
            reports.append(dict(index=request['index'], phase=request['phase'],
                usage=request['response']['usage'], wall_seconds=request['wall_seconds'],
                response_timing=request['response'].get('timings'), output_hash=request['output_sha256'],
                timer_tokens=sum(int(m[1]) for m in headers),
                gpu_timeline_ms=sum(float(m[2]) for m in headers),
                timer_wall_ms=sum(float(m[3]) for m in headers),
                host_staging_reported_ms=[float(m[4]) for m in headers],
                phases_ms=dict(phases_ms), timer_lines=lines,
                decode_lines=decode_lines,
                native_work_lines=[line for line in text.splitlines() if
                    'strata serve:' in line and any(s in line for s in
                        ('prompt ', 'cache hit rate', 'expert tiers:', 'KV streaming:', 'resident RAM:'))],
                limits=['Strata timer intervals include stream waits and host gaps.',
                        'Phase totals are rounded to milliseconds by the installed engine.',
                        'Host staging can be cumulative; do not sum across requests.',
                        'Timer tokens are native prefill work, separately recorded from frontend usage.']))
    return reports

def moe_stream_breakdown(folder):
    """Request/phase-owned stream dependencies; never infer whole-device idle."""
    path=folder/'server.log'
    data=path.read_bytes() if path.exists() else gzip.open(str(path)+'.gz','rb').read()
    requests=[json.loads(l) for l in (folder/'requests.jsonl').read_text().splitlines()]
    reports=[]
    for request in requests:
        begin,end=request['native_log_start_offset'],request['native_log_end_offset']
        if not 0<=begin<end<=len(data):raise ValueError('invalid stream request byte window')
        records=[]
        for line in data[begin:end].decode().splitlines():
            if line.startswith('{'):
                try:records.append(json.loads(line))
                except json.JSONDecodeError:continue
        expected=defaultdict(Counter)
        native_phases=defaultdict(lambda:dict(rounds=0,counters=Counter(),host_spans_us=Counter(),worker_jobs=Counter(),worker_us=Counter()))
        prompt_columns=0
        for row in records:
            if row.get('kind')!='round':continue
            if row['status']!='ok':raise ValueError('failed execution in stream request')
            context=row['context']
            key=(context['executor'],context['transaction'],row['phase'])
            native=native_phases[row['phase']]
            native['rounds']+=1
            if row['phase']=='prefill':prompt_columns+=context['input_columns']
            for layer in row['layers']:
                expected[key].update(layer.get('counters',{}))
                native['counters'].update(layer.get('counters',{}))
                native['host_spans_us'].update(layer.get('host_us',{}))
                for worker in layer.get('workers',[]):
                    if not isinstance(worker,dict):continue
                    for name in ('full_jobs','gate_up_jobs','down_jobs'):
                        if name in worker:native['worker_jobs'][name]+=worker[name]
                    for name in ('full_us','gate_up_us','down_us'):
                        if name in worker:native['worker_us'][name]+=worker[name]
        if prompt_columns!=request['response']['usage']['prompt_tokens']:
            raise ValueError('stream report does not cover complete prompt')
        observed=defaultdict(Counter)
        phases=defaultdict(lambda:dict(experts=0,routes=0,gemm_experts=0,gemm_routes=0,
            copy_ms=0,compute_wait_ms=0,kernel_interval_ms=0,route_buckets=[0]*6))
        stream_rows=[r for r in records if r.get('kind')=='expert_stream_timing']
        if not stream_rows:raise ValueError('missing phase-owned stream timing')
        for row in stream_rows:
            context=row.get('context') or {}
            key=(context.get('executor'),context.get('transaction'),context.get('phase'))
            if key not in expected:raise ValueError('stream timing has no matching execution owner')
            if (sum(row['route_buckets'])!=row['experts'] or
                sum(row['route_buckets'][1:])!=row['gemm_experts'] or
                not 0<=row['gemm_routes']<=row['routes']):
                raise ValueError('stream group geometry does not conserve experts/routes')
            observed[key].update(routes=row['routes'],experts=row['experts'])
            dst=phases[key[2]]
            for name in ('experts','routes','gemm_experts','gemm_routes','copy_ms',
                         'compute_wait_ms','kernel_interval_ms'):
                value=row[name]
                if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                    raise ValueError('invalid stream measurement')
                dst[name]+=value
            for i,value in enumerate(row['route_buckets']):dst['route_buckets'][i]+=value
        slot_bytes=(2_764_808+255)//256*256
        for key,counters in expected.items():
            if (observed[key]['routes']!=counters.get('nonresident_gpu_routes',0) or
                observed[key]['experts']*slot_bytes!=counters.get('expert_h2d_bytes',0)):
                raise ValueError('phase-owned stream routes/weight bytes do not conserve native work')
        for phase,dst in phases.items():
            dst['weight_h2d_bytes']=dst['experts']*slot_bytes
            dst['modeled_fp16_expansion_write_bytes']=dst['gemm_experts']*(1280*2560+2560*640)*2
            dst['useful_gemm_flops']=dst['gemm_routes']*2*(1280*2560+2560*640)
        reports.append(dict(index=request['index'],phase=request['phase'],
            usage=request['response']['usage'],output_hash=request['output_sha256'],
            speculative=request.get('native_speculative'),wall_seconds=request['wall_seconds'],
            native_timing=request['native_timing'],stream_phases=dict(phases),
            native_phases=dict(native_phases),
            prefill_wait_pct_of_instrumented_native=100*phases['prefill']['compute_wait_ms']/
                (request['native_timing']['prefill']*1000),
            route_bucket_bounds=['1-31','32-63','64-127','128-255','256-511','512+'],
            limits=['Compute-stream dependency wait is not whole-GPU idle or an uninstrumented critical path.',
                    'Copy and compute intervals overlap; do not add them into wall time.',
                    'Kernel interval includes expansion, GEMM/SIMT, joins and submission gaps.',
                    'Expansion write bytes and useful FLOPs are modeled work, not hardware counters.',
                    'Host scopes are inclusive and worker durations overlap; neither sums to request latency.',
                    'Ring event collection and stage/SV0 telemetry perturb timing.']))
    return reports


def compress_logs(folder):
    for path in folder.glob('*.log'):
        with path.open('rb') as src, gzip.open(str(path)+'.gz', 'wb', compresslevel=1) as dst:
            import shutil
            shutil.copyfileobj(src, dst)
        path.unlink()


def qualify_full_qsa_dispatch(work):
    events=work['qsa_score_dispatch']
    prompt=[e for e in events if e.get('tokens')==7111]
    if len(prompt)!=12 or not all(e.get('fp8') and e.get('full_mma') for e in prompt):
        raise ValueError('full QSA prompt dispatch coverage changed')
    if sum(bool(e.get('full_mma')) for e in events)!=12:
        raise ValueError('full MMA escaped the twelve prompt dispatches')
    comparisons=[r for r in work['qsa_real_comparison'] if r.get('tokens')==7111]
    if len(comparisons)!=12 or not all(r.get('full_mma') and r.get('sample_oracle_pass') for r in comparisons):
        raise ValueError('full QSA real-layer independent oracle admission failed')
    return dict(prompt_events=12, full_mma=12, sampled_queries=36, sampled_heads=864,
                real_layer_oracles=comparisons)


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
    p.add_argument('--final-comparison', action='store_true',
                   help='step 5: qualified adaptive156 versus installed Strata; larger-context feasibility')
    p.add_argument('--prefill-breakdown', action='store_true',
                   help='six-step plan step 1: paired qualified adaptive156 timing and stage attribution')
    p.add_argument('--controlled-comparison', action='store_true',
                   help='step 6 initial serving controls: 8K capacity, 16-bit KV, 2048 chunk, MTP2, physical cores')
    p.add_argument('--controlled-strata-followup', action='store_true',
                   help='correct Strata byte budget/draft controls; reuse valid NInfer comparison reference')
    p.add_argument('--dense-prefill-gemm-ab', action='store_true',
                   help='Step 2: qualified QPN versus full-chunk FP16 dense projection GEMM')
    p.add_argument('--dense-baseline-reference', type=Path)
    p.add_argument('--moe-work-breakdown', action='store_true',
                   help='Step 3: phase-owned stream waits/group geometry with dense GEMM opt-in')
    p.add_argument('--expert-pair-expand-ab', action='store_true',
                   help='Step 3: scalar/two-launch versus packed-pair/fused expert expansion')
    p.add_argument('--expert-pair-expand-reference', type=Path)
    p.add_argument('--context-residency-ab', action='store_true',
                   help='Step 4: static64 versus allocator-clipped adaptive156 at 262K BF16 capacity')
    p.add_argument('--fp8-long-context', action='store_true',
                   help='Step 5: qualified FP8 KV calibration then actual 128K/near-262K prompts')
    p.add_argument('--large-prefill-race', action='store_true',
                   help='Renewed optimization: 2048/8192 NInfer chunks and contemporary native Strata')
    p.add_argument('--large-profile-breakdown', action='store_true',
                   help='Attribute the tested 8192/FP8/262K/dense+fused profile with paired reference and diagnostics')
    p.add_argument('--qsa-full-mma-race', action='store_true',
                   help='Independent real-input admission of full QK/PV MMA then best-profile practical race')
    p.add_argument('--hyper-cublas-race', action='store_true',
                   help='Qualified staged-FP16 cuBLAS hyper projections versus tiled Volta MMA and native Strata')
    p.add_argument('--shared-cublas-race', action='store_true',
                   help='Source-directed shared expert SIMT versus large cuBLAS projections and native Strata')
    p.add_argument('--qsa-late-half-race', action='store_true',
                   help='Bounded 8192-profile SIMT versus admitted late-half score-MMA and native Strata')
    p.add_argument('--pinned-expert-source-race', action='store_true',
                   help='Best-profile staged-copy versus page-locked direct expert-source DMA and native Strata')
    p.add_argument('--architecture-breakdown', action='store_true',
                   help='Source-directed shared/routed MoE attribution plus installed Strata prefill/decode timers')
    p.add_argument('--comparison-reference', type=Path)
    a = p.parse_args()
    if a.large_profile_breakdown:
        a.moe_work_breakdown=True
    if a.controlled_strata_followup:
        if a.comparison_reference is None:
            p.error('--controlled-strata-followup requires --comparison-reference')
        a.controlled_comparison = True
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
    if a.context_residency_ab or a.fp8_long_context or a.large_prefill_race or a.large_profile_breakdown or a.qsa_late_half_race or a.pinned_expert_source_race or a.architecture_breakdown or a.qsa_full_mma_race or a.hyper_cublas_race or a.shared_cublas_race:
        a.ninfer_flags = ab.option(a.ninfer_flags, '--max-context', '262144')
        a.ninfer_flags = ab.option(a.ninfer_flags, '--kv-capacity', '262144')
    if a.fp8_long_context or a.large_prefill_race or a.large_profile_breakdown or a.qsa_late_half_race or a.pinned_expert_source_race or a.architecture_breakdown or a.qsa_full_mma_race or a.hyper_cublas_race or a.shared_cublas_race:
        a.ninfer_flags = ab.option(a.ninfer_flags, '--kv-dtype', 'fp8')
        a.request_timeout = 5400
    if a.large_profile_breakdown or a.qsa_late_half_race or a.pinned_expert_source_race or a.architecture_breakdown or a.qsa_full_mma_race or a.hyper_cublas_race or a.shared_cublas_race:
        a.ninfer_flags=ab.option(a.ninfer_flags,'--prefill-chunk','8192')
    if a.pinned_expert_source_race or a.architecture_breakdown or a.qsa_full_mma_race or a.hyper_cublas_race or a.shared_cublas_race:
        a.ninfer_flags=[f for f in a.ninfer_flags if f!='--no-qsa-prefill-mma']+['--qsa-prefill-mma']
    a.long_prompt = LONG
    a.max_output_tokens = 64 if a.fp8_long_context else 128
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
    if a.controlled_comparison:
        # Installed serving rejects spec<2. Preserve that engine contract and
        # record this limitation; a no-spec native-route comparison remains separate.
        cfg = json.loads(a.strata_config.read_text())
        flags = cfg['args']
        if '--kv-resident' in flags:
            index = flags.index('--kv-resident'); del flags[index:index+2]
        for key, value in [('--max-context','8192'),('--kv','fp16'),('--prefill','2048'),
                           ('--spec','2'),('--expert-cache','6606')]:
            flags = ab.option(flags,key,value)
        if a.controlled_strata_followup:
            # Strata count is multiplied by MAX native blob size before ranked
            # smaller blobs are packed. 5184*3993600 nearly matches NInfer bytes.
            for key,value in [('--expert-cache','5184'),('--spec','3'),
                              ('--mtp-max-t','3'),('--suffix-draft','0'),('--spec-min-p','0')]:
                flags = ab.option(flags,key,value)
        cfg['args'] = flags
        path = a.output/'controlled-strata-config.json'
        path.write_text(json.dumps(cfg,indent=2)+'\n')
        a.strata_config = path
    dense_arms=[('ninfer',name,dict(residency_arms()[1][2],NINFER_V100_FP8_PREFILL_GEMM=value))
              for name,value in [('dense-qpn','0'),('dense-full-chunk','1')]]
    pair_arms=[('ninfer',name,dict(dense_arms[1][2],NINFER_V100_EXPERT_PAIR_EXPAND=value))
               for name,value in [('expand-scalar','0'),('expand-pair','1')]]
    context_arms=[(engine,arm,dict(flags,NINFER_V100_FP8_PREFILL_GEMM='1',
                                  NINFER_V100_EXPERT_PAIR_EXPAND='1'))
                  for engine,arm,flags in residency_arms()]
    race_arms=[('ninfer','chunk2048',context_arms[1][2]),
               ('ninfer','chunk8192',context_arms[1][2]),ARMS[1]]
    qsa_arms=[('ninfer','qsa-simt',dict(context_arms[1][2],NINFER_V100_QSA_SCORE_MMA='0')),
              ('ninfer','qsa-late-half-mma',dict(context_arms[1][2],NINFER_V100_QSA_SCORE_MMA='1',
                                                NINFER_V100_QSA_SCORE_MMA_MIN_QSA='6')),
              ARMS[1]]
    pinned_base=dict(context_arms[1][2],NINFER_V100_QSA_SCORE_MMA='1',
                     NINFER_V100_QSA_SCORE_MMA_MIN_QSA='6',
                     NINFER_FLASH_NEXT_EXPERT_CACHE_MAX_SLOTS='102',
                     NINFER_V100_EXPERT_SOURCE_PINNED='0')
    pinned_arms=[('ninfer','staged-source-copy',pinned_base),
                 ('ninfer','pinned-source-dma',dict(pinned_base,NINFER_V100_EXPERT_SOURCE_PINNED='1')),
                 ARMS[1]]
    full_arms=[('ninfer','qsa-late-half-pinned',dict(pinned_arms[1][2],NINFER_V100_QSA_FULL_MMA='0')),
               ('ninfer','qsa-full-mma-pinned',dict(pinned_arms[1][2],NINFER_V100_QSA_FULL_MMA='1')),ARMS[1]]
    hyper_arms=[('ninfer','hyper-tiled-mma',dict(full_arms[1][2],NINFER_V100_HYPER_CUBLAS='0')),
                ('ninfer','hyper-staged-cublas',dict(full_arms[1][2],NINFER_V100_HYPER_CUBLAS='1')),ARMS[1]]
    shared_arms=[('ninfer',name,dict(hyper_arms[1][2],NINFER_V100_SHARED_CUBLAS=value))
                 for name,value in [('shared-simt','0'),('shared-staged-cublas','1')]]+[ARMS[1]]
    manifest_arms = (shared_arms if a.shared_cublas_race else hyper_arms if a.hyper_cublas_race else full_arms if a.qsa_full_mma_race else pinned_arms if a.pinned_expert_source_race else qsa_arms if a.qsa_late_half_race else [context_arms[1]] if a.large_profile_breakdown else race_arms if a.large_prefill_race else [context_arms[1]] if a.fp8_long_context else context_arms if a.context_residency_ab else pair_arms if a.expert_pair_expand_ab else [dense_arms[1]] if a.moe_work_breakdown else dense_arms if a.dense_prefill_gemm_ab else [('ninfer', 'adaptive156-qualified', residency_arms()[1][2])] if a.prefill_breakdown else
                     [default_arm] if a.qualified_default_smoke else
                     [("ninfer", "adaptive156-qualified", residency_arms()[1][2]), ARMS[1]] if (a.final_comparison or a.controlled_comparison) else
                     residency_arms() if a.residency_gemm_ab else
                     [('ninfer', backend, flags) for backend, flags in gemm_arms]
                     if (a.expert_gemm_ab or a.resident_gemm_ab or a.cpu_stream_overlap_ab) else ARMS)
    if a.architecture_breakdown:
        manifest_arms=[shared_arms[1], ARMS[1]]
    (a.output/'manifest.json').write_text(json.dumps(dict(prompt=None if a.fp8_long_context else LONG, short_prompt=ab.SHORT,
        prompt_definition='Serving-calibrated repeated prose; actual per-cell payloads in requests.jsonl' if a.fp8_long_context else 'fixed 360 numbered records',
        output_limit=32 if (a.stream_attribution_only or a.qualified_default_smoke) else a.max_output_tokens,
        physical_cpus=cpus,
        arms=[ARMS[0]] if a.stream_attribution_only else manifest_arms,
        fresh_servers_per_arm=2 if a.architecture_breakdown else 4 if a.fp8_long_context else 2 if (a.prefill_breakdown or a.moe_work_breakdown) else 1 if (a.stream_attribution_only or a.qualified_default_smoke) else 3,
        moe_work_breakdown=a.moe_work_breakdown,
        expert_pair_expand_ab=a.expert_pair_expand_ab, context_residency_ab=a.context_residency_ab,
        fp8_long_context=a.fp8_long_context, large_prefill_race=a.large_prefill_race,
        large_profile_breakdown=a.large_profile_breakdown,
        qsa_late_half_race=a.qsa_late_half_race, qsa_full_mma_race=a.qsa_full_mma_race,
        hyper_cublas_race=a.hyper_cublas_race,
        shared_cublas_race=a.shared_cublas_race,
        pinned_expert_source_race=a.pinned_expert_source_race,
        architecture_breakdown=a.architecture_breakdown,
        request_timeout=getattr(a,'request_timeout',900),
        prefill_breakdown=a.prefill_breakdown, dense_prefill_gemm_ab=a.dense_prefill_gemm_ab,
        stream_attribution_only=a.stream_attribution_only, expert_gemm_ab=a.expert_gemm_ab, resident_gemm_ab=a.resident_gemm_ab, cpu_stream_overlap_ab=a.cpu_stream_overlap_ab,
        qualified_default_smoke=a.qualified_default_smoke,
        residency_gemm_ab=a.residency_gemm_ab, final_comparison=a.final_comparison,
        controlled_comparison=a.controlled_comparison,
        controlled_strata_followup=a.controlled_strata_followup,
        timing_telemetry_level=0, diagnostic_telemetry_level=0 if a.architecture_breakdown else 2,
        cache_budget='actual allocations logged; requested maxima are not equal-memory controls',
        cold_definition='first request after fresh server; host file cache preserved',
        cross_quant_comparison='same represented artifact; exact-output expansion work reduction' if a.expert_pair_expand_ab else 'same represented artifact; cache residency can change CPU/GPU arithmetic' if a.context_residency_ab else 'same represented artifact; arithmetic-changing expert route' if (a.expert_gemm_ab or a.resident_gemm_ab or a.cpu_stream_overlap_ab or a.residency_gemm_ab or a.prefill_breakdown or a.dense_prefill_gemm_ab or a.moe_work_breakdown)
            else 'descriptive; source artifacts and frontend token counts differ',
        ninfer_flags=a.ninfer_flags), indent=2)+'\n')
    cells = []
    def run(engine, arm, overrides, tag, mode):
        if ab.gpu_free_mib() < 28000: raise RuntimeError('GPU occupied; no processes were stopped')
        # Preserve each engine's evidenced placement: NInfer physical-core mask,
        # Strata native inherited CPU set. Record both rather than retune Strata.
        a.launch_prefix = ninfer_placement if (engine == 'ninfer' or a.controlled_comparison) else []
        try:
            result = ab.run_case(a, engine, tag, overrides)
            folder = a.output/(engine+'-'+tag)
            cell = dict(engine=engine, arm=arm, mode=mode, folder=str(folder), result=result)
            if a.residency_gemm_ab or a.context_residency_ab or a.fp8_long_context:
                cell['residency'] = residency_observation(folder, a.max_output_tokens,
                    None if a.fp8_long_context else 7111)
            if a.final_comparison or a.prefill_breakdown or a.controlled_comparison or a.dense_prefill_gemm_ab or a.moe_work_breakdown or a.expert_pair_expand_ab or a.large_prefill_race or a.qsa_late_half_race or a.pinned_expert_source_race or a.architecture_breakdown or a.qsa_full_mma_race or a.hyper_cublas_race or a.shared_cublas_race:
                samples=[json.loads(line) for line in (folder/'system.jsonl').read_text().splitlines()]
                peaks=[r.get('gpu',{}).get('metrics',{}).get('memory_used_bytes') for r in samples
                       if r.get('kind')=='system_sample']
                cell['sampled_peak_gpu_bytes']=max((v for v in peaks if v is not None),default=None)
                if engine=='ninfer':
                    cell['residency']=residency_observation(folder,a.max_output_tokens,
                                                         7111 if mode=='timing' else None)
                    if a.pinned_expert_source_race or a.architecture_breakdown or a.qsa_full_mma_race or a.hyper_cublas_race or a.shared_cublas_race:
                        log=(folder/'server.log').read_text(errors='replace')
                        match=re.search(r'v100\.expert_source_pinned=1 ranges=(\d+) registered_bytes=(\d+)',log)
                        cell['expert_source_registration']=(dict(ranges=int(match.group(1)),
                            registered_bytes=int(match.group(2))) if match else None)
            if mode == 'diagnostic' and a.architecture_breakdown:
                cell['architecture_requests'] = architecture_requests(folder, engine)
                native_log=folder/('server.log' if engine=='ninfer' else 'native-engine.log')
                cell['architecture_startup']=[line for line in native_log.read_text().splitlines()
                    if any(s in line for s in ('"runtime_memory"', 'strata generate:',
                        'FileExpertSource:', 'prompt chunk auto:', 'prompt path borrows',
                        'prompt kernels:', 'prompt experts on', 'strata hc:', 'v100.expert_source_pinned='))]
            elif mode == 'diagnostic':
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
    if a.architecture_breakdown:
        paired = [(shared_arms[1], dict(NINFER_V100_MOE_ARCH_TIMING='1',
                      NINFER_FLASH_NEXT_STAGE_LEDGER='1', NINFER_FLASH_NEXT_STAGE_LEDGER_JSON='1')),
                  (ARMS[1], dict(STRATA_PREFILL_TIMING='1', STRATA_DECODE_TIMING='1'))]
        report = dict(status='in_progress', telemetry_level=0, comparisons=[],
            objective='Explain the practical serving gap using source-directed whole-request attribution',
            limits=['One clean and one diagnostic fresh server per engine; attribution, not a new win claim.',
                    'Different weights, KV representation, residency and speculative algorithms remain.',
                    'CUDA intervals include waits/host gaps; host spans must not be added to them.',
                    'No per-layer expert stream drain or worker telemetry; overhead is still measured.',
                    'The current admitted full-QSA, hyper-cuBLAS and shared-cuBLAS routes are included.',
                    'Packed-reduce candidate is operator-admitted but is not part of this audit.'])
        output = a.output/'architecture-breakdown-report.json'
        output.write_text(json.dumps(report, indent=2)+'\n')
        for (engine, arm, flags), diagnostics in paired:
            if engine=='ninfer': flags=dict(flags,NINFER_V100_EXPERT_PACKED_REDUCE='0')
            run(engine, arm, flags, 'architecture-reference', 'timing')
            run(engine, arm, dict(flags, **diagnostics), 'architecture-diagnostic', 'diagnostic')
            before, after = cells[-2:]
            requests = [[json.loads(l) for l in (Path(c['folder'])/'requests.jsonl').read_text().splitlines()]
                        for c in (before, after)]
            if any(len(rows) != 6 for rows in requests):
                raise ValueError('incomplete architecture schedule')
            comparisons = []
            for b, d in zip(*requests):
                for row in (b,d):
                    usage=row['response']['usage']
                    if (usage['prompt_tokens']!=7111 or usage['completion_tokens']!=128
                            or usage.get('prompt_tokens_details',{}).get('cached_tokens',0)!=0
                            or row['response']['choices'][0]['finish_reason']!='length'):
                        raise ValueError('architecture serving workload changed')
                def speculative(row):
                    if engine=='ninfer': return row.get('native_speculative')
                    timings=row['response'].get('timings') or {}
                    return {k:timings[k] for k in ('draft_n','draft_n_accepted') if k in timings}
                baseline_spec, diagnostic_spec=speculative(b),speculative(d)
                exact = (b['output_sha256'] == d['output_sha256'] and
                         b['response']['usage'] == d['response']['usage'] and
                         baseline_spec == diagnostic_spec)
                comparisons.append(dict(index=b['index'], phase=b['phase'], exact_output_usage_mtp=exact,
                    http_overhead_seconds=d['wall_seconds']-b['wall_seconds'],
                    baseline_speculative_work=baseline_spec, diagnostic_speculative_work=diagnostic_spec,
                    baseline_native_timing=b.get('native_timing') or b['response'].get('timings'),
                    diagnostic_native_timing=d.get('native_timing') or d['response'].get('timings'),
                    baseline_output_hash=b['output_sha256'], diagnostic_output_hash=d['output_sha256']))
            comparison = dict(engine=engine, requests=comparisons,
                diagnostic=after['architecture_requests'],
                startup=after['architecture_startup'],
                residency=[c.get('residency') for c in (before, after)],
                sampled_peak_gpu_bytes=[c.get('sampled_peak_gpu_bytes') for c in (before, after)])
            report['comparisons'].append(comparison)
            output.write_text(json.dumps(report, indent=2)+'\n')
            if engine == 'ninfer' and not all(r['exact_output_usage_mtp'] for r in comparisons):
                raise ValueError('lightweight instrumentation changed NInfer corresponding output/usage/MTP')
        report['status'] = 'complete'
        report['reference_timing'] = summarize(cells)
        output.write_text(json.dumps(report, indent=2)+'\n')
        print('Architecture attribution complete; no arithmetic candidate/default change.', flush=True)
        return
    if a.shared_cublas_race:
        # Independent shared-expert gate/up, represented activation, down and
        # merged-output oracles plus actual-shape operator admission precede this.
        for repeat in range(3):
            for engine,arm,flags in shared_arms[repeat:]+shared_arms[:repeat]:
                run(engine,arm,flags,arm+'-r'+str(repeat),'timing')
                cell=cells[-1]
                folder=Path(cell['folder'])
                rows=[json.loads(l) for l in (folder/'requests.jsonl').read_text().splitlines()]
                if [r['phase'] for r in rows]!=['cold']+['warmup']*4+['measured']:
                    raise ValueError('shared cuBLAS race request schedule incomplete')
                if any(r['response']['usage'].get('prompt_tokens')!=7111 or
                       r['response']['usage']['completion_tokens']!=128 or
                       r['response']['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)!=0
                       for r in rows):
                    raise ValueError('shared cuBLAS race prompt/output/reuse work changed')
                if engine=='ninfer':
                    marker='v100.shared_cublas=1 tokens=7111 gate_up_and_down=1'
                    log=folder/'server.log'
                    text=log.read_text() if log.exists() else gzip.open(str(log)+'.gz','rt').read()
                    selected=marker in text
                    cell['shared_cublas_selected']=selected
                    if selected!=(arm=='shared-staged-cublas'):
                        raise ValueError('shared cuBLAS real serving dispatch changed')
        residency=[c['residency'] for c in cells if c['engine']=='ninfer']
        if len({r['actual_cache']['slots_per_layer'] for r in residency})!=1:
            raise ValueError('shared projection arms did not retain matched expert residency')
        report=dict(status='shared_cublas_race_complete',timing=summarize(cells),
            observations=[dict(engine=c['engine'],arm=c['arm'],peak=c.get('sampled_peak_gpu_bytes'),
                               residency=c.get('residency'),selected=c.get('shared_cublas_selected')) for c in cells],
            limits=['Independent affected operator qualification does not close historical full-model drift.',
                    'Shared projections remain opt-in; cross-arm output and MTP work can change.',
                    'Strata weights, KV codec, residency and speculative arithmetic differ; practical comparison is not matched quality.'])
        (a.output/'shared-cublas-race-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        return
    if a.hyper_cublas_race:
        # The target-GPU hyper fixture has already admitted the represented-BF16
        # arithmetic and actual 7,111-token operator shape. This branch collects
        # only clean request-level evidence on the otherwise identical best profile.
        for repeat in range(3):
            for engine,arm,flags in hyper_arms[repeat:]+hyper_arms[:repeat]:
                run(engine,arm,flags,arm+'-r'+str(repeat),'timing')
                rows=[json.loads(l) for l in (Path(cells[-1]['folder'])/'requests.jsonl').read_text().splitlines()]
                if [r['phase'] for r in rows]!=['cold']+['warmup']*4+['measured']:
                    raise ValueError('hyper cuBLAS race request schedule incomplete')
                if any(r['response']['usage']['completion_tokens']!=128 or
                       r['response']['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)!=0
                       for r in rows):
                    raise ValueError('hyper cuBLAS race output/reuse work changed')
        report=dict(status='hyper_cublas_race_complete',timing=summarize(cells),
            observations=[dict(engine=c['engine'],arm=c['arm'],peak=c.get('sampled_peak_gpu_bytes'),
                               residency=c.get('residency')) for c in cells],
            limits=['Independent hyper operator qualification does not close historical full-model drift.',
                    'The cuBLAS hyper backend is explicitly opt-in; cross-arm output and MTP work can change.',
                    'Strata differs in weights, KV codec, residency and speculative work; practical comparison is not matched quality.'])
        (a.output/'hyper-cublas-race-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        return
    if a.qsa_full_mma_race:
        # Real represented query/KV/selection qualification for every prompt QSA
        # layer precedes clean timing. Independent complete FP64 formula for all
        # heads at beginning/middle/end; unchanged public BF16 output thresholds.
        a.request_schedule=[('cold','long')]
        a.max_output_tokens=32
        a.telemetry_level=2
        diagnostic=dict(full_arms[1][2],NINFER_V100_TELEMETRY='1',
                        NINFER_FLASH_NEXT_STAGE_LEDGER='1',NINFER_V100_QSA_SCORE_COMPARE='1')
        run('ninfer','qsa-full-mma-pinned',diagnostic,'qsa-full-mma-admission','diagnostic')
        admission=qualify_full_qsa_dispatch(cells[-1]['work'])
        a.request_schedule=[('cold','long')]+[('warmup','long')]*4+[('measured','long')]
        a.max_output_tokens=128
        a.telemetry_level=0
        for repeat in range(3):
            for engine,arm,flags in full_arms[repeat:]+full_arms[:repeat]:
                run(engine,arm,flags,arm+'-r'+str(repeat),'timing')
                rows=[json.loads(l) for l in (Path(cells[-1]['folder'])/'requests.jsonl').read_text().splitlines()]
                if [r['phase'] for r in rows]!=['cold']+['warmup']*4+['measured']:
                    raise ValueError('full QSA race request schedule incomplete')
                if any(r['response']['usage']['completion_tokens']!=128 or
                       r['response']['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)!=0
                       for r in rows):
                    raise ValueError('full QSA race output/reuse work changed')
        report=dict(status='qsa_full_mma_race_complete',timing=summarize(cells),admission=admission,
            observations=[dict(engine=c['engine'],arm=c['arm'],peak=c.get('sampled_peak_gpu_bytes'),
                               residency=c.get('residency')) for c in cells if c['mode']=='timing'],
            limits=['Independent operator and sampled real-layer admission do not close historical full-model drift.',
                    'Full QK/PV MMA is explicitly opt-in; cross-arm output and MTP work can change.',
                    'Strata differs in weights, KV codec, residency and speculative work; practical comparison is not matched quality.'])
        (a.output/'qsa-full-mma-race-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        return
    if a.qsa_late_half_race:
        # The late-half scope was admitted by the existing 4,096-position paired
        # numerical screen. Confirm that this exact serving profile actually
        # dispatches six MMA and six SIMT QSA layers before collecting clean timings.
        original_flags=list(a.ninfer_flags)
        a.ninfer_flags=[f for f in original_flags if f!='--no-qsa-prefill-mma']+['--qsa-prefill-mma']
        a.request_schedule=[('cold','long')]
        a.max_output_tokens=32
        a.telemetry_level=2
        diagnostic=dict(qsa_arms[1][2],NINFER_V100_TELEMETRY='1',
                        NINFER_FLASH_NEXT_STAGE_LEDGER='1')
        run('ninfer','qsa-late-half-mma',diagnostic,'qsa-late-half-dispatch','diagnostic')
        events=cells[-1]['work']['qsa_score_dispatch']
        prompt_events=[e for e in events if e.get('tokens')==7111]
        if (len(prompt_events),sum(bool(e.get('mma')) for e in prompt_events))!=(12,6):
            raise ValueError(f'late-half prompt dispatch coverage changed: {len(prompt_events)} '
                             f'7111-token events, {sum(bool(e.get("mma")) for e in prompt_events)} MMA')
        if not all(e.get('fp8') for e in prompt_events):
            raise ValueError('late-half prompt dispatch did not use FP8 KV')
        if sum(bool(e.get('mma')) for e in events)!=6:
            raise ValueError('MMA escaped the admitted six late-half prompt dispatches')
        a.request_schedule=[('cold','long')]+[('warmup','long')]*4+[('measured','long')]
        a.max_output_tokens=128
        a.telemetry_level=0
        for repeat in range(3):
            for engine,arm,flags in qsa_arms[repeat:]+qsa_arms[:repeat]:
                run(engine,arm,flags,arm+'-r'+str(repeat),'timing')
                rows=[json.loads(l) for l in (Path(cells[-1]['folder'])/'requests.jsonl').read_text().splitlines()]
                if [r['phase'] for r in rows]!=['cold']+['warmup']*4+['measured']:
                    raise ValueError('QSA race request schedule incomplete')
                if any(r['response']['usage']['completion_tokens']!=128 or
                       r['response']['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)!=0
                       for r in rows):
                    raise ValueError('QSA race output/reuse work changed')
        report=dict(status='qsa_late_half_race_complete',timing=summarize(cells),
            dispatch=dict(events=len(events),mma=sum(bool(e['mma']) for e in events),
                          simt=sum(not bool(e['mma']) for e in events),
                          prompt_events=len(prompt_events),
                          prompt_mma=sum(bool(e['mma']) for e in prompt_events),
                          prompt_simt=sum(not bool(e['mma']) for e in prompt_events)),
            observations=[dict(engine=c['engine'],arm=c['arm'],peak=c.get('sampled_peak_gpu_bytes'),
                               residency=c.get('residency')) for c in cells if c['mode']=='timing'],
            limits=['The unchanged late-half candidate has bounded prior paired numerical admission, not default-on qualification.',
                    'Cross-arm output and MTP work may differ because score arithmetic changes.',
                    'Native Strata uses different weights, KV codec and speculative algorithm; this is practical, not matched-quality performance.'])
        (a.output/'qsa-late-half-race-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        return
    if a.pinned_expert_source_race:
        for repeat in range(3):
            for engine,arm,flags in pinned_arms[repeat:]+pinned_arms[:repeat]:
                run(engine,arm,flags,arm+'-r'+str(repeat),'timing')
                rows=[json.loads(l) for l in (Path(cells[-1]['folder'])/'requests.jsonl').read_text().splitlines()]
                if [r['phase'] for r in rows]!=['cold']+['warmup']*4+['measured']:
                    raise ValueError('pinned expert source race request schedule incomplete')
                if any(r['response']['usage']['completion_tokens']!=128 or
                       r['response']['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)!=0
                       for r in rows):
                    raise ValueError('pinned expert source race output/reuse work changed')
                if engine=='ninfer':
                    observed=cells[-1]['expert_source_registration'] is not None
                    if observed != (arm=='pinned-source-dma'):
                        raise ValueError(f'expert source registration evidence changed for {arm}: {observed}')
        for repeat in range(3):
            by_arm={c['arm']:c for c in cells if c['mode']=='timing' and
                    c['arm'] in ('staged-source-copy','pinned-source-dma') and
                    c['folder'].endswith('-r'+str(repeat))}
            baseline=[json.loads(l) for l in (Path(by_arm['staged-source-copy']['folder'])/'requests.jsonl').read_text().splitlines()]
            candidate=[json.loads(l) for l in (Path(by_arm['pinned-source-dma']['folder'])/'requests.jsonl').read_text().splitlines()]
            for left,right in zip(baseline,candidate,strict=True):
                if (left['phase'],left['output_sha256'],left['response']['usage'],left['native_speculative']) != \
                   (right['phase'],right['output_sha256'],right['response']['usage'],right['native_speculative']):
                    raise ValueError(f'pinned source changed output or native work at repeat {repeat} '
                                     f'ordinal {left["index"]}')
        report=dict(status='pinned_expert_source_race_complete',timing=summarize(cells),
            observations=[dict(engine=c['engine'],arm=c['arm'],peak=c.get('sampled_peak_gpu_bytes'),
                               residency=c.get('residency'),
                               expert_source_registration=c.get('expert_source_registration')) for c in cells],
            limits=['The page-locked expert source route is experimental and remains default-off.',
                    'Both NInfer arms are capped at 102 resident experts per layer to match the registered-source allocation.',
                    'Corresponding NInfer requests require exact output, usage and native MTP work parity.',
                    'Native Strata uses different weights, KV codec and speculative algorithm; this is practical, not matched-quality performance.'])
        (a.output/'pinned-expert-source-race-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        return
    if a.large_prefill_race:
        original=list(a.ninfer_flags)
        for repeat in range(3):
            # Rotate all three engines/arms; no historical control reuse.
            order=race_arms[repeat:]+race_arms[:repeat]
            for engine,arm,flags in order:
                chunk='8192' if arm=='chunk8192' else '2048'
                a.ninfer_flags=ab.option(original,'--prefill-chunk',chunk)
                run(engine,arm,flags,arm+'-r'+str(repeat),'timing')
                rows=[json.loads(l) for l in (Path(cells[-1]['folder'])/'requests.jsonl').read_text().splitlines()]
                if [r['phase'] for r in rows]!=['cold']+['warmup']*4+['measured']:
                    raise ValueError('large-prefill race schedule incomplete')
                if any(r['response']['usage']['completion_tokens']!=128 or
                       r['response']['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)!=0
                       for r in rows):
                    raise ValueError('large-prefill race output/reuse work changed')
        report=dict(status='large_prefill_race_complete',timing=summarize(cells),
            observations=[dict(engine=c['engine'],arm=c['arm'],peak=c['sampled_peak_gpu_bytes'],
                               residency=c.get('residency')) for c in cells],
            limits=['Native practical comparison: representations, KV codecs and MTP work differ.',
                    'Chunk size may change residency and floating-point/MTP behavior.',
                    'No arithmetic default promotion or full-model accuracy equivalence claim.'])
        (a.output/'large-prefill-race-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        return
    if a.fp8_long_context:
        # Calibrate repeated prose using this artifact's serving frontend, rather
        # than estimate large token counts from characters or a foreign tokenizer.
        paragraph='The solar station stores energy during daylight and supplies the village after sunset. '
        suffix='In at least 500 words, explain how the battery storage works and its limitations.'
        engine,arm,flags=context_arms[1]
        a.max_output_tokens=64
        report=dict(step=5,status='running',context_capacity=262144,kv_dtype='fp8',cells=[],
            limits=['Synthetic repeated-prose capacity screen, not a long-document quality test.',
                    'Single cold request at each full size; no throughput comparison claim.',
                    'Flash-Next QSA uses unscaled E4M3 storage; no arithmetic default promotion.'])
        path=a.output/'fp8-long-context-report.json'
        def save():
            path.write_text(json.dumps(report,indent=2)+'\n')
        def prompt_count(cell):
            rows=[json.loads(l) for l in (Path(cell['folder'])/'requests.jsonl').read_text().splitlines()]
            counts={r['response']['usage']['prompt_tokens'] for r in rows}
            if len(counts)!=1:raise ValueError('calibration prompt count changed between requests')
            return counts.pop()
        save()
        try:
            calibration=[]
            for records in (360,361):
                a.long_prompt=paragraph*records+suffix
                a.request_schedule=[('cold','long'),('measured','long')]
                run(engine,arm,flags,'fp8-calibrate-'+str(records),'capacity')
                cell=cells[-1]; cell['prompt_tokens']=prompt_count(cell)
                report['cells'].append(cell);calibration.append(cell['prompt_tokens']);save()
            slope=calibration[1]-calibration[0]
            if not 1<=slope<=64:raise ValueError('unexpected repeated-prose token increment')
            intercept=calibration[0]-360*slope
            report['calibration']=dict(tokens_per_record=slope,intercept=intercept)
            for target in (131072,262016):
                records=math.ceil((target-intercept)/slope)
                predicted=records*slope+intercept
                if predicted+64>=262144:raise ValueError('insufficient context headroom')
                a.long_prompt=paragraph*records+suffix
                a.request_schedule=[('cold','long')]
                run(engine,arm,flags,'fp8-prompt-'+str(target),'capacity')
                cell=cells[-1];cell.update(prompt_tokens=prompt_count(cell),target_prompt_tokens=target,
                                           predicted_prompt_tokens=predicted)
                report['cells'].append(cell);save()
                if cell['prompt_tokens']!=predicted or not target<=cell['prompt_tokens']<target+slope:
                    raise ValueError('actual large prompt tokens did not match calibrated target')
                if cell['prompt_tokens']+64>=262144:raise ValueError('actual context headroom failed')
            report['status']='full_prompt_capacity_complete';save()
        except Exception as error:
            report['status']='failed';report['error']=str(error);save()
            raise
        print(json.dumps(report,indent=2),flush=True)
        return
    if a.context_residency_ab:
        for repeat in range(3):
            for engine,arm,flags in (context_arms if repeat!=1 else list(reversed(context_arms))):
                run(engine,arm,flags,arm+'-262k-r'+str(repeat),'timing')
                folder=Path(cells[-1]['folder'])
                rows=[json.loads(l) for l in (folder/'requests.jsonl').read_text().splitlines()]
                if [r['phase'] for r in rows]!=['cold']+['warmup']*4+['measured']:
                    raise ValueError('context-residency schedule incomplete')
                log=gzip.open(folder/'server.log.gz','rt').read()
                if 'v100.expert_pair_expand=1' not in log:
                    raise ValueError('qualified fused expansion dispatch missing')
        report=dict(step=4,status='context_residency_screen_complete',timing=summarize(cells),
            context_capacity=262144,prompt_tokens=7111,output_tokens=128,
            observations=[dict(arm=c['arm'],**c['residency']) for c in cells],
            limits=['Combined cache policy/capacity screen; allocator reserve determines actual slots.',
                    'Full allocated 262K context, not a 262K prompt.',
                    'Outputs and native MTP work recorded; cache policy may change arithmetic.',
                    'Static64 remains default; adaptive residency remains opt-in.'])
        (a.output/'context-residency-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        return
    if a.expert_pair_expand_ab:
        if a.expert_pair_expand_reference:
            cells.append(reuse_pair_expand_baseline(a.expert_pair_expand_reference,a.output,
                pair_arms[0][2],cpus,a.ninfer_flags))
        for repeat in range(3):
            for engine,arm,flags in (pair_arms if repeat!=1 else list(reversed(pair_arms))):
                if repeat==0 and arm=='expand-scalar' and a.expert_pair_expand_reference:continue
                run(engine,arm,flags,arm+'-r'+str(repeat),'timing')
                folder=Path(cells[-1]['folder'])
                text=gzip.open(folder/'server.log.gz','rt').read()
                if ('v100.expert_pair_expand=1' in text)!=(arm=='expand-pair'):
                    raise ValueError('expert expansion dispatch does not match selected arm')
        signatures=defaultdict(set)
        for cell in cells:
            rows=[json.loads(l) for l in (Path(cell['folder'])/'requests.jsonl').read_text().splitlines()]
            if [r['phase'] for r in rows]!=['cold']+['warmup']*4+['measured']:
                raise ValueError('expert expansion A/B schedule incomplete')
            for row in rows:
                signatures[row['index']].add(json.dumps([row['output_sha256'],row['response']['usage'],
                    row.get('native_speculative')],sort_keys=True))
        if len(signatures)!=6 or any(len(s)!=1 for s in signatures.values()):
            raise ValueError('expert expansion changed corresponding output/usage/native MTP work')
        report=dict(step=3,status='pair_expansion_comparison_complete',timing=summarize(cells),
            exact_corresponding_output_usage_mtp=True,
            observations=[dict(arm=c['arm'],source_run=c.get('source_run'),reused=c.get('reused',False),
                               **c['residency']) for c in cells],
            limits=['First unchanged scalar control reused from run38102677923 when provided.',
                    'No production default promotion; request gain must justify retaining candidate.'])
        (a.output/'expert-pair-expand-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        return
    if a.dense_prefill_gemm_ab:
        if a.dense_baseline_reference:
            cells.append(reuse_dense_baseline(a.dense_baseline_reference,a.output))
            (a.output/'checkpoint.json').write_text(json.dumps(cells,indent=2)+'\n')
        for repeat in range(3):
            for engine,arm,flags in (dense_arms if repeat!=1 else list(reversed(dense_arms))):
                if repeat==0 and arm=='dense-qpn' and a.dense_baseline_reference:continue
                run(engine,arm,flags,arm+'-r'+str(repeat),'timing')
        report=dict(step=2,status='timing_complete',timing=summarize(cells),
            observations=[dict(arm=c['arm'],source_run=c.get('source_run'),reused=c.get('reused',False),**c['residency']) for c in cells],
            limits=['Same represented weights; arithmetic changes may alter output/routes/MTP work.',
                    'Request/native phase gains must be measured; operator gains are not request gains.',
                    'No default promotion without full-model numerical qualification.',
                    'First unchanged QPN control reused from run38098932339 when reference provided.'])
        path=a.output/'dense-prefill-gemm-report.json'
        path.write_text(json.dumps(report,indent=2)+'\n')
        a.max_output_tokens=32;a.telemetry_level=2
        a.request_schedule=[('diagnostic','long'),('diagnostic','long')]
        for engine,arm,flags in dense_arms:
            run(engine,arm,dict(flags,NINFER_FLASH_NEXT_STAGE_LEDGER='1'),arm+'-diagnostic','diagnostic')
            observed=cells[-1]['work']['dense_prefill_gemm']
            for n,k,layers in [(16384,2560,36),(2560,6144,48)]:
                rows=[r for r in observed if (r['n'],r['k'])==(n,k)]
                if (not rows or sum(r['tokens'] for r in rows)!=2*7111*layers or
                    any(r['expanded']!=(arm=='dense-full-chunk') for r in rows)):
                    raise ValueError('missing dense prefill dispatch or token/layer conservation')
        report['status']='complete'
        report['diagnostics']=[dict(arm=c['arm'],residency=c['residency'],work=c['work'])
                               for c in cells if c['mode']=='diagnostic']
        path.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report['timing'],indent=2),flush=True)
        print('Step 2 request comparison complete; arithmetic defaults unchanged.',flush=True)
        return
    if a.controlled_comparison:
        arms=([ARMS[1]] if a.controlled_strata_followup else
              [('ninfer','adaptive156-qualified',residency_arms()[1][2]),ARMS[1]])
        for repeat in range(3):
            for engine,arm,flags in (arms if repeat!=1 else list(reversed(arms))):
                run(engine,arm,flags,arm+'-controlled-r'+str(repeat),'timing')
        cache=[]
        for c in cells:
            folder=Path(c['folder'])
            if c['engine']=='strata':
                text=gzip.open(folder/'native-engine.log.gz','rt').read()
                cache.append(dict(engine='strata',arm=c['arm'],
                    requested_slots=5184 if a.controlled_strata_followup else 6606, actual_cache_lines=[line for line in text.splitlines()
                        if ('expert cache' in line.lower() or 'prompt chunk' in line.lower())
                        and ('slots' in line.lower() or 'chunk' in line.lower())],
                    sampled_peak_gpu_bytes=c['sampled_peak_gpu_bytes']))
            else:
                cache.append(dict(engine='ninfer',arm=c['arm'],residency=c['residency']))
        timings=summarize(cells)
        if a.controlled_strata_followup:
            prior=json.loads(a.comparison_reference.read_text())
            for key,value in prior['timing'].items():
                if key.startswith('ninfer/'):
                    timings[key]=value
            cache.extend(c for c in prior['cache'] if c['engine']=='ninfer')
        report=dict(step=6,status='serving_controls_complete',timing=timings,cache=cache,
            controls=dict(context=8192,kv_storage_bits=16,prefill_chunk_requested=2048,
                          draft_window_requested=2,physical_cpus=cpus,prompt_tokens=7111,
                          output_limit=128,prefix_reuse=False),
            limitations=['Installed Strata serving rejects spec<2; no-spec native baseline remains pending.',
                         'BF16 versus FP16 KV are different representations; only storage width is matched.',
                         'Expert weights differ; equal quantization/quality is not asserted.',
                         'Strata 6606 requested slots target about 20.7 GB using mean expert size; actual bytes in logs govern.',
                         'MTP algorithms/min-p and accepted work differ; decode is not equal-work.',
                         'Requested chunk size must be checked against actual Strata lending/fallback logs.'])
        if a.controlled_strata_followup:
            report['status']='controlled_followup_complete'
            report['ninfer_reference_run']=38090951809
            report['strata_controls']=dict(requested_slots=5184,requested_byte_budget=5184*3993600,
                                          spec_window=3,max_mtp_drafts=2,suffix_draft=0,spec_min_p=0)
            report['limitations']=[
                'Installed native pack rejects spec<2 in BOTH serving and native CLI; no-spec comparison unavailable without engine change.',
                'Strata window3 includes anchor and allows2 drafts; NInfer draft-tokens2 allows2 drafts. Algorithms/accepted work still differ.',
                'NInfer reference reused from run38090951809; Strata changed controls measured separately, not alternating contemporaneously.',
                'BF16/FP16 KV and expert quantization differ; matched storage width/budgets do not establish equal quality.',
                'Strata count5184 times max_blob3993600 targets20702822400B versus NInfer20704739328B; actual ranked allocation logs govern.']
        (a.output/'controlled-comparison-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('Step 6 serving control evidence retained; installed pack no-spec limitation recorded.',flush=True)
        return
    if a.prefill_breakdown or a.moe_work_breakdown:
        engine, arm, flags = (context_arms[1] if a.large_profile_breakdown else dense_arms[1] if a.moe_work_breakdown else
                              ('ninfer', 'adaptive156-qualified', residency_arms()[1][2]))
        run(engine, arm, flags, arm+'-reference', 'timing')
        a.telemetry_level = 2
        diagnostic = dict(flags, NINFER_FLASH_NEXT_STAGE_LEDGER='1',
                          NINFER_V100_EXPERT_STREAM_TIMING='1')
        run(engine, arm, diagnostic, arm+'-breakdown', 'diagnostic')
        reference, observed = [Path(c['folder']) for c in cells]
        base = [json.loads(line) for line in (reference/'requests.jsonl').read_text().splitlines()]
        breakdown = prefill_breakdown(observed)
        stream_breakdown = moe_stream_breakdown(observed) if a.moe_work_breakdown else None
        if len(base) != 6 or len(breakdown) != len(base):
            raise ValueError('incomplete reference/diagnostic schedule')
        for before, after in zip(base, breakdown):
            if (before['index'] != after['index'] or before['phase'] != after['phase']
                    or before['output_sha256'] != after['output_hash']
                    or before['response']['usage'] != after['usage']
                    or before.get('native_speculative') != after['speculative']):
                raise ValueError('instrumentation changed corresponding output/usage/MTP work')
        report = dict(step=3 if a.moe_work_breakdown else 1, plan='six-step performance investigation',
            status='stream_measurement_complete' if a.moe_work_breakdown else 'complete',
            reference_timing=summarize(cells), diagnostic_requests=breakdown,
            large_profile_breakdown=a.large_profile_breakdown,ninfer_flags=a.ninfer_flags,
            exact_corresponding_output_usage_mtp=True,
            residency=[dict(mode=c['mode'], observation=c['residency']) for c in cells],
            telemetry_overhead=[dict(index=b['index'], phase=b['phase'],
                http_seconds=d['wall_seconds']-b['wall_seconds'],
                prefill_seconds=d['native_timing']['prefill']-b['native_timing']['prefill'])
                for b,d in zip(base,breakdown)],
            profiler='existing SV0 and CUDA event ledgers; nsys availability recorded separately')
        if stream_breakdown is not None:
            report['stream_requests']=stream_breakdown
        name='moe-work-breakdown-report.json' if a.moe_work_breakdown else 'prefill-breakdown-report.json'
        (a.output/name).write_text(json.dumps(report,indent=2)+'\n')
        for row in breakdown:
            if row['phase'] in ('cold','measured'):
                print(json.dumps(row,indent=2),flush=True)
        print('Step 3 stream measurement complete; work reduction remains open.' if a.moe_work_breakdown
              else 'Six-step plan Step 1 complete: paired prefill attribution retained.',flush=True)
        return
    if a.final_comparison:
        arms=[('ninfer','adaptive156-qualified',residency_arms()[1][2]),ARMS[1]]
        for repeat in range(3):
            for engine,arm,flags in (arms if repeat!=1 else list(reversed(arms))):
                run(engine,arm,flags,arm+'-final-r'+str(repeat),'timing')
        report=dict(step=5,status='comparison_complete',timing=summarize(cells),
            comparison='native practical configurations; different quantization, KV, context, MTP and output work',
            memory=[dict(engine=c['engine'],arm=c['arm'],peak=c['sampled_peak_gpu_bytes'],
                         residency=c.get('residency')) for c in cells])
        (a.output/'final-comparison-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        # Measure a longer request at two NInfer capacity budgets, then installed
        # Strata at its native 262K/int8/32K-resident settings. This does not claim
        # to measure a full 262K prompt or isolate selected-block attention math.
        a.long_prompt=' '.join(f'Record {i}: the solar station stores energy during daylight and supplies the village after sunset.'
                               for i in range(1400))+' Explain how its battery storage works in detail.'
        a.max_output_tokens=64; a.request_schedule=[('cold','long'),('warmup','long'),('measured','long')]
        cases=[('ninfer','context32768',32768),('ninfer','context262144',262144),('strata','native262144',262144)]
        capacity=[]
        for engine,arm,context in cases:
            if engine=='ninfer':
                a.ninfer_flags=ab.option(a.ninfer_flags,'--max-context',str(context))
                a.ninfer_flags=ab.option(a.ninfer_flags,'--kv-capacity',str(context))
            flags=residency_arms()[1][2] if engine=='ninfer' else {}
            try:
                run(engine,arm,flags,arm+'-longer-request','capacity')
                cells[-1]['requested_context']=context
                capacity.append(cells[-1])
            except Exception as error:
                # A capacity failure must not erase the completed comparison.
                folder=a.output/(engine+'-'+arm+'-longer-request')
                if folder.exists(): compress_logs(folder)
                failure=dict(engine=engine,arm=arm,requested_context=context,status='failed',error=str(error))
                capacity.append(failure)
                print(json.dumps(failure),flush=True)
                ab.release_gpu()
            (a.output/'capacity-report.json').write_text(json.dumps(capacity,indent=2)+'\n')
        report['status']='complete'; report['capacity']=capacity
        (a.output/'final-comparison-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('Step 5 contemporary comparison and larger-context checks completed.',flush=True)
        return
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
