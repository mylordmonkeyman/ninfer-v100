"""SV2 production HTTP screen with fixed expert residency, prefix reuse and MTP."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request

from v100_expert_profile import atomic_json
from v100_sv0_benchmark import gpu_snapshot


def request(base, path, payload=None):
    encoded = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base+path, encoded, {'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def response_signature(response):
    choice = response['choices'][0]
    return dict(message=choice['message'], finish_reason=choice['finish_reason'],
                completion_tokens=response['usage']['completion_tokens'])


def validate_events(events, responses, mtp, large_prefill=False, multichunk_prefill=False,
                    long_context_prefill=False):
    done = [e for e in events if e.get('event') == 'request_done']
    if any(e.get('event') in ('request_error','request_rejected') for e in events):
        raise ValueError('server reported request failure')
    if len(done) != 4 or len(responses) != 4:
        raise ValueError('missing production requests or completion records')
    for index, (event, response) in enumerate(zip(done, responses)):
        result, timings = event['result'], event['timings_seconds']
        usage = response['usage']
        if result['completion_tokens'] <= 0 or result['completion_tokens'] != usage['completion_tokens']:
            raise ValueError('HTTP/Engine output token accounting differs')
        if result['prompt_tokens'] != usage['prompt_tokens'] or result['prefix_cache_hit_tokens'] != usage['prompt_tokens_details']['cached_tokens']:
            raise ValueError('HTTP/Engine prefix accounting differs')
        if index == 0 and result['prefix_cache_hit_tokens'] != 0:
            raise ValueError('first workload request unexpectedly reused a prefix')
        if index > 0 and result['prefix_cache_hit_tokens'] <= 0:
            raise ValueError('repeat/continuation did not exercise prefix reuse')
        for key in ('prefill','decode','total','ttft'):
            if not math.isfinite(timings[key]) or timings[key] < 0:
                raise ValueError('invalid production timing')
        if timings['decode'] <= 0:
            raise ValueError('missing decode timing')
    if response_signature(responses[0]) != response_signature(responses[1]):
        raise ValueError('read-only prefix replay changed greedy response')
    if response_signature(responses[2]) != response_signature(responses[3]):
        raise ValueError('continuation replay changed greedy response')
    if mtp:
        if any(e['speculative']['backend'] != 'mtp' for e in done):
            raise ValueError('MTP capability was not active')
        if sum(e['speculative']['drafted_tokens'] for e in done) <= 0:
            raise ValueError('no actual MTP drafting occurred')
    elif any(e['speculative']['drafted_tokens'] for e in done):
        raise ValueError('non-MTP control drafted tokens')
    prompt_tokens = done[0]['result']['prompt_tokens']
    if long_context_prefill:
        # Exercise four prefill chunks, preserving 8K context headroom.
        if not 6400 <= prompt_tokens <= 7600:
            raise ValueError('long-context prefill must contain 6400..7600 tokens')
    elif multichunk_prefill:
        # Two chunks must be exercised; the final continuation still fits the
        # fixed 4096-token context after the 64-token first completion.
        if not 2560 <= prompt_tokens <= 3500:
            raise ValueError('production multi-chunk prompt outside bounded 2560..3500 tokens')
    elif large_prefill and not 1024 <= prompt_tokens <= 2048:
        raise ValueError("production prompt did not exercise a single large prefill chunk")
    return done


def validate_handoff_peers(peers):
    # A scheduling change must retain speculation behavior, not just final text.
    counts = [tuple((event['speculative']['drafted_tokens'],
                     event['speculative']['accepted_tokens']) for event in row['requests'])
              for row in peers]
    if any(value != counts[0] for value in counts):
        raise ValueError('route handoff changed MTP draft/accept counts')


def validate_qsa_dispatch(text, mode):
    if mode not in ('simt','score-mma'):
        raise ValueError('unknown QSA score mode')
    records=[]
    for line in text.splitlines():
        try: row=json.loads(line)
        except json.JSONDecodeError: continue
        if row.get('kind')=='qsa_score_dispatch': records.append(row)
    large=[row for row in records if row.get('tokens',0)>=512]
    if not large or any(not row.get('fp8') for row in records):
        raise ValueError('missing actual large FP8 QSA prefill dispatch')
    mma_large=sum(bool(row.get('mma')) for row in large)
    if mode=='simt':
        if mma_large:
            raise ValueError('SIMT QSA control unexpectedly used MMA')
    elif mma_large*2 != len(large):
        raise ValueError('QSA candidate did not restrict MMA to the late half of QSA layers')
    if any(row.get('mma') and row.get('tokens',0)<512 for row in records):
        raise ValueError('QSA MMA escaped the measured batch range')
    return records


def validate_qsa_comparisons(text, dispatch):
    records=[]
    for line in text.splitlines():
        try: row=json.loads(line)
        except json.JSONDecodeError: continue
        if row.get('kind')=='qsa_real_comparison': records.append(row)
    large=[row for row in dispatch if row['tokens']>=512]
    if not records or [row['tokens'] for row in records] != [row['tokens'] for row in large]:
        raise ValueError('missing same-input QSA comparison for an eligible dispatch')
    for row in records:
        if row.get('sample_queries')!=3 or row.get('sample_heads')!=24 or row.get('sample_oracle_pass') is not True:
            raise ValueError('real QSA sampled independent oracle failed or incomplete')
        for path in ('simt','mma'):
            relative=row[path+'_fp64_relative_l2'];pointwise=row[path+'_pointwise_ratio']
            if not math.isfinite(relative) or relative>1e-3 or relative<0 or not math.isfinite(pointwise) or not 0<=pointwise<=1:
                raise ValueError('real QSA independent thresholds failed')
        for key in ('relative_l2_difference','max_absolute_difference'):
            if not math.isfinite(row[key]) or row[key]<0:
                raise ValueError('invalid real QSA comparison metric')
    return records


def require_gpu_headroom():
    """Optional fail-closed V100 free-memory guard before EACH fresh server launch."""
    requested = os.environ.get('NINFER_V100_AB_MIN_FREE_GPU_MIB')
    if requested is None:
        return
    if not requested.isascii() or not requested.isdecimal() or int(requested) < 1:
        raise ValueError('NINFER_V100_AB_MIN_FREE_GPU_MIB must be a positive integer')
    free = int(subprocess.check_output(
        ['nvidia-smi', '--id=0', '--query-gpu=memory.free',
         '--format=csv,noheader,nounits'], text=True).strip())
    if free < int(requested):
        raise RuntimeError(f'V100 occupied before case: {free} MiB free, '
                           f'need {requested}; unrelated processes were not stopped')


def run_server(executable, artifact, profile, output, mode, mtp, repeat,
               prefill_screen=False, cpu_group_screen=False, route_handoff_screen=False,
               route_handoff_policy="all", qsa_score_screen=False, diagnostic=False, qsa_score_attribution=False,
               sv7_tc_screen=False, sv7_stage_screen=False, moe_policy_screen=False, ple_io_screen=False,
               moe_minroutes_screen=False, moe_long_prefill_screen=False,
               moe_threshold_sweep_screen=False, cpu_workers_screen=False,
               cpu_workers_long_screen=False):
    require_gpu_headroom()
    minroutes_screen = (moe_minroutes_screen or moe_long_prefill_screen or
                        moe_threshold_sweep_screen or cpu_workers_screen or
                        cpu_workers_long_screen)
    name = f'{mode}-mtp{int(mtp)}-{repeat}'
    log_path, request_path = output/f'{name}.log', output/f'{name}-requests.jsonl'
    env = os.environ.copy()
    for key in tuple(env):
        if (key.startswith('NINFER_PHASE') or key.startswith('NINFER_V100_EXPERT_') or
                key.startswith('NINFER_V100_PREFILL_') or
                key.startswith('NINFER_V100_CPU_EXPERT_GROUP') or
                key.startswith('NINFER_V100_ROUTE_HANDOFF') or
                key.startswith('NINFER_V100_QSA_SCORE_') or
                key.startswith('NINFER_V100_PLE_') or
                key == 'NINFER_FLASH_NEXT_QSA_PREFILL_MMA'):
            env.pop(key)
    large_prefill = prefill_screen or cpu_group_screen or route_handoff_screen or qsa_score_screen or sv7_tc_screen or sv7_stage_screen or moe_policy_screen or ple_io_screen or minroutes_screen
    env.update(NINFER_V100_ROUTE_HANDOFF=('prefill' if route_handoff_policy == 'prefill' else '1')
               if route_handoff_screen and mode == 'handoff' else '0',
               NINFER_V100_DEVICE_ROUTE_COMBINE='1' if large_prefill or mode == 'device' else '0',
               NINFER_V100_PREFILL_EXPERT_POLICY=mode if prefill_screen else (('auto' if mode == 'auto-grouped' else 'stream') if (sv7_stage_screen or moe_policy_screen) and mode in ('stream-grouped','auto-grouped') else 'auto' if ple_io_screen or minroutes_screen else 'cpu-cache'),
               NINFER_V100_PREFILL_STREAM_MIN_TOKENS='256' if (moe_policy_screen and mode == 'auto-grouped') or ple_io_screen or minroutes_screen else '',
               NINFER_V100_PREFILL_EXPERT_STREAM_MIN_ROUTES=(
                    '20' if cpu_workers_screen or cpu_workers_long_screen else
                    {'auto-min14':'14','auto-min20':'20','auto-min28':'28'}.get(mode,'')
                    if minroutes_screen else ''),
               NINFER_FLASH_NEXT_CPU_EXPERT_WORKERS=(
                    {'cpu-workers32':'32','cpu-workers48':'48','cpu-workers64':'64'}[mode]
                    if cpu_workers_screen or cpu_workers_long_screen else
                     env.get('NINFER_FLASH_NEXT_CPU_EXPERT_WORKERS','32')),
               NINFER_V100_CPU_EXPERT_GROUP='1' if sv7_tc_screen or route_handoff_screen or (cpu_group_screen and mode == 'grouped') or ((sv7_stage_screen or moe_policy_screen) and mode in ('cpu-cache-grouped','stream-grouped','auto-grouped')) or ple_io_screen or minroutes_screen else '0',
               NINFER_V100_EXPERT_PROFILE=str(profile.resolve()),NINFER_V100_EXPERT_POLICY='static',
               NINFER_FLASH_NEXT_EXPERT_CACHE='1',NINFER_FLASH_NEXT_EXPERT_CACHE_MAX_SLOTS='64',
               NINFER_FLASH_NEXT_EXPERT_CACHE_SERIAL='0',NINFER_FLASH_NEXT_EXPERT_CACHE_PREFILL='1',
               NINFER_FLASH_NEXT_EXPERT_CACHE_BATCHED_DECODE='0',
               NINFER_FLASH_NEXT_EXPERT_CACHE_GROUPED_PREFILL='1',
               NINFER_V100_TELEMETRY='1' if diagnostic or sv7_stage_screen else '0',NINFER_FLASH_NEXT_EXPERT_CACHE_TIMING='0',
               NINFER_V100_QSA_SCORE_MMA='1' if qsa_score_screen and mode=='score-mma' else '0',
               NINFER_V100_QSA_SCORE_MMA_MIN_QSA='6' if qsa_score_screen and mode=='score-mma' else '0',
               NINFER_V100_QSA_SCORE_COMPARE='1' if qsa_score_attribution else '0',
               NINFER_V100_PLE_IO='direct' if ple_io_screen and mode == 'ple-direct' else 'mmap',
               NINFER_V100_PLE_STRICT_DIRECT='1' if ple_io_screen and mode == 'ple-direct' else '0',
               NINFER_V100_PLE_QUEUE_DEPTH='64',
               NINFER_V100_SV7_FP16_TC=('1' if mode == 'fp16-tc' else '0') if sv7_tc_screen else '0' if sv7_stage_screen else env.get('NINFER_V100_SV7_FP16_TC','0'),
               NINFER_FLASH_NEXT_STAGE_LEDGER='1' if sv7_stage_screen else '0',NINFER_FLASH_NEXT_FP32_MOE_ROUTED_INPUT='0',
               NINFER_FLASH_NEXT_CPU_EXPERT_FP32_INTERMEDIATE='0')
    # Bind only a loopback port. The subprocess is the only process this tool stops.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0)); port=probe.getsockname()[1]
    # Preserve the .ninfer alias: resolve() would strip the suffix of the mounted file.
    command=[str(executable.resolve()),str(artifact.absolute()),'--host','127.0.0.1',
             '--port',str(port),'--max-context','8192' if cpu_workers_long_screen else '4096',
              '--kv-capacity','8192' if cpu_workers_long_screen else '4096',
             '--max-concurrency','1','--prefill-chunk','2048' if large_prefill else '128','--kv-dtype','fp8' if qsa_score_screen else 'bf16',
             '--device-state-slots','2','--host-state-slots','2','--host-kv-mib','256',
             '--max-private-continuations','2','--max-shared-prefixes','2',
             '--no-cuda-graph','--qsa-prefill-mma' if qsa_score_screen else '--no-qsa-prefill-mma','--no-thinking',
             '--request-log-jsonl',str(request_path)]
    if mtp:
        command += ['--spec','mtp','--draft-tokens','3','--lm-head-draft']
    snapshots, monitor_errors = [gpu_snapshot()], []
    stopped=threading.Event()
    def monitor():
        while not stopped.wait(5):
            try: snapshots.append(gpu_snapshot())
            except Exception as error: monitor_errors.append(str(error))
    worker=threading.Thread(target=monitor); worker.start()
    responses=[]
    try:
        with log_path.open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                base=f'http://127.0.0.1:{port}'
                deadline=time.monotonic()+600
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'{name}: server exited before readiness; see {log_path}')
                    try:
                        models=request(base,'/v1/models'); break
                    except (urllib.error.URLError,TimeoutError,ConnectionResetError):
                        if time.monotonic()>deadline: raise TimeoutError('server readiness timeout')
                        time.sleep(1)
                model=models['data'][0]['id']
                paragraphs=' '.join(f'Record {i}: the solar station stores energy during daylight and supplies the village after sunset.' for i in range(360 if cpu_workers_long_screen else 160 if moe_long_prefill_screen or moe_threshold_sweep_screen or cpu_workers_screen else 64 if large_prefill else 16))
                messages=[{'role':'user','content':paragraphs+' Explain how its battery storage works in detail.'}]
                payload=dict(model=model,messages=messages,max_tokens=64,temperature=0,seed=42,enable_thinking=False)
                for readonly in (False,True):
                    body=dict(payload,prompt_cache_read_only=readonly)
                    responses.append(request(base,'/v1/chat/completions',body))
                content=responses[0]['choices'][0]['message'].get('content')
                if not isinstance(content,str) or not content:
                    raise ValueError('no usable assistant continuation')
                messages += [{'role':'assistant','content':content},
                             {'role':'user','content':'Now describe two practical maintenance tasks and explain why they matter.'}]
                for readonly in (False,True):
                    body=dict(payload,messages=messages,prompt_cache_read_only=readonly)
                    responses.append(request(base,'/v1/chat/completions',body))
            finally:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=10)
    finally:
        stopped.set(); worker.join()
        snapshots.append(gpu_snapshot())
        atomic_json(output/f'{name}-hardware.json',dict(snapshots=snapshots,errors=monitor_errors,
                    command=command,environment={k:v for k,v in env.items() if k.startswith('NINFER_')}))
        atomic_json(output/f'{name}-responses.json',responses)
    if monitor_errors or any(not s['thermal_status_observed'] or s['thermal_throttled'] for s in snapshots):
        raise ValueError('production thermal evidence missing or throttled')
    events=[json.loads(line) for line in request_path.read_text().splitlines() if line.strip()]
    done=validate_events(events,responses,mtp,large_prefill,
                         moe_long_prefill_screen or moe_threshold_sweep_screen or cpu_workers_screen,
                          long_context_prefill=cpu_workers_long_screen)
    text=log_path.read_text()
    if sv7_tc_screen or sv7_stage_screen:
        actual = 'sv7.fp16_tc.dispatch=1' in text
        if actual != (mode == 'fp16-tc'):
            raise ValueError(f'{name}: SV7 FP16 TC actual dispatch={actual}; expected={mode == "fp16-tc"}')
    if sv7_stage_screen and '"kind":"prefill_stage_ledger"' not in text:
        raise ValueError(f'{name}: missing stage ledger prefill evidence')
    dispatch=validate_qsa_dispatch(text,mode) if qsa_score_screen and diagnostic else None
    comparisons=validate_qsa_comparisons(text,dispatch) if qsa_score_attribution else None
    slots=re.findall(r'phase13.cache.slots_per_layer=(\d+)',text)
    size=re.findall(r'phase13.cache.bytes=(\d+)',text)
    seeded=re.findall(r'v100.profile.seeded=(\d+)',text)
    if slots!=['64'] or len(size)!=1 or seeded!=[str(64*48)]:
        raise ValueError('server did not publish expected fixed resident expert cache')
    if 'v100.profile.save_failed=' in text:
        raise ValueError('unexpected profile save failure')
    startup=[e for e in events if e.get('event')=='server_start']
    if len(startup)!=1 or not startup[0]['engine']['prefix_reuse']:
        raise ValueError('missing production Engine/prefix capability record')
    result=dict(name=name,mode=mode,mtp=mtp,repeat=repeat,diagnostic=diagnostic,
                qsa_dispatch=dispatch,qsa_comparisons=comparisons,cache_slots_per_layer=64,
                cache_bytes=int(size[0]),startup=startup[0],requests=done,
                response_signatures=[response_signature(r) for r in responses])
    print(f'{name}: production prefix/continuation and drafting checks passed',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prefill-screen',action='store_true',
                        help='compare fixed-cache cpu-cache/stream with a large production prompt')
    parser.add_argument('--cpu-group-screen',action='store_true',
                        help='compare fixed-cache single/grouped CPU misses in production')
    parser.add_argument('--route-handoff-screen',action='store_true',
                        help='compare serial/route-ready with grouping enabled in both arms')
    parser.add_argument('--route-handoff-policy',choices=('all','prefill'),default='all',
                        help='phase eligibility for the route-ready candidate; serial control remains off')
    parser.add_argument('--moe-policy-screen',action='store_true',
                        help='end-to-end HTTP three-arm grouped CPU, GPU stream, adaptive auto256 performance A/B/C')
    parser.add_argument('--cpu-workers-screen',action='store_true',
                        help='production 3K HTTP fixed minroutes20 with 32/48/64 CPU expert workers')
    parser.add_argument('--moe-threshold-sweep-screen',action='store_true',
                        help='production 3K HTTP auto256: grouped minimum expert routes 14/20/28')
    parser.add_argument('--moe-long-prefill-screen',action='store_true',
                        help='production HTTP 3K-token multi-chunk auto256 baseline vs grouped minroutes14')
    parser.add_argument('--moe-minroutes-screen',action='store_true',
                        help='production HTTP auto256: all streaming versus opt-in >=14 routes per expert')
    parser.add_argument('--ple-io-screen',action='store_true',
                        help='SV6 strict direct PLE I/O versus warm mmap in identical auto256 MoE production HTTP')
    parser.add_argument('--sv7-stage-screen',action='store_true',
                        help='two-process, bounded stage-level production cold-prefill attribution')
    parser.add_argument('--sv7-tc-screen',action='store_true',
                        help='compare opt-in FP16 tensor-core BF16 projection against SIMT in production HTTP')
    parser.add_argument('--qsa-score-screen',action='store_true',
                        help='compare SIMT/MMA FP8 KV scores with actual large dispatch evidence')
    parser.add_argument('--qsa-score-attribution',action='store_true',
                        help='one SIMT serving process with same-input MMA/FP64 diagnostic comparisons; no timing')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.artifact.suffix != '.ninfer' or not args.artifact.is_file():
        parser.error('--artifact requires an explicit readable .ninfer file')
    if sum((args.prefill_screen,args.cpu_group_screen,args.route_handoff_screen,args.qsa_score_screen,args.qsa_score_attribution,args.sv7_tc_screen,args.sv7_stage_screen,args.moe_policy_screen,args.ple_io_screen,args.moe_minroutes_screen,args.moe_long_prefill_screen,args.moe_threshold_sweep_screen,args.cpu_workers_screen)) > 1:
        parser.error('performance screen flags are mutually exclusive')
    if args.repeats<3: parser.error('need three fresh-process observations per arm')
    args.output.mkdir(parents=True,exist_ok=True)
    profile=json.loads(args.profile.read_text())
    if profile['magic']!='NINFER_V100_EXPERT_PROFILE' or profile['version']!=2:
        raise ValueError('need identity-bound fixed profile')
    if args.qsa_score_attribution:
        row=run_server(args.executable,args.artifact,args.profile,args.output,'simt',False,
                       -2,qsa_score_screen=True,diagnostic=True,qsa_score_attribution=True)
        atomic_json(args.output/'attribution.json',dict(schema=1,milestone='SV7',qualified=False,
                    scope='same_input_real_fp8_qsa_sampled_oracle_attribution',observation=row,
                    limitations=['three query positions and all heads per eligible call; not a complete model oracle',
                                 'failed exact continuation screen remains failed; no timing or MTP qualification']))
        print(f"Same-input real QSA sampled FP64 checks passed: {len(row['qsa_comparisons'])} eligible calls; no serving qualification",flush=True)
        return
    if args.sv7_stage_screen:
        modes=('cpu-cache-single','cpu-cache-grouped','stream-grouped')
    elif args.moe_policy_screen:
        modes=('cpu-cache-grouped','stream-grouped','auto-grouped')
    elif args.ple_io_screen:
        modes=('ple-mmap','ple-direct')
    elif args.moe_minroutes_screen or args.moe_long_prefill_screen:
        modes=('auto-all','auto-min14')
    elif args.moe_threshold_sweep_screen:
        modes=('auto-min14','auto-min20','auto-min28')
    elif args.cpu_workers_screen:
        modes=('cpu-workers32','cpu-workers48','cpu-workers64')
    elif args.sv7_tc_screen:
        modes=('bf16-simt','fp16-tc')
    elif args.qsa_score_screen:
        modes=('simt','score-mma')
    elif args.route_handoff_screen:
        modes=('serial','handoff')
    elif args.cpu_group_screen:
        modes=('single','grouped')
    else:
        modes=('cpu-cache','stream') if args.prefill_screen else ('legacy','device')
    if args.sv7_stage_screen:
        summaries = {}
        for mode in modes:
            row = run_server(args.executable,args.artifact,args.profile,args.output,
                             mode,False,0,sv7_stage_screen=True)
            log = (args.output/f'{mode}-mtp0-0.log').read_text()
            events = []
            for line in log.splitlines():
                try: obj = json.loads(line)
                except json.JSONDecodeError: continue
                if obj.get('kind') == 'prefill_stage_ledger': events.append(obj)
            if not events:
                raise ValueError(f'{mode}: no stage events')
            summary = {'chunks':len(events),'total_ms':sum(e['total_chunk_ms'] for e in events),
                       'stages':{}}
            for event in events:
                for stage in event['stages']:
                    key=stage['stage']
                    item=summary['stages'].setdefault(key,{'calls':0,'ms':0.0})
                    item['calls']+=stage['calls']
                    item['ms']+=stage['interval_ms']
            summaries[mode]=summary
        atomic_json(args.output/'stage-attribution.json',
                    dict(schema=2,scope='instrumented_prefill_and_followup_cpu_moe_policy_attribution',
                         results=summaries,limitations=[
                             'CUDA event stage ledger is invasive; timings are diagnostic not a throughput benchmark',
                             'stage intervals may include CPU stalls and stream waits, not pure GPU kernel time',
                             'one fresh server per arm; production defaults unchanged',
                              'stage ledger includes multiple prefill/continuation chunks; do not interpret stage total as cold TTFT',
                              'compare cpu-cache-single vs cpu-cache-grouped vs stream-grouped, BF16 projection path fixed']))
        print('Stage attribution (instrumented; no performance claim):')
        for mode, summary in summaries.items():
            print(mode, 'total_ms',round(summary['total_ms'],2))
            for name, value in sorted(summary['stages'].items(),key=lambda x:-x[1]['ms'])[:12]:
                print(' ',name,round(value['ms'],2),'ms',value['calls'],'calls')
        return
    observations=[]
    diagnostic_cross_path_exact=None
    if args.ple_io_screen:
        # Verify the actual PLE storage backend once per arm before uninstrumented timing.
        # Strict-direct must not fall back to mmap. The diagnostic servers are excluded
        # from timings, and neither arm evicts or changes the model's page cache.
        dispatch={}
        for mode in modes:
            run_server(args.executable,args.artifact,args.profile,args.output,mode,False,
                       -1,ple_io_screen=True,diagnostic=True)
            text=(args.output/f'{mode}-mtp0--1.log').read_text()
            rows=[]
            for line in text.splitlines():
                try: entry=json.loads(line)
                except json.JSONDecodeError: continue
                if entry.get('kind')=='ple_gather' and entry.get('compressed'):
                    rows.append(entry)
            backend='direct' if mode=='ple-direct' else 'mmap'
            if not rows or any(row.get('storage_backend')!=backend or
                               row.get('storage_fallback') for row in rows):
                raise ValueError(f'{mode}: failed to verify strict {backend} PLE backend')
            if backend=='direct' and not any(row.get('coalesced_pages',0)>0
                                             and row.get('page_read_us') is not None
                                             for row in rows):
                raise ValueError('strict direct PLE dispatched no page reads')
            dispatch[mode]={'backend':backend,'ple_gather_calls':len(rows),
                            'max_tokens':max(row['tokens'] for row in rows),
                            'actual_direct_page_reads':sum(row.get('coalesced_pages',0) for row in rows)}
            atomic_json(args.output/'ple-backend-dispatch.json',dispatch)
        print('SV6 mmap/direct storage backends verified separately from timing',flush=True)
    if args.moe_minroutes_screen or args.moe_long_prefill_screen or args.moe_threshold_sweep_screen or args.cpu_workers_screen:
        # Inspect one uninstrumented-equivalent prompt under explicit telemetry
        # separately from all timing observations; no stage ledger or broad profiling.
        traffic={}
        for mode in modes:
            run_server(args.executable,args.artifact,args.profile,args.output,mode,False,
                       -1,moe_minroutes_screen=args.moe_minroutes_screen,
                       moe_long_prefill_screen=args.moe_long_prefill_screen,
                       moe_threshold_sweep_screen=args.moe_threshold_sweep_screen,
                       cpu_workers_screen=args.cpu_workers_screen,diagnostic=True)
            lines=(args.output/f'{mode}-mtp0--1.log').read_text().splitlines()
            records=[]
            minimum_tokens=256 if args.moe_long_prefill_screen or args.moe_threshold_sweep_screen or args.cpu_workers_screen else 1024
            for line in lines:
                try: entry=json.loads(line)
                except json.JSONDecodeError: continue
                if entry.get('kind')=='expert_layer' and entry.get('prefill') is True and entry.get('tokens',0)>=minimum_tokens:
                    records.append(entry)
            # 3K tokens fit in two 2048-token chunks; each has 48 MoE layers.
            expected_records=96 if args.moe_long_prefill_screen or args.moe_threshold_sweep_screen or args.cpu_workers_screen else 48
            if len(records)!=expected_records:
                raise ValueError(f'{mode}: expected {expected_records} large MoE layer telemetry records; got {len(records)}')
            traffic[mode]={
                'layers':len(records),
                'stream_expert_h2d_bytes':sum(row['stream_expert_h2d_bytes'] for row in records),
                'stream_routes':sum(row['stream_routes'] for row in records),
                'cpu_miss_routes':sum(row['cpu_miss_routes'] for row in records),
                'cpu_weight_read_bytes':sum(row['cpu_weight_read_bytes'] for row in records),
                'cpu_branch_us_total':sum(row.get('cpu_branch_us',0) for row in records)}
            atomic_json(args.output/'minroutes-traffic.json',traffic)
        if args.cpu_workers_screen:
            # A worker-count A/B must keep route policy, GPU cache and weight
            # traffic fixed. These diagnostic counters are not the HTTP timing.
            expected=['cpu-workers32','cpu-workers48','cpu-workers64']
            if list(traffic)!=expected or len({traffic[m]['layers'] for m in modes})!=1:
                raise ValueError('worker screen missing comparable 96-layer traces')
            atomic_json(args.output/'cpu-worker-traffic.json',traffic)
            print('CPU expert worker 32/48/64 traffic diagnostics captured',flush=True)
        elif args.moe_threshold_sweep_screen:
            gpu=[traffic[m]['stream_expert_h2d_bytes'] for m in modes]
            cpu=[traffic[m]['cpu_miss_routes'] for m in modes]
            if not (gpu[0] > gpu[1] > gpu[2] > 0 and
                    0 < cpu[0] < cpu[1] < cpu[2]):
                raise ValueError('minroutes14/20/28 did not produce strictly selective GPU/CPU traffic')
            print('Minroutes14/20/28 transfer and CPU fallback selectivity verified',flush=True)
        else:
            if not (0 < traffic['auto-min14']['stream_expert_h2d_bytes'] <
                        traffic['auto-all']['stream_expert_h2d_bytes'] and
                    traffic['auto-min14']['cpu_miss_routes'] >
                        traffic['auto-all']['cpu_miss_routes']):
                raise ValueError('minroutes14 failed to reduce GPU streaming and increase CPU fallback')
            print('Minroutes14 selectivity verified separately from HTTP timing',flush=True)
    if args.qsa_score_screen:
        # Hard dispatch/accounting/replay failures still stop. Arithmetic cross-path
        # text differences are preserved as diagnostics after paired accuracy admission.
        diagnostics=[]
        for mode in modes:
            row=run_server(args.executable,args.artifact,args.profile,args.output,mode,False,
                           -1,qsa_score_screen=True,diagnostic=True)
            diagnostics.append(row);atomic_json(args.output/'dispatch-observations.json',diagnostics)
        diagnostic_cross_path_exact=(diagnostics[0]['response_signatures']==
                                     diagnostics[1]['response_signatures'])
    for repeat in range(args.repeats):
        for mtp in (False,True):
            for mode in (modes if repeat%2==0 else tuple(reversed(modes))):
                row=run_server(args.executable,args.artifact,args.profile,args.output,mode,mtp,
                               repeat,args.prefill_screen,args.cpu_group_screen,args.route_handoff_screen,
                               args.route_handoff_policy,qsa_score_screen=args.qsa_score_screen,
                               sv7_tc_screen=args.sv7_tc_screen,moe_policy_screen=args.moe_policy_screen,
                               ple_io_screen=args.ple_io_screen,
                               moe_minroutes_screen=args.moe_minroutes_screen,
                               moe_long_prefill_screen=args.moe_long_prefill_screen,
                               moe_threshold_sweep_screen=args.moe_threshold_sweep_screen,
                               cpu_workers_screen=args.cpu_workers_screen)
                observations.append(row); atomic_json(args.output/'observations.json',observations)
                peers=[r for r in observations if r['mtp']==mtp]
                same_mode=[r for r in peers if r['mode']==mode]
                if any(r['response_signatures']!=same_mode[0]['response_signatures']
                       for r in same_mode):
                    raise ValueError('same-path fresh-process greedy response or finish accounting differs')
                if (not (args.qsa_score_screen or args.sv7_tc_screen or args.moe_policy_screen or args.moe_minroutes_screen or args.moe_long_prefill_screen or args.moe_threshold_sweep_screen or args.cpu_workers_screen) and
                        any(r['response_signatures']!=peers[0]['response_signatures'] for r in peers)):
                    raise ValueError('cross-path greedy production response or finish accounting differs')
                if args.route_handoff_screen:
                    validate_handoff_peers(peers)
                if len({r['cache_bytes'] for r in peers})!=1:
                    raise ValueError('production cache capacity differs between paths')
    cross_path_response_matches=[]
    if args.qsa_score_screen or args.sv7_tc_screen:
        for repeat in range(args.repeats):
            for mtp in (False,True):
                pair=[r for r in observations if r['repeat']==repeat and r['mtp']==mtp]
                if len(pair)!=2:
                    raise ValueError('missing QSA A/B observation pair')
                cross_path_response_matches.append(dict(
                    repeat=repeat,mtp=mtp,
                    exact=pair[0]['response_signatures']==pair[1]['response_signatures'],
                    speculative_counts={r['mode']:[
                        dict(drafted=e['speculative']['drafted_tokens'],
                             accepted=e['speculative']['accepted_tokens'])
                        for e in r['requests']] for r in pair}))
    if args.moe_policy_screen or args.moe_minroutes_screen or args.moe_long_prefill_screen or args.moe_threshold_sweep_screen or args.cpu_workers_screen:
        for repeat in range(args.repeats):
            for mtp in (False,True):
                arms=[r for r in observations if r['repeat']==repeat and r['mtp']==mtp]
                if len(arms)!=len(modes) or {r['mode'] for r in arms}!=set(modes):
                    raise ValueError('missing complete MoE production crossover')
                cross_path_response_matches.append(dict(
                    repeat=repeat,mtp=mtp,
                    exact=all(r['response_signatures']==arms[0]['response_signatures'] for r in arms),
                    modes=[r['mode'] for r in arms]))
    results=[]
    labels=('cold','prefix-replay','continuation','continuation-replay')
    for mtp in (False,True):
        for mode in modes:
            rows=[r for r in observations if r['mtp']==mtp and r['mode']==mode]
            for index,label in enumerate(labels):
                values=[r['requests'][index]['result']['completion_tokens']/r['requests'][index]['timings_seconds']['decode'] for r in rows]
                fresh=[r['requests'][index]['result']['prompt_tokens'] -
                       r['requests'][index]['result']['prefix_cache_hit_tokens'] for r in rows]
                prefill=[tokens/r['requests'][index]['timings_seconds']['prefill']
                         if r['requests'][index]['timings_seconds']['prefill']>0 else 0
                         for tokens,r in zip(fresh,rows)]
                ttft=[r['requests'][index]['timings_seconds']['ttft'] for r in rows]
                results.append(dict(mode=mode,mtp=mtp,request=label,
                    ttft_seconds_median=statistics.median(ttft),ttft_seconds_min=min(ttft),ttft_seconds_max=max(ttft),
                    prefill_tokens_per_s_median=statistics.median(prefill),
                    fresh_prompt_tokens=fresh,decode_tokens_per_s_median=statistics.median(values),
                    minimum=min(values),maximum=max(values),process_observations=len(rows)))
    milestone=('SV5' if args.route_handoff_screen else
               ('SV4' if args.cpu_group_screen else ('SV3' if args.prefill_screen else 'SV2')))
    scope=('production_http_route_handoff_prefix_mtp_screen' if args.route_handoff_screen else
           ('production_http_grouped_cpu_prefix_mtp_screen' if args.cpu_group_screen else
            ('production_http_prefill_prefix_mtp_screen' if args.prefill_screen else
             'production_http_prefix_mtp_screen')))
    if args.qsa_score_screen:
        milestone='SV7';scope='production_http_fp8_qsa_score_prefix_mtp_screen'
    if args.sv7_tc_screen:
        milestone='SV7';scope='production_http_bf16_fp16_tc_prefix_mtp_screen'
    if args.moe_policy_screen:
        milestone='MoE';scope='production_http_cpu_grouped_streamed_auto256_prefix_mtp_screen'
    if args.ple_io_screen:
        milestone='SV6';scope='production_http_warm_mmap_vs_strict_direct_ple_auto256_prefix_mtp_screen'
    if args.moe_minroutes_screen:
        milestone='MoE';scope='production_http_auto256_all_stream_vs_minroutes14_grouped_prefix_mtp_screen'
    if args.moe_long_prefill_screen:
        milestone='MoE';scope='production_http_three_k_multichunk_auto256_all_vs_minroutes14_prefix_mtp_screen'
    if args.moe_threshold_sweep_screen:
        milestone='MoE';scope='production_http_three_k_multichunk_minroutes14_20_28_prefix_mtp_screen'
    if args.cpu_workers_screen:
        milestone='MoE';scope='production_http_three_k_multichunk_minroutes20_cpu_workers32_48_64_screen'
    report=dict(schema=1,milestone=milestone,scope=scope,qualified=False,
                route_handoff_policy=args.route_handoff_policy if args.route_handoff_screen else None,
                candidate_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),results=results,
                exact_compared_responses=(all(x['exact'] for x in cross_path_response_matches)
                                          if args.qsa_score_screen or args.sv7_tc_screen or args.moe_policy_screen or args.moe_minroutes_screen or args.moe_long_prefill_screen or args.moe_threshold_sweep_screen or args.cpu_workers_screen else True),
                diagnostic_cross_path_exact=diagnostic_cross_path_exact,
                cross_path_response_matches=cross_path_response_matches,
                limitations=[
                    f'fixed 64 slots per layer, {"FP8" if args.qsa_score_screen else "BF16"} KV, one active request, short 4096 context',
                    'small two-turn corpus; no concurrent cancellation, long context, vision or full production matrix',
                    'MTP drafting required; acceptance counts retained, no minimum acceptance coefficient imposed',
                    'no independent oracle thresholds changed; accepted baseline numerical failure remains separate'] +
                    (['grouped CPU experts remain opt-in; no concurrent-request matrix in this screen']
                     if args.cpu_group_screen else []) +
                    (['three-arm MoE policies remain opt-in; auto streams only >=256-token chunks; streaming may alter resident expert cache behavior',
                      'cross-path greedy output differences are diagnostic and are not numerical qualification; same-mode replay and accounting remain hard gates']
                     if args.moe_policy_screen else []) +
                    (['SV6 strict-direct backend validated in separate diagnostic processes; no file cold eviction or model copy',
                      'warm mmap vs strict O_DIRECT only; kernel page residency is not equalized and startup times are not the measured TTFT',
                      'adaptive prefill auto256 held fixed in both arms, CPU grouping on, 64 device expert slots/layer']
                     if args.ple_io_screen else []) +
                    (['auto256 prefill held fixed; minroutes14 streams only nonresident experts with >=14 routes, others use grouped CPU',
                      '48 cold large-chunk MoE layers inspected per arm only in separate telemetry diagnostics; not included in timing',
                      'within-policy outputs/replay must match; cross-policy greedy differences diagnostic, independent Phase11 numerical gate unchanged']
                     if args.moe_minroutes_screen else []) +
                    (['~3K initial prompt crosses exactly two 2048-token prefill chunks within 4096 context',
                      'auto256 held fixed, per-expert >=14 routed-token selective GPU stream versus stream all misses',
                      'both large prefill chunks separately instrumented for expert H2D and CPU weights, 96 expert-layer records per arm',
                      'repeated one-topic synthetic prompt and one active request; numerical Phase11 gates unchanged, no default promotion']
                     if args.moe_long_prefill_screen else []) +
                    (['3K input uses exactly two 2048-token prefill chunks and fixed 4096 context',
                      'three experimental per-expert route cutoffs 14/20/28 only; no production defaults changed',
                      '96 expert-layer traffic records per arm collected only in separate telemetry processes',
                      'cold and continuation MTP/no-MTP outputs compared diagnostically; independent Phase11 qualification remains required']
                     if args.moe_threshold_sweep_screen else []) +
                    (['fixed minroutes20 and auto256; vary only CPU expert workers 32/48/64',
                      '3111-token two-chunk prompt, 4096 context, static 64 expert slots per layer, BF16 KV, mmap PLE',
                      'traffic diagnostics from separate processes, not the uninstrumented timed HTTP processes',
                      'NUMA scheduling and SMT not controlled; benchmark cannot distinguish worker count and OS placement effects',
                      'Phase11 numerical gates unchanged; cross-policy text differences are diagnostic only']
                     if args.cpu_workers_screen else []) +
                    (['route handoff remains opt-in; both arms use grouped CPU experts; no concurrent-request matrix']
                     if args.route_handoff_screen else []) +
                    (['attention remains opt-in; dispatch diagnostics are excluded from timing; paired 4096-position accuracy admission is separate',
                       'cross-path greedy text and MTP accept-count differences are diagnostic, while same-path replay/accounting remain hard gates']
                     if args.qsa_score_screen else []) +
                    (['SV7 FP16 TensorOp candidate is opt-in; actual dispatch verified in server log',
                      'cross-path greedy differences are diagnostic only, not numerical qualification; same-mode replay and accounting remain hard gates',
                      'short-context single-request profile; independent full-model oracle and long-prefix validation still required']
                     if args.sv7_tc_screen else []))
    atomic_json(args.output/'report.json',report)
    lines=['Production HTTP prefill/prefix/MTP screen; defaults unchanged.','',
           '| Mode | MTP | Request | TTFT s median (range) | Prefill t/s median | Decode t/s median (range) |','|---|---|---|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['mode']} | {r['mtp']} | {r['request']} | {r['ttft_seconds_median']:.3f} ({r['ttft_seconds_min']:.3f}–{r['ttft_seconds_max']:.3f}) | {r['prefill_tokens_per_s_median']:.3f} | {r['decode_tokens_per_s_median']:.3f} ({r['minimum']:.3f}–{r['maximum']:.3f}) |")
    (args.output/'report.txt').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))


if __name__=='__main__': main()
