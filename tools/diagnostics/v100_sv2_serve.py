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


def validate_events(events, responses, mtp, prefill_screen=False):
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
    if prefill_screen and not 1024 <= done[0]["result"]["prompt_tokens"] <= 2048:
        raise ValueError("production prompt did not exercise a single large prefill chunk")
    return done


def run_server(executable, artifact, profile, output, mode, mtp, repeat, prefill_screen=False):
    name = f'{mode}-mtp{int(mtp)}-{repeat}'
    log_path, request_path = output/f'{name}.log', output/f'{name}-requests.jsonl'
    env = os.environ.copy()
    for key in tuple(env):
        if (key.startswith('NINFER_PHASE') or key.startswith('NINFER_V100_EXPERT_') or
                key.startswith('NINFER_V100_PREFILL_')):
            env.pop(key)
    env.update(NINFER_V100_DEVICE_ROUTE_COMBINE='1' if prefill_screen or mode == 'device' else '0',
               NINFER_V100_PREFILL_EXPERT_POLICY=mode if prefill_screen else 'cpu-cache',
               NINFER_V100_EXPERT_PROFILE=str(profile.resolve()),NINFER_V100_EXPERT_POLICY='static',
               NINFER_FLASH_NEXT_EXPERT_CACHE='1',NINFER_FLASH_NEXT_EXPERT_CACHE_MAX_SLOTS='64',
               NINFER_FLASH_NEXT_EXPERT_CACHE_SERIAL='0',NINFER_FLASH_NEXT_EXPERT_CACHE_PREFILL='1',
               NINFER_FLASH_NEXT_EXPERT_CACHE_BATCHED_DECODE='0',
               NINFER_FLASH_NEXT_EXPERT_CACHE_GROUPED_PREFILL='1',
               NINFER_V100_TELEMETRY='0',NINFER_FLASH_NEXT_EXPERT_CACHE_TIMING='0',
               NINFER_FLASH_NEXT_STAGE_LEDGER='0',NINFER_FLASH_NEXT_FP32_MOE_ROUTED_INPUT='0',
               NINFER_FLASH_NEXT_CPU_EXPERT_FP32_INTERMEDIATE='0')
    # Bind only a loopback port. The subprocess is the only process this tool stops.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0)); port=probe.getsockname()[1]
    # Preserve the .ninfer alias: resolve() would strip the suffix of the mounted file.
    command=[str(executable.resolve()),str(artifact.absolute()),'--host','127.0.0.1',
             '--port',str(port),'--max-context','4096','--kv-capacity','4096',
             '--max-concurrency','1','--prefill-chunk','2048' if prefill_screen else '128','--kv-dtype','bf16',
             '--device-state-slots','2','--host-state-slots','2','--host-kv-mib','256',
             '--max-private-continuations','2','--max-shared-prefixes','2',
             '--no-cuda-graph','--no-qsa-prefill-mma','--no-thinking',
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
                paragraphs=' '.join(f'Record {i}: the solar station stores energy during daylight and supplies the village after sunset.' for i in range(64 if prefill_screen else 16))
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
    done=validate_events(events,responses,mtp,prefill_screen)
    text=log_path.read_text()
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
    result=dict(name=name,mode=mode,mtp=mtp,repeat=repeat,cache_slots_per_layer=64,
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
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.artifact.suffix != '.ninfer' or not args.artifact.is_file():
        parser.error('--artifact requires an explicit readable .ninfer file')
    if args.repeats<3: parser.error('need three fresh-process observations per arm')
    args.output.mkdir(parents=True,exist_ok=True)
    profile=json.loads(args.profile.read_text())
    if profile['magic']!='NINFER_V100_EXPERT_PROFILE' or profile['version']!=2:
        raise ValueError('need identity-bound fixed profile')
    modes=('cpu-cache','stream') if args.prefill_screen else ('legacy','device')
    observations=[]
    for repeat in range(args.repeats):
        for mtp in (False,True):
            for mode in (modes if repeat%2==0 else tuple(reversed(modes))):
                row=run_server(args.executable,args.artifact,args.profile,args.output,mode,mtp,repeat,args.prefill_screen)
                observations.append(row); atomic_json(args.output/'observations.json',observations)
                peers=[r for r in observations if r['mtp']==mtp]
                if any(r['response_signatures']!=peers[0]['response_signatures'] for r in peers):
                    raise ValueError('legacy/device greedy production response or finish accounting differs')
                if len({r['cache_bytes'] for r in peers})!=1:
                    raise ValueError('production cache capacity differs between paths')
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
                results.append(dict(mode=mode,mtp=mtp,request=label,
                    prefill_tokens_per_s_median=statistics.median(prefill),
                    fresh_prompt_tokens=fresh,decode_tokens_per_s_median=statistics.median(values),
                    minimum=min(values),maximum=max(values),process_observations=len(rows)))
    report=dict(schema=1,milestone='SV3' if args.prefill_screen else 'SV2',scope='production_http_prefill_prefix_mtp_screen' if args.prefill_screen else 'production_http_prefix_mtp_screen',qualified=False,
                candidate_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),results=results,
                exact_compared_responses=True,limitations=[
                    'fixed 64 slots per layer, BF16 KV, one active request, short 4096 context',
                    'small two-turn corpus; no concurrent cancellation, long context, vision or full production matrix',
                    'MTP drafting required; acceptance counts retained, no minimum acceptance coefficient imposed',
                    'no independent oracle thresholds changed; accepted baseline numerical failure remains separate'])
    atomic_json(args.output/'report.json',report)
    lines=['Production HTTP prefill/prefix/MTP screen; defaults unchanged.','',
           '| Mode | MTP | Request | Prefill t/s median | Decode t/s median (range) |','|---|---|---|---:|---:|']
    for r in results:
        lines.append(f"| {r['mode']} | {r['mtp']} | {r['request']} | {r['prefill_tokens_per_s_median']:.3f} | {r['decode_tokens_per_s_median']:.3f} ({r['minimum']:.3f}–{r['maximum']:.3f}) |")
    (args.output/'report.txt').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))


if __name__=='__main__': main()
