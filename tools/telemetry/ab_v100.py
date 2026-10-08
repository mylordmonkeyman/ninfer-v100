"""Controlled, one-at-a-time V100 NInfer/Strata A/B campaign.

Cross-engine times are descriptive because model weights, quantization, MTP policy
and kernel implementations differ. Only within-engine ratios are experimental.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

from qualify_v100 import PROMPT, api, free_port, stop, strata_config

SHORT = ' '.join(f'Record {i}: solar batteries store energy to supply the village after sunset.'
                 for i in range(3)) + ' Explain this system in detail.'
CASES = [
    ('ninfer', 'baseline', {}),
    ('strata', 'baseline', {}),
    ('ninfer', 'cache-off', {'NINFER_FLASH_NEXT_EXPERT_CACHE': '0'}),
    ('ninfer', 'cache-lru', {'NINFER_V100_EXPERT_POLICY': 'lru'}),
    ('strata', 'cache-off', {'cache': '0'}),
    ('ninfer', 'workers-16', {'NINFER_FLASH_NEXT_CPU_EXPERT_WORKERS': '16'}),
    ('strata', 'workers-16', {'workers': '16'}),
    ('ninfer', 'prefill-no-group', {'NINFER_FLASH_NEXT_EXPERT_CACHE_GROUPED_PREFILL': '0'}),
    ('strata', 'prefill-256', {'prefill': '256'}),
    ('strata', 'mtp-window-2', {'spec': '2'}),
    ('ninfer', 'cpu-node0', {'affinity': 'node0'}),
    ('strata', 'cpu-node0', {'affinity': 'node0'}),
    ('ninfer', 'baseline-repeat', {}),
    ('strata', 'baseline-repeat', {}),
]
NINFER_FLAGS = ['--max-context', '4096', '--kv-capacity', '4096', '--max-concurrency', '1',
                '--prefill-chunk', '2048', '--kv-dtype', 'fp8', '--no-prefix-reuse',
                '--no-thinking', '--spec', 'mtp', '--draft-tokens', '1', '--lm-head-draft']


def option(flags, name, value):
    flags = list(flags)
    if name in flags:
        at = flags.index(name)
        if at + 1 >= len(flags):
            raise ValueError(f'missing value for {name}')
        flags[at+1] = str(value)
    else:
        flags.extend((name, str(value)))
    return flags


def remove_option(flags, name):
    flags = list(flags)
    while name in flags:
        at = flags.index(name)
        if at + 1 >= len(flags):
            raise ValueError(f'missing value for {name}')
        del flags[at:at+2]
    return flags


def gpu_free_mib():
    cp = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                         '--format=csv,noheader,nounits'], capture_output=True,
                        text=True, check=True)
    return int(cp.stdout.strip())


def release_gpu():
    # Never stop someone else's process. Only our launched server process group
    # is terminated by qualify_v100.stop() before this postcondition is checked.
    for _ in range(20):
        if gpu_free_mib() >= 28000:
            return
        time.sleep(2)
    raise RuntimeError('V100 still occupied after terminating our process; stop campaign')


def node0_allowed_cpus():
    # Cgroup-bound GitHub Actions containers cannot call set_mempolicy
    # (--preferred=0 fails EPERM). Limit CPU placement only, and name the
    # control accurately. Do not claim that memory affinity is enforced.
    allowed = os.sched_getaffinity(0)
    cpulist = Path('/sys/devices/system/node/node0/cpulist').read_text().strip()
    cpus = set()
    for segment in cpulist.split(','):
        bounds = segment.split('-')
        if len(bounds) == 1:
            cpus.add(int(bounds[0]))
        elif len(bounds) == 2:
            cpus.update(range(int(bounds[0]), int(bounds[1]) + 1))
        else:
            raise ValueError('invalid node0 CPU range')
    subset = sorted(allowed & cpus)
    if len(subset) < 2:
        raise RuntimeError('node0 contains fewer than 2 allowed CPUs in this container')
    return subset


def native_ninfer_requests(path):
    if not path.exists():
        return []
    events = []
    with path.open() as stream:
        for line in stream:
            if '"event":"request_done"' in line or '"event": "request_done"' in line:
                rec = json.loads(line)
                events.append(dict(request_id=rec.get('request', {}).get('request_id'),
                                   timings_seconds=rec.get('timings_seconds'),
                                   result=rec.get('result'),
                                   speculative=rec.get('speculative'),
                                   engine_timing=rec.get('engine_timing')))
    return events


def summary_for(rows):
    result = {}
    for prompt in ('long', 'short'):
        selected = [r for r in rows if r['phase'] == 'measured' and r['prompt'] == prompt]
        if not selected:
            continue
        wall = [r['wall_seconds'] for r in selected]
        usage = [r['response']['usage'] for r in selected]
        median_tokens = statistics.median(u['completion_tokens'] for u in usage)
        times = [r.get('native_timing') or r['response'].get('timings') for r in selected]
        result[prompt] = dict(wall_median_seconds=statistics.median(wall),
            wall_range_fraction=(max(wall)-min(wall))/statistics.median(wall),
            completion_tokens_median=median_tokens,
            prompt_tokens_median=statistics.median(u['prompt_tokens'] for u in usage),
            approximate_output_tokens_per_http_second=median_tokens/statistics.median(wall),
            output_hashes=sorted({r['output_sha256'] for r in selected}),
            native_timings=times)
    return result


def run_case(args, engine, variant, overrides):
    tag = f'{engine}-{variant}'
    folder = args.output/tag
    folder.mkdir()
    env = os.environ.copy()
    env.update(V100_COMPARE_TELEMETRY_LEVEL='1', V100_COMPARE_RUN_ID=tag,
               CUDA_VISIBLE_DEVICES='0', STRATA_REQUEST_LINES='1')
    port = free_port()
    if engine == 'ninfer':
        env.update(NINFER_V100_TELEMETRY='0', NINFER_FLASH_NEXT_STAGE_LEDGER='0',
                   NINFER_V100_EXPERT_PROFILE=str(args.profile.resolve()),
                   NINFER_V100_EXPERT_POLICY='static',
                   NINFER_FLASH_NEXT_EXPERT_CACHE='1',
                   NINFER_FLASH_NEXT_EXPERT_CACHE_MAX_SLOTS='156',
                   NINFER_V100_PLE_IO='mmap')
        for key, value in overrides.items():
            if key.startswith('NINFER_'):
                env[key] = value
        cmd = ([str(args.ninfer.resolve()), str(args.artifact.absolute()),
                '--host', '127.0.0.1', '--port', str(port)] + NINFER_FLAGS +
               ['--request-log-jsonl', str((folder/'native-requests.jsonl').resolve())])
        native_log = folder/'server.log'
    else:
        native_log = folder/'native-engine.log'
        cfg = strata_config(args.strata_config, args.strata_binary, native_log, 1, tag)
        flags = option(cfg['args'], '--prompt-cache', '0')
        if 'cache' in overrides:
            flags = option(flags, '--expert-cache', overrides['cache'])
            # Keep the static expert profile even with cache disabled:
            # Strata's resident-CPU expert mode requires that ranking.
        if 'workers' in overrides:
            flags = option(flags, '--pool-workers', overrides['workers'])
        if 'prefill' in overrides:
            flags = option(flags, '--prefill', overrides['prefill'])
        if 'spec' in overrides:
            flags = option(flags, '--spec', overrides['spec'])
        cfg['args'] = flags
        config_path = folder/'config.json'
        config_path.write_text(json.dumps(cfg, indent=2)+'\n')
        cmd = [str(args.strata_python.absolute()), str(args.strata_server.resolve()),
               '--engine', 'strata', '--config', str(config_path.resolve()),
               '--host', '127.0.0.1', '--port', str(port)]
    if overrides.get('affinity') == 'node0':
        if not shutil.which('taskset'):
            raise RuntimeError('taskset is unavailable')
        node0 = node0_allowed_cpus()
        cmd = ['taskset', '-c', ','.join(str(n) for n in node0)] + cmd
    (folder/'launch.json').write_text(json.dumps(dict(command=cmd,
        overrides=overrides, env={k:v for k,v in env.items() if
            k.startswith(('NINFER_', 'V100_', 'CUDA_', 'STRATA_'))}),
        indent=2)+'\n')
    rows, sampler = [], None
    from qualify_v100 import api as query
    import signal
    with (folder/'server.log').open('w') as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            start_up = time.monotonic()
            model = None
            while time.monotonic()-start_up < 600:
                if proc.poll() is not None:
                    raise RuntimeError(f'{tag}: exited before HTTP ready, inspect server.log')
                try:
                    models = query(f'http://127.0.0.1:{port}', '/v1/models')
                    model = models['data'][0]['id']
                    break
                except (OSError, ValueError, KeyError, IndexError):
                    time.sleep(1)
            if model is None:
                raise TimeoutError(f'{tag}: model did not become ready in 600s')
            sampler = subprocess.Popen([sys.executable, str(Path(__file__).with_name('sample_system.py')),
                '--pid', str(proc.pid), '--engine', engine, '--run-id', tag, '--level', '1',
                '--gpu-uuid', args.gpu_uuid, '--interval-ms', '500',
                '--output', str(folder/'system.jsonl')])
            # The five long-prompt warmups build expert/cache residency.
            schedule = [('warmup', 'long')]*args.warmups + [
                ('measured', 'long')]*args.repeats + [('measured', 'short')]*args.repeats
            with (folder/'requests.jsonl').open('w') as out:
                for index, (phase, which) in enumerate(schedule):
                    payload = dict(model=model, messages=[dict(
                        role='user', content=PROMPT if which == 'long' else SHORT)],
                        max_tokens=64, temperature=0, top_p=1, seed=42,
                        enable_thinking=False)
                    if engine == 'strata':
                        payload['chat_template_kwargs'] = {'enable_thinking': False}
                    begun=time.monotonic_ns()
                    response = query(f'http://127.0.0.1:{port}', '/v1/chat/completions', payload)
                    ended=time.monotonic_ns()
                    choice=response['choices'][0]
                    msg=choice['message']
                    content=msg.get('content') or ''
                    reasoning=msg.get('reasoning_content') or msg.get('reasoning') or ''
                    if choice.get('finish_reason') not in ('length','stop') or (not content and not reasoning):
                        raise RuntimeError(f'{tag}: invalid or empty inference response')
                    if engine == 'strata' and not content:
                        raise RuntimeError(f'{tag}: no-thinking contract returned no answer content')
                    digest=hashlib.sha256(json.dumps(dict(content=content,reasoning=reasoning,
                        tool_calls=msg.get('tool_calls')),sort_keys=True).encode()).hexdigest()
                    row=dict(schema='v100-strata-ab-request-v1',engine=engine,variant=variant,
                        index=index,phase=phase,prompt=which,
                        wall_seconds=(ended-begun)/1e9,output_sha256=digest,
                        response=response)
                    out.write(json.dumps(row,allow_nan=False)+'\n');out.flush()
                    rows.append(row)
                    print(f'{tag} {phase} {which} {index+1}/{len(schedule)} '
                          f'{row["wall_seconds"]:.2f}s',flush=True)
        finally:
            stop(proc)
            if sampler:
                sampler.terminate()
                try: sampler.wait(timeout=10)
                except subprocess.TimeoutExpired: sampler.kill();sampler.wait(timeout=5)
    natives = native_ninfer_requests(folder/'native-requests.jsonl')
    if natives and len(natives) == len(rows):
        for row, native in zip(rows, natives):
            row['native_timing'] = native['timings_seconds']
            row['native_speculative'] = native['speculative']
    (folder/'result.json').write_text(json.dumps(summary_for(rows),indent=2,allow_nan=False)+'\n')
    return dict(engine=engine,variant=variant,overrides=overrides,status='ok',
                request_count=len(rows),summary=summary_for(rows),
                ninfer_native_request_count=len(natives))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('ninfer','artifact','profile','strata-python','strata-server',
                 'strata-config','strata-binary','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--warmups',type=int,default=5)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--case',action='append',metavar='ENGINE/VARIANT',default=[],
                   help='Repeat to run only specified controls, in original matrix order.')
    a=p.parse_args()
    if a.warmups<2 or a.repeats<2:
        p.error('at least 2 warmups and 2 measured repetitions required')
    a.output.mkdir(parents=True,exist_ok=True)
    selection = set(a.case)
    available = {engine+'/'+variant for engine,variant,_ in CASES}
    if selection - available:
        p.error('unknown A/B case(s): '+', '.join(sorted(selection-available)))
    cases = [c for c in CASES if not selection or c[0]+'/'+c[1] in selection]
    if not all((e,'baseline') in [(x[0],x[1]) for x in cases] for e in ('ninfer','strata')):
        p.error('both engines require their baseline for a controlled comparison')
    (a.output/'matrix.json').write_text(json.dumps(dict(cases=cases,warmups=a.warmups,
        repeats=a.repeats,cross_quant_perf_is_not_controlled=True,
        main_config='same model and quantization within each engine',
        absolute_baseline_comparison='descriptive only'),indent=2)+'\n')
    results=[]
    try:
        for engine,variant,overrides in cases:
            if gpu_free_mib()<28000:
                raise RuntimeError('GPU occupied before A/B case; no process was killed')
            try:
                row=run_case(a,engine,variant,overrides)
            except Exception as error:
                row=dict(engine=engine,variant=variant,overrides=overrides,status='failed',
                         error=str(error),traceback=traceback.format_exc())
                print('A/B cell FAILED: '+engine+'/'+variant+': '+str(error),flush=True)
                if variant=='baseline':
                    results.append(row)
                    raise
            results.append(row)
            (a.output/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
            release_gpu()
    finally:
        (a.output/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    baselines={e:next((x for x in results if x['engine']==e and x['variant']=='baseline'
                         and x['status']=='ok'),None) for e in ('ninfer','strata')}
    ratios=[]
    for r in results:
        base=baselines[r['engine']]
        if r['status']!='ok' or base is None:
            continue
        for prompt in ('long','short'):
            first=base['summary'][prompt]['wall_median_seconds']
            current=r['summary'][prompt]['wall_median_seconds']
            ratios.append(dict(engine=r['engine'],variant=r['variant'],prompt=prompt,
                               median_wall_s=current,
                               ratio_to_same_engine_baseline=current/first,
                               speedup_vs_same_engine_baseline=first/current))
    (a.output/'ratios.json').write_text(json.dumps(ratios,indent=2)+'\n')
    print(json.dumps(dict(results=[dict(engine=r['engine'],variant=r['variant'],
              status=r['status']) for r in results],ratios=ratios),indent=2),flush=True)
    # Experimental cells may be unsupported, but the baseline and evidence are
    # still useful. The summary marks failures; never silently omit them.


if __name__=='__main__':
    main()
