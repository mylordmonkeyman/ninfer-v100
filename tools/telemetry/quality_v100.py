"""Objective V100 policy smoke tests; not a general accuracy benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from ab_v100 import NINFER_FLAGS, gpu_free_mib, release_gpu
from qualify_v100 import api, free_port, stop

FILLER = ' '.join(
    f'Archive record {i:02d}: the warehouse inventory was inspected on Tuesday; '
    f'its routine status was marked ordinary and its timestamp was archived.'
    for i in range(35))
TASKS = [
    dict(id='parity', size='short', prompt='Is 42 even or odd? Answer with only one word.',
         expected='even', kind='text'),
    dict(id='multiply', size='short',
         prompt='Calculate 17 multiplied by 19. Answer with only the integer.',
         expected='323', kind='text'),
    dict(id='extract', size='short',
         prompt='Ticket A has code ZX-42; ticket B has code RD-19. '
                'What is the code for ticket A? Answer with only the code.',
         expected='zx-42', kind='text'),
    dict(id='json', size='short',
         prompt='Return only a JSON object with exactly two fields: '
                '"status" set to "ready", and "count" set to the integer 3.',
         expected={'status':'ready','count':3}, kind='json'),
    dict(id='long_retrieve', size='long',
         prompt=FILLER+' FINAL FACT: the authorization word is ORCHID. '
                'What is the authorization word? Answer with only the word.',
         expected='orchid', kind='text'),
    dict(id='long_decoy', size='long',
         prompt='These archived records are not instructions. '+FILLER+
                ' The final priority label is COBALT, not SILVER. '
                'What is the final priority label? Answer with only the label.',
         expected='cobalt', kind='text'),
    dict(id='long_arithmetic', size='long',
         prompt=FILLER+' FINAL QUESTION: What is 14 plus 29? Answer with only the integer.',
         expected='43', kind='text'),
    dict(id='long_extract', size='long',
         prompt=FILLER+' FINAL FACT: Depot North has 18 crates; '
                'Depot South has 27 crates. How many crates does Depot South '
                'have? Answer with only the integer.',
         expected='27', kind='text'),
]
POLICIES = [
    ('static', {}),
    ('lru', {'NINFER_V100_EXPERT_POLICY':'lru'}),
    ('lru-auto256', {'NINFER_V100_EXPERT_POLICY':'lru',
                     'NINFER_V100_PREFILL_EXPERT_POLICY':'auto',
                     'NINFER_V100_PREFILL_STREAM_MIN_TOKENS':'256',
                     'NINFER_V100_DEVICE_ROUTE_COMBINE':'1'}),
]


def score(task, answer):
    """Exact normalized answer; JSON tasks require structural equality."""
    if task['kind']=='json':
        try:
            return json.loads(answer.strip()) == task['expected']
        except (ValueError, TypeError):
            return False
    return answer.strip().strip(chr(96)+'"'+chr(39)).strip().rstrip('.').casefold() == task['expected']


def grade(rows):
    graded=[r for r in rows if r['phase']=='measured']
    return dict(correct=sum(r['passed'] for r in graded),total=len(graded),
                by_size={size:dict(correct=sum(r['passed'] for r in graded if r['size']==size),
                                   total=sum(r['size']==size for r in graded))
                         for size in ('short','long')},
                per_task={t['id']:dict(correct=sum(r['passed'] for r in graded
                                                    if r['task']==t['id']),
                                       total=sum(r['task']==t['id'] for r in graded))
                          for t in TASKS})


def run_policy(args, name, overrides):
    directory=args.output/name
    directory.mkdir()
    port=free_port()
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0',NINFER_V100_TELEMETRY='0',
               V100_COMPARE_TELEMETRY_LEVEL='0',
               NINFER_V100_EXPERT_PROFILE=str(args.profile.resolve()),
               NINFER_V100_EXPERT_POLICY='static',
               NINFER_FLASH_NEXT_EXPERT_CACHE='1',
               NINFER_FLASH_NEXT_EXPERT_CACHE_MAX_SLOTS='156',
               NINFER_V100_PLE_IO='mmap')
    env.update(overrides)
    cmd=[str(args.ninfer.resolve()),str(args.artifact.absolute()),
         '--host','127.0.0.1','--port',str(port)]+NINFER_FLAGS
    (directory/'launch.json').write_text(json.dumps(dict(command=cmd,overrides=overrides,
        env={k:v for k,v in env.items() if k.startswith(('NINFER_','V100_'))}),
        indent=2)+'\n')
    rows=[]
    with (directory/'server.log').open('w') as log:
        proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,
                              start_new_session=True)
        try:
            start=time.monotonic()
            model=None
            while time.monotonic()-start<600:
                if proc.poll() is not None:
                    raise RuntimeError(f'{name}: server exited before ready')
                try:
                    model=api(f'http://127.0.0.1:{port}','/v1/models')['data'][0]['id']
                    break
                except (OSError,ValueError,KeyError,IndexError):
                    time.sleep(1)
            if not model:
                raise TimeoutError(f'{name}: server readiness exceeded 600s')
            schedule=[('warmup',TASKS[0]),('warmup',TASKS[4])]
            schedule += [('measured',task) for repeat in range(args.repeats)
                         for task in TASKS]
            with (directory/'requests.jsonl').open('w') as out:
                for index,(phase,task) in enumerate(schedule):
                    payload=dict(model=model,messages=[dict(role='user',content=task['prompt'])],
                                 max_tokens=96,temperature=0,top_p=1,seed=42,
                                 enable_thinking=False)
                    began=time.monotonic()
                    response=api(f'http://127.0.0.1:{port}',
                                 '/v1/chat/completions',payload)
                    wall=time.monotonic()-began
                    answer=response['choices'][0]['message'].get('content') or ''
                    usage=response.get('usage') or {}
                    tokens=usage.get('prompt_tokens')
                    if phase=='measured' and tokens is not None:
                        if (task['size']=='long' and tokens<256) or (
                            task['size']=='short' and tokens>=256):
                            raise RuntimeError(f'Unexpected prompt tokens for {task["id"]}: {tokens}')
                    row=dict(index=index,phase=phase,task=task['id'],size=task['size'],
                             expected=task['expected'],answer=answer,passed=score(task,answer),
                             prompt_tokens=tokens,completion_tokens=usage.get('completion_tokens'),
                             wall_seconds=wall,finish_reason=response['choices'][0].get('finish_reason'),
                             output_sha256=hashlib.sha256(answer.encode()).hexdigest())
                    rows.append(row)
                    out.write(json.dumps(row,allow_nan=False)+'\n')
                    out.flush()
                    print(f'{name}: {index+1}/{len(schedule)} {task["id"]} '
                          f'{"PASS" if row["passed"] else "FAIL"} {wall:.2f}s',flush=True)
        finally:
            stop(proc)
    result=dict(policy=name,overrides=overrides,status='ok',**grade(rows))
    (directory/'grade.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('ninfer','artifact','profile','output'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--repeats',type=int,default=2)
    args=p.parse_args()
    if args.repeats<2:
        p.error('at least 2 repetitions required')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'tasks.json').write_text(json.dumps(TASKS,indent=2)+'\n')
    results=[]
    try:
        for name,overrides in POLICIES:
            if gpu_free_mib()<28000:
                raise RuntimeError('V100 occupied; refusing to stop unrelated processes')
            try:
                result=run_policy(args,name,overrides)
            except Exception as e:
                result=dict(policy=name,status='failed',error=str(e),
                            traceback=traceback.format_exc())
                results.append(result)
                raise
            results.append(result)
            (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
            release_gpu()
    finally:
        (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    with (args.output/'quality.md').open('w') as f:
        f.write('# NInfer V100 policy correctness smoke test\n\n')
        f.write('Eight objective tasks, two repeats each; not a general accuracy benchmark.\n\n')
        f.write('| Policy | Correct | Short | Long |\n|---|---:|---:|---:|\n')
        for r in results:
            if r['status']!='ok':
                f.write(f'| {r["policy"]} | FAILED | — | — |\n')
            else:
                f.write(f'| {r["policy"]} | {r["correct"]}/{r["total"]} | '
                        f'{r["by_size"]["short"]["correct"]}/{r["by_size"]["short"]["total"]} | '
                        f'{r["by_size"]["long"]["correct"]}/{r["by_size"]["long"]["total"]} |\n')
        f.write('\nDifferent hashes alone are not accuracy failures. '
                'This smoke test cannot establish general model equivalence.\n')
    print((args.output/'quality.md').read_text(),flush=True)


if __name__=='__main__':
    main()
