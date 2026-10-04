"""Real-artifact SV1 training/persistence smoke and held-out policy screening."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading

from v100_expert_profile import atomic_json, build_profile, evaluate, frequencies
from v100_sv0_benchmark import gpu_snapshot
from v100_sv0_report import read_records, summarize

ARMS = ('off', 'lru', 'static', 'heat', 'decay', 'profile')


def parse_run(path):
    text = path.read_text()
    if 'PASS: SV1 real-artifact workload finite and committed' not in text:
        raise ValueError(f'{path}: incomplete workload or missing state/finite check')
    rows = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    rows = [r for r in rows if 'sv1' in r]
    startup = [r for r in rows if r['sv1'] == 'startup']
    complete = [r for r in rows if r['sv1'] == 'complete']
    measurements = [r for r in rows if r['sv1'] == 'measurement']
    if len(startup) != 1 or len(complete) != 1 or sorted(r['shape'] for r in measurements) != ['decode', 'prefill']:
        raise ValueError('missing or duplicate SV1 records')
    for row in measurements:
        if row['seconds'] <= 0 or not math.isfinite(row['seconds']) or not math.isclose(
                row['tokens_per_s'], row['tokens']/row['seconds'], rel_tol=1e-8):
            raise ValueError('invalid workload timing')
        if row['maximum_outstanding'] > 4:
            raise ValueError('fill queue bound exceeded')
    if sum(r['tokens'] for r in measurements) != complete[0]['tokens']:
        raise ValueError('incomplete token workload')
    decode = next(r for r in measurements if r['shape'] == 'decode')
    if len(complete[0]['decode_top1']) != decode['tokens']:
        raise ValueError('incomplete decode output checks')
    return dict(startup=startup[0], complete=complete[0], measurements=measurements)


def diagnostic_summary(path, token_count):
    records = read_records(path)
    report = summarize(records)
    rows = [r for r in records if r['kind'] == 'expert_layer']
    for layer in range(48):
        if sum(r['routes'] for r in rows if r['layer'] == layer) != token_count*10:
            raise ValueError(f'{path}: incomplete real-model routes in layer {layer}')
    result = []
    for prefill in (True, False):
        cells = [r for r in rows if r['prefill'] == prefill]
        hits = sum(r['gpu_hit_routes'] for r in cells)
        routes = sum(r['routes'] for r in cells)
        result.append(dict(shape='prefill' if prefill else 'decode', routes=routes, hits=hits,
                           hit_fraction=hits/routes, cpu_branch_us=sum(r['cpu_branch_us'] for r in cells),
                           by_layer=[r for r in report['layer_summary'] if r['prefill'] == prefill]))
    return result


def read_logits(path, words):
    values = array.array('H')
    values.frombytes(path.read_bytes())
    if sys.byteorder != 'little':
        values.byteswap()
    if len(values) != words:
        raise ValueError('incomplete final logits')
    import struct
    logits = [struct.unpack('<f', struct.pack('<I', v << 16))[0] for v in values]
    if not all(math.isfinite(v) for v in logits):
        raise ValueError('nonfinite final logits')
    return logits


def compare_logits(expected, actual):
    if len(expected) != len(actual):
        raise ValueError('logit shape mismatch')
    def logs(values):
        maximum = max(values)
        normalizer = maximum + math.log(sum(math.exp(v-maximum) for v in values))
        return [v-normalizer for v in values]
    p, q = logs(expected), logs(actual)
    return dict(final_position_kl=sum(math.exp(x)*(x-y) for x, y in zip(p, q)),
                final_position_maximum_error=max(abs(x-y) for x, y in zip(expected, actual)),
                final_position_nrmse=math.sqrt(sum((x-y)**2 for x, y in zip(expected, actual))/
                                              max(sum(x*x for x in expected), 1e-30)),
                bitwise_equal=expected == actual)


def run(executable, output, name, workload, arm, slots, profile=None, learned=None, telemetry=False, discover=False):
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith('NINFER_PHASE') or key.startswith('NINFER_V100_EXPERT_') or key.startswith('NINFER_V100_SV1_'):
            env.pop(key)
    env.update(NINFER_V100_SV1_WORKLOAD=str(workload.resolve()),
               NINFER_V100_SV1_LOGITS=str((output/f'{name}.bf16').resolve()),
               NINFER_FLASH_NEXT_EXPERT_CACHE='0' if arm == 'off' else '1',
               NINFER_FLASH_NEXT_EXPERT_CACHE_MAX_SLOTS=str(slots),
               NINFER_FLASH_NEXT_EXPERT_CACHE_ADMISSION_CAP='1',
               NINFER_FLASH_NEXT_EXPERT_CACHE_SERIAL='0',
               NINFER_FLASH_NEXT_EXPERT_CACHE_PREFILL='1',
               NINFER_FLASH_NEXT_EXPERT_CACHE_BATCHED_PREFILL='0',
               NINFER_FLASH_NEXT_EXPERT_CACHE_GROUPED_PREFILL='1',
               NINFER_FLASH_NEXT_EXPERT_CACHE_BATCHED_DECODE='0',
               NINFER_V100_EXPERT_POLICY='lru' if arm == 'off' else arm,
               NINFER_V100_TELEMETRY='1' if telemetry else '0',
               NINFER_FLASH_NEXT_EXPERT_CACHE_TIMING='1' if telemetry else '0',
               NINFER_FLASH_NEXT_STAGE_LEDGER='0',
               NINFER_FLASH_NEXT_FP32_MOE_ROUTED_INPUT='0',
               NINFER_FLASH_NEXT_CPU_EXPERT_FP32_INTERMEDIATE='0')
    if profile:
        env['NINFER_V100_EXPERT_PROFILE'] = str(profile.resolve())
    if learned:
        env['NINFER_V100_EXPERT_PROFILE_SAVE'] = str(learned.resolve())
    if arm == 'decay':
        env.update(NINFER_V100_EXPERT_DECAY='0.9', NINFER_V100_EXPERT_DECAY_INTERVAL='32')
    if arm == 'profile':
        env['NINFER_V100_EXPERT_PRIOR_WEIGHT'] = '32'
    snapshots = [gpu_snapshot()]
    done, errors = threading.Event(), []
    def monitor():
        while not done.wait(5):
            try:
                snapshots.append(gpu_snapshot())
            except Exception as error:
                errors.append(str(error))
    worker = threading.Thread(target=monitor)
    worker.start()
    try:
        with (output/f'{name}.log').open('w') as log:
            process = subprocess.run([str(executable.resolve())], env=env, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=1200)
    finally:
        done.set()
        worker.join()
    snapshots.append(gpu_snapshot())
    atomic_json(output/f'{name}-hardware.json', dict(snapshots=snapshots, errors=errors,
                environment={k:v for k,v in env.items() if k.startswith('NINFER_')}))
    if process.returncode or errors:
        raise RuntimeError(f'{name}: executable or hardware monitor failed; see preserved logs')
    if any(not s['thermal_status_observed'] or s['thermal_throttled'] for s in snapshots):
        raise ValueError(f'{name}: thermal evidence unavailable or throttled')
    result = parse_run(output/f'{name}.log')
    result.update(name=name, arm=arm, telemetry=telemetry)
    if not discover and arm != 'off' and result['startup']['slots_per_layer'] != slots:
        raise ValueError(f'{name}: unequal cache capacity')
    print(f'{name}: completed', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('at least three independent process observations required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(os.environ['NINFER_FLASH_NEXT_ORACLE_MANIFEST'])
    positions = json.loads(manifest.read_text())['positions']
    # Explicit nonoverlapping contiguous corpus segments, each with fresh state.
    if len(positions) < 2432 or any(row['position'] != i for i,row in enumerate(positions)):
        raise ValueError('need contiguous source corpus through position 2431')
    for name, start in (('training', 0), ('held-out', 2048)):
        atomic_json(output/f'{name}.json', dict(name=name, source_manifest=str(manifest),
                    source_start=start, source_end=start+384, prefill_tokens=128,
                    tokens=[r['token_id'] for r in positions[start:start+384]]))
    train, held = (output/'training.json', output/'held-out.json')
    train_tokens, held_tokens = (json.loads(p.read_text())['tokens'] for p in (train,held))
    if train_tokens == held_tokens:
        raise ValueError('training and evaluation token sequences must differ')
    # Training discovers the planner's safe capacity; all later cached arms use it.
    learned = output/'training-learned.json'
    # Discover the planner's safe capacity; enforce this exact cap in evaluation.
    training_env_slots = 512
    discovery = run(args.executable,output,'training',train,'lru',training_env_slots,
                    learned=learned,telemetry=True, discover=True)
    slots = discovery['startup']['slots_per_layer']
    if slots <= 0:
        raise ValueError('no cache capacity available')
    counts = frequencies(output/'training.log','all')
    diagnostic_summary(output/'training.log',384)
    runtime_profile = json.loads(learned.read_text())
    if runtime_profile['frequency'] != counts or runtime_profile['model_id'] != discovery['startup']['model_id'] or runtime_profile['weights_id'] != discovery['startup']['weights_id']:
        raise ValueError('real-model shutdown profile identity/frequencies disagree with measured routes')
    if learned.with_name(learned.name+'.tmp').exists():
        raise ValueError('shutdown left an unfinished profile write')
    with Path(os.environ['NINFER_WEIGHTS']).open('rb') as stream:
        artifact_sha = hashlib.file_digest(stream,'sha256').hexdigest()
    profile = build_profile(counts,artifact_sha,'all',discovery['startup']['model_id'],
                            discovery['startup']['weights_id'])
    profile['training_trace_sha256'] = hashlib.sha256(train.read_bytes()).hexdigest()
    ranked = output/'ranked-profile.json'
    atomic_json(ranked,profile)
    # Persistence smoke consumes the actual runtime-produced profile, distinct from
    # the frequency-ranked proposal used for policy comparisons.
    smoke = run(args.executable,output,'learned-reload',held,'static',slots,learned,telemetry=True)
    smoke['diagnostics'] = diagnostic_summary(output/'learned-reload.log',384)
    atomic_json(output/'persistence-smoke.json',dict(training=discovery,reload=smoke,qualified=False))
    diagnostic = {}
    for arm in ARMS:
        name=f'diagnostic-{arm}'
        observation=run(args.executable,output,name,held,arm,slots,
                        ranked if arm not in ('off','lru') else None,telemetry=True)
        observation['diagnostics']=diagnostic_summary(output/f'{name}.log',384)
        diagnostic[arm]=observation
        atomic_json(output/'diagnostic-observations.json',diagnostic)
    static_projection=evaluate(profile,frequencies(output/'diagnostic-static.log','all'),[slots]*48)
    static_projection['cpu_only_route_projection']=evaluate(
        profile,frequencies(output/'diagnostic-off.log','all'),[slots]*48)
    static_hits=sum(r['hits'] for r in diagnostic['static']['diagnostics'])
    if static_hits != static_projection['projected_hits']:
        raise ValueError('actual static GPU hits disagree with held-out training-profile projection')
    atomic_json(output/'profile-coverage.json',static_projection)
    observations=[]
    for repeat in range(args.repeats):
        # Alternate process order to reduce order bias; each starts with fresh state/cache.
        for arm in (ARMS if repeat%2 == 0 else tuple(reversed(ARMS))):
            name=f'timing-{repeat}-{arm}'
            observation=run(args.executable,output,name,held,arm,slots,
                            ranked if arm not in ('off','lru') else None)
            observations.append(observation)
            atomic_json(output/'timing-observations.json',observations)
    cache_rows=[r['startup'] for r in [*diagnostic.values(),*observations] if r['arm']!='off']
    for key in ('cache_bytes','transfer_budget_bytes','runtime_plan_device_bytes','slots_per_layer'):
        if len({r[key] for r in cache_rows}) != 1:
            raise ValueError(f'cached policy VRAM allocation category differs: {key}')
    results=[]
    for arm in ARMS:
        rows=[r for r in observations if r['arm']==arm]
        rates={}
        for shape in ('prefill','decode'):
            values=[next(m['tokens_per_s'] for m in r['measurements'] if m['shape']==shape) for r in rows]
            rates[shape]=dict(observations=len(values),median=statistics.median(values),
                              minimum=min(values),maximum=max(values))
        results.append(dict(arm=arm,throughput=rates,diagnostic=diagnostic[arm]))
    off=diagnostic['off']
    reference=read_logits(output/'diagnostic-off.bf16',off['complete']['final_logit_words'])
    numerical=[]
    for observation in [*diagnostic.values(),*observations,smoke]:
        actual=read_logits(output/f"{observation['name']}.bf16",observation['complete']['final_logit_words'])
        comparison=compare_logits(reference,actual)
        comparison.update(name=observation['name'],
            decode_top1_agreement=sum(a==b for a,b in zip(off['complete']['decode_top1'],
                observation['complete']['decode_top1']))/len(off['complete']['decode_top1']))
        numerical.append(comparison)
    report=dict(schema=1,milestone='SV1',qualified=False,
                scope='held_out_teacher_forced_policy_screening',results=results,numerical_screen=numerical,
                artifact_sha256=artifact_sha,candidate_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                train_token_sha256=hashlib.sha256(json.dumps(train_tokens).encode()).hexdigest(),
                evaluation_token_sha256=hashlib.sha256(json.dumps(held_tokens).encode()).hexdigest(),
                cache_slots_per_layer=slots,cache_capacity_equal=True,
                experimental_coefficients=dict(decay=0.9,decay_interval=32,prior_weight=32),
                limitations=['teacher-forced executor with logits downloads and finite checks; not production server throughput',
                    'single held-out corpus segment, fresh recurrent state, 128 prefill + 256 decode tokens',
                    'PLE/filesystem warm; telemetry collected separately from timing',
                    'diagnostic adaptive cache trajectories may differ from timing trajectories',
                    'whole-model independent oracle is unchanged and remains a separate preexisting failure',
                    'pairwise numerical screening is supplementary evidence, not independent oracle qualification',
                    'MTP, continuation, longer-context matrix and default promotion remain pending'])
    atomic_json(output/'report.json',report)
    lines=['SV1 held-out screening; no policy promotion.','',
           '| Policy | Decode t/s median (range) | Decode hit rate (diagnostic) | CPU miss branch ms (diagnostic) |',
           '|---|---:|---:|---:|']
    for row in results:
        rate=row['throughput']['decode']; d=next(r for r in row['diagnostic']['diagnostics'] if r['shape']=='decode')
        lines.append(f"| {row['arm']} | {rate['median']:.3f} ({rate['minimum']:.3f}–{rate['maximum']:.3f}) | {d['hit_fraction']:.2%} | {d['cpu_branch_us']/1000:.1f} |")
    lines.extend(['','Equal cache capacity for all cached arms. Cache-off intentionally allocates no cache.',
                  'Finite/state/persistence checks passed. Independent numerical and production performance qualification remain separate.'])
    (output/'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
