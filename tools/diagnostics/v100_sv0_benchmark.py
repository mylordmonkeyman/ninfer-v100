"""ABBA baseline/candidate screening through the existing teacher-forced harness."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--id=0', '-q', '-d', 'PERFORMANCE,TEMPERATURE,CLOCK'],
                            capture_output=True, text=True, check=True)
    text = result.stdout
    thermal = re.findall(r'(?:HW|SW) Thermal Slowdown\s*:\s*(Not Active|Active)', text)
    return dict(monotonic_seconds=time.monotonic(), thermal_status_observed=bool(thermal),
                thermal_throttled='Active' in thermal, detail=text)


def parse_benchmark(text):
    marker = 'PASS: Phase 14 warmed benchmark and fixed-cache exact schedule parity'
    if marker not in text:
        raise ValueError('missing correctness/state-frontier marker')
    rows = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    rows = [r for r in rows if r.get('phase14') == 'benchmark']
    if len(rows) != 8:
        raise ValueError('expected four observations for each of decode and prefill')
    for shape in ('decode', 'prefill'):
        selected = [r for r in rows if r['shape'] == shape]
        if sorted(r['sample'] for r in selected) != [0, 1, 2, 3]:
            raise ValueError('missing or duplicate benchmark observation')
    for row in rows:
        seconds, rate = row['seconds'], row['tokens_per_s']
        if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(rate) or rate <= 0:
            raise ValueError('invalid benchmark timing')
        if not math.isclose(rate, row['tokens']/seconds, rel_tol=1e-8):
            raise ValueError('benchmark rate disagrees with wall time')
    return rows


def summary(observations):
    cells = {}
    for observation in observations:
        snapshots = observation['gpu_snapshots']
        if not snapshots or any(not s['thermal_status_observed'] or s['thermal_throttled'] for s in snapshots):
            raise ValueError('thermal status missing or throttling observed: comparison invalid')
        grouped = {}
        for row in observation['rows']:
            key = (row['shape'], row['mode'])
            grouped.setdefault(key, []).append(row)
        for key, rows in grouped.items():
            # One process contributes one median. Inner harness samples are not independent
            # process replications and must not inflate the reported observation count.
            cell = cells.setdefault(key, {'baseline': [], 'candidate': [], 'slots': set()})
            cell[observation['arm']].append(statistics.median(r['tokens_per_s'] for r in rows))
            cell['slots'].update(r['slots_per_layer'] for r in rows)
    results = []
    for (shape, mode), cell in sorted(cells.items()):
        if len(cell['slots']) != 1:
            raise ValueError('baseline/candidate cache capacity differs')
        if min(len(cell[a]) for a in ('baseline', 'candidate')) < 3:
            raise ValueError('fewer than three process observations per comparison cell')
        result = dict(shape=shape, mode=mode, slots_per_layer=next(iter(cell['slots'])))
        for arm in ('baseline', 'candidate'):
            values = cell[arm]
            result[arm] = dict(observations=len(values), median=statistics.median(values),
                               minimum=min(values), maximum=max(values))
        result['median_change_percent'] = 100*(result['candidate']['median']/result['baseline']['median']-1)
        results.append(result)
    if not results:
        raise ValueError('no valid comparison cells')
    return dict(schema=1, milestone='SV0', scope='teacher_forced_screening_including_logit_downloads',
                qualified=False, results=results,
                limitations=['not production request throughput', 'short 64-token screening workload',
                             'PLE warmed at model startup', 'does not establish the complete SV0 request matrix'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=2)
    args = parser.parse_args()
    if args.cycles < 2:
        parser.error('at least two ABBA cycles are needed for three or more observations per arm')
    args.output.mkdir(parents=True, exist_ok=True)
    artifact = Path(os.environ['NINFER_WEIGHTS'])
    with artifact.open('rb') as stream:
        artifact_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    provenance = dict(artifact=str(artifact), artifact_sha256=artifact_sha,
                      baseline_sha=(args.output/'baseline-sha.txt').read_text().strip(),
                      candidate_sha=(args.output/'candidate-sha.txt').read_text().strip(),
                      runtime_environment={k: v for k, v in os.environ.items() if k.startswith('NINFER_')},
                      numa_policy='inherited runner placement; numa.txt records host topology',
                      cache_state='fresh process, harness warms and freezes the Ready set',
                      filesystem_state='warm; artifact hashing and oracle runs precede screening')
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    observations = []
    for cache in (0, 1):
        for cycle in range(args.cycles):
            for order, arm in enumerate(('baseline', 'candidate', 'candidate', 'baseline')):
                executable = args.baseline if arm == 'baseline' else args.candidate
                env = os.environ.copy()
                for key in tuple(env):
                    if key.startswith('NINFER_PHASE'):
                        env.pop(key)
                env.update(NINFER_PHASE14_BENCHMARK='1', NINFER_PHASE11_ORACLE_SMOKE_POSITIONS='64',
                           NINFER_V100_TELEMETRY='0', NINFER_FLASH_NEXT_STAGE_LEDGER='0',
                           NINFER_FLASH_NEXT_EXPERT_CACHE_TIMING='0', NINFER_FLASH_NEXT_EXPERT_CACHE=str(cache))
                name = f'cache{cache}-cycle{cycle}-{order}-{arm}'
                snapshots = [gpu_snapshot()]
                done = threading.Event()
                errors = []
                def monitor():
                    while not done.wait(5):
                        try:
                            snapshots.append(gpu_snapshot())
                        except Exception as error:
                            errors.append(str(error))
                worker = threading.Thread(target=monitor)
                worker.start()
                try:
                    with (args.output/f'{name}.log').open('w') as output:
                        result = subprocess.run([str(executable.resolve())], env=env, stdout=output, stderr=subprocess.STDOUT)
                finally:
                    done.set()
                    worker.join()
                snapshots.append(gpu_snapshot())
                observation = dict(arm=arm, cache=cache, cycle=cycle, order=order,
                                   gpu_snapshots=snapshots, monitor_errors=errors)
                (args.output/f'{name}-hardware.json').write_text(json.dumps(observation, indent=2)+'\n')
                if result.returncode != 0 or errors:
                    raise RuntimeError(f'{name}: benchmark or hardware monitor failed')
                observation['rows'] = parse_benchmark((args.output/f'{name}.log').read_text())
                observations.append(observation)
                (args.output/'observations.json').write_text(json.dumps(observations, indent=2, allow_nan=False)+'\n')
                print(f'{name}: passed', flush=True)
    report = summary(observations)
    report['provenance'] = provenance
    (args.output/'screening.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['SV0 teacher-forced screening; diagnostics disabled.',
             '| Shape | Cache schedule | Baseline median (range) | Candidate median (range) | Median change |',
             '|---|---|---:|---:|---:|']
    for row in report['results']:
        b, c = row['baseline'], row['candidate']
        lines.append(f"| {row['shape']} | {row['mode']} | {b['median']:.3f} ({b['minimum']:.3f}–{b['maximum']:.3f}) | "
                     f"{c['median']:.3f} ({c['minimum']:.3f}–{c['maximum']:.3f}) | {row['median_change_percent']:+.2f}% |")
    lines.append('Screening only. Production request matrix and SV0 qualification remain pending.')
    (args.output/'screening.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
