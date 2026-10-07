"""Check known CPU-pool job counts and level-dependent measurement coverage."""
import argparse
import json
from pathlib import Path
from schema import validate


def check(engine, level, path):
    records = [validate(json.loads(line)) for line in Path(path).read_text().splitlines()]
    if level == 0:
        if records:
            raise ValueError('disabled telemetry emitted records')
        return
    if len(records) != (1 if engine == 'ninfer' else 4):
        raise ValueError('missing CPU fixture rounds')
    for record in records:
        if record['engine'] != engine or record['level'] != level or record['status'] != 'ok':
            raise ValueError('invalid CPU fixture envelope')
        workers = record['layers'][0]['workers']
        if not all(('gate_up_us' in w) == (level >= 2) for w in workers):
            raise ValueError('worker timing coverage does not match level')
        if engine == 'ninfer':
            # Three grouped width-2/3/4 calls, each split into four row jobs
            # in each phase. Other pools exercise reuse and recovery.
            observed = [w for w in workers if w['pool_id'] == 0]
            expected = (0, 12, 12)
            if len(observed) != 4:
                raise ValueError('missing NInfer configured worker observations')
        else:
            _, configured, host = record['round_id'].split(':')
            configured, host = int(configured), int(host)
            observed = workers
            # Four whole-expert batches of four; three split-multi calls.
            phase_jobs = 9 * (configured + host)
            expected = (16, phase_jobs, phase_jobs)
            if len(observed) != configured + 1:
                raise ValueError('missing Strata worker or host observations')
            if configured > 1 and not host:
                drainer = [w for w in observed if w['role'] == 'host']
                if any(w['full_jobs'] + w['gate_up_jobs'] + w['down_jobs'] for w in drainer):
                    raise ValueError('disabled Strata host drainer claimed work')
        actual = tuple(sum(w[field] for w in observed)
                       for field in ('full_jobs', 'gate_up_jobs', 'down_jobs'))
        if actual != expected:
            raise ValueError(f'{engine}: lost/duplicated jobs: {actual} != {expected}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine', choices=('ninfer', 'strata'))
    parser.add_argument('level', type=int, choices=(0, 1, 2, 3))
    parser.add_argument('path')
    args = parser.parse_args()
    check(args.engine, args.level, args.path)
    print(f'{args.engine} Level {args.level}: CPU output fixture and worker coverage validated')
