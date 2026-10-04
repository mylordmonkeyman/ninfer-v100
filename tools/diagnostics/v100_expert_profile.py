"""Build and evaluate static SV1 residency proposals from separate SV0 traces."""
import argparse
import hashlib
import json
from pathlib import Path
from v100_sv0_report import read_records, summarize


def frequencies(path, phase):
    records = read_records(path)
    # Validate complete SV0 records before selecting the requested phase.
    summarize(records)
    rows = [r for r in records if r['kind'] == 'expert_layer' and
            (phase == 'all' or r['prefill'] == (phase == 'prefill'))]
    if not rows:
        raise ValueError(f'{path}: no expert observations for {phase}')
    counts = [[0] * 512 for _ in range(48)]
    for row in rows:
        for expert, count in enumerate(row['frequency']):
            counts[row['layer']][expert] += count
    return counts


def build_profile(counts, artifact_sha256, phase):
    if len(artifact_sha256) != 64 or any(c not in '0123456789abcdef' for c in artifact_sha256):
        raise ValueError('artifact SHA256 must be 64 lowercase hexadecimal characters')
    return dict(magic='NINFER_V100_EXPERT_PROFILE', version=1,
                artifact_sha256=artifact_sha256, layers=48, experts_per_layer=512,
                phase=phase, ranking=[sorted(range(512), key=lambda e: (-row[e], e))
                                     for row in counts], frequency=counts)


def evaluate(profile, counts, slots):
    if len(slots) != 48 or any(type(n) is not int or not 0 <= n <= 512 for n in slots):
        raise ValueError('slot budget must contain 48 integers in [0, 512]')
    layers = []
    for layer, row in enumerate(counts):
        selected = set(profile['ranking'][layer][:slots[layer]])
        hits = sum(count for expert, count in enumerate(row) if expert in selected)
        routes = sum(row)
        layers.append(dict(layer=layer, slots=slots[layer], routes=routes,
                           projected_hits=hits, hit_fraction=hits/routes if routes else None))
    total = sum(r['routes'] for r in layers)
    hits = sum(r['projected_hits'] for r in layers)
    return dict(schema=1, scope='static_residency_trace_projection', qualified=False,
                layers=layers, routes=total, projected_hits=hits,
                hit_fraction=hits/total if total else None,
                unavailable=['throughput', 'cpu_miss_time', 'fill_time', 'evictions',
                             'actual_cache_hits', 'oracle_acceptance'])


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-log', type=Path, required=True)
    parser.add_argument('--evaluation-log', type=Path, required=True)
    parser.add_argument('--artifact-sha256', required=True)
    parser.add_argument('--slots', type=Path, required=True, help='JSON array of 48 per-layer slot capacities')
    parser.add_argument('--phase', choices=['decode', 'prefill', 'all'], default='decode')
    parser.add_argument('--profile-output', type=Path, required=True)
    parser.add_argument('--evaluation-output', type=Path, required=True)
    args = parser.parse_args()
    train_hash = hashlib.sha256(args.train_log.read_bytes()).hexdigest()
    evaluation_hash = hashlib.sha256(args.evaluation_log.read_bytes()).hexdigest()
    if train_hash == evaluation_hash:
        raise ValueError('training and evaluation traces must differ')
    if args.profile_output.resolve() == args.evaluation_output.resolve():
        raise ValueError('profile and evaluation outputs must differ')
    profile = build_profile(frequencies(args.train_log, args.phase), args.artifact_sha256, args.phase)
    profile['training_trace_sha256'] = train_hash
    result = evaluate(profile, frequencies(args.evaluation_log, args.phase), json.loads(args.slots.read_text()))
    result.update(artifact_sha256=args.artifact_sha256, training_trace_sha256=train_hash,
                  evaluation_trace_sha256=evaluation_hash, phase=args.phase)
    atomic_json(args.profile_output, profile)
    atomic_json(args.evaluation_output, result)
    print('Static residency proposal written; hardware policy qualification remains pending.')


if __name__ == '__main__':
    main()
