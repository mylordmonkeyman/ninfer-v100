"""Re-analyze a frozen first-hardware qualification ZIP without starting a GPU.

Only preserved HTTP-observer evidence is used. This cannot prove deterministic
native token execution or GPU critical-path attribution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from qualify_v100 import report

ENGINES = ('ninfer', 'strata')
LEVELS = (0, 1, 2)


def replay(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        manifest = json.loads(archive.read('matrix/manifest.json'))
        expected = manifest['warmups'] + manifest['repeats']
        original = json.loads(archive.read('matrix/summary.json'))
        rows = []
        for engine in ENGINES:
            for level in LEVELS:
                name = f'matrix/qualification-{engine}-l{level}/requests.jsonl'
                samples = [json.loads(line) for line in archive.read(name).splitlines()]
                if len(samples) != expected:
                    raise ValueError(f'{name}: {len(samples)} requests, expected {expected}')
                for position, row in enumerate(samples):
                    phase = 'warmup' if position < manifest['warmups'] else 'measured'
                    if row['engine'] != engine or row['level'] != level or row['phase'] != phase:
                        raise ValueError(f'{name}:{position}: engine, level or phase differs')
                    if row['completed_monotonic_ns'] <= row['started_monotonic_ns']:
                        raise ValueError(f'{name}:{position}: invalid time boundaries')
                    response = row['response']['choices'][0]['message']
                    identity = dict(content=response.get('content') or '',
                                    reasoning=response.get('reasoning_content') or response.get('reasoning') or '',
                                    tool_calls=response.get('tool_calls'))
                    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                    if digest != row['output_sha256']:
                        raise ValueError(f'{name}:{position}: response fingerprint mismatch')
                rows.extend(samples)
    refreshed = report(rows)
    if refreshed['software_output_fidelity_passed'] != original['software_output_fidelity_passed']:
        raise ValueError('replayed strict fidelity result differs from frozen original')
    for current in refreshed['engines']:
        old = next(x for x in original['engines'] if x['engine'] == current['engine'])
        for key in ('baseline_output_stable', 'median_http_wall_us',
                    'relative_sample_range', 'level_1_overhead_fraction', 'overhead_qualified'):
            # JSON object keys are strings; the in-memory report deliberately
            # retains integer level keys. Compare the serialized wire shape.
            normalized = json.loads(json.dumps(current[key], sort_keys=True))
            if normalized != old[key]:
                raise ValueError(f'{current["engine"]}: replay changed frozen {key}')
    return dict(source_artifact=str(zip_path), requests_verified=len(rows),
                frozen_strict_fidelity_agreed=True, analysis=refreshed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path, help='Existing qualification ZIP')
    parser.add_argument('--output', type=Path, help='Optional analysis-only JSON destination')
    options = parser.parse_args()
    result = replay(options.artifact)
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if options.output:
        options.output.write_text(payload)
    print(payload, end='')


if __name__ == '__main__':
    main()
