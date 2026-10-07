"""Extract/validate the common JSONL schema from engine stderr logs."""
import argparse
import json
from pathlib import Path
from schema import SCHEMA, validate


def read_records(path):
    records = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if SCHEMA not in line:
            continue
        try:
            record = json.loads(line)
            validate(record)
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f'{path}:{i}: {e}') from e
        records.append(record)
    if not records:
        raise ValueError(f'{path}: no common telemetry records')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('logs', nargs='+', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    records = [r for path in args.logs for r in read_records(path)]
    if args.output:
        args.output.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
    print(f'Validated {len(records)} common telemetry records')


if __name__ == '__main__': main()
