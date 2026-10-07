"""Compare observed subsystem work without inventing GPU or request attribution."""
import argparse
import json
from pathlib import Path
from validate_records import read_records


def summarize(records):
    rows = {}
    for r in records:
        if r['status'] != 'ok':
            continue
        for layer in r['layers']:
            key = (r['engine'], r['level'], r['phase'], layer['layer'])
            row = rows.setdefault(key, {'observations': 0, 'counters': {}, 'host_us': {},
                                        'coverage': {}, 'normalizing_tokens': {}})
            row['observations'] += 1
            for field, value in layer['counters'].items():
                row['counters'][field] = row['counters'].get(field, 0) + value
                row['coverage'][field] = row['coverage'].get(field, 0) + 1
            for stage, value in layer['host_us'].items():
                row['host_us'][stage] = row['host_us'].get(stage, 0) + value
                name = f'host_us.{stage}'
                row['coverage'][name] = row['coverage'].get(name, 0) + 1
                tokens = layer['counters'].get('routed_tokens')
                if tokens is not None:
                    prior = row['normalizing_tokens'].get(stage, 0)
                    row['normalizing_tokens'][stage] = prior + tokens if prior is not None else None
                else:
                    row['normalizing_tokens'][stage] = None
    out = []
    for (engine, level, phase, layer), row in sorted(rows.items()):
        c = row['counters']
        coverage = row['coverage']
        routes = c.get('total_routes')
        observed = row['observations']
        hit = c.get('resident_routes')
        row['resident_route_fraction'] = (hit / routes if routes and hit is not None and
            coverage.get('total_routes') == observed and coverage.get('resident_routes') == observed else None)
        row['host_us_per_routed_token'] = {}
        for stage, us in row['host_us'].items():
            tokens = row['normalizing_tokens'].get(stage)
            complete = coverage.get('host_us.' + stage) == observed
            row['host_us_per_routed_token'][stage] = us / tokens if complete and tokens else None
        out.append(dict(engine=engine, level=level, phase=phase, layer=layer, **row))
    return out


def compare(records):
    summary = summarize(records)
    lookup = {(r['engine'], r['level'], r['phase'], r['layer']): r for r in summary}
    matrix = []
    for r in summary:
        if r['engine'] != 'ninfer': continue
        s = lookup.get(('strata', r['level'], r['phase'], r['layer']))
        if s is None: continue
        for stage in sorted(set(r['host_us_per_routed_token']) | set(s['host_us_per_routed_token'])):
            a = r['host_us_per_routed_token'].get(stage)
            b = s['host_us_per_routed_token'].get(stage)
            matrix.append(dict(subsystem=stage, layer=r['layer'], level=r['level'], phase=r['phase'],
                ninfer_host_us_per_routed_token=a, strata_host_us_per_routed_token=b,
                normalized_host_difference_us=a-b if a is not None and b is not None else None,
                estimated_contribution_to_request_gap=None,
                confidence='host observation only; inputs/quantization and GPU overlap require qualification',
                candidate_ab_experiment=None))
    return dict(schema='ninfer-strata-v100-comparison-v1', attribution_ready=False,
                limitations=['Host spans are inclusive and overlap; do not sum them.',
                             'Routed tokens include verification work, not accepted generation tokens.',
                             'Request throughput, GPU busy time, critical path and MTP coverage remain required.',
                             'Validate fixed token inputs, engine/model/config hashes and measurement repetitions.'],
                layers=summary, difference_matrix=matrix,
                failed_rounds=sum(r['status'] != 'ok' for r in records))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('logs', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    records = [r for path in args.logs for r in read_records(path)]
    report = compare(records)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"Compared {len(report['layers'])} observed layer/configuration rows; full attribution remains unqualified")


if __name__ == '__main__': main()
