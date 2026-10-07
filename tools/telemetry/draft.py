"""Report actual MTP production separately from offered verification proposals."""
import argparse
import json
from pathlib import Path
from schema import number

SCHEMA = 'ninfer-strata-v100-draft-v1'


def validate(r):
    if r.get('schema') != SCHEMA or r.get('kind') != 'draft':
        raise ValueError('unsupported draft record')
    number(r.get('level'), 'level', True)
    if r.get('engine') not in ('ninfer', 'strata') or r['level'] not in (1, 2, 3):
        raise ValueError('invalid engine/level')
    for field in ('run_id', 'draft_id', 'sequence_trace_id'):
        if not isinstance(r.get(field), str) or not r[field]: raise ValueError(f'missing {field}')
    if 'after_round_id' in r and (not isinstance(r['after_round_id'], str) or not r['after_round_id']):
        raise ValueError('invalid predecessor round')
    number(r.get('first_token_index'), 'first_token_index', True)
    if r.get('proposal_source') != 'mtp' or r.get('status') not in ('ok', 'failed'):
        raise ValueError('invalid draft source/status')
    if r['status'] == 'ok': number(r.get('produced_drafts'), 'produced_drafts', True)
    elif 'produced_drafts' in r or 'token_ids' in r:
        raise ValueError('failed draft production is unknown')
    if 'host_call_us' in r:
        if r['level'] < 2: raise ValueError('level 1 must not time draft calls')
        number(r['host_call_us'], 'host_call_us')
    if 'token_ids' in r:
        if r['level'] < 2 or not isinstance(r['token_ids'], list) or len(r['token_ids']) != r['produced_drafts']:
            raise ValueError('invalid produced draft trace')
        for token in r['token_ids']: number(token, 'token_id', True)
    return r


def read_records(path):
    records = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if SCHEMA not in line: continue
        try: records.append(validate(json.loads(line)))
        except (ValueError, TypeError, KeyError) as e: raise ValueError(f'{path}:{i}: {e}') from e
    return records


def summarize(records, lifecycle_records=None):
    rows, seen = {}, set()
    for r in records:
        validate(r)
        identity = (r['engine'], r['run_id'], r['level'], r['draft_id'])
        if identity in seen: raise ValueError('duplicate draft identity')
        seen.add(identity)
        key = (r['engine'], r['run_id'], r['level'], r['sequence_trace_id'])
        row = rows.setdefault(key, dict(observed_draft_calls=0, failed_draft_calls=0,
            observed_successful_produced_drafts=0, host_call_us=0, timed_calls=0))
        row['observed_draft_calls'] += 1
        if r['status'] == 'failed': row['failed_draft_calls'] += 1
        else: row['observed_successful_produced_drafts'] += r['produced_drafts']
        if 'host_call_us' in r:
            row['host_call_us'] += r['host_call_us']; row['timed_calls'] += 1
    offered = {}
    if lifecycle_records is not None:
        from lifecycle import validate as validate_lifecycle
        verified_seen = set()
        for r in lifecycle_records:
            validate_lifecycle(r)
            if r['event'] != 'verified': continue
            key = (r['engine'], r['run_id'], r['level'], r['sequence_trace_id'])
            identity = (*key, r['round_id'])
            if identity in verified_seen: raise ValueError('duplicate verified identity')
            verified_seen.add(identity)
            sources = offered.setdefault(key, {})
            source = r['proposal_source']
            sources[source] = sources.get(source, 0) + r['proposed_drafts']
    output = []
    for key in sorted(rows.keys() | offered.keys()):
        row = rows.get(key, dict(observed_draft_calls=0, failed_draft_calls=0,
            observed_successful_produced_drafts=0, host_call_us=0, timed_calls=0)).copy()
        if not row['observed_draft_calls'] or row['timed_calls'] != row['observed_draft_calls']:
            row['host_call_us'] = None
        output.append(dict(zip(('engine', 'run_id', 'level', 'sequence_trace_id'), key), **row,
            offered_drafts_by_source=offered.get(key) if lifecycle_records is not None else None))
    return dict(schema='ninfer-strata-v100-draft-summary-v1', request_attribution_ready=False,
        limitations=['Host call duration includes existing waits and overlap; it is not isolated GPU time.',
                     'Failed calls retain unknown production, not zero.',
                     'Production and verifier offers are separate observations; no discarded-token count is inferred.',
                     'Missing calls and request boundaries prevent complete request cost attribution.'], sequences=output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('logs', nargs='+', type=Path)
    p.add_argument('--lifecycle-logs', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    records = [r for path in args.logs for r in read_records(path)]
    if not records: p.error('no draft records')
    from lifecycle import read_records as read_lifecycle
    lifecycle = [r for path in args.lifecycle_logs for r in read_lifecycle(path)] if args.lifecycle_logs else None
    args.output.write_text(json.dumps(summarize(records, lifecycle), indent=2, allow_nan=False) + '\n')
    print(f'Validated {len(records)} draft calls; complete request costs remain unqualified')


if __name__ == '__main__': main()
