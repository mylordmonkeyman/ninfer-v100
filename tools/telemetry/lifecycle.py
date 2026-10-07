"""Validate native speculative decisions without equating committed and emitted work."""
import argparse
import json
from pathlib import Path
from schema import number

SCHEMA = 'ninfer-strata-v100-lifecycle-v1'


def validate(r):
    if r.get('schema') != SCHEMA or r.get('kind') != 'lifecycle':
        raise ValueError('unsupported lifecycle record')
    number(r.get('level'), 'level', True)
    if r.get('engine') not in ('ninfer', 'strata') or r.get('level') not in (1, 2, 3):
        raise ValueError('invalid engine/level')
    for field in ('run_id', 'round_id', 'sequence_trace_id'):
        if not isinstance(r.get(field), str) or not r[field]: raise ValueError(f'missing {field}')
    if r.get('event') not in ('verified', 'commit_returned', 'emitted', 'aborted'):
        raise ValueError('invalid lifecycle event')
    number(r.get('first_token_index'), 'first_token_index', True)
    if ('lane' in r) != ('epoch' in r): raise ValueError('incomplete lane identity')
    for field in ('lane', 'epoch', 'proposed_drafts', 'accepted_drafts', 'verified_tokens', 'token_count'):
        if field in r: number(r[field], field, True)
    if r.get('token_semantics') not in ('unknown', 'verified_output_candidates', 'input_state_prefix', 'output_tokens'):
        raise ValueError('invalid token semantics')
    if r.get('proposal_source') not in ('unknown', 'none', 'mtp', 'suffix', 'oracle'):
        raise ValueError('invalid proposal source')
    if r['event'] == 'verified':
        for field in ('proposed_drafts', 'accepted_drafts', 'verified_tokens', 'token_count'):
            if field not in r: raise ValueError(f'missing {field}')
        if r['accepted_drafts'] > r['proposed_drafts'] or r['verified_tokens'] != r['accepted_drafts'] + 1:
            raise ValueError('invalid acceptance accounting')
        if r['token_count'] != r['verified_tokens'] or r['token_semantics'] != 'verified_output_candidates':
            raise ValueError('invalid verified candidates')
    elif r['event'] == 'aborted':
        if r.get('token_count') != 0: raise ValueError('abort must commit zero')
    else:
        number(r.get('token_count'), 'token_count', True)
        expected = ('input_state_prefix', 'output_tokens') if r['event'] == 'commit_returned' else ('output_tokens',)
        if r['token_semantics'] not in expected: raise ValueError('invalid commit/emission semantics')
    if 'token_ids' in r:
        if r['level'] < 2 or not isinstance(r['token_ids'], list) or len(r['token_ids']) != r.get('token_count'):
            raise ValueError('invalid lifecycle token trace coverage')
        for token in r['token_ids']: number(token, 'token_id', True)
    return r


def read_records(path):
    records = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if SCHEMA not in line: continue
        try: records.append(validate(json.loads(line)))
        except (ValueError, TypeError, KeyError) as e: raise ValueError(f'{path}:{i}: {e}') from e
    return records



def qualify_links(records, execution_records):
    """Check actual native input widths and sampled prefixes when observed."""
    lookup = {}
    for r in execution_records:
        key = (r['engine'], r['run_id'], r['level'], r['round_id'])
        if key in lookup: raise ValueError('duplicate execution round identity')
        lookup[key] = r
    links = []
    for v in records:
        if v['event'] != 'verified': continue
        key = (v['engine'], v['run_id'], v['level'], v['round_id'])
        execution = lookup.get(key)
        width_match = prefix_match = None
        if execution and 'context' in execution:
            context = execution['context']
            spans = context['spans']
            if 'lane' in v:
                spans = [s for s in spans if s.get('lane') == v['lane'] and s.get('epoch') == v['epoch']]
            if len(spans) == 1:
                span = spans[0]
                width_match = span['columns'] == v['proposed_drafts'] + 1
                if not width_match: raise ValueError('lifecycle proposals differ from native execution width')
                if 'sampled_token_ids' in context and 'token_ids' in v:
                    first = span['first_column']
                    prefix_match = context['sampled_token_ids'][first:first + v['verified_tokens']] == v['token_ids']
                    if not prefix_match: raise ValueError('lifecycle candidates differ from native sampled prefix')
        links.append(dict(engine=v['engine'], run_id=v['run_id'], level=v['level'],
            sequence_trace_id=v['sequence_trace_id'], round_id=v['round_id'],
            execution_status=execution['status'] if execution else 'missing',
            input_width_matches=width_match, candidate_prefix_matches=prefix_match))
    return links


def summarize(records, execution_records=None):
    rounds = {}
    for r in records:
        validate(r)
        key = (r['engine'], r['run_id'], r['level'], r['sequence_trace_id'], r['round_id'])
        events = rounds.setdefault(key, {})
        if r['event'] in events: raise ValueError('duplicate lifecycle event under round/sequence identity')
        events[r['event']] = r
    rows = {}
    for (engine, run, level, sequence, _), events in rounds.items():
        row = rows.setdefault((engine, run, level, sequence), dict(verified_rounds=0, proposed_drafts=0,
            accepted_drafts=0, verify_input_tokens=0, acceptance_histogram={}, commit_token_counts={}, emitted_tokens=0,
            observed_commits=0, observed_emissions=0, aborted_rounds=0, unlinked_terminal_events=0))
        v = events.get('verified'); c = events.get('commit_returned'); a = events.get('aborted'); e = events.get('emitted')
        if c and a: raise ValueError('round both committed and aborted')
        if v:
            row['verified_rounds'] += 1
            row['proposed_drafts'] += v['proposed_drafts']; row['accepted_drafts'] += v['accepted_drafts']
            row['verify_input_tokens'] += v['proposed_drafts'] + 1
            accepted = str(v['accepted_drafts'])
            row['acceptance_histogram'][accepted] = row['acceptance_histogram'].get(accepted, 0) + 1
            for terminal in (c, e):
                if terminal and any(field in terminal and terminal[field] != v[field]
                                    for field in ('proposed_drafts', 'accepted_drafts', 'verified_tokens')):
                    raise ValueError('terminal acceptance metadata differs from verification')
                if terminal and terminal['token_count'] > v['verified_tokens']:
                    raise ValueError('commit/emission exceeds verified candidates')
                if terminal and terminal['token_semantics'] == 'output_tokens' and 'token_ids' in terminal and 'token_ids' in v:
                    if terminal['token_ids'] != v['token_ids'][:terminal['token_count']]:
                        raise ValueError('committed/emitted outputs are not a verified prefix')
            if c or a: row['observed_commits'] += 1
            if e: row['observed_emissions'] += 1
        else:
            row['unlinked_terminal_events'] += len(events)
        if a: row['aborted_rounds'] += 1
        if c:
            semantics = c['token_semantics']
            row['commit_token_counts'][semantics] = row['commit_token_counts'].get(semantics, 0) + c['token_count']
        if e: row['emitted_tokens'] += e['token_count']
    out = []
    for (engine, run, level, sequence), row in sorted(rows.items()):
        count = row['verified_rounds']
        row['acceptance_fraction'] = row['accepted_drafts']/row['proposed_drafts'] if row['proposed_drafts'] else None
        row['commit_decision_coverage_complete'] = bool(count) and row['observed_commits'] == count and not row['unlinked_terminal_events']
        row['emission_coverage_complete'] = bool(count) and row['observed_emissions'] == count and not row['unlinked_terminal_events']
        if not row['emission_coverage_complete']: row['emitted_tokens'] = None
        out.append(dict(engine=engine, run_id=run, level=level, sequence_trace_id=sequence, **row))
    return dict(schema='ninfer-strata-v100-lifecycle-summary-v1', request_attribution_ready=False,
        execution_links=qualify_links(records, execution_records) if execution_records is not None else [],
        limitations=['Trace labels are process-local correlation, not external request IDs.',
                     'Verified candidates, input-state commits and emitted outputs are distinct.',
                     'A native commit return may precede GPU completion; no extra synchronization is added.',
                     'Missing terminal events remain incomplete; request boundaries/timings and draft costs are required.',
                     'NInfer Program commits do not prove external output emission.'], sequences=out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('logs', nargs='+', type=Path); p.add_argument('--output', required=True, type=Path)
    p.add_argument('--round-logs', nargs='+', type=Path, help='Optional native execution logs for linkage checks')
    args = p.parse_args()
    records = [r for path in args.logs for r in read_records(path)]
    if not records: p.error('no lifecycle records')
    from validate_records import read_records as read_rounds
    rounds = [r for path in args.round_logs for r in read_rounds(path)] if args.round_logs else None
    args.output.write_text(json.dumps(summarize(records, rounds), indent=2, allow_nan=False) + '\n')
    print(f'Validated {len(records)} lifecycle events; request attribution remains unqualified')


if __name__ == '__main__': main()
