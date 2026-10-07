"""Validation for comparable engine records; missing measurements remain unknown."""
import math

SCHEMA = 'ninfer-strata-v100-telemetry-v1'
STAGES = ('request', 'tokenize', 'prefill', 'decode', 'layer', 'normalization',
          'mixer', 'qsa', 'gdn', 'ple', 'dense', 'router', 'moe', 'shared_expert',
          'route_combine', 'residual', 'head', 'sampling', 'cpu_expert',
          'gpu_resident', 'gpu_nonresident', 'expert_transfer', 'host_wait',
          'cache_admission', 'cache_adaptation', 'mtp_draft', 'verify', 'commit')


def number(value, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name}: expected number')
    if not math.isfinite(value) or value < 0 or (integer and not isinstance(value, int)):
        raise ValueError(f'{name}: invalid nonnegative value')
    return value


CACHE_TOTALS = ('hits_total', 'misses_total', 'admissions_total', 'fills_total',
                'evictions_total', 'fill_bytes_total')


def validate_context(context, level):
    if not isinstance(context, dict): raise ValueError('invalid context')
    columns = number(context.get('input_columns'), 'input_columns', True)
    if columns < 1: raise ValueError('empty input context')
    if context.get('execution_mode') not in ('unknown', 'eager', 'cuda_graph'):
        raise ValueError('invalid execution mode')
    for name in ('executor', 'transaction'):
        if name in context: number(context[name], name, True)
    end = 0
    if not isinstance(context.get('spans'), list): raise ValueError('missing input spans')
    for span in context['spans']:
        first = number(span.get('first_column'), 'first_column', True)
        count = number(span.get('columns'), 'columns', True)
        number(span.get('first_token_index'), 'first_token_index', True)
        if first != end or count < 1: raise ValueError('noncontiguous input spans')
        if ('lane' in span) != ('epoch' in span): raise ValueError('incomplete lane identity')
        for name in ('lane', 'epoch'):
            if name in span: number(span[name], name, True)
        end += count
    if end != columns: raise ValueError('input span coverage mismatch')
    for name in ('input_token_ids', 'sampled_token_ids'):
        if name not in context: continue
        ids = context[name]
        if level < 2 or not isinstance(ids, list) or len(ids) != columns:
            raise ValueError('invalid token trace coverage')
        for token in ids: number(token, name, True)


def validate_cache(window, level):
    for boundary in ('begin', 'end'):
        c = window.get(boundary)
        if not isinstance(c, dict): raise ValueError('missing cache boundary')
        for name in ('generation', 'capacity_bytes', 'capacity_experts', 'ready', 'uploading', 'leased') + CACHE_TOTALS:
            number(c.get(name), name, True)
        if c['ready'] + c['uploading'] > c['capacity_experts'] or c['leased'] > c['ready']:
            raise ValueError('invalid cache occupancy')
        if 'resident_ids' in c:
            ids = c['resident_ids']
            if level < 2 or not isinstance(ids, list) or len(ids) != c['ready'] or len(set(ids)) != len(ids):
                raise ValueError('invalid cache resident coverage')
            for expert in ids:
                if number(expert, 'resident id', True) >= 512: raise ValueError('invalid cache expert')
    a, b = window['begin'], window['end']
    if b['generation'] < a['generation']: raise ValueError('cache generation reversed')
    if a['generation'] == b['generation']:
        if any(b[name] < a[name] for name in CACHE_TOTALS): raise ValueError('cache total decreased without reset')
        if any(a[name] != b[name] for name in ('capacity_bytes', 'capacity_experts')):
            raise ValueError('cache capacity changed without reset')


def validate(record):
    if record.get('schema') != SCHEMA:
        raise ValueError('unsupported schema')
    if record.get('engine') not in ('ninfer', 'strata'):
        raise ValueError('invalid engine')
    for name in ('run_id', 'round_id', 'phase'):
        if not isinstance(record.get(name), str) or not record[name]:
            raise ValueError(f'missing {name}')
    if record.get('kind') != 'round':
        raise ValueError('expected round record')
    number(record.get('level'), 'level', True)
    if record.get('level') not in (1, 2, 3):
        raise ValueError('invalid telemetry level')
    if record.get('status') not in ('ok', 'failed'):
        raise ValueError('invalid status')
    if record.get('timing_semantics') != 'host_observed_not_gpu_execution':
        raise ValueError('unknown timing semantics')
    number(record.get('host_wall_us'), 'host_wall_us')
    if 'context' in record: validate_context(record['context'], record['level'])
    if not isinstance(record.get('layers'), list): raise ValueError('missing layers')
    seen = set()
    for layer in record.get('layers', []):
        i = number(layer.get('layer'), 'layer', True)
        if i >= 48 or i in seen:
            raise ValueError('invalid/duplicate layer')
        seen.add(i)
        if not isinstance(layer.get('counters'), dict) or not isinstance(layer.get('host_us'), dict):
            raise ValueError('missing layer metrics')
        for key, value in layer['counters'].items():
            number(value, key, True)
        for key, value in layer['host_us'].items():
            if key not in STAGES:
                raise ValueError(f'unknown stage {key}')
            number(value, key)
        for window in layer.get('cache_windows', []): validate_cache(window, record['level'])
        workers_seen = set()
        for worker in layer.get('workers', []):
            pool = number(worker.get('pool_id'), 'pool_id', True)
            index = number(worker.get('worker_id'), 'worker_id', True)
            configured = number(worker.get('configured_workers'), 'configured_workers', True)
            role = worker.get('role')
            key = (pool, index, role)
            if configured < 1 or role not in ('host', 'worker') or key in workers_seen:
                raise ValueError('invalid/duplicate worker')
            if index >= configured and not (role == 'host' and index == configured):
                raise ValueError('invalid worker index')
            workers_seen.add(key)
            for field in ('full_jobs', 'gate_up_jobs', 'down_jobs'):
                number(worker.get(field), field, True)
            times = ('full_us', 'gate_up_us', 'down_us')
            if any(field in worker for field in times):
                if record['level'] < 2 or not all(field in worker for field in times):
                    raise ValueError('invalid worker timing coverage')
                for field in times:
                    number(worker[field], field)
        c = layer['counters']
        route_fields = ('total_routes', 'resident_routes', 'cpu_routes', 'nonresident_gpu_routes')
        if all(key in c for key in route_fields):
            if c['total_routes'] != c['resident_routes'] + c['cpu_routes'] + c['nonresident_gpu_routes']:
                raise ValueError('route conservation failed')
        # A round can call the same layer repeatedly (chunks/verify groups).
        # These counters sum per-call distinct work, not a request-wide union.
        for distinct, bound in (
                ('distinct_experts', c.get('total_routes')),
                ('resident_distinct_experts', c.get('resident_routes')),
                ('distinct_missed_experts',
                 c['cpu_routes'] + c['nonresident_gpu_routes']
                 if 'cpu_routes' in c and 'nonresident_gpu_routes' in c else None)):
            if distinct in c and bound is not None and c[distinct] > bound:
                raise ValueError('invalid distinct expert count')
    return record
