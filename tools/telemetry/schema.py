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
