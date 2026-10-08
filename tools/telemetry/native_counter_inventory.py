"""Compact aggregate of native Level-1 work from saved telemetry logs.

Count route work, not request throughput. This is NOT per-token normalized
fidelity, physical memory bandwidth or a GPU critical-path estimate.
"""
from collections import Counter
import json

SCHEMA = b'ninfer-strata-v100-telemetry-v1'
ROUTES = ('total_routes', 'resident_routes', 'cpu_routes', 'nonresident_gpu_routes')
WEIGHT_COUNTERS = ('cpu_weight_read_bytes', 'expert_h2d_bytes')


def inventory(archive, engine, level=1):
    if engine not in ('ninfer', 'strata') or level != 1:
        raise ValueError('only frozen Level-1 NInfer and Strata inventory is supported')
    relative = f'matrix/qualification-{engine}-l{level}/'
    log = relative + ('server.log' if engine == 'ninfer' else 'native-engine.log')
    totals, phases, widths, jobs = Counter(), Counter(), Counter(), Counter()
    rounds, layer_observations, routes_covered, classified_routes = 0, 0, 0, 0
    with archive.open(log) as stream:
        for line_no, raw in enumerate(stream, 1):
            if SCHEMA not in raw:
                continue
            try:
                item = json.loads(raw)
            except ValueError as exc:
                raise ValueError(f'{log}:{line_no}: malformed native telemetry') from exc
            if item.get('schema') != SCHEMA.decode() or item.get('kind') != 'round':
                continue
            if item.get('engine') != engine or item.get('level') != level:
                raise ValueError(f'{log}:{line_no}: unexpected engine or level')
            rounds += 1
            phases[item['phase']] += 1
            widths[str(item.get('context', {}).get('input_columns', 'unobserved'))] += 1
            for layer in item['layers']:
                layer_observations += 1
                counters = layer['counters']
                for name in (*ROUTES, *WEIGHT_COUNTERS):
                    if name in counters:
                        totals[name] += counters[name]
                if all(name in counters for name in ROUTES):
                    if counters['total_routes'] != sum(counters[k] for k in ROUTES[1:]):
                        raise ValueError(f'{log}:{line_no}: route conservation failed')
                    routes_covered += 1
                    classified_routes += counters['total_routes']
                for worker in layer.get('workers_compact', []):
                    if not isinstance(worker, list) or len(worker) != 7:
                        raise ValueError(f'{log}:{line_no}: malformed compact worker')
                    for index, key in ((4, 'full_jobs'), (5, 'gate_up_jobs'), (6, 'down_jobs')):
                        jobs[key] += worker[index]
    if not rounds or not routes_covered:
        raise ValueError(f'{log}: no native Level-1 work observed')
    route_total = totals['total_routes']
    complete = routes_covered == layer_observations
    return dict(engine=engine, level=level, log_bytes=archive.getinfo(log).file_size,
                rounds=rounds, phases=dict(phases), input_width_rounds=dict(widths),
                observed_layers=layer_observations, route_complete_layer_observations=routes_covered,
                all_layer_route_coverage=complete,
                classified_route_total=classified_routes,
                classification_route_fraction=(classified_routes / route_total if route_total else None),
                counter_totals={name: totals[name] for name in (*ROUTES, *WEIGHT_COUNTERS)},
                worker_jobs=dict(jobs),
                resident_route_fraction_among_observed=(totals['resident_routes'] / classified_routes if classified_routes else None),
                cpu_route_fraction_among_observed=(totals['cpu_routes'] / classified_routes if classified_routes else None),
                route_fraction_complete=complete,
                notes=['Cumulative counters are sums of observed round/layer work including prefill and verify.',
                       'Route totals are not normalized by accepted tokens, model quantization or prompt tokens.',
                       'cpu_weight_read_bytes is an engine counter, not measured physical memory bandwidth.',
                       'Route fractions use only fully classified route work, never unclassified prefill total routes; coverage is reported separately.',
                       'Missing layers/routes or asynchronous cache events cannot be inferred from the totals.'])
