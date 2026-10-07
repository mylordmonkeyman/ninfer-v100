"""Schema for external observations, separate from inference round records."""
import argparse
import json
from pathlib import Path
from schema import number

SYSTEM_SCHEMA = 'ninfer-strata-v100-system-v1'


def validate(record):
    if record.get('schema') != SYSTEM_SCHEMA or record.get('engine') not in ('ninfer', 'strata'):
        raise ValueError('invalid system envelope')
    if not isinstance(record.get('run_id'), str) or not record['run_id']:
        raise ValueError('missing run_id')
    for name in ('process_id', 'process_start_ticks', 'clock_hz', 'monotonic_ns', 'wall_time_ns'):
        number(record.get(name), name, True)
    if not record['process_id'] or not record['clock_hz']:
        raise ValueError('invalid process identity/clock')
    if record.get('level') not in (0, 1, 2, 3) or isinstance(record.get('level'), bool):
        raise ValueError('invalid level')
    interval = number(record.get('interval_ms'), 'interval_ms', True)
    detail = number(record.get('detail_interval_ms'), 'detail_interval_ms', True)
    if not 100 <= interval <= 200 or detail < interval:
        raise ValueError('invalid sample interval')
    if record.get('kind') == 'system_end':
        if record.get('reason') not in ('duration_elapsed', 'target_exit', 'pid_reused', 'interrupted'):
            raise ValueError('invalid stop reason')
        number(record.get('samples'), 'samples', True)
        return record
    if record.get('kind') != 'system_sample': raise ValueError('invalid system kind')
    number(record.get('sample_id'), 'sample_id', True)
    number(record.get('sampling_duration_us'), 'sampling_duration_us')
    if record.get('previous_output_us') is not None:
        number(record['previous_output_us'], 'previous_output_us')
    if not isinstance(record.get('details_observed'), bool): raise ValueError('invalid detail coverage')
    def errors(value):
        if not isinstance(value, dict) or not all(isinstance(v, str) for v in value.values()):
            raise ValueError('invalid unavailable map')
    def stat(value):
        if not isinstance(value, dict) or not isinstance(value.get('state'), str):
            raise ValueError('missing process stat')
        for key, item in value.items():
            if key != 'state': number(item, key, True)
    def status(value):
        if value is None: return
        if not isinstance(value, dict): raise ValueError('invalid process status')
        for key, item in value.items():
            if key.endswith('_bytes') or key.endswith('_total'): number(item, key, True)
            elif not isinstance(item, str): raise ValueError('invalid status string')
    process = record.get('process')
    if not isinstance(process, dict): raise ValueError('missing process')
    stat(process.get('stat'))
    if process['stat'].get('start_ticks') != record['process_start_ticks']:
        raise ValueError('mixed process identities')
    status(process.get('status'))
    errors(process.get('unavailable'))
    for name in ('io', 'numa_mapped_bytes_by_node'):
        values = process.get(name)
        if values is not None:
            if not isinstance(values, dict): raise ValueError(f'invalid {name}')
            for key, item in values.items(): number(item, key, True)
    tasks = process.get('tasks')
    if tasks is not None:
        if not record['details_observed'] or not isinstance(tasks, list):
            raise ValueError('invalid task coverage')
        seen = set()
        for task in tasks:
            if not isinstance(task, dict) or task.get('tid') in seen: raise ValueError('duplicate task')
            number(task.get('tid'), 'tid', True); seen.add(task['tid'])
            status(task.get('status'))
            stat({key: item for key, item in task.items() if key not in ('tid', 'status')})
    host = record.get('host_memory')
    if host is not None:
        if not isinstance(host, dict): raise ValueError('invalid host memory')
        for key, item in host.items(): number(item, key, True)
    gpu = record.get('gpu')
    if not isinstance(gpu, dict) or not isinstance(gpu.get('metrics'), dict):
        raise ValueError('missing GPU coverage')
    errors(gpu.get('unavailable')); errors(record.get('unavailable'))
    for key, item in gpu['metrics'].items():
        number(item, key, True)
        if key.endswith('_percent') and item > 100: raise ValueError('invalid GPU percent')
    return record


def read_records(path):
    records = []
    identity, previous, ended = None, None, False
    for line in Path(path).read_text().splitlines():
        record = validate(json.loads(line))
        key = tuple(record[field] for field in ('engine', 'run_id', 'level', 'process_id', 'process_start_ticks'))
        if identity is not None and key != identity: raise ValueError('mixed sampler owners')
        if ended or (previous is not None and record['monotonic_ns'] < previous):
            raise ValueError('invalid sample ordering')
        if record['kind'] == 'system_sample':
            if record['sample_id'] != len(records): raise ValueError('missing/duplicate sample')
        else:
            ended = True
            if record['samples'] != len(records): raise ValueError('invalid sample total')
        identity, previous = key, record['monotonic_ns']
        records.append(record)
    if not records or not ended: raise ValueError('missing sampler end record')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    print(f'Validated {len(read_records(args.path)) - 1} external system samples')
