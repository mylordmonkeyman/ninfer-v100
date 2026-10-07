"""Sample a named engine process and optional GPU without a CUDA context."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import signal
import time
from system_schema import SYSTEM_SCHEMA


def parse_stat(text, clock_hz, page_bytes):
    # comm can contain spaces and closing parentheses. Fields follow its LAST ')'.
    fields = text[text.rindex(')') + 2:].split()
    return dict(state=fields[0], minor_faults_total=int(fields[7]),
                major_faults_total=int(fields[9]),
                cpu_user_us_total=int(fields[11]) * 1_000_000 // clock_hz,
                cpu_system_us_total=int(fields[12]) * 1_000_000 // clock_hz,
                threads=int(fields[17]), start_ticks=int(fields[19]),
                virtual_bytes=int(fields[20]), rss_stat_bytes=int(fields[21]) * page_bytes,
                last_cpu=int(fields[36]))


def parse_status(text):
    result = {}
    memory = {'VmRSS': 'rss_bytes', 'RssAnon': 'rss_anon_bytes',
              'RssFile': 'rss_file_bytes', 'RssShmem': 'rss_shmem_bytes',
              'VmSize': 'virtual_bytes', 'VmSwap': 'swap_bytes', 'VmLck': 'locked_bytes',
              'VmPin': 'pinned_bytes'}
    strings = {'Name': 'name', 'Cpus_allowed_list': 'cpus_allowed_list',
               'Mems_allowed_list': 'mems_allowed_list'}
    for line in text.splitlines():
        key, _, value = line.partition(':')
        if key in memory:
            number, unit = value.split()
            if unit != 'kB': raise ValueError('unknown /proc memory unit')
            result[memory[key]] = int(number) * 1024
        elif key in strings:
            result[strings[key]] = value.strip()
        elif key in ('voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches'):
            result[key + '_total'] = int(value)
    return result


def parse_numa(text, page_bytes):
    nodes = {}
    for line in text.splitlines():
        tokens = line.split()
        size = next((int(t.split('=')[1]) * 1024 for t in tokens
                     if t.startswith('kernelpagesize_kB=')), page_bytes)
        for token in tokens:
            key, separator, value = token.partition('=')
            if separator and key.startswith('N') and key[1:].isdigit():
                nodes[key[1:]] = nodes.get(key[1:], 0) + int(value) * size
    return nodes


class ProcessChanged(Exception):
    pass


class ProcSampler:
    def __init__(self, pid, root=Path('/proc')):
        self.pid, self.root = pid, Path(root)
        self.clock_hz, self.page_bytes = os.sysconf('SC_CLK_TCK'), os.sysconf('SC_PAGESIZE')
        self.directory = self.root / str(pid)
        self.start_ticks = self.stat(self.directory)['start_ticks']

    def stat(self, directory):
        return parse_stat((directory / 'stat').read_text(), self.clock_hz, self.page_bytes)

    def optional(self, path, parser, unavailable, key):
        try: return parser(path.read_text())
        except (OSError, ValueError, IndexError) as error:
            unavailable[key] = f'{type(error).__name__}: {error}'
            return None

    def sample(self, details=False):
        stats = self.stat(self.directory)
        if stats['start_ticks'] != self.start_ticks: raise ProcessChanged('pid_reused')
        unavailable = {}
        result = dict(stat=stats, unavailable=unavailable)
        result['status'] = self.optional(self.directory / 'status', parse_status, unavailable, 'status')
        result['io'] = self.optional(self.directory / 'io',
            lambda s: {key: int(value) for key, value in
                       (line.split(':', 1) for line in s.splitlines())}, unavailable, 'io')
        if details:
            result['numa_mapped_bytes_by_node'] = self.optional(self.directory / 'numa_maps',
                lambda s: parse_numa(s, self.page_bytes), unavailable, 'numa_maps')
            tasks = []
            try:
                for directory in sorted((self.directory / 'task').iterdir()):
                    try:
                        task = self.stat(directory)
                        task.update(tid=int(directory.name),
                                    status=parse_status((directory / 'status').read_text()))
                        if self.stat(directory)['start_ticks'] != task['start_ticks']:
                            unavailable['task:' + directory.name] = 'TID reused during observation'
                            continue
                        tasks.append(task)
                    except FileNotFoundError:
                        # A worker exited while being observed; do not invent a zero.
                        unavailable['task:' + directory.name] = 'exited during observation'
                result['tasks'] = tasks
            except OSError as error:
                unavailable['tasks'] = f'{type(error).__name__}: {error}'
        # Bracket the multi-file observation so PID reuse cannot combine owners.
        if self.stat(self.directory)['start_ticks'] != self.start_ticks:
            raise ProcessChanged('pid_reused')
        return result


class NvmlMemory(C.Structure):
    _fields_ = [('total', C.c_ulonglong), ('free', C.c_ulonglong), ('used', C.c_ulonglong)]


class NvmlUtilization(C.Structure):
    _fields_ = [('gpu', C.c_uint), ('memory', C.c_uint)]


class NvmlSampler:
    def __init__(self, uuid, library=None):
        self.uuid, self.library, self.initialized, self.error = uuid, None, False, None
        self.handle = C.c_void_p()
        if uuid is None:
            self.error = 'GPU not selected'
            return
        try:
            self.library = library if library is not None else C.CDLL('libnvidia-ml.so.1')
            self.call('nvmlInit_v2', [], [])
            self.initialized = True
            self.call('nvmlDeviceGetHandleByUUID', [C.c_char_p, C.POINTER(C.c_void_p)],
                      [uuid.encode(), C.byref(self.handle)])
        except (OSError, AttributeError, RuntimeError) as error:
            self.error = str(error)

    def call(self, name, types, args):
        function = getattr(self.library, name)
        function.argtypes, function.restype = types, C.c_int
        code = function(*args)
        if code != 0: raise RuntimeError(f'{name}: NVML error {code}')

    def sample(self):
        result = dict(requested_uuid=self.uuid, metrics={}, unavailable={})
        if self.error:
            result['unavailable']['nvml'] = self.error
            return result
        metrics, errors = result['metrics'], result['unavailable']
        queries = [
            ('memory', 'nvmlDeviceGetMemoryInfo', NvmlMemory, [], []),
            ('utilization', 'nvmlDeviceGetUtilizationRates', NvmlUtilization, [], []),
            ('power_mw', 'nvmlDeviceGetPowerUsage', C.c_uint, [], []),
            ('temperature_c', 'nvmlDeviceGetTemperature', C.c_uint, [C.c_uint], [0]),
            ('sm_clock_mhz', 'nvmlDeviceGetClockInfo', C.c_uint, [C.c_uint], [1]),
            ('memory_clock_mhz', 'nvmlDeviceGetClockInfo', C.c_uint, [C.c_uint], [2]),
            ('pcie_tx_kb_per_s_nvml', 'nvmlDeviceGetPcieThroughput', C.c_uint, [C.c_uint], [0]),
            ('pcie_rx_kb_per_s_nvml', 'nvmlDeviceGetPcieThroughput', C.c_uint, [C.c_uint], [1]),
        ]
        for key, name, output_type, types, args in queries:
            output = output_type()
            try:
                self.call(name, [C.c_void_p, *types, C.POINTER(output_type)],
                          [self.handle, *args, C.byref(output)])
                if key == 'memory':
                    metrics.update(memory_total_bytes=output.total, memory_free_bytes=output.free,
                                   memory_used_bytes=output.used)
                elif key == 'utilization':
                    metrics.update(gpu_utilization_percent=output.gpu,
                                   memory_interface_utilization_percent=output.memory)
                else: metrics[key] = output.value
            except (AttributeError, RuntimeError) as error:
                errors[key] = str(error)
        return result

    def close(self):
        if self.initialized:
            self.call('nvmlShutdown', [], [])
            self.initialized = False


def host_memory():
    keys = {'MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree', 'Cached', 'Buffers', 'Dirty', 'Writeback'}
    result = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in keys:
            count, unit = value.split()
            if unit != 'kB': raise ValueError('unknown host memory unit')
            result[key + '_bytes'] = int(count) * 1024
    return result


def run(args):
    process = ProcSampler(args.pid)
    gpu = NvmlSampler(args.gpu_uuid)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    interval = args.interval_ms / 1000
    next_sample, next_details = time.monotonic(), 0
    deadline = next_sample + args.duration_seconds if args.duration_seconds else None
    index, reason, previous_output_us = 0, 'interrupted', None
    envelope = dict(schema=SYSTEM_SCHEMA, engine=args.engine, run_id=args.run_id,
                    level=args.level, process_id=args.pid, process_start_ticks=process.start_ticks,
                    clock_hz=process.clock_hz, interval_ms=args.interval_ms,
                    detail_interval_ms=args.detail_interval_ms)
    try:
        with args.output.open('x') as output:
            while not stopping:
                begin = time.monotonic_ns()
                wall = time.time_ns()
                now = begin / 1e9
                if deadline is not None and now >= deadline:
                    reason = 'duration_elapsed'; break
                details = now >= next_details
                try:
                    state = process.sample(details)
                    if state['stat']['state'] == 'Z':
                        reason = 'target_exit'; break
                except FileNotFoundError:
                    reason = 'target_exit'; break
                except ProcessChanged as error:
                    reason = str(error); break
                host, errors = None, {}
                try: host = host_memory()
                except (OSError, ValueError) as error: errors['host_memory'] = str(error)
                record = dict(envelope, kind='system_sample', sample_id=index,
                              monotonic_ns=begin, wall_time_ns=wall,
                              process=state, host_memory=host, gpu=gpu.sample(), unavailable=errors,
                              details_observed=details, previous_output_us=previous_output_us)
                record['sampling_duration_us'] = (time.monotonic_ns() - begin) / 1000
                write_started = time.monotonic_ns()
                output.write(json.dumps(record, allow_nan=False) + '\n'); output.flush()
                previous_output_us = (time.monotonic_ns() - write_started) / 1000
                index += 1
                if details: next_details = now + args.detail_interval_ms / 1000
                next_sample += interval
                # Do not produce a catch-up burst after an expensive observation.
                if next_sample < time.monotonic(): next_sample = time.monotonic() + interval
                pause = next_sample - time.monotonic()
                if pause > 0: time.sleep(pause)
            output.write(json.dumps(dict(envelope, kind='system_end', reason=reason,
                             samples=index, monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns())) + '\n')
    finally:
        try: gpu.close()
        finally:
            for signum, handler in previous.items(): signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--engine', choices=('ninfer', 'strata'), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--level', type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--interval-ms', type=int, choices=range(100, 201), default=200)
    parser.add_argument('--detail-interval-ms', type=int, default=1000)
    parser.add_argument('--duration-seconds', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.pid < 1 or args.detail_interval_ms < args.interval_ms or not args.run_id:
        parser.error('positive PID/run ID and detail interval >= sample interval required')
    if args.duration_seconds is not None and (not 0 < args.duration_seconds < float('inf')):
        parser.error('duration must be positive and finite')
    run(args)


if __name__ == '__main__': main()
