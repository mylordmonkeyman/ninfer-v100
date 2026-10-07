import ctypes
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from sample_system import NvmlSampler, ProcSampler, ProcessChanged, parse_numa, parse_stat, parse_status
from system_schema import read_records, validate

ROOT = Path(__file__).resolve().parents[2]


def stat_text(start=123):
    fields = ['0'] * 37
    fields[0], fields[7], fields[9], fields[11], fields[12] = 'S', '5', '2', '100', '50'
    fields[17], fields[19], fields[20], fields[21], fields[36] = '1', str(start), '8192', '2', '3'
    return '42 (a tricky ) name) ' + ' '.join(fields)


class SystemSamplerTests(unittest.TestCase):
    def test_proc_units_and_comm_parsing(self):
        stat = parse_stat(stat_text(), 100, 4096)
        self.assertEqual(stat['start_ticks'], 123)
        self.assertEqual(stat['major_faults_total'], 2)
        self.assertEqual(stat['cpu_user_us_total'], 1_000_000)
        self.assertEqual(stat['rss_stat_bytes'], 8192)
        self.assertEqual(stat['last_cpu'], 3)
        status = parse_status('VmRSS:\t12 kB\nCpus_allowed_list:\t0-3,8\n')
        self.assertEqual(status['rss_bytes'], 12 * 1024)
        self.assertEqual(status['cpus_allowed_list'], '0-3,8')
        self.assertNotIn('pinned_bytes', status)
        self.assertEqual(parse_numa('0 default N0=2 N1=3 kernelpagesize_kB=4\n'
                                   '1 huge N1=1 kernelpagesize_kB=2048\n', 4096),
                         {'0': 8192, '1': 3 * 4096 + 2 * 1024 * 1024})

    def test_pid_reuse_is_not_a_new_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); process = root / '42'; process.mkdir()
            (process / 'stat').write_text(stat_text())
            sampler = ProcSampler(42, root)
            (process / 'stat').write_text(stat_text(124))
            with self.assertRaisesRegex(ProcessChanged, 'pid_reused'): sampler.sample()

    def test_missing_files_stay_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); process = root / '42'; process.mkdir()
            (process / 'stat').write_text(stat_text())
            sample = ProcSampler(42, root).sample()
            self.assertIsNone(sample['status'])
            self.assertIsNone(sample['io'])
            self.assertIn('status', sample['unavailable'])
            self.assertNotIn('rss_bytes', sample)
        with mock.patch('sample_system.C.CDLL') as loader:
            gpu = NvmlSampler(None).sample()
            loader.assert_not_called()
            self.assertEqual(gpu['metrics'], {})
            self.assertIn('nvml', gpu['unavailable'])

    def test_nvml_abi_and_unsupported_measurements(self):
        # A C shared-library fixture exercises pointer width and actual ABI writes,
        # not a GPU. Unsupported calls must not become measured zeros.
        source = r'''
        #include <stdint.h>
        typedef struct {uint64_t total, free, used;} Memory;
        typedef struct {unsigned gpu, memory;} Util;
        int nvmlInit_v2(void) {return 0;}
        int nvmlShutdown(void) {return 0;}
        int nvmlDeviceGetHandleByUUID(const char* uuid, void** h) {*h=(void*)0x12345678;return 0;}
        int nvmlDeviceGetMemoryInfo(void* h, Memory* m) {
          if(h!=(void*)0x12345678) return 2;
          m->total=1000;m->free=750;m->used=250;return 0;
        }
        int nvmlDeviceGetUtilizationRates(void* h, Util* u) {u->gpu=0;u->memory=42;return 0;}
        int nvmlDeviceGetPowerUsage(void* h, unsigned* v) {return 3;}
        '''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); (path / 'nvml.c').write_text(source)
            subprocess.run(['gcc', '-shared', '-fPIC', str(path / 'nvml.c'),
                            '-o', str(path / 'nvml.so')], check=True)
            gpu = NvmlSampler('GPU-fixture', ctypes.CDLL(str(path / 'nvml.so')))
            sample = gpu.sample()
            self.assertEqual(sample['metrics']['memory_used_bytes'], 250)
            self.assertEqual(sample['metrics']['memory_free_bytes'], 750)
            self.assertEqual(sample['metrics']['gpu_utilization_percent'], 0)
            self.assertEqual(sample['metrics']['memory_interface_utilization_percent'], 42)
            self.assertNotIn('power_mw', sample['metrics'])
            self.assertIn('power_mw', sample['unavailable'])
            self.assertIn('pcie_tx_kb_per_s_nvml', sample['unavailable'])
            gpu.close(); self.assertFalse(gpu.initialized)

    def collect(self, target_seconds, duration):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'samples.jsonl'
            target = subprocess.Popen([sys.executable, '-c',
                'import time; from pathlib import Path; '
                "print(Path('/proc/self/stat').read_text().split('(',1)[0].strip(),flush=True); "
                f'time.sleep({target_seconds})'], stdout=subprocess.PIPE, text=True)
            try:
                # /proc may be mounted from a parent PID namespace. Supply its
                # visible identity, never guess that subprocess.pid is that ID.
                observed_pid = int(target.stdout.readline())
                subprocess.run([sys.executable, str(ROOT / 'tools/telemetry/sample_system.py'),
                    '--pid', str(observed_pid), '--engine', 'ninfer', '--run-id', 'fixture', '--level', '2',
                    '--interval-ms', '100', '--detail-interval-ms', '200',
                    '--duration-seconds', str(duration), '--output', str(output)], check=True, timeout=5)
                records = read_records(output)
                for r in records: validate(r)
                self.assertEqual(records[0]['process_id'], observed_pid)
                self.assertGreaterEqual(len(records), 2)
                self.assertFalse(any(r.get('gpu', {}).get('metrics') for r in records))
                return records, target.poll()
            finally:
                if target.poll() is None: target.terminate()
                target.wait()
                target.stdout.close()

    def test_live_collection_and_bounded_stop(self):
        records, target_status = self.collect(5, 0.35)
        self.assertEqual(records[-1]['reason'], 'duration_elapsed')
        self.assertIsNone(target_status)  # Reader never stops its observed process.
        self.assertTrue(any('tasks' in r.get('process', {}) for r in records))
        samples = records[:-1]
        self.assertGreaterEqual(samples[-1]['monotonic_ns'] - samples[0]['monotonic_ns'], 200_000_000)
        self.assertTrue(all(r['sampling_duration_us'] >= 0 for r in samples))

    def test_target_exit_finishes_stream(self):
        records, _ = self.collect(0.5, 3)
        self.assertEqual(records[-1]['reason'], 'target_exit')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'partial.jsonl'
            output.write_text('\n'.join(json.dumps(r) for r in records[:-1]))
            with self.assertRaisesRegex(ValueError, 'end record'): read_records(output)


if __name__ == '__main__': unittest.main()
