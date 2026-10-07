"""Test emitted C++ records against the common contract, including unwind failure."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from schema import validate

ROOT = Path(__file__).resolve().parents[2]


class CollectorTests(unittest.TestCase):
    def test_ninfer_native_context_scope(self):
        with tempfile.TemporaryDirectory() as d:
            binary = Path(d) / 'scope'
            subprocess.run(['g++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-I' + str(ROOT / 'src'),
                            str(ROOT / 'tests/targets/qwen3_8_flash_next/test_perf_telemetry.cpp'),
                            '-o', str(binary)], check=True)
            for level in (0, 1, 2):
                env = dict(os.environ, V100_COMPARE_TELEMETRY_LEVEL=str(level))
                p = subprocess.run([str(binary)], env=env, text=True, capture_output=True, check=True)
                records = [validate(json.loads(s)) for s in p.stderr.splitlines()]
                if level == 0:
                    self.assertEqual(records, [])
                    continue
                self.assertEqual([r['status'] for r in records], ['failed', 'ok'])
                for r in records:
                    c = r['context']
                    self.assertEqual((c['executor'], c['transaction']), (1, 9))
                    self.assertEqual(c['spans'], [dict(first_column=0, columns=1,
                        first_token_index=42, lane=0, epoch=2)])
                    self.assertEqual('input_token_ids' in c, level == 2)
                    if level == 2: self.assertEqual(c['input_token_ids'], [101])

    def test_emitted_records(self):
        with tempfile.TemporaryDirectory() as d:
            binary = Path(d) / 'collector'
            subprocess.run(['g++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-I' + str(ROOT / 'src'),
                            str(ROOT / 'tests/targets/qwen3_8_flash_next/test_compare_telemetry.cpp'),
                            '-o', str(binary)], check=True)
            env = dict(os.environ, V100_COMPARE_TELEMETRY_LEVEL='1',
                       V100_COMPARE_RUN_ID='quoted"run\n')
            p = subprocess.run([str(binary)], env=env, text=True, capture_output=True, check=True)
            main = validate(json.loads(p.stdout))
            self.assertEqual(main['run_id'], 'quoted"run\n')
            self.assertEqual(main['layers'][0]['counters']['total_routes'], 10)
            self.assertNotIn('gpu_us', main['layers'][0])
            self.assertEqual(main['layers'][0]['workers'][0]['gate_up_jobs'], 2)
            self.assertNotIn('gate_up_us', main['layers'][0]['workers'][0])
            records = [validate(json.loads(s)) for s in p.stderr.splitlines()]
            self.assertEqual(len(records), 2)
            context = main['context']
            self.assertEqual(context['executor'], 7)
            self.assertEqual(context['transaction'], 19)
            self.assertEqual([s['epoch'] for s in context['spans']], [3, 8])
            self.assertNotIn('input_token_ids', context)
            self.assertNotIn('resident_ids', main['layers'][0]['cache_windows'][0]['begin'])
            env['V100_COMPARE_TELEMETRY_LEVEL'] = '2'
            detailed = subprocess.run([str(binary)], env=env, text=True, capture_output=True, check=True)
            trace = validate(json.loads(detailed.stdout))
            self.assertEqual(trace['context']['input_token_ids'], [101, 202])
            self.assertEqual(trace['context']['sampled_token_ids'], [303, 404])
            self.assertEqual(trace['layers'][0]['cache_windows'][0]['end']['resident_ids'], [7, 9])
            env['V100_COMPARE_TELEMETRY_LEVEL'] = 'invalid'
            bad = subprocess.run([str(binary)], env=env, capture_output=True)
            self.assertNotEqual(bad.returncode, 0)


if __name__ == '__main__': unittest.main()
