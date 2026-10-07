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
            env['V100_COMPARE_TELEMETRY_LEVEL'] = 'invalid'
            bad = subprocess.run([str(binary)], env=env, capture_output=True)
            self.assertNotEqual(bad.returncode, 0)


if __name__ == '__main__': unittest.main()
