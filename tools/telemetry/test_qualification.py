"""Qualification harness regression tests; hosted-only, no GPU campaign."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import subprocess
import sys
from qualify_v100 import report, strata_config, run_cell
from test_schema import fixture


def rows():
    return [dict(engine=engine, level=level, phase='measured', output_sha256=engine, wall_us=1000)
            for engine in ('ninfer', 'strata') for level in (0,1,2) for _ in range(3)]


class QualificationTests(unittest.TestCase):
    def test_within_engine_fidelity_and_overhead_gate(self):
        data = rows(); self.assertTrue(report(data)['software_output_fidelity_passed'])
        data[3]['output_sha256'] = 'changed'
        self.assertFalse(report(data)['software_output_fidelity_passed'])
        data = rows(); data[0]['wall_us'] = 1300
        result = report(data)
        self.assertTrue(result['software_output_fidelity_passed'])
        self.assertFalse(result['engines'][0]['overhead_qualified'])
        self.assertFalse(result['comprehensive_attribution_ready'])

    def test_warmups_excluded_and_baseline_variation_fails_fidelity(self):
        data = rows()+[dict(engine='ninfer', level=0, phase='warmup', output_sha256='different', wall_us=100000)]
        self.assertTrue(report(data)['software_output_fidelity_passed'])
        data[0]['output_sha256'] = 'different'
        self.assertFalse(report(data)['software_output_fidelity_passed'])

    def test_nondeterministic_baseline_is_not_misclassified_as_telemetry_drift(self):
        data = rows()
        # Strata naturally produces two output variants at Level 0.
        # Both variants recur when instrumentation is enabled.
        for row in data:
            if row['engine'] == 'strata':
                row['output_sha256'] = 'variant-a' if row['level'] != 1 else 'variant-b'
        data[10]['output_sha256'] = 'variant-b'
        data[12]['output_sha256'] = 'variant-a'
        data[16]['output_sha256'] = 'variant-b'
        result = report(data)
        strata = next(row for row in result['engines'] if row['engine'] == 'strata')
        self.assertFalse(result['software_output_fidelity_passed'])
        self.assertFalse(strata['baseline_output_stable'])
        self.assertEqual(strata['output_fidelity_failure_reason'], 'baseline_nondeterministic')
        self.assertEqual(strata['new_output_hashes_vs_baseline'][1], 0)
        self.assertEqual(strata['new_output_hashes_vs_baseline'][2], 0)

    def test_latency_drift_is_reported_without_weakening_output_gate(self):
        data = rows()
        # The measured cells trend systematically downward as caches warm.
        for engine in ('ninfer', 'strata'):
            for level in (0, 1, 2):
                matching = [r for r in data if r['engine'] == engine and r['level'] == level]
                for row, wall in zip(matching, (1250, 1100, 1000)):
                    row['wall_us'] = wall
        result = report(data)
        self.assertTrue(result['software_output_fidelity_passed'])
        for row in result['engines']:
            self.assertTrue(row['measured_request_drift_exceeds_5pct'])
            self.assertFalse(row['overhead_qualified'])
            self.assertLess(row['first_to_last_measured_wall_change_fraction'][0], -.05)

    def test_installed_config_preserved_and_native_instrumentation_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'installed.json'
            original = dict(exe='/installed/strata', cwd='/installed', args=['--native', '/existing/pack'],
                            env={'V100_COMPARE_TELEMETRY_LEVEL':'0','OTHER':'kept'})
            path.write_text(json.dumps(original))
            modified = strata_config(path, Path(tmp)/'new-engine', Path(tmp)/'new-log', 2, 'run')
            self.assertEqual(json.loads(path.read_text()), original)
            self.assertEqual(modified['args'], original['args'])
            self.assertEqual(modified['cwd'], original['cwd'])
            self.assertEqual(modified['env']['V100_COMPARE_TELEMETRY_LEVEL'], '2')
            self.assertEqual(modified['env']['OTHER'], 'kept')

    def test_http_cell_preserves_boundaries_payloads_and_native_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); binary = root/'fake-server'
            binary.write_text('''#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
record = '''+repr(fixture())+'''
config = json.load(open(sys.argv[sys.argv.index('--config')+1])) if '--config' in sys.argv else None
class Handler(BaseHTTPRequestHandler):
    counter = 0
    def log_message(self, *args): pass
    def reply(self, body):
        self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers()
        self.wfile.write(json.dumps(body).encode())
    def do_GET(self): self.reply({'data':[{'id':'fixture-model'}]})
    def do_POST(self):
        payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert payload['temperature']==0 and payload['seed']==42 and payload['max_tokens']==64
        if int(os.environ['V100_COMPARE_TELEMETRY_LEVEL']):
            record.update(level=int(os.environ['V100_COMPARE_TELEMETRY_LEVEL']),
                          run_id=os.environ['V100_COMPARE_RUN_ID'], round_id=str(Handler.counter),
                          engine='strata' if config else 'ninfer')
            if config:
                with open(config['log'],'a') as log: print(json.dumps(record),file=log,flush=True)
            else: print(json.dumps(record), file=sys.stderr, flush=True)
        Handler.counter += 1
        self.reply({'choices':[{'finish_reason':'length','message':{'content':'exact output'}}]})
HTTPServer(('127.0.0.1',int(sys.argv[sys.argv.index('--port')+1])),Handler).serve_forever()
''')
            binary.chmod(0o755)
            output = root/'results'; output.mkdir()
            a = SimpleNamespace(output=output, ninfer=binary, artifact=root/'model', profile=root/'profile',
                                gpu_uuid='fixture', warmups=1, repeats=3, strata_python=Path(sys.executable),
                                strata_server=binary, strata_binary=binary, strata_config=root/'installed.json')
            a.strata_config.write_text(json.dumps(dict(exe=str(binary), args=[], cwd=str(root))))
            original_popen = subprocess.Popen
            def launch(command, **kwargs):
                if any(str(x).endswith('sample_system.py') for x in command):
                    return SimpleNamespace(terminate=lambda: None, wait=lambda **kw: 0)
                return original_popen(command, **kwargs)
            with patch('qualify_v100.subprocess.Popen', side_effect=launch):
                for engine, level in (('ninfer',0),('ninfer',2),('strata',0),('strata',2)):
                    result = run_cell(a, engine, level)
                    self.assertEqual(len(result), 4)
                    self.assertEqual([r['phase'] for r in result], ['warmup']+['measured']*3)
                    self.assertTrue(all(r['completed_monotonic_ns'] > r['started_monotonic_ns'] for r in result))
                    self.assertTrue(all(r['native_request_id'] is None for r in result))
                    self.assertEqual(len({r['output_sha256'] for r in result}), 1)
                    self.assertEqual(result[0]['output_content_bytes'], len('exact output'))


if __name__ == '__main__': unittest.main()
