"""Hosted-only tests of frozen qualification artifact replay."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from qualify_v100 import report
from replay_qualification import replay


def make_zip(path, corrupt=False):
    rows = []
    files = {}
    warmups, repeats = 2, 3
    for engine in ('ninfer', 'strata'):
        for level in (0, 1, 2):
            samples = []
            for position in range(warmups + repeats):
                content = (f'{engine} variant {position % 2}' if engine == 'strata'
                           else 'stable answer')
                identity = dict(content=content, reasoning='', tool_calls=None)
                digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                response = dict(choices=[dict(message=dict(content=content))])
                row = dict(engine=engine, level=level,
                           phase='warmup' if position < warmups else 'measured',
                           response=response, output_sha256=digest,
                           request_payload=dict(seed=42, same_input=True),
                           wall_us=1000, started_monotonic_ns=100, completed_monotonic_ns=1100)
                if corrupt and engine == 'strata' and level == 2 and position == 3:
                    row['output_sha256'] = 'wrong'
                samples.append(row)
                rows.append(row)
            files[f'matrix/qualification-{engine}-l{level}/requests.jsonl'] = (
                ''.join(json.dumps(r) + '\n' for r in samples))
    files['matrix/manifest.json'] = json.dumps(dict(warmups=warmups, repeats=repeats))
    files['matrix/summary.json'] = json.dumps(report(rows if not corrupt else [
        dict(r, output_sha256=hashlib.sha256(json.dumps(
            dict(content=r['response']['choices'][0]['message']['content'],
                 reasoning='', tool_calls=None), sort_keys=True).encode()).hexdigest())
        for r in rows]))
    with zipfile.ZipFile(path, 'w') as artifact:
        for filename, value in files.items():
            artifact.writestr(filename, value)


class ReplayTests(unittest.TestCase):
    def test_offline_replay_preserves_original_gate_but_detects_indexed_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evidence.zip'
            make_zip(path)
            result = replay(path)
            self.assertEqual(result['requests_verified'], 30)
            self.assertTrue(result['frozen_strict_fidelity_agreed'])
            self.assertFalse(result['analysis']['software_output_fidelity_passed'])
            self.assertTrue(result['analysis']['indexed_telemetry_output_parity_observed'])
            strata = next(x for x in result['analysis']['engines'] if x['engine'] == 'strata')
            self.assertFalse(strata['baseline_output_stable'])
            self.assertEqual(strata['indexed_request_mismatches'], {1: [], 2: []})

    def test_offline_replay_rejects_tampered_observer_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'corrupted.zip'
            make_zip(path, corrupt=True)
            with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                replay(path)


if __name__ == '__main__':
    unittest.main()
