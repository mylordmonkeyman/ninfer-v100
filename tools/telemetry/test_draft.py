import copy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from draft import SCHEMA, validate, read_records, summarize
from lifecycle import read_records as read_lifecycle
from test_lifecycle import event


def fixture():
    return dict(schema=SCHEMA, kind='draft', engine='ninfer', run_id='run', level=2,
        draft_id='d1', sequence_trace_id='1:0:3', first_token_index=42,
        proposal_source='mtp', status='ok', produced_drafts=3, host_call_us=5.5,
        token_ids=[101, 202, 303])


class DraftTests(unittest.TestCase):
    def test_production_and_offers_remain_distinct(self):
        v = event(proposed_drafts=1, accepted_drafts=0, verified_tokens=1,
                  token_count=1, token_ids=[101])
        row = summarize([fixture()], [v])['sequences'][0]
        self.assertEqual(row['observed_successful_produced_drafts'], 3)
        self.assertEqual(row['offered_drafts_by_source'], {'mtp': 1})
        self.assertEqual(row['host_call_us'], 5.5)

    def test_failure_and_missing_timing_stay_unknown(self):
        failed = fixture(); failed.update(status='failed', draft_id='d2')
        del failed['produced_drafts']; del failed['token_ids']; del failed['host_call_us']
        row = summarize([fixture(), failed])['sequences'][0]
        self.assertEqual(row['failed_draft_calls'], 1)
        self.assertEqual(row['observed_successful_produced_drafts'], 3)
        self.assertIsNone(row['host_call_us'])
        self.assertIsNone(row['offered_drafts_by_source'])
        self.assertEqual(summarize([], [event()])['sequences'][0]['observed_draft_calls'], 0)

    def test_bad_records_and_duplicate_identity(self):
        for change in (dict(level=True), dict(produced_drafts=-1), dict(status='failed'),
                       dict(host_call_us=float('nan')), dict(token_ids=[101]), dict(level=1)):
            bad = copy.deepcopy(fixture()); bad.update(change)
            with self.assertRaises(ValueError): validate(bad)
        with self.assertRaisesRegex(ValueError, 'duplicate'): summarize([fixture(), fixture()])

    def test_cpp_records_and_cli(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp)/'draft'
            subprocess.run(['g++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-I'+str(root/'src'),
                str(root/'tests/targets/qwen3_8_flash_next/test_lifecycle_telemetry.cpp'), '-o', str(binary)], check=True)
            for level in (0, 1, 2):
                result = subprocess.run([str(binary)], capture_output=True, text=True, check=True,
                    env=dict(os.environ, V100_COMPARE_TELEMETRY_LEVEL=str(level)))
                log = Path(tmp)/'records.jsonl'; log.write_text(result.stderr)
                records = read_records(log)
                if level == 0: self.assertEqual(records, []); continue
                self.assertEqual(len(records), 4)
                for r in records:
                    self.assertEqual('host_call_us' in r, level == 2)
                    self.assertEqual('token_ids' in r, level == 2 and r['status'] == 'ok')
                report = summarize(records, read_lifecycle(log))
                for row in report['sequences']:
                    self.assertEqual(row['observed_successful_produced_drafts'], 3)
                    self.assertEqual(row['failed_draft_calls'], 1)
                    self.assertEqual(row['offered_drafts_by_source'], {'mtp': 3})
                output = Path(tmp)/'summary.json'
                subprocess.run(['python3', str(root/'tools/telemetry/draft.py'), str(log),
                    '--lifecycle-logs', str(log), '--output', str(output)], check=True, capture_output=True)
                self.assertTrue(output.exists())


if __name__ == '__main__': unittest.main()
