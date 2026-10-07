import copy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from lifecycle import SCHEMA, validate, summarize, read_records, qualify_links
from test_schema import fixture
from validate_records import read_records as read_rounds


def event(kind='verified', **kwargs):
    r = dict(schema=SCHEMA, kind='lifecycle', engine='ninfer', run_id='run', level=2,
             round_id='1:9', sequence_trace_id='1:0:3', first_token_index=42, event=kind,
             token_semantics='verified_output_candidates', proposal_source='mtp',
             proposed_drafts=3, accepted_drafts=2, verified_tokens=3, token_count=3, token_ids=[101,202,303])
    r.update(kwargs); return r


class LifecycleTests(unittest.TestCase):
    def test_trimmed_commits_are_not_emissions(self):
        v = event(); c = event('commit_returned', token_semantics='output_tokens', token_count=2, token_ids=[101,202])
        row = summarize([v,c])['sequences'][0]
        self.assertEqual(row['accepted_drafts'], 2)
        self.assertEqual(row['commit_token_counts'], {'output_tokens':2})
        self.assertTrue(row['commit_decision_coverage_complete']); self.assertFalse(row['emission_coverage_complete'])
        self.assertIsNone(row['emitted_tokens'])
        self.assertEqual(row['acceptance_histogram'], {'2':1})

    def test_input_state_commit_and_eos_truncated_emission(self):
        v = event(engine='strata')
        c = event('commit_returned', engine='strata', token_semantics='input_state_prefix', token_ids=[99,101,202])
        e = event('emitted', engine='strata', token_semantics='output_tokens', token_count=1, token_ids=[101])
        row = summarize([v,c,e])['sequences'][0]
        self.assertEqual(row['commit_token_counts'], {'input_state_prefix':3})
        self.assertEqual(row['emitted_tokens'], 1)
        self.assertTrue(row['emission_coverage_complete'])
        e['token_ids'] = [202]
        with self.assertRaisesRegex(ValueError, 'prefix'): summarize([v,c,e])

    def test_abort_partial_and_no_draft_round(self):
        v = event(proposed_drafts=0, accepted_drafts=0, verified_tokens=1, token_count=1, token_ids=[101])
        a = event('aborted', token_count=0, token_ids=[], token_semantics='unknown')
        row = summarize([v,a])['sequences'][0]
        self.assertTrue(row['commit_decision_coverage_complete']); self.assertIsNone(row['acceptance_fraction'])
        self.assertEqual(row['aborted_rounds'],1)
        self.assertFalse(summarize([v])['sequences'][0]['commit_decision_coverage_complete'])
        c = event('commit_returned', token_semantics='output_tokens', proposed_drafts=0, accepted_drafts=0, verified_tokens=1)
        with self.assertRaisesRegex(ValueError, 'exceeds'): summarize([v,c])
        with self.assertRaisesRegex(ValueError, 'duplicate'): summarize([v,v])

    def test_bad_accounting_and_identity(self):
        for bad in (dict(accepted_drafts=4), dict(token_count=2), dict(epoch=3), dict(level=1),
                    dict(first_token_index=True), dict(level=True), dict(accepted_drafts=float('nan'))):
            with self.assertRaises(ValueError): validate(event(**bad))
        v = event(); other = copy.deepcopy(v); other['sequence_trace_id']='1:1:5'
        self.assertEqual(len(summarize([v,other])['sequences']),2)

    def test_native_linkage_and_missing_coverage(self):
        v=event(); r=fixture(); r['level']=2; r['round_id']='1:9'
        r['context']=dict(input_columns=4,execution_mode='cuda_graph',
            spans=[dict(first_column=0,columns=4,first_token_index=42)],
            sampled_token_ids=[101,202,303,404])
        link=qualify_links([v],[r])[0]
        self.assertTrue(link['input_width_matches']); self.assertTrue(link['candidate_prefix_matches'])
        self.assertEqual(qualify_links([v],[])[0]['execution_status'],'missing')
        r['context']['spans'][0]['columns']=3
        with self.assertRaisesRegex(ValueError,'execution width'): qualify_links([v],[r])
        r['context']['spans'][0]['columns']=4; r['context']['sampled_token_ids'][0]=999
        with self.assertRaisesRegex(ValueError,'sampled prefix'): qualify_links([v],[r])

    def test_batched_lane_linkage_uses_actual_lane_epoch(self):
        v=event(lane=1,epoch=8,proposed_drafts=0,accepted_drafts=0,verified_tokens=1,token_count=1,token_ids=[202])
        r=fixture(); r['level']=2; r['round_id']='1:9'
        r['context']=dict(input_columns=2,execution_mode='cuda_graph',spans=[
            dict(first_column=0,columns=1,first_token_index=42,lane=0,epoch=3),
            dict(first_column=1,columns=1,first_token_index=9,lane=1,epoch=8)],sampled_token_ids=[101,202])
        self.assertTrue(qualify_links([v],[r])[0]['candidate_prefix_matches'])
        v['epoch']=9
        self.assertIsNone(qualify_links([v],[r])[0]['input_width_matches'])

    def test_real_cpp_records_levels(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp)/'lifecycle'
            subprocess.run(['g++','-std=c++20','-O2','-Wall','-Wextra','-Werror','-I'+str(root/'src'),
                str(root/'tests/targets/qwen3_8_flash_next/test_lifecycle_telemetry.cpp'),'-o',str(binary)],check=True)
            for level in (0,1,2):
                p=subprocess.run([str(binary)],env=dict(os.environ,V100_COMPARE_TELEMETRY_LEVEL=str(level)),
                                 capture_output=True,text=True,check=True)
                log=Path(tmp)/'records.jsonl'; log.write_text(p.stderr)
                records=read_records(log)
                if level==0: self.assertEqual(records,[]); continue
                self.assertEqual(len(records),5)
                self.assertEqual('token_ids' in records[0],level==2)
                report=summarize(records,read_rounds(log))
                self.assertTrue(all(x['input_width_matches'] for x in report['execution_links']))
                self.assertEqual(all(x['candidate_prefix_matches'] for x in report['execution_links']),level==2)
                rows=report['sequences']
                self.assertEqual(rows[0]['engine'],'ninfer'); self.assertIsNone(rows[0]['emitted_tokens'])
                self.assertEqual(rows[1]['emitted_tokens'],1)


if __name__=='__main__': unittest.main()
