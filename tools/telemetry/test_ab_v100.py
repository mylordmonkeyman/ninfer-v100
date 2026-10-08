"""Hosted-only tests of the one-run hardware A/B campaign plan."""
import json
from pathlib import Path
import tempfile
import unittest

from ab_v100 import CASES, option, remove_option, native_ninfer_requests, summary_for


class ABV100Tests(unittest.TestCase):
    def test_matrix_has_named_controls_and_repeat_baselines(self):
        names = [(engine, name) for engine, name, _ in CASES]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[:2], [('ninfer', 'baseline'), ('strata', 'baseline')])
        self.assertIn(('ninfer', 'cache-off'), names)
        self.assertIn(('strata', 'cache-off'), names)
        self.assertIn(('ninfer', 'workers-16'), names)
        self.assertIn(('strata', 'workers-16'), names)
        self.assertIn(('ninfer', 'prefill-no-group'), names)
        self.assertIn(('strata', 'prefill-256'), names)
        self.assertIn(('strata', 'mtp-window-1'), names)
        self.assertIn(('ninfer', 'numa-node0'), names)
        self.assertIn(('strata', 'numa-node0'), names)
        self.assertEqual(names[-2:],
                         [('ninfer', 'baseline-repeat'), ('strata', 'baseline-repeat')])
        for _, name, overrides in CASES:
            self.assertEqual(bool(overrides), name not in ('baseline', 'baseline-repeat'))
            self.assertLessEqual(len(overrides), 1)

    def test_flags_replace_existing_preserve_native_assets(self):
        baseline=['--pack','/model/pack','--native','/model/original.gguf',
                  '--expert-cache','auto','--prefill','auto',
                  '--expert-profile','/model/heat.bin','--spec','4']
        modified=option(baseline,'--expert-cache','0')
        modified=remove_option(modified,'--expert-profile')
        self.assertEqual(modified[modified.index('--expert-cache')+1],'0')
        self.assertNotIn('--expert-profile',modified)
        self.assertEqual(modified[modified.index('--native')+1],'/model/original.gguf')
        self.assertEqual(baseline[baseline.index('--expert-cache')+1],'auto')
        self.assertEqual(option(baseline,'--pool-workers','16')[-2:],
                         ['--pool-workers','16'])
        self.assertEqual(option(baseline,'--prefill','256')[
                         baseline.index('--prefill')+1],'256')

    def test_saved_native_request_timing_and_speculation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'requests.jsonl'
            p.write_text('\n'.join([
                json.dumps({'event':'server_start'}),
                json.dumps({'event':'request_done',
                            'request':{'request_id':1},
                            'timings_seconds':{'prefill':12.2,'decode':5.1},
                            'speculative':{'accepted_tokens':17}}),
                json.dumps({'event':'throughput'})])+'\n')
            found=native_ninfer_requests(p)
            self.assertEqual(len(found),1)
            self.assertEqual(found[0]['timings_seconds']['prefill'],12.2)
            self.assertEqual(found[0]['speculative']['accepted_tokens'],17)

    def test_summary_uses_http_observer_and_reports_native_timing(self):
        def observation(phase, prompt, sec, tokens):
            return dict(phase=phase,prompt=prompt,wall_seconds=sec,
                        output_sha256='abc',response={'usage':{
                            'completion_tokens':tokens, 'prompt_tokens':500 if prompt=='long' else 50},
                            'timings':{'prompt_ms':123.0}})
        rows=[observation('warmup','long',40,64),
              observation('measured','long',13,64),
              observation('measured','long',14,64),
              observation('measured','long',15,64),
              observation('measured','short',2,64),
              observation('measured','short',3,64)]
        summary=summary_for(rows)
        self.assertEqual(summary['long']['wall_median_seconds'],14)
        self.assertEqual(summary['short']['wall_median_seconds'],2.5)
        self.assertEqual(summary['long']['prompt_tokens_median'],500)
        self.assertAlmostEqual(summary['long']['approximate_output_tokens_per_http_second'],64/14)
        self.assertEqual(len(summary['long']['native_timings']),3)


if __name__ == '__main__':
    unittest.main()
