"""Hosted-only tests of the one-run hardware A/B campaign plan."""
import json
from pathlib import Path
import tempfile
import unittest

from ab_v100 import CASES, option, remove_option, native_ninfer_requests, summary_for, node0_allowed_cpus


class ABV100Tests(unittest.TestCase):
    def test_matrix_has_named_controls_and_repeat_baselines(self):
        names = [(engine, name) for engine, name, _ in CASES]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[:2], [('ninfer', 'baseline'), ('strata', 'baseline')])
        self.assertIn(('ninfer', 'cache-off'), names)
        self.assertIn(('strata', 'cache-off'), names)
        self.assertIn(('strata', 'cache-4096'), names)
        self.assertIn(('ninfer', 'cache-lru'), names)
        self.assertIn(('ninfer', 'lru-admission-2'), names)
        self.assertIn(('ninfer', 'lru-prefill-stream'), names)
        self.assertIn(('ninfer', 'lru-prefill-auto256'), names)
        for variant in ('lru-auto256-mtp2','lru-auto256-mtp3',
                        'lru-auto256-decode-hybrid','lru-auto256-repeat'):
            self.assertIn(('ninfer',variant), names)
        controls={v:overrides for engine,v,overrides in CASES if engine=='ninfer'}
        self.assertEqual(controls['lru-auto256-mtp2']['draft_tokens'],'2')
        self.assertEqual(controls['lru-auto256-mtp3']['draft_tokens'],'3')
        self.assertEqual(controls['lru-auto256-decode-hybrid']
                         ['NINFER_V100_DECODE_EXPERT_POLICY'],'hybrid')
        self.assertEqual(controls['lru-auto256-ring8']
                         ['NINFER_V100_EXPERT_STREAM_RING_SLOTS'],'8')
        self.assertEqual(controls['lru-auto256-busy-first']
                         ['NINFER_V100_STREAM_EXPERT_ORDER'],'busy-first')
        self.assertEqual(controls['lru-auto256-pipelined-reuse']
                         ['NINFER_V100_EXPERT_STREAM_SLOT_REUSE'],'pipelined')
        self.assertEqual(controls['lru-auto256-gpu90']
                         ['NINFER_V100_PREFILL_EXPERT_STREAM_FRACTION'],'0.9')
        self.assertEqual(controls['lru-auto256-gpu75']
                         ['NINFER_V100_PREFILL_EXPERT_STREAM_FRACTION'],'0.75')
        self.assertEqual(controls['lru-auto256-gpu70']
                         ['NINFER_V100_PREFILL_EXPERT_STREAM_FRACTION'],'0.70')
        self.assertEqual(controls['lru-auto256-gpu60']
                         ['NINFER_V100_PREFILL_EXPERT_STREAM_FRACTION'],'0.60')
        self.assertEqual(controls['lru-auto256-gpu50']
                         ['NINFER_V100_PREFILL_EXPERT_STREAM_FRACTION'],'0.50')
        self.assertEqual(controls['lru-auto256-gpu50-group-prefill']
                         ['NINFER_V100_CPU_EXPERT_GROUP'],'prefill')
        self.assertEqual(controls['lru-auto256-gpu60-group-prefill']
                         ['NINFER_V100_CPU_EXPERT_GROUP'],'prefill')
        self.assertEqual(controls['lru-auto256-gpu50-repeat'],
                         controls['lru-auto256-gpu50'])
        self.assertEqual(controls['lru-auto256-gpu50-group-prefill-repeat'],
                         controls['lru-auto256-gpu50-group-prefill'])
        hot_admit=controls['lru-auto256-gpu50-group-prefill-hot-admit']
        self.assertEqual(hot_admit['NINFER_V100_PREFILL_STREAM_ADMIT'],'hot')
        self.assertEqual(
            {k:v for k,v in hot_admit.items()
             if k!='NINFER_V100_PREFILL_STREAM_ADMIT'},
            controls['lru-auto256-gpu50-group-prefill'])
        for routes in (10,12,14):
            variant=f'lru-auto256-minroutes{routes}-group-prefill'
            self.assertEqual(controls[variant]['NINFER_V100_PREFILL_EXPERT_STREAM_MIN_ROUTES'],str(routes))
            self.assertEqual(controls[variant]['NINFER_V100_CPU_EXPERT_GROUP'],'prefill')
            self.assertNotIn('NINFER_V100_PREFILL_EXPERT_STREAM_FRACTION',controls[variant])
        self.assertEqual(controls['lru-auto256-repeat'],
                         controls['lru-prefill-auto256'])
        self.assertIn(('ninfer', 'profile-prior-50'), names)
        self.assertIn(('ninfer', 'profile-prior-50-auto256'), names)
        self.assertLess(names.index(('ninfer','lru-prefill-auto256')),
                        names.index(('ninfer','profile-prior-50-auto256')))
        self.assertLess(names.index(('ninfer','profile-prior-50-auto256')),
                        names.index(('ninfer','lru-auto256-repeat')))
        profile_auto = controls['profile-prior-50-auto256']
        self.assertEqual(profile_auto['NINFER_V100_EXPERT_POLICY'], 'profile')
        self.assertEqual(profile_auto['NINFER_V100_EXPERT_PRIOR_WEIGHT'], '50')
        for key in ('NINFER_V100_PREFILL_EXPERT_POLICY',
                    'NINFER_V100_PREFILL_STREAM_MIN_TOKENS',
                    'NINFER_V100_DEVICE_ROUTE_COMBINE'):
            self.assertEqual(profile_auto[key], controls['lru-prefill-auto256'][key])
        self.assertIn(('ninfer', 'workers-16'), names)
        self.assertIn(('strata', 'workers-16'), names)
        self.assertIn(('ninfer', 'prefill-no-group'), names)
        self.assertIn(('strata', 'prefill-256'), names)
        self.assertIn(('strata', 'mtp-window-2'), names)
        self.assertIn(('ninfer', 'cpu-node0'), names)
        self.assertIn(('strata', 'cpu-node0'), names)
        self.assertEqual(names[-2:],
                         [('ninfer', 'baseline-repeat'), ('strata', 'baseline-repeat')])
        for _, name, overrides in CASES:
            self.assertEqual(bool(overrides), name not in ('baseline', 'baseline-repeat'))
            if name in ('lru-admission-2','lru-prefill-stream','lru-prefill-auto256',
                        'lru-auto256-mtp2','lru-auto256-mtp3','lru-auto256-decode-hybrid',
                        'lru-auto256-repeat','lru-auto256-ring8','lru-auto256-busy-first',
                         'lru-auto256-pipelined-reuse','lru-auto256-gpu90',
                         'lru-auto256-gpu75','lru-auto256-gpu70','lru-auto256-gpu60',
                         'lru-auto256-gpu60-group-prefill','lru-auto256-gpu50',
                         'lru-auto256-gpu50-group-prefill','lru-auto256-gpu50-repeat',
                         'lru-auto256-gpu50-group-prefill-repeat',
                         'lru-auto256-minroutes10-group-prefill',
                         'lru-auto256-minroutes12-group-prefill',
                         'lru-auto256-minroutes14-group-prefill',
                         'lru-auto256-gpu50-group-prefill-hot-admit',
                         'profile-prior-50',
                         'profile-prior-50-auto256'):
                self.assertIn('NINFER_V100_EXPERT_POLICY', overrides)
            else:
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

    def test_affinity_is_only_allowed_node0_cpu_set(self):
        from unittest.mock import patch
        with patch('ab_v100.os.sched_getaffinity',return_value={0,1,3,7}), \
             patch('ab_v100.Path.read_text',return_value='0-3,6-7\n'):
            self.assertEqual(node0_allowed_cpus(),[0,1,3,7])
        with patch('ab_v100.os.sched_getaffinity',return_value={7}), \
             patch('ab_v100.Path.read_text',return_value='0-7\n'):
            with self.assertRaisesRegex(RuntimeError,'fewer than 2'):
                node0_allowed_cpus()

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
