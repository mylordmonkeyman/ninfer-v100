import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from v100_sv2_serve import run_server, validate_events, validate_handoff_peers, validate_qsa_dispatch, validate_qsa_comparisons


class ProductionEvidenceTest(unittest.TestCase):
    def fixtures(self, mtp=True):
        responses=[];events=[]
        for i in range(4):
            cached=0 if i==0 else 64
            response={'choices':[{'message':{'role':'assistant','content':'same' if i<2 else 'followup'},
                                    'finish_reason':'length'}],
                      'usage':{'prompt_tokens':128,'completion_tokens':32,
                               'prompt_tokens_details':{'cached_tokens':cached}}}
            event={'event':'request_done','result':{'prompt_tokens':128,'completion_tokens':32,
                    'prefix_cache_hit_tokens':cached},
                   'timings_seconds':{'prefill':1.,'decode':2.,'total':3.,'ttft':1.},
                   'speculative':{'backend':'mtp' if mtp else 'none','drafted_tokens':24 if mtp else 0}}
            responses.append(response);events.append(event)
        return events,responses

    def test_real_qsa_comparison_requires_every_dispatch_and_unchanged_thresholds(self):
        import json
        dispatch=[dict(tokens=1220,mma=False,fp8=True),dict(tokens=64,mma=False,fp8=True)]
        row=dict(kind='qsa_real_comparison',tokens=1220,sample_queries=3,sample_heads=24,
                 sample_oracle_pass=True,simt_fp64_relative_l2=1e-5,mma_fp64_relative_l2=1e-5,
                 simt_pointwise_ratio=.1,mma_pointwise_ratio=.1,
                 relative_l2_difference=1e-5,max_absolute_difference=.001)
        self.assertEqual(validate_qsa_comparisons(json.dumps(row),dispatch),[row])
        with self.assertRaises(ValueError):validate_qsa_comparisons('',dispatch)
        with self.assertRaises(ValueError):validate_qsa_comparisons(json.dumps(row),dispatch+dispatch[:1])
        for key,value in [('sample_oracle_pass',False),('sample_heads',12),
                          ('mma_fp64_relative_l2',.002),('simt_pointwise_ratio',1.01),
                          ('relative_l2_difference',float('nan'))]:
            altered=dict(row);altered[key]=value
            with self.assertRaises(ValueError):validate_qsa_comparisons(json.dumps(altered),dispatch)

    def test_qsa_dispatch_rejects_inactive_or_wrong_paths(self):
        import json
        large=[dict(kind='qsa_score_dispatch',tokens=1227,fp8=True,mma=i>=6)
               for i in range(12)]
        rows=large+[dict(kind='qsa_score_dispatch',tokens=64,fp8=True,mma=False)]
        encode=lambda values: '\n'.join(json.dumps(x) for x in values)
        self.assertEqual(validate_qsa_dispatch(encode(rows),'score-mma'),rows)
        with self.assertRaises(ValueError): validate_qsa_dispatch('', 'score-mma')
        with self.assertRaises(ValueError): validate_qsa_dispatch(encode(rows[-1:]), 'score-mma')
        with self.assertRaises(ValueError): validate_qsa_dispatch(encode(rows), 'simt')
        changed=copy.deepcopy(rows);changed[0]['fp8']=False
        with self.assertRaises(ValueError): validate_qsa_dispatch(encode(changed),'score-mma')
        changed=copy.deepcopy(rows);changed[6]['mma']=False
        with self.assertRaises(ValueError): validate_qsa_dispatch(encode(changed),'score-mma')
        changed=copy.deepcopy(rows);changed[0]['mma']=True
        with self.assertRaises(ValueError): validate_qsa_dispatch(encode(changed),'score-mma')
        simt=copy.deepcopy(rows)
        for row in simt: row['mma']=False
        self.assertEqual(validate_qsa_dispatch(encode(simt),'simt'),simt)

    def test_qsa_screen_isolates_score_flag_and_uses_fp8_batch(self):
        for mode,flag,attribution in (('simt','0',False),('score-mma','1',False),('simt','0',True)):
            with tempfile.TemporaryDirectory() as directory, \
                    mock.patch('v100_sv2_serve.gpu_snapshot', return_value={
                        'thermal_status_observed': True, 'thermal_throttled': False}), \
                    mock.patch('v100_sv2_serve.subprocess.Popen') as popen, \
                    mock.patch('v100_sv2_serve.request') as request_call, \
                    mock.patch.dict('os.environ', {'NINFER_FLASH_NEXT_QSA_PREFILL_MMA':'0',
                                                  'NINFER_V100_PLE_IO':'direct'}):
                process=popen.return_value;process.poll.return_value=None
                request_call.side_effect=[{'data':[{'id':'model'}]},RuntimeError('stop')]
                output=Path(directory)
                with self.assertRaisesRegex(RuntimeError,'stop'):
                    run_server(Path('/bin/true'),output/'model.ninfer',output/'profile.json',
                               output,mode,False,-1,qsa_score_screen=True,diagnostic=True,
                               qsa_score_attribution=attribution)
                command=popen.call_args.args[0];environment=popen.call_args.kwargs['env']
                self.assertEqual(command[command.index('--kv-dtype')+1],'fp8')
                self.assertEqual(command[command.index('--prefill-chunk')+1],'2048')
                self.assertIn('--qsa-prefill-mma',command)
                self.assertNotIn('NINFER_FLASH_NEXT_QSA_PREFILL_MMA',environment)
                self.assertEqual(environment['NINFER_V100_QSA_SCORE_MMA'],flag)
                self.assertEqual(environment['NINFER_V100_QSA_SCORE_MMA_MIN_QSA'],
                                 '6' if mode=='score-mma' else '0')
                self.assertEqual(environment['NINFER_V100_TELEMETRY'],'1')
                self.assertEqual(environment['NINFER_V100_QSA_SCORE_COMPARE'],'1' if attribution else '0')
                self.assertEqual(environment['NINFER_V100_PLE_IO'],'mmap')

    def test_large_prefill_screen_requires_actual_prompt_extent(self):
        events,responses=self.fixtures(False)
        with self.assertRaisesRegex(ValueError, 'large prefill chunk'):
            validate_events(events,responses,False,True)
        for event,response in zip(events,responses):
            event['result']['prompt_tokens']=1536
            response['usage']['prompt_tokens']=1536
        self.assertEqual(len(validate_events(events,responses,False,True)),4)

    def test_real_drafting_reuse_and_replay_required(self):
        events,responses=self.fixtures()
        self.assertEqual(len(validate_events(events,responses,True)),4)
        for field,value in [('prefix_cache_hit_tokens',0),('completion_tokens',0)]:
            altered=copy.deepcopy(events);altered[1]['result'][field]=value
            with self.assertRaises(ValueError): validate_events(altered,responses,True)
        altered=copy.deepcopy(events)
        for e in altered: e['speculative']['drafted_tokens']=0
        with self.assertRaises(ValueError): validate_events(altered,responses,True)
        altered=copy.deepcopy(responses);altered[3]['choices'][0]['message']['content']='changed'
        with self.assertRaises(ValueError): validate_events(events,altered,True)
        with self.assertRaises(ValueError): validate_events(events[:-1],responses,True)
        with self.assertRaises(ValueError): validate_events(events+[{'event':'request_error'}],responses,True)
        events,responses=self.fixtures(False)
        self.assertEqual(len(validate_events(events,responses,False)),4)

    def test_handoff_preserves_speculation_counts(self):
        events, _ = self.fixtures()
        for event in events:
            event['speculative']['accepted_tokens'] = 12
        peers = [dict(requests=events), dict(requests=copy.deepcopy(events))]
        validate_handoff_peers(peers)
        for key in ('drafted_tokens', 'accepted_tokens'):
            changed = copy.deepcopy(peers)
            changed[1]['requests'][2]['speculative'][key] += 1
            with self.assertRaisesRegex(ValueError, 'draft/accept'):
                validate_handoff_peers(changed)

    def test_handoff_screen_isolates_route_readiness(self):
        for mode, policy, flag in (('serial', 'all', '0'), ('handoff', 'all', '1'),
                                   ('serial', 'prefill', '0'), ('handoff', 'prefill', 'prefill')):
            with tempfile.TemporaryDirectory() as directory, \
                    mock.patch('v100_sv2_serve.gpu_snapshot', return_value={
                        'thermal_status_observed': True, 'thermal_throttled': False}), \
                    mock.patch('v100_sv2_serve.subprocess.Popen') as popen, \
                    mock.patch('v100_sv2_serve.request') as request_call:
                process = popen.return_value
                process.poll.return_value = None
                process.wait.return_value = 0
                request_call.side_effect = [{'data': [{'id': 'model'}]}, RuntimeError('stop')]
                output = Path(directory)
                with self.assertRaisesRegex(RuntimeError, 'stop'):
                    run_server(Path('/bin/true'), output/'model.ninfer', output/'profile.json',
                               output, mode, True, 0, route_handoff_screen=True, route_handoff_policy=policy)
                environment = popen.call_args.kwargs['env']
                self.assertEqual(environment['NINFER_V100_ROUTE_HANDOFF'], flag)
                self.assertEqual(environment['NINFER_V100_CPU_EXPERT_GROUP'], '1')
                self.assertEqual(environment['NINFER_V100_DEVICE_ROUTE_COMBINE'], '1')
                self.assertEqual(environment['NINFER_V100_PREFILL_EXPERT_POLICY'], 'cpu-cache')
                command = popen.call_args.args[0]
                self.assertIn('--no-cuda-graph', command)
                self.assertIn('mtp', command)

    def test_group_screen_selects_only_the_grouping_environment(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch('v100_sv2_serve.gpu_snapshot', return_value={
                    'thermal_status_observed': True, 'thermal_throttled': False}), \
                mock.patch('v100_sv2_serve.subprocess.Popen') as popen, \
                mock.patch('v100_sv2_serve.request') as request_call:
            process=popen.return_value
            process.poll.side_effect=[None]+[None]*20
            process.wait.return_value=0
            # Stop after readiness; the captured environment is the contract under test.
            request_call.side_effect=[{'data':[{'id':'model'}]},RuntimeError('stop')]
            output=Path(directory)
            artifact=output/'model.ninfer';artifact.write_bytes(b'x')
            profile=output/'profile.json';profile.write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'stop'):
                run_server(Path('/bin/true'),artifact,profile,output,'grouped',False,0,
                           cpu_group_screen=True)
            environment=popen.call_args.kwargs['env']
            self.assertEqual(environment['NINFER_V100_CPU_EXPERT_GROUP'],'1')
            self.assertEqual(environment['NINFER_V100_PREFILL_EXPERT_POLICY'],'cpu-cache')
            self.assertEqual(environment['NINFER_V100_DEVICE_ROUTE_COMBINE'],'1')
            command=popen.call_args.args[0]
            self.assertEqual(command[command.index('--prefill-chunk')+1],'2048')


if __name__=='__main__': unittest.main()
