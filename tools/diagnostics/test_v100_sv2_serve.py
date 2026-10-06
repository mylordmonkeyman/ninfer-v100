import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from v100_sv2_serve import run_server, validate_events


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
