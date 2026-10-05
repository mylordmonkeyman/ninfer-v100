import copy
import unittest
from v100_sv2_serve import validate_events


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


if __name__=='__main__': unittest.main()
