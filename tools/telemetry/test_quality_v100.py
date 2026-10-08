"""Hosted-only tests for objective, fixed-policy quality smoke suite."""
import unittest
from quality_v100 import TASKS, POLICIES, FILLER, score, grade


class QualityTests(unittest.TestCase):
    def test_tasks_have_unambiguous_ground_truth_and_both_sizes(self):
        self.assertEqual(len(TASKS), 8)
        self.assertEqual(len({t['id'] for t in TASKS}), 8)
        self.assertEqual(sum(t['size']=='short' for t in TASKS),4)
        self.assertEqual(sum(t['size']=='long' for t in TASKS),4)
        self.assertGreater(len(FILLER.split()),256)
        for t in TASKS:
            self.assertIn(t['kind'],('text','json'))
            self.assertTrue(t['expected'])

    def test_policies_change_only_expert_execution(self):
        self.assertEqual([x[0] for x in POLICIES],['static','lru','lru-auto256'])
        self.assertEqual(POLICIES[1][1],{'NINFER_V100_EXPERT_POLICY':'lru'})
        self.assertEqual(POLICIES[2][1]['NINFER_V100_PREFILL_STREAM_MIN_TOKENS'],'256')
        self.assertEqual(POLICIES[2][1]['NINFER_V100_EXPERT_POLICY'],'lru')

    def test_objective_scoring_and_no_partial_credit(self):
        short=TASKS[0]
        self.assertTrue(score(short,' Even. '))
        self.assertFalse(score(short,'The number is even.'))
        self.assertFalse(score(short,'odd'))
        json_task=next(t for t in TASKS if t['kind']=='json')
        self.assertTrue(score(json_task,'{"count":3,"status":"ready"}'))
        self.assertFalse(score(json_task,'{"count":"3","status":"ready"}'))
        self.assertFalse(score(json_task,'some JSON {"count":3,"status":"ready"}'))

    def test_grading_excludes_warmups_and_tracks_each_task(self):
        rows=[dict(phase='warmup',task='parity',size='short',passed=False),
              dict(phase='measured',task='parity',size='short',passed=True),
              dict(phase='measured',task='parity',size='short',passed=False),
              dict(phase='measured',task='long_retrieve',size='long',passed=True)]
        result=grade(rows)
        self.assertEqual((result['correct'],result['total']),(2,3))
        self.assertEqual(result['by_size']['short'],{'correct':1,'total':2})
        self.assertEqual(result['per_task']['parity'],{'correct':1,'total':2})


if __name__=='__main__':
    unittest.main()
