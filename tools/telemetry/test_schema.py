import copy
import unittest
from schema import SCHEMA, validate


def fixture():
    return dict(schema=SCHEMA, engine='ninfer', kind='round', level=1,
                run_id='run', round_id='owner:1', phase='decode', status='ok',
                timing_semantics='host_observed_not_gpu_execution', host_wall_us=100,
                layers=[dict(layer=0, counters=dict(total_routes=10, resident_routes=4,
                            cpu_routes=6, nonresident_gpu_routes=0), host_us={'moe': 90})])


class SchemaTests(unittest.TestCase):
    def test_route_accounting(self):
        validate(fixture())
        r = fixture(); r['layers'][0]['counters']['cpu_routes'] = 7
        with self.assertRaisesRegex(ValueError, 'conservation'): validate(r)

    def test_unmeasured_metrics_are_not_zero(self):
        r = fixture(); del r['layers'][0]['counters']['cpu_routes']
        validate(r)
        self.assertNotIn('cpu_routes', r['layers'][0]['counters'])

    def test_distinct_work_across_repeated_layer_calls(self):
        r = fixture()
        c = r['layers'][0]['counters']
        c.update(total_routes=1200, resident_routes=600, cpu_routes=600,
                 distinct_experts=1000, resident_distinct_experts=600,
                 distinct_missed_experts=600)
        validate(r)
        c['distinct_missed_experts'] = 601
        with self.assertRaisesRegex(ValueError, 'distinct'): validate(r)

    def test_bad_measurements(self):
        for v in (-1, float('nan'), float('inf'), True):
            r = fixture(); r['host_wall_us'] = v
            with self.assertRaises(ValueError): validate(r)
        r = fixture(); r['layers'].append(copy.deepcopy(r['layers'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'): validate(r)

    def test_worker_coverage(self):
        r = fixture()
        w = dict(pool_id=0, worker_id=4, role='host', configured_workers=4,
                 full_jobs=1, gate_up_jobs=2, down_jobs=2)
        r['layers'][0]['workers'] = [w]
        validate(r)
        w['full_us'] = 0
        with self.assertRaisesRegex(ValueError, 'timing coverage'): validate(r)
        r['level'] = 2
        w.update(gate_up_us=10, down_us=20)
        validate(r)
        r['layers'][0]['workers'].append(copy.deepcopy(w))
        with self.assertRaisesRegex(ValueError, 'duplicate worker'): validate(r)


if __name__ == '__main__': unittest.main()
