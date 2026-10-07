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

    def test_input_trace_and_lane_coverage(self):
        r = fixture()
        r['context'] = dict(input_columns=2, execution_mode='eager',
            spans=[dict(first_column=0, columns=2, first_token_index=42, lane=0, epoch=3)])
        validate(r)
        r['context']['input_token_ids'] = [101, 202]
        with self.assertRaisesRegex(ValueError, 'trace coverage'): validate(r)
        r['level'] = 2; validate(r)
        r['context']['sampled_token_ids'] = [303]
        with self.assertRaisesRegex(ValueError, 'trace coverage'): validate(r)
        del r['context']['sampled_token_ids']
        r['context']['spans'][0]['first_column'] = 1
        with self.assertRaisesRegex(ValueError, 'noncontiguous'): validate(r)

    def test_cache_resets_and_monotonicity(self):
        r = fixture()
        c = dict(generation=0, capacity_bytes=400, capacity_experts=4, ready=1, uploading=0, leased=0,
            hits_total=4, misses_total=2, admissions_total=2, fills_total=1, evictions_total=0, fill_bytes_total=100)
        r['layers'][0]['cache_windows'] = [dict(begin=c, end=dict(c, hits_total=5))]
        validate(r)
        end = r['layers'][0]['cache_windows'][0]['end']
        end['hits_total'] = 0
        with self.assertRaisesRegex(ValueError, 'without reset'): validate(r)
        end['generation'] = 1; validate(r)
        end['ready'] = 5
        with self.assertRaisesRegex(ValueError, 'occupancy'): validate(r)

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
