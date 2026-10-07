import copy
import unittest
from test_schema import fixture
from compare_ninfer_strata import compare


class CompareTests(unittest.TestCase):
    def test_observed_host_difference_not_critical_path(self):
        a = fixture(); a['layers'][0]['counters']['routed_tokens'] = 1
        b = copy.deepcopy(a); b['engine'] = 'strata'; b['layers'][0]['host_us']['moe'] = 80
        r = compare([a, b])
        self.assertEqual(r['difference_matrix'][0]['normalized_host_difference_us'], 10)
        self.assertIsNone(r['difference_matrix'][0]['estimated_contribution_to_request_gap'])
        self.assertFalse(r['attribution_ready'])

    def test_failures_and_partial_coverage(self):
        a = fixture(); a['layers'][0]['counters']['routed_tokens'] = 1
        b = copy.deepcopy(a); del b['layers'][0]['counters']['resident_routes']
        c = fixture(); c['status'] = 'failed'
        r = compare([a, b, c])
        self.assertEqual(r['failed_rounds'], 1)
        self.assertIsNone(r['layers'][0]['resident_route_fraction'])

    def test_worker_distribution_retains_identity_and_unknown_timing(self):
        a = fixture(); a['level'] = 2
        a['layers'][0]['workers'] = [dict(pool_id=0, worker_id=0, role='worker', configured_workers=4,
            full_jobs=2, gate_up_jobs=1, down_jobs=1, full_us=8, gate_up_us=2, down_us=3)]
        b = copy.deepcopy(a); b['run_id'] = 'other-process'
        partial = copy.deepcopy(a)
        for field in ('full_us', 'gate_up_us', 'down_us'): del partial['layers'][0]['workers'][0][field]
        layer = compare([a, b, partial])['layers'][0]
        self.assertEqual(len(layer['workers']), 2)
        by_run = {w['run_id']: w for w in layer['workers']}
        self.assertEqual(by_run['run']['full_jobs'], 4)
        self.assertIsNone(by_run['run']['observed_active_us'])
        self.assertEqual(by_run['other-process']['observed_active_us'], 13)
        self.assertIsNone(by_run['other-process']['idle_us'])
        self.assertTrue(layer['worker_coverage_complete'])
        missing = fixture(); missing['level'] = 2
        self.assertFalse(compare([a, missing])['layers'][0]['worker_coverage_complete'])

    def test_cache_window_deltas_do_not_sum_cumulative_totals(self):
        a = fixture()
        c = dict(generation=0, hits_total=100, misses_total=10, admissions_total=20,
                 fills_total=19, evictions_total=5, fill_bytes_total=1900)
        a['layers'][0]['cache_windows'] = [dict(begin=c, end=dict(c, hits_total=104, fill_bytes_total=2000))]
        layer = compare([a, copy.deepcopy(a)])['layers'][0]
        self.assertEqual(layer['cache_window_deltas']['hits'], 8)
        self.assertEqual(layer['cache_window_deltas']['fill_bytes'], 200)
        self.assertIsNone(layer['cache_window_coverage_complete'])
        reset = copy.deepcopy(a); reset['layers'][0]['cache_windows'][0]['end']['generation'] = 1
        layer = compare([a, reset])['layers'][0]
        self.assertEqual(layer['cache_reset_windows'], 1)
        self.assertIsNone(layer['cache_window_deltas']['hits'])

    def test_missing_normalizer_is_unknown(self):
        r = compare([fixture()])
        self.assertIsNone(r['layers'][0]['host_us_per_routed_token']['moe'])


if __name__ == '__main__': unittest.main()
