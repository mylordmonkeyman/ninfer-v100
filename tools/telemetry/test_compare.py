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

    def test_missing_normalizer_is_unknown(self):
        r = compare([fixture()])
        self.assertIsNone(r['layers'][0]['host_us_per_routed_token']['moe'])


if __name__ == '__main__': unittest.main()
