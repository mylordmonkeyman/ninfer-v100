import copy
import json
from pathlib import Path
import tempfile
import unittest

from v100_sv0_report import read_records, summarize


def measurement(hits=5):
    frequency = [0]*512
    frequency[0], frequency[511] = 4, 6
    return dict(sv=0, schema=1, kind='expert_layer', layer=47, prefill=False,
                tokens=1, routes=10, frequency=frequency, distinct_experts=2,
                routes_per_distinct_expert=5, gpu_hit_routes=hits, cpu_miss_routes=10-hits,
                cache_result_d2h_bytes=102400 if hits else 0, routed_sum_h2d_bytes=10240,
                route_input_d2h_bytes=5200, cpu_branch_us=10, gpu_hit_window_us=None,
                merge_wait_us=None, branch_wall_us=15, router_rendezvous_us=3,
                overlap_lower_bound_us=None, cache=None)


class ReportTest(unittest.TestCase):
    def test_zero_mixed_full_hits_and_histogram(self):
        report = summarize([measurement(0), measurement(5), measurement(10)])
        layer = report['layer_summary'][0]
        self.assertEqual((layer['hits'], layer['misses']), (15, 15))
        self.assertEqual(layer['cache_result_d2h_bytes'], 204800)
        self.assertEqual(layer['routed_sum_h2d_bytes'], 30720)
        self.assertEqual(report['routing_frequency'][47][511], 18)
        self.assertEqual(len(report['routing_frequency']), 48)
        self.assertEqual(len(report['routing_frequency'][0]), 512)
        self.assertIsNone(layer['gpu_hit_window_us'])
        self.assertFalse(report['qualified'])

    def test_reject_incorrect_hit_only_download_accounting(self):
        row = measurement()
        row['cache_result_d2h_bytes'] //= 2
        with self.assertRaisesRegex(ValueError, 'D2H bytes'):
            summarize([row])

    def test_reject_inconsistent_routes_and_nonfinite_timing(self):
        row = measurement()
        row['frequency'][0] += 1
        with self.assertRaisesRegex(ValueError, 'histogram'):
            summarize([row])
        row = measurement()
        row['cpu_branch_us'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            summarize([row])

    def test_cumulative_background_snapshots_are_not_summed(self):
        first = measurement()
        first['cache'] = dict(fills_total=10, expert_staging_bytes_total=27648000)
        second = copy.deepcopy(first)
        second['cache']['fills_total'] = 12
        report = summarize([first, second])
        self.assertEqual(report['layer_summary'][0]['last_cache_snapshot']['fills_total'], 12)

    def test_log_interval_and_no_data_failure(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'run.log'
            path.write_text('human log\n{"phase14":"benchmark"}\n'+json.dumps(measurement())+'\n')
            self.assertEqual(len(read_records(path)), 1)
            path.write_text('human log\n')
            with self.assertRaisesRegex(ValueError, 'no SV0 measurements'):
                read_records(path)


if __name__ == '__main__':
    unittest.main()
