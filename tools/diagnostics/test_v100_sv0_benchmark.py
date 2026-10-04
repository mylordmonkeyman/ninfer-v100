import copy
import json
import unittest
from v100_sv0_benchmark import parse_benchmark, summary


def rows():
    return [dict(phase14='benchmark', shape=shape, mode='cache_off', sample=i,
                 tokens=64, seconds=4, tokens_per_s=16, slots_per_layer=0)
            for shape in ('decode', 'prefill') for i in range(4)]


class ScreeningTest(unittest.TestCase):
    def test_inner_samples_do_not_inflate_replications(self):
        observations = [dict(arm=arm, rows=rows(),
                             gpu_snapshots=[dict(thermal_status_observed=True, thermal_throttled=False)])
                        for arm in ('baseline', 'candidate') for _ in range(3)]
        report = summary(observations)
        self.assertEqual(report['results'][0]['baseline']['observations'], 3)
        self.assertFalse(report['qualified'])
        bad = copy.deepcopy(observations)
        bad[0]['gpu_snapshots'][0]['thermal_throttled'] = True
        with self.assertRaisesRegex(ValueError, 'thermal'):
            summary(bad)

    def test_missing_thermal_evidence_and_capacity_mismatch(self):
        observations = [dict(arm=arm, rows=rows(),
                             gpu_snapshots=[dict(thermal_status_observed=True, thermal_throttled=False)])
                        for arm in ('baseline', 'candidate') for _ in range(3)]
        bad = copy.deepcopy(observations)
        bad[0]['gpu_snapshots'] = []
        with self.assertRaisesRegex(ValueError, 'thermal'):
            summary(bad)
        bad = copy.deepcopy(observations)
        bad[0]['rows'][0]['slots_per_layer'] = 1
        with self.assertRaisesRegex(ValueError, 'capacity'):
            summary(bad)

    def test_correctness_marker_and_rate_consistency(self):
        text = '\n'.join(json.dumps(r) for r in rows())
        with self.assertRaisesRegex(ValueError, 'correctness'):
            parse_benchmark(text)
        marker = '\nPASS: Phase 14 warmed benchmark and fixed-cache exact schedule parity'
        self.assertEqual(len(parse_benchmark(text+marker)), 8)
        with self.assertRaisesRegex(ValueError, 'rate'):
            parse_benchmark(text.replace('"tokens_per_s": 16', '"tokens_per_s": 17')+marker)


if __name__ == '__main__':
    unittest.main()
