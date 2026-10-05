import copy
import unittest

from v100_sv2_runtime import compare_oracle, ORACLE_PARITY_METRICS


class RouteReductionOracleTest(unittest.TestCase):
    def test_accepted_baseline_failure_is_visible_and_parity_is_strict(self):
        legacy = dict(exit_status=1, independent_oracle_passed=False,
                      metrics={key:1.0 for key in ORACLE_PARITY_METRICS})
        legacy['metrics']['vram.observed_total_gib'] = 28.0
        device = copy.deepcopy(legacy)
        device['metrics']['vram.observed_total_gib'] = 28.1
        self.assertFalse(compare_oracle(legacy, device)['independent_oracle_passed'])
        for key in ORACLE_PARITY_METRICS:
            changed = copy.deepcopy(device)
            changed['metrics'][key] += 0.001
            with self.assertRaises(ValueError):
                compare_oracle(legacy, changed)
        for key, value in [('exit_status',0), ('independent_oracle_passed',True)]:
            changed = copy.deepcopy(device)
            changed[key] = value
            with self.assertRaises(ValueError):
                compare_oracle(legacy, changed)


if __name__ == '__main__':
    unittest.main()
