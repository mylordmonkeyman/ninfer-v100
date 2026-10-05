import copy
import json
import tempfile
import unittest
from pathlib import Path

from v100_sv2_runtime import compare_diagnostics, compare_oracle, ORACLE_PARITY_METRICS


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

    def test_diagnostic_output_requires_matching_cache_provenance(self):
        complete = {'decode_top1':[1,2,3]}
        row = dict(sv=0,schema=1,kind='expert_layer',layer=0,prefill=False,tokens=1,
                   gpu_hit_routes=5,cpu_miss_routes=5,frequency=[0]*512)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            legacy_log, device_log = directory/'legacy.log', directory/'device.log'
            legacy_log.write_text(json.dumps(row)+'\n')
            device_log.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError, 'equal cache provenance'):
                compare_diagnostics({'complete':complete},{'complete':complete},
                                    [1],[2],legacy_log,device_log,False)
            changed = copy.deepcopy(row)
            changed['gpu_hit_routes'], changed['cpu_miss_routes'] = 4, 6
            device_log.write_text(json.dumps(changed)+'\n')
            result = compare_diagnostics({'complete':complete},{'complete':complete},
                                         [1],[2],legacy_log,device_log,False)
            self.assertFalse(result['route_and_cache_provenance_exact'])
            with self.assertRaisesRegex(ValueError, 'cache-off telemetry'):
                compare_diagnostics({'complete':complete},{'complete':complete},
                                    [1],[2],legacy_log,device_log,True)
if __name__ == '__main__':
    unittest.main()
