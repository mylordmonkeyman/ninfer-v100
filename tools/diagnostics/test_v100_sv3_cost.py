import json
import unittest
from tools.diagnostics.v100_sv3_cost import collect,EXPECTED_ROUTES,REPEATS,EXPERT_BYTES

class CostTest(unittest.TestCase):
    def fixture(self):
        rows=[]
        for routes in EXPECTED_ROUTES:
            for sample in range(REPEATS):
                rows.append("sv3.cost="+json.dumps({
                    "routes":routes,"sample":sample,
                    "cpu_us":100+routes*10+sample,
                    "gpu_us":300+routes+sample,
                    "expert_h2d_bytes":EXPERT_BYTES,
                    "cpu_result_h2d_bytes":routes*2560*4}))
        return "\n".join(rows)
    def test_collects_bounded_crossover_without_promoting(self):
        report=collect(self.fixture())
        self.assertEqual(report["observed_separate_range_crossover"],32)
        self.assertIsNone(report["automatic_route_threshold"])
        self.assertFalse(report["qualified"])
    def test_rejects_incomplete_or_bad_accounting(self):
        with self.assertRaises(ValueError): collect(self.fixture().split("\n",1)[1])
        bad=self.fixture().replace(str(EXPERT_BYTES),str(EXPERT_BYTES-1),1)
        with self.assertRaises(ValueError): collect(bad)

if __name__=="__main__": unittest.main()
