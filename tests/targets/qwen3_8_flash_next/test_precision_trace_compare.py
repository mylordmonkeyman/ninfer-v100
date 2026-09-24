"""End-to-end comparison of small synthetic three-way traces."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] /
                       "tools/reference/qwen3_8_flash_next/oracle"))

from compare_precision_traces import compare


class TraceComparisonTest(unittest.TestCase):
    def test_three_way_stage_and_router_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, values in (("oracle", [1.0, 2.0]),
                                 ("cpu", [1.0, 2.1])):
                place = root / name / "pos0000"
                place.mkdir(parents=True)
                np.asarray(values, dtype="<f4").tofile(place / "final_hidden.bin")
                np.asarray([1, 2], dtype="<f4").tofile(place / "L00_moe_router_ids.bin")
                tensors = [{"name": stage, "file": f"pos0000/{stage}.bin"}
                           for stage in ("final_hidden", "L00_moe_router_ids")]
                (root / name / "manifest.json").write_text(json.dumps(
                    {"positions": [{"position": 0, "token_id": 42, "tensors": tensors}]}
                ))
            place = root / "v100" / "pos0000"
            place.mkdir(parents=True)
            np.asarray([1.0, 2.2], dtype="<f4").tofile(place / "final_hidden.bin")
            np.asarray([2, 1], dtype="<f4").tofile(place / "L00_moe_router_ids.bin")
            (root / "v100" / "stages.jsonl").write_text("\n".join(json.dumps(
                {"position": 0, "name": stage, "file": f"pos0000/{stage}.bin"}
            ) for stage in ("final_hidden", "L00_moe_router_ids")))
            report = compare(root / "oracle", root / "cpu", root / "report",
                             root / "v100")
            self.assertEqual(report["stage_rows"], 2)
            self.assertEqual(report["router_set_flips_v100"], 0)
            self.assertGreater(report["worst_cpu_stage"]["cpu_vs_oracle_nrmse"], 0)
            self.assertTrue((root / "report" / "stage_comparison.csv").is_file())


if __name__ == "__main__":
    unittest.main()
