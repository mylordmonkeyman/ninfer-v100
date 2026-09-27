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

    def test_router_scores_use_labeled_incremental_fp32_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = "L20_moe_router_scores"
            for name, values in (("incremental", [1.0, 0.5]),
                                 ("cpu", [1.0, 0.6])):
                place = root / name / "pos0000"
                place.mkdir(parents=True)
                np.asarray(values, dtype="<f4").tofile(place / (stage + ".bin"))
                (root / name / "manifest.json").write_text(json.dumps({
                    "positions": [{"position": 0, "token_id": 42, "tensors": [
                        {"name": stage, "file": f"pos0000/{stage}.bin"}]}]
                }))
            (root / "oracle").mkdir()
            (root / "oracle" / "manifest.json").write_text(
                json.dumps({"positions": []}))
            report = compare(root / "oracle", root / "cpu", root / "report",
                             incremental_fp32_root=root / "incremental")
            self.assertEqual(report["incremental_router_score_rows"], 1)
            import csv
            with (root / "report" / "stage_comparison.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["reference_source"], "incremental-fp32-scores")

    def test_unrounded_mlp_stage_uses_same_oracle_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oracle_stage = "L01_mlp_block_input"
            raw_stage = "L01_mlp_block_input_fp32"
            for folder, stage, value in (("oracle", oracle_stage, 1.0),
                                         ("cpu", raw_stage, 1.25),
                                         ("v100", raw_stage, 1.5)):
                pos = root / folder / "pos0000"
                pos.mkdir(parents=True)
                np.asarray([value], dtype="<f4").tofile(pos / f"{stage}.bin")
                record = {"position": 0, "token_id": 42, "tensors": [
                    {"name": stage, "file": f"pos0000/{stage}.bin"}]}
                if folder == "v100":
                    (root / folder / "stages.jsonl").write_text(json.dumps({
                        "position": 0, "name": stage,
                        "file": f"pos0000/{stage}.bin"}))
                else:
                    (root / folder / "manifest.json").write_text(json.dumps(
                        {"positions": [record]}))
            compare(root / "oracle", root / "cpu", root / "report", root / "v100")
            import csv
            with (root / "report" / "stage_comparison.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["reference_source"],
                             "independent-fp32-unrounded-mixer")
            self.assertAlmostEqual(float(row["cpu_vs_oracle_nrmse"]), 0.25)
            self.assertAlmostEqual(float(row["v100_vs_oracle_nrmse"]), 0.5)


if __name__ == "__main__":
    unittest.main()
