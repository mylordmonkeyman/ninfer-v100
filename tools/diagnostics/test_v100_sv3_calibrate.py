import copy
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import v100_sv3_calibrate as MODULE


def probe_line(positions=128, top1=7, elapsed=2.0):
    return (f"phase11.prefill_probe.positions={positions} final_position={positions-1} "
            f"candidate_top1={top1} oracle_top1=9 kl=0.1 relative_nll_delta=0.2 "
            f"max_logit_error=0.3 elapsed_s={elapsed} "
            f"expert_pairs={positions*10*48}\n")


def memory(reserved):
    return dict(kind="runtime_memory", runtime_reserved_bytes=reserved,
                persistent_allocation_bytes=1, workspace_allocation_bytes=2,
                attention_kv_bytes=3, indexer_kv_bytes=4, block_tables_bytes=5,
                recurrent_state_bytes=6, mtp_persistent_state_bytes=7,
                round_tensors_bytes=8, mtp_round_tensors_bytes=9,
                sampling_workspace_allocation_bytes=10, graph_allowance_bytes=0)


def diagnostic_rows(mode, positions=2):
    routes = positions * 10
    rows = [memory(100 if mode == "cpu-cache" else 200)]
    for layer in range(48):
        row = dict(kind="expert_layer", prefill=True, layer=layer,
                   tokens=positions, routes=routes, distinct_experts=5,
                   gpu_hit_routes=0, cache_result_d2h_bytes=0,
                   routed_sum_h2d_bytes=0)
        if mode == "stream":
            row.update(cpu_miss_routes=0, stream_routes=routes, stream_experts=5,
                       stream_expert_h2d_bytes=5*MODULE.EXPERT_SLOT_BYTES,
                       route_input_d2h_bytes=routes*4, cpu_miss_h2d_bytes=0)
        else:
            row.update(cpu_miss_routes=routes, stream_routes=0, stream_experts=0,
                       stream_expert_h2d_bytes=0,
                       route_input_d2h_bytes=positions*MODULE.EXPERT_HIDDEN*2+routes*4,
                       cpu_miss_h2d_bytes=routes*MODULE.EXPERT_HIDDEN*4)
        rows.append(row)
    return "\n".join(json.dumps(row) for row in rows)


class Sv3CalibrationTest(unittest.TestCase):
    def test_parse_probe(self):
        row = MODULE.parse_probe(probe_line(), 128)
        self.assertEqual(row["tokens_per_s"], 64)
        with self.assertRaises(ValueError):
            MODULE.parse_probe(probe_line(64), 128)

    def test_transfer_accounting_for_both_modes(self):
        cpu = MODULE.validate_diagnostic(diagnostic_rows("cpu-cache"), 2, "cpu-cache")
        stream = MODULE.validate_diagnostic(diagnostic_rows("stream"), 2, "stream")
        self.assertEqual(cpu["routes"], stream["routes"])
        self.assertGreater(stream["stream_expert_h2d_bytes"], 0)
        self.assertGreater(cpu["cpu_miss_h2d_bytes"], 0)

    def test_rejects_transfer_regression(self):
        text = diagnostic_rows("stream")
        rows = [json.loads(line) for line in text.splitlines()]
        rows[1]["routed_sum_h2d_bytes"] = 1
        with self.assertRaisesRegex(ValueError, "SV2 transfer"):
            MODULE.validate_diagnostic("\n".join(json.dumps(row) for row in rows), 2, "stream")

    def test_equal_capacity_excludes_separate_ring_cost(self):
        cpu, stream = memory(100), memory(200)
        result = MODULE.equal_capacity(cpu, stream)
        self.assertEqual(result["stream_runtime_overhead_bytes"], 100)
        changed = copy.deepcopy(stream)
        changed["attention_kv_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "attention_kv_bytes"):
            MODULE.equal_capacity(cpu, changed)

    def test_summary_reports_but_never_selects_threshold(self):
        diagnostics = {}
        for mode in MODULE.MODES:
            diagnostics[f"{mode}-128"] = {"diagnostic": {
                "runtime_memory": memory(100 if mode == "cpu-cache" else 200),
                "routes": 128*10*48, "distinct_experts": 100,
                "stream_expert_h2d_bytes": 0 if mode == "cpu-cache" else 10,
                "route_input_d2h_bytes": 1, "cpu_miss_h2d_bytes": 2,
            }}
        observations = []
        for mode, elapsed in (("cpu-cache", 2.0), ("stream", 1.0)):
            for repeat in range(3):
                observations.append({"mode": mode, "positions": 128,
                    "probe": MODULE.parse_probe(probe_line(elapsed=elapsed), 128)})
        report = MODULE.summarize(observations, diagnostics, 3)
        self.assertIsNone(report["automatic_threshold"])
        self.assertEqual(report["results"][0]["median_change_percent"], 100)


if __name__ == "__main__":
    unittest.main()
