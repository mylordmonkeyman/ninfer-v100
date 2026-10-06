import copy
import json
import unittest

from test_v100_sv3_calibrate import diagnostic_rows
from v100_sv3_calibrate import EXPERT_HIDDEN

from pathlib import Path
from v100_sv4_runtime import EXPERT_BYTES, LAYERS, environment, summarize, validate_layers, validate_handoff


class Sv4RuntimeTest(unittest.TestCase):
    def records(self, grouped):
        rows = []
        for layer in range(LAYERS):
            routes = 1280
            row = {"kind": "expert_layer", "prefill": True, "layer": layer,
                   "tokens": 128, "routes": routes, "cpu_miss_routes": routes,
                   "gpu_hit_routes": 0, "stream_routes": 0,
                   "cache_result_d2h_bytes": 0, "routed_sum_h2d_bytes": 0,
                   "cpu_groups": 20 if grouped else 0,
                   "cpu_grouped_pairs": 70 if grouped else 0,
                   "cpu_weight_read_bytes": (900 if grouped else routes) * EXPERT_BYTES,
                   "cpu_branch_us": 1000.0, "frequency": [0] * 512}
            rows.append(row)
        return "\n".join(__import__("json").dumps(row) for row in rows)

    def cached_rows(self, grouped):
        rows = [json.loads(line) for line in diagnostic_rows("cpu-cache", 128).splitlines()]
        for row in rows[1:]:
            row.update(gpu_hit_routes=640, cpu_miss_routes=640,
                       cpu_miss_h2d_bytes=640 * EXPERT_HIDDEN * 4,
                       frequency=[0] * 512, cpu_branch_us=1000.0,
                       cpu_groups=160 if grouped else 0,
                       cpu_grouped_pairs=640 if grouped else 0,
                       cpu_weight_read_bytes=(160 if grouped else 640) * EXPERT_BYTES,
                       cache=dict(admissions_total=3072, fills_total=3072,
                                  evictions_total=0, leased_experts=0, ready_experts=64,
                                  uploading_experts=0, cache_bytes=64 * 48 * EXPERT_BYTES))
        return rows + copy.deepcopy(rows[1:])

    def test_cached_mixed_routes_and_weight_reads(self):
        reports = [validate_layers("\n".join(map(json.dumps, self.cached_rows(grouped))),
                                   128, grouped, cached=True) for grouped in (False, True)]
        self.assertEqual(reports[0]["passes"], 2)
        self.assertEqual(reports[0]["gpu_hit_routes"], 640 * 96)
        self.assertEqual(reports[0]["provenance"], reports[1]["provenance"])
        self.assertEqual(reports[1]["effective_weight_read_bytes"] * 4,
                         reports[0]["effective_weight_read_bytes"])

    def test_cached_rejects_pollution_and_replay_routing_changes(self):
        for field, value, message in (("evictions_total", 1, "fixed resident set"),
                                      ("gpu_hit_routes", 639, "provenance")):
            rows = self.cached_rows(True)
            if field == "evictions_total":
                rows[-1]["cache"][field] = value
            else:
                rows[-1][field] = value
            with self.assertRaisesRegex(ValueError, message):
                validate_layers("\n".join(map(json.dumps, rows)), 128, True, cached=True)

    def test_handoff_readiness_provenance(self):
        rows = [dict(kind="expert_layer", prefill=True, route_handoff=True,
                     route_sequence=i + 1, router_rendezvous_us=2.0) for i in range(96)]
        report = validate_handoff("\n".join(map(json.dumps, rows)), True)
        self.assertEqual(report["route_sequence_last"], 96)
        self.assertEqual(report["router_rendezvous_us"], 192)
        rows[-1]["route_sequence"] = 1
        with self.assertRaisesRegex(ValueError, "nonmonotonic"):
            validate_handoff("\n".join(map(json.dumps, rows)), True)
        with self.assertRaisesRegex(ValueError, "mode"):
            validate_handoff("\n".join(map(json.dumps, rows)), False)
        for row in rows:
            row.update(route_handoff=False, route_sequence=0)
        validate_handoff("\n".join(map(json.dumps, rows)), False)

    def test_handoff_screen_keeps_grouping_equal(self):
        for mode, ready in (("serial", "0"), ("handoff", "1")):
            env = environment(mode, 512, False, Path("/tmp/logits"), handoff_screen=True)
            self.assertEqual(env["NINFER_V100_CPU_EXPERT_GROUP"], "1")
            self.assertEqual(env["NINFER_V100_ROUTE_HANDOFF"], ready)
            self.assertEqual(env["NINFER_V100_DEVICE_ROUTE_COMBINE"], "1")
        self.assertEqual(environment("grouped", 512, False, Path("/tmp/logits"))
                         ["NINFER_V100_ROUTE_HANDOFF"], "0")

    def test_single_and_grouped_accounting(self):
        single = validate_layers(self.records(False), 128, False)
        grouped = validate_layers(self.records(True), 128, True)
        self.assertEqual(single["groups"], 0)
        self.assertGreater(grouped["groups"], 0)
        self.assertLess(grouped["effective_weight_read_bytes"],
                        single["effective_weight_read_bytes"])

    def test_rejects_false_grouping(self):
        with self.assertRaises(ValueError):
            validate_layers(self.records(False), 128, True)

    def test_cached_summary_rejects_capacity_or_diagnostic_changes(self):
        probe = dict(candidate_top1=1, oracle_top1=2, kl=0.1,
                     relative_nll_delta=0.2, max_logit_error=0.3,
                     tokens_per_s=10.0, expert_pairs=480, elapsed_s=1.0)
        observations = [dict(mode=mode, probe=probe, logits_sha256="same")
                        for mode in ("single", "grouped") for _ in range(3)]
        diagnostics = {mode: dict(logits_sha256="same", diagnostic=validate_layers(
            "\n".join(map(json.dumps, self.cached_rows(mode == "grouped"))),
            128, mode == "grouped", cached=True)) for mode in ("single", "grouped")}
        report = summarize(observations, diagnostics, 3, cached=True)
        self.assertEqual(report["scope"], "fixed_cache_represented_full_model_prefill")
        self.assertFalse(report["qualified"])
        handoff_rows = [dict(row, mode="serial" if row["mode"] == "single" else "handoff")
                        for row in observations]
        handoff_diags = dict(serial=diagnostics["single"], handoff=diagnostics["grouped"])
        handoff_report = summarize(handoff_rows, handoff_diags, 3, cached=True, handoff_screen=True)
        self.assertEqual(handoff_report["milestone"], "SV5")
        self.assertFalse(handoff_report["qualified"])
        changed = copy.deepcopy(diagnostics)
        changed["grouped"]["diagnostic"]["cached_accounting"]["runtime_memory"]["attention_kv_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "capacity"):
            summarize(observations, changed, 3, cached=True)
        changed = copy.deepcopy(diagnostics)
        changed["grouped"]["logits_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "uninstrumented"):
            summarize(observations, changed, 3, cached=True)

    def test_summary_requires_exact_cross_arm_logits(self):
        probe = {"candidate_top1": 1, "oracle_top1": 2, "kl": 0.1,
                 "relative_nll_delta": 0.2, "max_logit_error": 0.3,
                 "tokens_per_s": 10.0, "expert_pairs": 480,
                 "elapsed_s": 1.0}
        observations = []
        for mode, digest in (("single", "a"), ("grouped", "b")):
            observations.extend({"mode": mode, "probe": probe,
                                 "logits_sha256": digest} for _ in range(3))
        diagnostics = {
            mode: {"diagnostic": {"provenance": [(0, (1,))],
                                   "effective_weight_read_bytes": 1,
                                   "cpu_branch_wall_us": 1.0}}
            for mode in ("single", "grouped")
        }
        with self.assertRaisesRegex(ValueError, "final BF16 logits"):
            summarize(observations, diagnostics, 3)


if __name__ == "__main__":
    unittest.main()
