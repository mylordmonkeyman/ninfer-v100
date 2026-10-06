import unittest

from v100_sv4_runtime import EXPERT_BYTES, LAYERS, summarize, validate_layers


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
