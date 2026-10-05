import importlib.util
import json
import pathlib
import unittest

PATH = pathlib.Path(__file__).with_name("v100_sv3_runtime.py")
SPEC = importlib.util.spec_from_file_location("v100_sv3_runtime", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Sv3RuntimeTest(unittest.TestCase):
    def test_stream_accounting(self):
        rows = []
        for layer in range(48):
            rows.append(json.dumps({
                "kind": "expert_layer", "prefill": True, "layer": layer,
                "tokens": 32, "routes": 320, "stream_routes": 320,
                "stream_experts": 7,
                "stream_expert_h2d_bytes": 7 * MODULE.EXPERT_SLOT_BYTES,
                "gpu_hit_routes": 0, "cpu_miss_routes": 0,
                "route_input_d2h_bytes": 320 * 4,
            }))
        summary = MODULE.validate("\n".join(rows), 32)
        self.assertEqual(summary["layers"], 48)
        self.assertEqual(summary["routes"], 48 * 320)

    def test_rejects_cpu_fallback(self):
        row = json.dumps({"kind": "expert_layer", "prefill": True})
        with self.assertRaises(RuntimeError):
            MODULE.validate(row, 32)


if __name__ == "__main__":
    unittest.main()
