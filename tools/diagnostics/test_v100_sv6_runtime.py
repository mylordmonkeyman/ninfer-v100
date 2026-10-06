import json
import unittest

import v100_sv6_runtime as module


class Sv6RuntimeTests(unittest.TestCase):
    def test_parse_direct_record(self):
        record = {"kind": "ple_gather", "compressed": True, "tokens": 8,
                  "payload_bytes": 12800, "storage_backend": "direct",
                  "storage_fallback": False, "coalesced_pages": 17,
                  "page_read_us": 123.0}
        parsed = module.parse_ple(json.dumps(record), 8, "direct")
        self.assertEqual(parsed["coalesced_pages"], 17)

    def test_rejects_fallback(self):
        record = {"kind": "ple_gather", "compressed": True, "tokens": 8,
                  "payload_bytes": 12800, "storage_backend": "mmap",
                  "storage_fallback": True, "coalesced_pages": 0,
                  "page_read_us": None}
        with self.assertRaisesRegex(RuntimeError, "fallback"):
            module.parse_ple(json.dumps(record), 8, "direct")

    def test_parse_probe(self):
        line = ("phase11.prefill_probe.positions=32 final_position=31 candidate_top1=7 "
                "oracle_top1=8 kl=0.1 relative_nll_delta=0.2 max_logit_error=0.3 "
                "elapsed_s=1.25 expert_pairs=15360")
        self.assertEqual(module.parse_probe(line, 32)["elapsed_s"], 1.25)


if __name__ == "__main__":
    unittest.main()
