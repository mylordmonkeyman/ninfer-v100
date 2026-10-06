import json
import unittest

import v100_sv6_serve as module


class Sv6ServeTests(unittest.TestCase):
    def test_warm_startup(self):
        text = ("flash_next host_ple_warm mode=mmap bytes=100 duration_ms=2.5\n"
                "flash_next host_ple_cache mode=warm total_pages=7 resident_pages=7\n")
        self.assertEqual(module.parse_startup_log(text, "mmap-warm")["resident_pages"], 7)

    def test_cold_startup(self):
        text = ("flash_next host_ple_warm mode=mmap bytes=100 duration_ms=2.5\n"
                "flash_next host_ple_cache mode=cold total_pages=7 resident_pages=0\n")
        self.assertEqual(module.parse_startup_log(text, "mmap-cold")["resident_pages"], 0)

    def test_direct_record(self):
        row = {"kind": "ple_gather", "compressed": True, "tokens": 8,
               "payload_bytes": 12800, "storage_backend": "direct",
               "storage_fallback": False, "coalesced_pages": 5, "page_read_us": 10.0}
        self.assertEqual(module.parse_ple_records(
            json.dumps(row), "direct", True, 8)["tokens"], 8)

    def test_request_record_ignores_server_warmup_prefill(self):
        warmup = {"kind": "ple_gather", "compressed": True, "tokens": 13,
                  "payload_bytes": 20800, "storage_backend": "mmap",
                  "storage_fallback": False, "coalesced_pages": 0,
                  "page_read_us": None}
        request = {"kind": "ple_gather", "compressed": True, "tokens": 1227,
                   "payload_bytes": 1963200, "storage_backend": "mmap",
                   "storage_fallback": False, "coalesced_pages": 0,
                   "page_read_us": None}
        text = "\n".join(json.dumps(row) for row in (warmup, request))
        self.assertEqual(module.parse_ple_records(
            text, "mmap-warm", True, 1227), request)

    def test_request_record_still_validates_warmup_backend(self):
        warmup = {"kind": "ple_gather", "compressed": True, "tokens": 13,
                  "payload_bytes": 20800, "storage_backend": "direct",
                  "storage_fallback": False, "coalesced_pages": 1,
                  "page_read_us": 1.0}
        request = {"kind": "ple_gather", "compressed": True, "tokens": 1227,
                   "payload_bytes": 1963200, "storage_backend": "mmap",
                   "storage_fallback": False, "coalesced_pages": 0,
                   "page_read_us": None}
        with self.assertRaisesRegex(ValueError, "wrong storage backend"):
            module.parse_ple_records(
                "\n".join(json.dumps(row) for row in (warmup, request)),
                "mmap-warm", True, 1227)

    def test_request_record_requires_matching_tokens(self):
        request = {"kind": "ple_gather", "compressed": True, "tokens": 1227,
                   "payload_bytes": 1963200, "storage_backend": "mmap",
                   "storage_fallback": False, "coalesced_pages": 0,
                   "page_read_us": None}
        with self.assertRaisesRegex(ValueError, "1228-token request PLE record"):
            module.parse_ple_records(
                json.dumps(request), "mmap-warm", True, 1228)

    def test_telemetry_off_rejects_ple_records(self):
        record = {"kind": "ple_gather", "compressed": True, "tokens": 13,
                  "payload_bytes": 20800, "storage_backend": "mmap",
                  "storage_fallback": False}
        with self.assertRaisesRegex(ValueError, "emitted PLE diagnostics"):
            module.parse_ple_records(json.dumps(record), "direct", False)

    def test_failure_digest(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            name = "mmap-warm-diagnostic"
            (output / f"{name}.log").write_text(
                "flash_next host_ple_warm mode=mmap bytes=100 duration_ms=2.5\n"
                "flash_next host_ple_cache mode=warm total_pages=7 resident_pages=7\n"
                '{"kind":"ple_gather","compressed":true,"tokens":8,"payload_bytes":12800,'
                '"storage_backend":"mmap","storage_fallback":false}\n')
            (output / f"{name}-requests.jsonl").write_text(
                json.dumps({"event": "request_done",
                            "result": {"prompt_tokens": 1500, "prefix_cache_hit_tokens": 0},
                            "timings_seconds": {"ttft": 1.5, "prefill": 1.0,
                                                "total": 2.0}}) + "\n")
            (output / f"{name}-response.json").write_text(json.dumps({"ok": True}))
            saved = dict(module.CURRENT_SERVER)
            module.CURRENT_SERVER.update(name=name, log=output / f"{name}.log")
            try:
                digest = module.failure_digest(output)
            finally:
                module.CURRENT_SERVER.update(saved)
            self.assertLessEqual(len(digest), 2800)
            self.assertIn("host_ple_cache mode=warm", digest)
            self.assertIn("ple_gather records: 1", digest)
            self.assertIn("prompt_tokens=1500", digest)
            self.assertIn("response: {\"ok\": true}", digest)
            self.assertEqual((output / "failure-digest.txt").read_text().rstrip("\n"),
                             digest)


if __name__ == "__main__":
    unittest.main()
