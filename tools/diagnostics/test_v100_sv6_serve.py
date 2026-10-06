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
        self.assertEqual(module.parse_ple_records(json.dumps(row), "direct", True)["tokens"], 8)


if __name__ == "__main__":
    unittest.main()
