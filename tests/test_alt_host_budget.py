import unittest
from cyclondex.host_budget import GIB, check_candidate, meminfo, parse_gpu


class HostBudgetTests(unittest.TestCase):
    def test_linux_meminfo(self):
        rows = "MemTotal: 131072000 kB\nMemAvailable: 60000000 kB\n"
        self.assertEqual(meminfo(rows)["available"], 60000000 * 1024)
        with self.assertRaises(ValueError):
            meminfo("MemTotal: 1 kB\n")

    def test_pascal_fallback_needed(self):
        gpus = parse_gpu("P106-100, 6144, 6.1\nRTX 4090, 24576, 8.9\n")
        self.assertEqual(len(gpus), 2)
        self.assertFalse(gpus[0]["hf_native_mxfp4_supported_by_cc"])
        self.assertTrue(gpus[1]["hf_native_mxfp4_supported_by_cc"])

    def test_memory_fit_is_not_inference_benchmark(self):
        host = {"ram_bytes": {"total": 128 * GIB, "available": 26 * GIB},
                "gpus": parse_gpu("P106-100, 6144, 6.1")}
        report = check_candidate(host, 25 * GIB,
                                 independent_sessions=2,
                                 kv_context_gib=1, reserve_gib=8)
        self.assertFalse(report["ram_memory_check_passed"])
        self.assertFalse(report["gpu_native_mxfp4_supported"])
        self.assertFalse(report["tokens_per_second_measured"])
        host["ram_bytes"]["available"] = 90 * GIB
        self.assertTrue(check_candidate(host, 25 * GIB)
                        ["ram_memory_check_passed"])

    def test_invalid_budgets_rejected(self):
        host = {"ram_bytes": {"available": 30 * GIB}, "gpus": []}
        with self.assertRaises(ValueError):
            check_candidate(host, 0)
        with self.assertRaises(ValueError):
            check_candidate(host, GIB, independent_sessions=0)


if __name__ == "__main__":
    unittest.main()
