import json
import tempfile
import unittest
from pathlib import Path
from cyclondex.expert_reduction import RoutingProfile, estimate_params
from tools.alt_subset_profile import evaluate


def event(layer, phase, domain, ids, weights=None):
    return {"layer": layer, "phase": phase, "domain": domain, "experts": ids,
            "weights": weights or [0.4, 0.3, 0.2, 0.1]}


class TestAltExpertReduction(unittest.TestCase):
    def test_parameter_estimate_and_active_cost(self):
        p48 = estimate_params(48)
        p32 = estimate_params(32)
        p16 = estimate_params(16)
        self.assertTrue(40 < p48["total_B"] < 48)
        self.assertTrue(29 < p32["total_B"] < 32)
        self.assertTrue(15 < p16["total_B"] < 17)
        self.assertEqual(p48["active_B"], p32["active_B"])
        self.assertEqual(p32["active_B"], p16["active_B"])
        self.assertEqual(p16["active_B"], 5.1)

    def test_balanced_domains_and_phases(self):
        p = RoutingProfile(layers=1, experts=16)
        # Prefill heavy code has a different favored quartet than decode math.
        for _ in range(100):
            p.observe(event(0, "prefill", "code", [0, 1, 2, 3]))
        p.observe(event(0, "decode", "math", [8, 9, 10, 11]))
        selected = p.choose(8)
        self.assertEqual(set(selected[0]), {0, 1, 2, 3, 8, 9, 10, 11})
        self.assertEqual(p.coverage(selected)["weighted_gate_retention"], 1.0)
        self.assertEqual(p.coverage(p.choose(4))["weighted_gate_retention"] < 1, True)

    def test_no_text_ingestion_and_invalid_router_traces(self):
        p = RoutingProfile(layers=1, experts=8)
        base = event(0, "decode", "code", [0, 1, 2, 3])
        for broken in [
            {**base, "prompt": "private token"},
            {**base, "experts": [0, 1, 2, 2]},
            {**base, "experts": [0, 1, 2, 100]},
            {**base, "weights": [1, 0, 0, float("nan")]},
            {**base, "layer": True},
            {**base, "domain": "../path"},
        ]:
            with self.subTest(broken=broken):
                with self.assertRaises(ValueError):
                    p.observe(broken)

    def test_missing_layer_fails_closed(self):
        p = RoutingProfile(layers=2, experts=8)
        p.observe(event(0, "decode", "code", [0, 1, 2, 3]))
        self.assertEqual(p.missing_layers(), [1])
        with self.assertRaises(ValueError):
            p.choose(4)

    def test_streaming_evaluation_with_holdout(self):
        with tempfile.TemporaryDirectory() as td:
            cal, ev = Path(td) / "train.jsonl", Path(td) / "test.jsonl"
            rows = [event(layer, phase, domain, ids)
                    for layer in range(2)
                    for phase, domain, ids in [
                        ("prefill", "code", [0, 1, 2, 3]),
                        ("decode", "math", [8, 9, 10, 11])
                    ]]
            cal.write_text("".join(json.dumps(row) + "\n" for row in rows),
                           encoding="utf-8")
            ev.write_text(cal.read_text(encoding="utf-8"), encoding="utf-8")
            data = evaluate(cal, ev, budgets=(8, 4), layers=2, experts=16)
            self.assertEqual(len(data["results"]), 2)
            self.assertEqual(data["results"][0]["heldout"]["weighted_gate_retention"], 1)
            self.assertLess(data["results"][1]["heldout"]["weighted_gate_retention"], 1)
            self.assertFalse(data["results"][0]["model_accuracy_evaluated"])
            with self.assertRaises(ValueError):
                evaluate(cal, cal, budgets=(8,), layers=2, experts=16)

    def test_no_contribution_if_uncalibrated(self):
        p = RoutingProfile(layers=1, experts=8)
        zero = event(0, "decode", "code", [0, 1, 2, 3])
        zero["contributions"] = [0, 0, 0, 0]
        p.observe(zero)
        self.assertEqual(len(p.choose(4)[0]), 4)


if __name__ == "__main__":
    unittest.main()
