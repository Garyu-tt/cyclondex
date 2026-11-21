import math
import unittest

from cyclondex.saliency import compare, select, validate_map


class SaliencyTests(unittest.TestCase):
    def test_frequency_and_saliency_differ(self):
        counts = {0: (9.0, 8.0, 7.0, 1.0, 1.0, 1.0),
                  1: (9.0, 8.0, 7.0, 1.0, 1.0, 1.0)}
        saliency = {0: (1.0, 1.0, 1.0, 10.0, 9.0, 8.0),
                    1: (1.0, 1.0, 1.0, 10.0, 9.0, 8.0)}
        report = compare(saliency, counts, budgets=(3,))
        result = report["results"][0]
        self.assertEqual(result["set_overlap_fraction"], 0)
        self.assertLess(result["saliency_selected_routing_frequency"],
                        result["frequency_selected_routing_frequency"])
        self.assertFalse(report["quality_evaluated"])

    def test_all_values_validated(self):
        valid = {str(i): [float(j+1) for j in range(6)]
                 for i in range(2)}
        output = validate_map(valid, layers=2, experts=6)
        self.assertEqual(output[0][0], 1.)
        for broken in [
            {"0": valid["0"]},
            {"0": valid["0"], "1": [float("nan")] * 6},
            {"0": valid["0"], "1": [-1.0] * 6},
            {"0": [0.0] * 6, "1": valid["1"]},
            {"0": [0.0, "secret", 1, 1, 1, 1], "1": valid["1"]},
        ]:
            with self.assertRaises(ValueError):
                validate_map(broken, layers=2, experts=6)

    def test_select_is_deterministic_on_ties(self):
        s = {0: (1., 1., 0., 0.)}
        self.assertEqual(select(s, 2), {0: [0, 1]})
        with self.assertRaises(ValueError):
            select(s, 5)


if __name__ == "__main__":
    unittest.main()
