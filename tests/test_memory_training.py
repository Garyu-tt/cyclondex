import json
import tempfile
import unittest
from pathlib import Path

from tools.train_memory_router import fit


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8")


class MemoryRouterTrainingTests(unittest.TestCase):
    def test_supervised_streaming_and_holdout(self):
        with tempfile.TemporaryDirectory() as d:
            train, evaluation = Path(d) / "train.jsonl", Path(d) / "eval.jsonl"
            task = "GitHub previous PR decision"
            write_rows(train, [
                {"task": task, "useful": False, "independently_reviewed": True}
                for _ in range(80)
            ] + [{"task": "unreviewed task", "useful": True,
                   "independently_reviewed": False}])
            write_rows(evaluation, [
                {"task": task, "useful": False, "independently_reviewed": True}
            ])
            stats = fit(train, evaluation)
            self.assertEqual(stats["train_reviewed"], 80)
            self.assertEqual(stats["train_skipped"], 1)
            self.assertLess(stats["eval_before"]["accuracy"],
                            stats["eval_after"]["accuracy"])

    def test_no_training_on_same_file_or_malformed_data(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            write_rows(path, [{"task": "x", "useful": "yes",
                               "independently_reviewed": True}])
            with self.assertRaises(ValueError):
                fit(path, path)
            other = Path(d) / "eval.jsonl"
            other.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                fit(path, other)


if __name__ == "__main__":
    unittest.main()
