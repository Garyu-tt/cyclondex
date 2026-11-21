"""Offline streaming training for the CYCLON memory retrieval gate.

Use independently audited, locally held JSONL. Does NOT train an 80B LLM.
Input is never uploaded or printed; the operator protects source files.
"""
import argparse
import json
from pathlib import Path

from cyclondex.policy import RetrievalPolicy


def records(path, max_rows=100000):
    count = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > 8192:
                raise ValueError("sample too large")
            count += 1
            if count > max_rows:
                raise ValueError("row budget exceeded")
            row = json.loads(line)
            if (not isinstance(row, dict)
                    or type(row.get("useful")) is not bool
                    or type(row.get("independently_reviewed")) is not bool
                    or not isinstance(row.get("task"), str)
                    or not 1 <= len(row["task"]) <= 4096):
                raise ValueError("invalid reviewed trajectory record")
            yield row


def evaluate(policy, path):
    correct, total = 0, 0
    for row in records(path):
        if not row["independently_reviewed"]:
            continue
        correct += policy.should_search(row["task"]) == row["useful"]
        total += 1
    return {"reviewed": total,
            "accuracy": round(correct / total, 4) if total else None}


def fit(train, evaluation, learning_rate=.3):
    if Path(train).resolve() == Path(evaluation).resolve():
        raise ValueError("train and evaluation must be separate files")
    policy = RetrievalPolicy(learning_rate)
    before = evaluate(policy, evaluation)
    skipped = 0
    for row in records(train):
        if not row["independently_reviewed"]:
            skipped += 1
            continue
        policy.learn(row["task"], row["useful"],
                     independently_reviewed=True)
    after = evaluate(policy, evaluation)
    return {"train_reviewed": policy.verified_updates,
            "train_skipped": skipped, "eval_before": before,
            "eval_after": after, "weight_count": len(policy.weights)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--lr", type=float, default=.3)
    args = p.parse_args()
    stats = fit(args.train, args.eval, args.lr)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    print("Metrics only; no model or private Noosphere data exported.")


if __name__ == "__main__":
    main()
