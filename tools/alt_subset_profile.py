"""Analyze ALT router traces, without downloading or modifying model weights.

Example:
PYTHONPATH=. python3 tools/alt_subset_profile.py --calibration train.jsonl
   --holdout heldout.jsonl --out runs/alt-profile.json

Every JSONL line must contain ONLY:
{"layer":0,"domain":"code","phase":"decode",
 "experts":[1,2,3,4],"weights":[.4,.3,.2,.1]}
Optional: "contributions":[1.0,1.0,1.0,1.0].
"""
import argparse
import json
from pathlib import Path
from cyclondex.expert_reduction import RoutingProfile, estimate_params


def read_traces(path, profile, max_rows=5_000_000):
    count = 0
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > 4096:
                raise ValueError("oversized trace row")
            count += 1
            if count > max_rows:
                raise ValueError("routing trace event budget exceeded")
            try:
                profile.observe(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError("invalid router metadata at line " +
                                 str(lineno)) from exc
    if count == 0:
        raise ValueError("empty routing trace")
    return count


def evaluate(calibration, holdout, budgets=(48, 32, 16),
             *, layers=36, experts=128, top_k=4):
    if Path(calibration).resolve() == Path(holdout).resolve():
        raise ValueError("evaluation and calibration must be separate files")
    train = RoutingProfile(layers, experts, top_k)
    test = RoutingProfile(layers, experts, top_k)
    read_traces(calibration, train)
    read_traces(holdout, test)
    results = []
    for count in budgets:
        selected = train.choose(count)
        measured = test.coverage(selected)
        results.append({
            "kept_per_layer": count,
            "estimated_parameters": estimate_params(count, experts=experts,
                                                      top_k=top_k),
            "selected_expert_ids": {str(k): v for k, v in selected.items()},
            "heldout": measured,
            "model_accuracy_evaluated": False,
        })
    return {
        "schema": "cyclon.alt-router-profile.v1",
        "architecture": {"layers": layers, "experts": experts, "top_k": top_k},
        "calibration_events": sum(train.samples.values()),
        "heldout_events": sum(test.samples.values()),
        "data_contains_prompts": False,
        "warning": "Router coverage != generation quality. No checkpoints pruned.",
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--budgets", default="48,32,16")
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    budgets = tuple(int(x) for x in args.budgets.split(","))
    output = Path(args.out)
    if output.resolve() in (Path(args.calibration).resolve(),
                            Path(args.holdout).resolve()):
        parser.error("output cannot overwrite input traces")
    report = evaluate(args.calibration, args.holdout, budgets, layers=args.layers,
                      experts=args.experts, top_k=args.top_k)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print("Routing metadata evaluated, budgets:",
          [r["kept_per_layer"] for r in report["results"]])
    print("Model weights modified: no; model quality evaluated: no")


if __name__ == "__main__":
    main()
