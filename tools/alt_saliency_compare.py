"""Compare existing REAP saliency & router count JSON aggregates.

Input files are e.g. published results/rtx5090-reap-{saliency,counts}.json
from bryanvine/gpt-oss-subset (Apache-2.0). No model checkpoint needed.
Results are aggregate statistics; there is NO benchmark of generation quality.
"""
import argparse
import json
from pathlib import Path
from cyclondex.saliency import validate_map, compare


def load(path):
    if Path(path).stat().st_size > 2_000_000:
        raise ValueError("aggregate file too large")
    return validate_map(json.loads(Path(path).read_text(encoding="utf-8")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saliency", required=True)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths = [Path(x).resolve() for x in
             (args.saliency, args.counts, args.out)]
    if len(set(paths)) != 3:
        parser.error("input/output paths must be distinct")
    report = compare(load(paths[0]), load(paths[1]))
    paths[2].parent.mkdir(parents=True, exist_ok=True)
    paths[2].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print("Compared published saliency and counts metadata across 36 layers.")
    print("Model quality tested: no; model weights modified: no")


if __name__ == "__main__":
    main()
