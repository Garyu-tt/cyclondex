"""Create immutable, dry-run remapping instructions without loading ALT weights.

Input files: authorized openai/gpt-oss-120b config.json,
model.safetensors.index.json and existing aggregate per-layer saliency JSON.
Source model weights are not needed for this structural planning stage.
"""
import argparse
import json
from pathlib import Path

from cyclondex.checkpoint_plan import build_plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--saliency", required=True)
    p.add_argument("--keep", required=True, type=int)
    p.add_argument("--out", required=True)
    p.add_argument("--model-revision")
    p.add_argument("--selection-source", default="external_calibration")
    args = p.parse_args()
    inputs = [Path(s).resolve() for s in (args.config, args.index, args.saliency)]
    out = Path(args.out).resolve()
    if out in inputs or out.exists():
        p.error("output must be a new file, not one of the inputs")
    for path in inputs:
        if path.stat().st_size > 2_000_000:
            p.error("metadata file exceeds size cap")
    config, index, ranking = [json.loads(p.read_text(encoding="utf-8"))
                              for p in inputs]
    plan = build_plan(config, index, ranking, args.keep,
                      model_revision=args.model_revision,
                      selection_source=args.selection_source)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({
        "schema": plan["schema"],
        "keep": args.keep,
        "layers": len(plan["expert_index_maps"]),
        "expert_indexed_tensor_operations": len(plan["tensor_slicing_plan"]),
        "source_shards": plan["source_index_audit"]["indexed_shards"],
        "predicted_bytes": plan["predicted_checkpoint_bytes"],
        "weights_loaded": False, "weights_modified": False,
        "quality_evaluated": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
