"""Offline, immutable expert remap plan for a gpt-oss-120b MXFP4 snapshot.

No tensors are loaded. No files are deleted, model weights never modified.
This module validates the HF safetensors index and config before producing
a reproducible list of source->target expert indices.
"""
import hashlib
import json
import re

from .saliency import select, validate_map

LAYERS = 36
EXPERTS = 128
TOP_K = 4
PER_EXPERT_PER_LAYER_BYTES = 13_236_480  # From published MXFP4 tensor study.
EXPERT_SUFFIXES = (
    "experts.gate_up_proj_blocks",
    "experts.gate_up_proj_scales",
    "experts.gate_up_proj_bias",
    "experts.down_proj_blocks",
    "experts.down_proj_scales",
    "experts.down_proj_bias",
    "router.weight",
    "router.bias",
)
SAFE_SHARD = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*\.safetensors$")
REVISION = re.compile(r"^[a-fA-F0-9]{40}$")


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False,
                                     sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def validate_source(config, index):
    if not isinstance(config, dict) or not isinstance(index, dict):
        raise ValueError("config and index must be objects")
    if (config.get("model_type") != "gpt_oss" or
            config.get("architectures") != ["GptOssForCausalLM"] or
            type(config.get("num_hidden_layers")) is not int or
            config["num_hidden_layers"] != LAYERS or
            type(config.get("num_local_experts")) is not int or
            config["num_local_experts"] != EXPERTS or
            type(config.get("experts_per_token")) is not int or
            config["experts_per_token"] != TOP_K or
            type(config.get("num_experts_per_tok")) is not int or
            config["num_experts_per_tok"] != TOP_K or
            config.get("quantization_config", {}).get("quant_method")
            != "mxfp4"):
        raise ValueError("not the expected untouched gpt-oss-120b MXFP4 config")
    if not isinstance(config.get("layer_types"), list) or len(config["layer_types"]) != LAYERS:
        raise ValueError("wrong layer type count")
    mapping = index.get("weight_map")
    metadata = index.get("metadata")
    if (not isinstance(mapping, dict) or not mapping or
            not isinstance(metadata, dict) or
            type(metadata.get("total_size")) is not int or
            metadata["total_size"] <= 0):
        raise ValueError("invalid safetensors index")
    for tensor, shard in mapping.items():
        if (not isinstance(tensor, str) or not tensor or
                not isinstance(shard, str) or not SAFE_SHARD.fullmatch(shard)):
            raise ValueError("invalid tensor or unsafe shard name")
    missing = []
    for layer in range(LAYERS):
        for suffix in EXPERT_SUFFIXES:
            name = f"model.layers.{layer}.mlp.{suffix}"
            if name not in mapping:
                missing.append(name)
    if missing:
        raise ValueError("missing indexed MoE tensors: " + str(len(missing)))
    return {"original_index_total_size_bytes": metadata["total_size"],
            "indexed_tensors": len(mapping),
            "indexed_shards": len(set(mapping.values())),
            "validated_moe_tensors": LAYERS * len(EXPERT_SUFFIXES)}


def build_plan(config, index, saliency, keep, *,
               model_revision=None, selection_source="external_calibration"):
    source = validate_source(config, index)
    if type(keep) is not int or not TOP_K <= keep <= EXPERTS:
        raise ValueError("invalid expert subset size")
    if model_revision is not None and (
            not isinstance(model_revision, str) or
            not REVISION.fullmatch(model_revision)):
        raise ValueError("model revision must be an immutable 40-char commit ID")
    if (not isinstance(selection_source, str) or
            not 1 <= len(selection_source) <= 200 or
            any(ch in selection_source for ch in "\r\n\x00")):
        raise ValueError("invalid source label")
    ranking = validate_map(saliency, layers=LAYERS, experts=EXPERTS)
    selected = select(ranking, keep)
    operations = []
    layers = {}
    for layer in range(LAYERS):
        old = selected[layer]
        layers[str(layer)] = {
            "new_to_old": old,
            "old_to_new": {str(src): dst for dst, src in enumerate(old)},
        }
        for suffix in EXPERT_SUFFIXES:
            tensor = f"model.layers.{layer}.mlp.{suffix}"
            operations.append({
                "tensor": tensor, "shard": index["weight_map"][tensor],
                "expert_axis": 0, "expected_before": EXPERTS,
                "expected_after": keep,
                "new_to_old": old,
            })
    removed = (EXPERTS - keep) * LAYERS * PER_EXPERT_PER_LAYER_BYTES
    estimated = source["original_index_total_size_bytes"] - removed
    if estimated <= 0:
        raise ValueError("source index size smaller than predicted removed experts")
    return {
        "schema": "cyclon.alt-gptoss-mxfp4-plan.v1",
        "source_model": "openai/gpt-oss-120b",
        "source_revision": model_revision,
        "source_config_sha256": digest(config),
        "source_index_sha256": digest(index),
        "ranking_sha256": digest(saliency),
        "selection_source": selection_source,
        "selection_kind": "top-k by supplied score; not quality verified",
        "architecture": {"layers": LAYERS, "original_experts": EXPERTS,
                         "new_experts": keep, "active_experts_per_token": TOP_K},
        "source_index_audit": source,
        "predicted_checkpoint_bytes": estimated,
        "predicted_size_is_unverified": True,
        "expert_index_maps": layers,
        "tensor_slicing_plan": operations,
        "requires_loaded_checkpoint_validation": True,
        "requires_generation_quality_benchmarks": True,
        "model_weights_loaded": False,
        "model_weights_modified": False,
    }
