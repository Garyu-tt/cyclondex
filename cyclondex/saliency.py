"""Compare published aggregated MoE routing counts with REAP saliency scores.

This is a *selection diagnostic*, NOT a claim of checkpoint correctness
or model quality. It accepts only 36x128 finite nonnegative numeric arrays.
"""
import math


def validate_map(raw, *, layers=36, experts=128, name="scores"):
    if (not isinstance(raw, dict) or
            set(raw) != {str(i) for i in range(layers)}):
        raise ValueError(name + " must contain all consecutive layer IDs")
    result = {}
    for layer in range(layers):
        arr = raw[str(layer)]
        if (not isinstance(arr, list) or len(arr) != experts or
                any(type(x) not in (float, int) or not math.isfinite(x)
                    or x < 0 for x in arr)):
            raise ValueError("invalid " + name + " for layer " + str(layer))
        if sum(arr) <= 0:
            raise ValueError("all-zero " + name + " for layer " + str(layer))
        result[layer] = tuple(float(x) for x in arr)
    return result


def select(scoremap, keep):
    experts = len(next(iter(scoremap.values())))
    if type(keep) is not int or not 1 <= keep <= experts:
        raise ValueError("invalid number of experts")
    return {layer: sorted(sorted(range(experts),
                         key=lambda i: (-scores[i], i))[:keep])
            for layer, scores in scoremap.items()}


def compare(saliency, counts, *, budgets=(48, 32, 16)):
    if set(saliency) != set(counts):
        raise ValueError("missing layer in comparison")
    for layer in saliency:
        if len(saliency[layer]) != len(counts[layer]):
            raise ValueError("incompatible expert counts")
    rows = []
    for keep in budgets:
        selected_saliency = select(saliency, keep)
        selected_frequency = select(counts, keep)
        retention_saliency, retention_frequency, overlaps = [], [], []
        for layer in sorted(saliency):
            frequencies = counts[layer]
            total = sum(frequencies)
            s = set(selected_saliency[layer])
            f = set(selected_frequency[layer])
            retention_saliency.append(sum(frequencies[i] for i in s) / total)
            retention_frequency.append(sum(frequencies[i] for i in f) / total)
            overlaps.append(len(s & f) / keep)
        rows.append({
            "kept_per_layer": keep,
            "saliency_selected_routing_frequency": round(
                sum(retention_saliency) / len(retention_saliency), 6),
            "frequency_selected_routing_frequency": round(
                sum(retention_frequency) / len(retention_frequency), 6),
            "saliency_worst_layer_routing_frequency": round(
                min(retention_saliency), 6),
            "set_overlap_fraction": round(sum(overlaps) / len(overlaps), 6),
        })
    return {
        "schema": "cyclon.alt-saliency-comparison.v1",
        "layers": len(saliency),
        "experts_per_layer": len(next(iter(saliency.values()))),
        "quality_evaluated": False,
        "warning": "Routing frequency is NOT predictive of model quality.",
        "results": rows,
    }
