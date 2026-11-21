"""Reproducible ALT/gpt-oss MoE router trace analysis (NO model pruning).

Only aggregate selected expert IDs, gate weights and optional contribution
metrics. Never take text or secrets as input. Selection is an experiment,
NOT a claimed REAP implementation or proof of model-quality preservation.
"""
from collections import defaultdict
import math
import re

FIELDS = frozenset({"layer", "domain", "phase", "experts", "weights",
                    "contributions"})
DOMAIN = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
PHASES = frozenset({"prefill", "decode"})
PARAM_TOTAL_B = 116.8
PARAM_ACTIVE_B = 5.1
GPT_OSS_EXPERTS = 128
GPT_OSS_TOP_K = 4
GPT_OSS_LAYERS = 36


def estimate_params(kept, *, experts=GPT_OSS_EXPERTS, top_k=GPT_OSS_TOP_K):
    """Analytical parameter-count estimate, not checkpoint/memory benchmark."""
    if experts != GPT_OSS_EXPERTS or top_k != GPT_OSS_TOP_K:
        return None
    if type(kept) is not int or not top_k <= kept <= experts:
        raise ValueError("invalid expert count")
    shared = (experts * PARAM_ACTIVE_B -
              top_k * PARAM_TOTAL_B) / (experts - top_k)
    per_expert = (PARAM_TOTAL_B - shared) / experts
    return {"total_B": round(shared + kept * per_expert, 3),
            "active_B": round(shared + top_k * per_expert, 3),
            "shared_B": round(shared, 3),
            "expert_B_per_layer_group": round(per_expert, 6),
            "q4_weights_GB_min": round((shared + kept * per_expert) / 2, 3)}


class RoutingProfile:
    """Bounded statistics by (layer, domain, phase), with no raw text."""

    def __init__(self, layers=GPT_OSS_LAYERS, experts=GPT_OSS_EXPERTS,
                 top_k=GPT_OSS_TOP_K):
        if (any(type(x) is not int for x in (layers, experts, top_k)) or
                not 1 <= layers <= 128 or not 4 <= experts <= 512 or
                not 1 <= top_k <= experts):
            raise ValueError("invalid architecture")
        self.layers, self.experts, self.top_k = layers, experts, top_k
        self.mass = defaultdict(lambda: [0.0] * experts)
        self.gates = defaultdict(lambda: [0.0] * experts)
        self.samples = defaultdict(int)

    def observe(self, item):
        if not isinstance(item, dict) or set(item) - FIELDS:
            raise ValueError("only routing metadata is accepted")
        layer, domain, phase = (item.get("layer"), item.get("domain"),
                                item.get("phase"))
        ids, weights = item.get("experts"), item.get("weights")
        if (type(layer) is not int or not 0 <= layer < self.layers or
                not isinstance(domain, str) or not DOMAIN.fullmatch(domain) or
                phase not in PHASES or not isinstance(ids, list) or
                not isinstance(weights, list) or
                len(ids) != self.top_k or len(weights) != self.top_k or
                any(type(i) is not int or not 0 <= i < self.experts
                    for i in ids) or len(set(ids)) != self.top_k or
                any(type(w) not in (float, int) or not math.isfinite(w)
                    or w < 0 for w in weights)):
            raise ValueError("invalid routing trace")
        denom = sum(weights)
        if not math.isfinite(denom) or denom <= 0:
            raise ValueError("nonpositive router weights")
        contributions = item.get("contributions", [1.] * self.top_k)
        if (not isinstance(contributions, list) or
                len(contributions) != self.top_k or
                any(type(c) not in (float, int) or not math.isfinite(c)
                    or not 0 <= c <= 1000 for c in contributions)):
            raise ValueError("invalid contribution values")
        key = (layer, domain, phase)
        # Do not store individual tokens, prompts, full states or activations.
        for index, weight, value in zip(ids, weights, contributions):
            g = weight / denom
            self.gates[key][index] += g
            self.mass[key][index] += g * value
        self.samples[key] += 1

    def missing_layers(self):
        present = {key[0] for key in self.samples}
        return [i for i in range(self.layers) if i not in present]

    def choose(self, keep):
        if type(keep) is not int or not self.top_k <= keep <= self.experts:
            raise ValueError("expert budget outside available range")
        if self.missing_layers():
            raise ValueError("calibration missing layers: " +
                             str(self.missing_layers()))
        selected = {}
        for layer in range(self.layers):
            groups = [k for k in self.samples if k[0] == layer]
            # Each domain/phase gets equal representation independent of its
            # token count, unlike raw-frequency-only pruning.
            weights = [0.0] * self.experts
            for key in groups:
                mass = self.mass[key]
                total = sum(mass)
                if total <= 0:
                    # Fall back to gates if contributions were all zero.
                    mass, total = self.gates[key], sum(self.gates[key])
                for idx, value in enumerate(mass):
                    weights[idx] += value / total / len(groups)
            selected[layer] = sorted(
                sorted(range(self.experts),
                       key=lambda i: (-weights[i], i))[:keep])
        return selected

    def coverage(self, selected):
        rows, total_weight, retained_weight = [], 0., 0.
        for (layer, domain, phase), gates in sorted(self.gates.items()):
            if layer not in selected:
                raise ValueError("selected set missing layer")
            kept = set(selected[layer])
            if not kept or any(type(i) is not int or
                               not 0 <= i < self.experts for i in kept):
                raise ValueError("invalid expert selection")
            total = sum(gates)
            retained = sum(gates[i] for i in kept)
            percent = retained / total
            rows.append({"layer": layer, "domain": domain, "phase": phase,
                         "samples": self.samples[(layer, domain, phase)],
                         "retained_gate_mass": round(percent, 6)})
            total_weight += total
            retained_weight += retained
        if not rows:
            raise ValueError("no evaluation samples")
        return {
            "weighted_gate_retention": round(retained_weight / total_weight, 6),
            "worst_group_retention": round(min(r["retained_gate_mass"]
                                                for r in rows), 6),
            "groups": rows, "token_layer_events": sum(self.samples.values()),
            "missing_layers": self.missing_layers(),
        }
