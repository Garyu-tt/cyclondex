import copy
import unittest

from cyclondex.checkpoint_plan import (
    build_plan, digest, validate_source, LAYERS, EXPERTS, EXPERT_SUFFIXES,
)


def fixture():
    config = {
        "model_type": "gpt_oss", "architectures": ["GptOssForCausalLM"],
        "num_hidden_layers": LAYERS, "num_local_experts": EXPERTS,
        "experts_per_token": 4, "num_experts_per_tok": 4,
        "quantization_config": {"quant_method": "mxfp4"},
        "layer_types": ["sliding_attention", "full_attention"] * 18,
    }
    weight_map = {
        f"model.layers.{layer}.mlp.{suffix}": "model-00001-of-00015.safetensors"
        for layer in range(LAYERS) for suffix in EXPERT_SUFFIXES
    }
    weight_map["model.embed_tokens.weight"] = "model-00015-of-00015.safetensors"
    index = {"metadata": {"total_size": 65_248_815_744}, "weight_map": weight_map}
    ranking = {str(layer): [float((i * 3 + layer) % 128)
                           for i in range(EXPERTS)] for layer in range(LAYERS)}
    return config, index, ranking


class PlanTests(unittest.TestCase):
    def test_complete_deterministic_remapping(self):
        config, index, scores = fixture()
        report = build_plan(config, index, scores, 48)
        again = build_plan(config, index, scores, 48)
        self.assertEqual(digest(report), digest(again))
        self.assertEqual(len(report["expert_index_maps"]), LAYERS)
        self.assertEqual(len(report["tensor_slicing_plan"]), 8 * LAYERS)
        self.assertEqual(report["source_index_audit"]["validated_moe_tensors"], 288)
        self.assertFalse(report["model_weights_loaded"])
        self.assertTrue(report["requires_generation_quality_benchmarks"])
        self.assertGreater(report["predicted_checkpoint_bytes"], 0)
        for layer, mapping in report["expert_index_maps"].items():
            self.assertEqual(len(mapping["new_to_old"]), 48)
            self.assertEqual(len(set(mapping["new_to_old"])), 48)
            self.assertEqual(list(mapping["new_to_old"]),
                             sorted(mapping["new_to_old"]))
            for new, old in enumerate(mapping["new_to_old"]):
                self.assertEqual(mapping["old_to_new"][str(old)], new)

    def test_missing_tensors_fail_before_plan(self):
        config, index, scores = fixture()
        del index["weight_map"]["model.layers.35.mlp.experts.down_proj_scales"]
        with self.assertRaisesRegex(ValueError, "missing indexed"):
            build_plan(config, index, scores, 48)

    def test_wrong_architecture_and_path_rejected(self):
        config, index, scores = fixture()
        wrong = copy.deepcopy(config)
        wrong["num_local_experts"] = 48
        with self.assertRaises(ValueError):
            build_plan(wrong, index, scores, 16)
        bad = copy.deepcopy(index)
        bad["weight_map"]["model.embed_tokens.weight"] = "../private.safetensors"
        with self.assertRaisesRegex(ValueError, "unsafe shard"):
            build_plan(config, bad, scores, 48)

    def test_revision_and_budget_validated(self):
        config, index, scores = fixture()
        with self.assertRaises(ValueError):
            build_plan(config, index, scores, 3)
        with self.assertRaises(ValueError):
            build_plan(config, index, scores, 48, model_revision="main")
        ok = build_plan(config, index, scores, 16,
                        model_revision="a" * 40)
        self.assertEqual(ok["source_revision"], "a" * 40)
        self.assertEqual(ok["architecture"]["new_experts"], 16)

    def test_rank_input_must_have_36_layers_and_128_experts(self):
        config, index, scores = fixture()
        scores.pop("35")
        with self.assertRaises(ValueError):
            build_plan(config, index, scores, 48)


if __name__ == "__main__":
    unittest.main()
