# CYCLON from ALT: expert reduction R0

**Status: profiling framework only. No checkpoint downloaded, copied, edited,
trained or run. No pruning quality claims.**

## Motivation and hypotheses

Goal: build **two CYCLON agents** sharing a single smaller ALT-derived
pretrained MoE backbone, with separately scoped contexts and Noosphere
memories. This must be measured against alternatives rather than assumed
equivalent to full gpt-oss-120b.

OpenAI's gpt-oss-120b: 36 layers; 128 experts per MoE layer; top-4 routing;
~116.8B total and ~5.1B active parameters per token.

Under a constant-shared-layer, equal-sized-expert **approximation**:

| Retained experts/layer | Estimated total | Approx active/token |
|--:|--:|--:|
| 128 | 116.8B | 5.1B |
| 48 | 44.7B | 5.1B |
| 32 | 30.3B | 5.1B |
| 16 | 15.9B | 5.1B |
| 8 | 8.7B | 5.1B |

These are *parameter estimates*, not real checkpoint sizes, model
quality, serving VRAM or tokens/sec. Q4 theoretical weight bytes in
the profiler exclude MXFP4 scales, non-MoE high-precision weights,
activations, KV cache, framework, offload and duplicated state.

**Critical:** equal top-4 means active expert FLOPs per token are not
reduced proportionally. The two agents must share base weights in the
same resident runtime if memory savings are desired. They still need
independent KV caches and will compete for compute.

## What real-world research supports

- [OpenAI gpt-oss](https://openai.com/index/introducing-gpt-oss/):
  documented original architecture.
- [bryanvine/gpt-oss-subset](https://github.com/bryanvine/gpt-oss-subset):
  public Apache-2.0 study with a 48-expert MXFP4-preserving reduced
  checkpoint performing well on a **specific private classification
  workload**. Its README explicitly says the 16 GB target is unsolved.
  Frequency-only 32/48 variants reportedly underperformed there.
- These claims do **not** establish that CYCLON-16 or -32 retains
  broad software-engineering and agent performance.
- Inspect license and source revision before reusing external code.
  Current implementation is independent research code, not copied
  from that project.

## Experiment pipeline

1. **Teacher baseline:** measure the original ALT model on diverse
   held-out GitHub software engineering tasks, reasoning/math, tool use,
   dialogues and memory-retrieval tasks. First record exact checkpoint
   hashes, license, tokenizer, runtime, seed, GPU/CPU and precision.
2. **Routing calibration:** instrument an authorized inference runtime
   to emit *only* selected expert IDs and normalized gate probabilities,
   optional nonnegative per-expert contribution strengths, plus
   layer/domain/prefill-vs-decode. No prompts, completions, file contents,
   user identity or Noosphere text in routing traces.
3. **Candidate selection:** this R0 averages normalized contribution/gate
   mass **per layer and per domain/phase** so high-volume domains do not
   entirely swamp rare ones. This is NOT yet the REAP saliency algorithm.
4. **Held-out coverage:** evaluate retained top-4 gate mass, worst
   domain/phase retention and missing layers on a separate trace file.
   Coverage is an advisory proxy, not a loss or generation metric.
5. **Checkpoint transformation:** future R1 must remap router indices and
   expert tensors *without violating MXFP4 representation*. Never
   mutate the only copy of ALT. A valid reduced checkpoint must load,
   route and produce outputs with structural tests.
6. **Quality gates:** teacher vs subset on held-out GitHub project tasks,
   real compile/test/review, format adherence, mathematics, safety,
   Noosphere source recall and independent reviewer verdicts. Include
   output distribution agreement when feasible.
7. **Mikoshi performance gates:** independently benchmark resident RAM,
   VRAM, PCIe transfer, power, tokens/s, time-to-first-token, contention
   with two active contexts, and idle draw. BitWeave is an optional
   validated backend, not an assumed LLM speed multiplier.
8. **Remediation:** if losing quality, use task-aware expert merging,
   router calibration and adapter distillation with measured holdouts.
   Keep a nonpruned fallback and never mix train/evaluation traffic.

## Baselines and candidate order

- **Candidate first:** 48 retained experts, then 32, then 16. No automatic
  promotion just because a candidate fits.
- **Essential comparison:** stock gpt-oss-20b, whose 24 layers and
  original 32 experts differ from a pruned 36-layer ALT-32.
  It may be faster or higher quality on Mikoshi for a given task.
- Always benchmark 1 vs 2 independent logical agent sessions sharing the
  same model memory; two separately resident 80B backbones are outside
  the initial CPU-first budget.

## Trace format / command

One line per *token/layer routing observation* (synthetic example):

```json
{"layer":0,"domain":"code","phase":"decode","experts":[7,8,3,42],"weights":[0.4,0.3,0.2,0.1]}
```

Optional contributions are finite nonnegative numbers, but must be from
a documented, reproducible runtime metric. Without contributions the
profiler is **gate-mass weighted**, not REAP.

```sh
PYTHONPATH=. python3 tools/alt_subset_profile.py \
  --calibration /private/calibration-router.jsonl \
  --holdout /private/heldout-router.jsonl \
  --out /private/cyclon-router-report.json
python3 -m unittest discover -s tests -v
```

Use independent calibration/heldout tasks and sources; supplying two
copies of the same sample as separate files is invalid evaluation in
real research. The script enforces distinct paths, but not semantic
non-overlap; the data curator must enforce that.

**Do not submit calibration traces or private prompts to GitHub.**
Reports contain only routing IDs, aggregate metrics and estimated sizes.

## Ship/no-ship conditions

No deployment or checkpoint conversion unless: all 36 layers represented,
separate heldouts, model functionality tests, benchmark regression within
an explicitly approved tolerance, resource and power ceilings, independent
review, privacy-safe dataset, and rollback to ALT or stock gpt-oss-20b.

This R0 covers only steps 2 (metadata shape), 3 and 4.


## Published aggregate comparison (no model loading)

A zero-download-of-weights sanity check was computed using the
**public aggregate JSON results** in
[bryanvine/gpt-oss-subset](https://github.com/bryanvine/gpt-oss-subset):

- `results/rtx5090-reap-saliency.json`, Git blob SHA
  `c85bab142f97dfe3010d938f5485ec69acaa729e`;
- `results/rtx5090-reap-counts.json`, Git blob SHA
  `2da534845f299a458981268a92f86a9699390912`.

These represent *different score types on a published calibration*
and have 36 layers × 128 experts. The tool selects the top-N from
saliency versus the top-N from **frequency**, then calculates what
fraction of the same published frequency counts each set retains.

| Keep | Top-N counts retention | Top-N saliency set's counts retention | Mean overlap |
|--:|--:|--:|--:|
| 48 | 88.30% | 44.62% | 41.49% |
| 32 | 78.77% | 31.33% | 30.03% |
| 16 | 60.88% | 18.39% | 17.19% |

**Interpretation:** the selections are very different. Frequency
retention is defined here using *routed-expert counts*, not the
quality of the next token and not percentage of unique tokens. In
particular low count-retention of a high-saliency set does NOT mean
that the model lost that fraction of its intelligence.

The upstream study found task-specific quality recovery using
REAP-style importance, despite frequency-only baselines failing.
This motivates testing meaningful contribution metrics, not just
count concentration. There is no held-out CYCLON model-quality test
yet, and no claims about 16-expert performance.

Reproduce with the two public JSON files downloaded into a private
test directory (do not commit a large checkpoint or a dataset):

```sh
PYTHONPATH=. python3 tools/alt_saliency_compare.py \
  --saliency /local/rtx5090-reap-saliency.json \
  --counts /local/rtx5090-reap-counts.json \
  --out /local/saliency-comparison.json
```


## Verified official *metadata-only* checkpoint inventory (21 Nov 2026)

Retrieved the public `openai/gpt-oss-120b/config.json` and all chunks of
`model.safetensors.index.json` through Hugging Face. This analysis
downloaded **no tensor weights**.

- Original model: `GptOssForCausalLM`, `gpt_oss`, 36 layers,
  128 experts/layer, `experts_per_token=4`,
  `num_experts_per_tok=4`, quantization `mxfp4`.
- Safetensors index: **687 entries in 15 shards**; source size index metadata:
  **65,248,815,744 bytes (60.768 GiB)**.
- All **288** required per-layer MoE projection/router tensor names
  for this checkpoint layout were present.
- With the published 13,236,480 bytes per expert per layer assumption,
  projected full checkpoint storage is:

| Keep | Predicted bytes | Predicted GiB |
|---:|---:|---:|
| 48 | 27,127,753,344 | 25.265 |
| 32 | 19,503,540,864 | 18.164 |
| 16 | 11,879,328,384 | 11.063 |

These are **unverified projections**, not actual exported artifacts.
Quantized checkpoint tensor sizes and temporary conversion headroom
still need end-to-end verification before deployment.

`cyclondex.checkpoint_plan` now validates the original config and
index, and generates an immutable 36-layer expert remapping plan
from an existing score map, without ever reading weights:

```sh
PYTHONPATH=. python3 tools/alt_checkpoint_plan.py \
  --config /local/model-config.json \
  --index /local/model.safetensors.index.json \
  --saliency /local/rtx5090-reap-saliency.json \
  --keep 48 \
  --out /local/alt48-plan.json
```

This is a **plan only**, not a weight exporter. A saliency ranking
from another task/repository must not be promoted directly to a
production GitHub-development model without calibration and independent
quality tests.

## Mikoshi hardware preflight

For a read-only check **inside the intended Linux runtime** (not on
the wrong Proxmox host or unrelated YodaBR server):

```sh
PYTHONPATH=. python3 tools/alt_host_preflight.py \
  --weight-bytes 27127753344 --sessions 2 \
  --kv-gib 1 --reserve-gib 8
```

The result reports current host memory and GPUs if `nvidia-smi` is
available, plus the conservative RAM-only capacity check. This does
**not** load a model, train, benchmark throughput, infer free VRAM,
or alter any host files.

**Pascal caveat:** the standard Hugging Face native MXFP4 backend
requires NVIDIA compute capability >= 7.5; P106-100 is Pascal
SM 6.1. Plan CPU-first inference/offload using a compatible
backend and validate BitWeave experimentally, never advertise that
native HF MXFP4 kernels run on these devices.

An SSH port being reachable from the JediControl notebook is not
proof that we can execute commands on Mikoshi in this session:
the subsequent remote command was blocked by the tool security layer.
No fresh Mikoshi GPU/VM200 resource measurement is claimed here.
