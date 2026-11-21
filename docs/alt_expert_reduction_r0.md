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
