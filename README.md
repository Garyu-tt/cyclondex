# CyclonDex · CYCLON Twins

**One repository for the CYCLON model and the software to operate it.**

CyclonDex aims to provide a lightweight, Codex-like workbench with GitHub-first
task graphs, a minimal code editor, local/remote agent adapters, and two logical
CYCLON agents that can exchange thoughts through a bounded recurrent loop.

> **Status:** research scaffold / MVP. No trained 80B model, no proven 80B
> dual-agent deployment, no claim of self-sufficient GitHub software development.

## Planned runtime

- **CYCLON A + CYCLON B:** two independent agent sessions/contexts with separate
  memories and optional adapters, sharing **one** 80B-class pretrained model.
  Two 80B agents do **not** imply two separately trained 80B parameter tensors.
- **Recurrent cycle:** model output can become a new input; bounded by token,
  latency, step and power budgets. Do not keep expensive inference spinning idle.
- **Collaboration:** async peer messages through versioned, bounded queues,
  causal event IDs, separate worktrees and permissions.
- **Mikoshi:** CPU-first inference/offload, optionally supported BitWare/BitWeave
  kernels **after** correctness and end-to-end benchmarking. GPU VRAM is not
  magically additive; production runtime must deliberately partition weights.
- **GitHub Academy:** training and evaluation from real project lifecycles,
  reconstructing issues, code, tests, reviews and releases **at the right time**.
- **Workbench:** project status, task graph, agent conversations, simple code
  editor, diagnostics and resource meter. GitHub remains task authority.

## Repository layout

```
cyclondex/                Python research runtime and service
tests/                    Unit and API checks
tools/                    Offline diagnostics and GitHub curriculum
web/                      Minimal workbench
docs/                     Architecture and Mikoshi deployment gates
```

## Research prototype

```sh
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 tools/cyclon_demo.py
python3 tools/cyclon_github_curriculum.py --repo karpathy/nanochat --limit 20
```

This initial prototype performs real, tiny CPU-based recurrent byte-level
training with shared weights, two independently adapted states and explicit
feedback. It is **not** a language model and cannot yet develop code.

The GitHub curriculum collector is read-only, limited and stores source URLs.
Do not execute untrusted project code on the host. Use sandboxes, review
licenses, keep API tokens out of datasets, and never give training processes
production SSH keys.

## Current host diagnostic (21 Nov 2026)

Verified over JediControl notebook SSH: `mikoshi`, Proxmox VE 9.2.2,
2×Xeon E5-2670 v2 (40 hardware threads), ~125 GiB RAM, VM200
`ubuntu-server` running with 96 GiB assigned. Host GPU driver tooling was
not present. This is a snapshot, not a promise of available compute.
Verify VM200 GPU passthrough and guest headroom before executing ML workloads.

## Source projects

- [Mikoshi](https://github.com/Garyu-tt/mikoshi), current neural substrate.
- [BitWare](https://github.com/Garyu-tt/BitWare), research bit-packed kernels.
- [AdamantiumToolBox](https://github.com/Garyu-tt/AdamantiumToolBox), optional
  capability-based orchestration.

Project policy: this repo is the single source of truth for CYCLON/CyclonDex.
Changes to external projects use explicit adapters or tracked pull requests.

License for new CyclonDex contributions: **TBD**. Reused code retains its
original copyright/license, including GPL requirements where applicable.

## Noosphere Memory Native R0

[Memory R0 architecture](docs/noosphere_memory_r0.md) · [Tracking issue #1](https://github.com/Garyu-tt/cyclondex/issues/1)

A *read-only*, source-grounded Noosphere tool adapter now supports two
independent CYCLON memory sessions, bounded untrusted memory retrieval and a
small retrieval policy that learns from independent review. This is a tested
research module, not an 80B model or an authenticated production integration.

```sh
python3 -m unittest discover -s tests -v
PYTHONPATH=. python3 tools/memory_demo.py
```

The demo uses synthetic source records. Never configure private vault credentials
in public CI or GitHub. A real Noosphere MCP transport and host ACL are still
required before accessing production memory.
