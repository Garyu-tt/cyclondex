# CYCLON Memory Native R0

Status: research prototype; **no private Noosphere deployment connected**.

## Goal

CYCLON A and B (eventual two sessions on one 80B-class backbone) use a
small task context. Noosphere is authoritative long-term state, not a big
system prompt. This R0 does **not** train an LLM or modify its weight tensors.

## Existing Noosphere contract

The `noosphere-lite-mcp` project exposes
`memory_source_grounded_context(query, scope, as_of, max_chars)`.
It returns `mode=source_grounded_only`, `facts`, `status`,
conflicts, and truncation. Source-matching is **not independent truth
verification**. Legacy notes are excluded by default.

The actual authenticated host must supply a callback to
`NoosphereReader(tool_call, allowed_scopes=...)`. The adapter does **not**
contact MCP, SSH, HTTP, databases or read tokens itself; no hardcoded
addresses, credentials or vault paths. A scope is only an **organizational
filter** in Noosphere, not an ACL: a real host-enforced permission gate is
mandatory before enabling production memory.

## Learning loops

1. **Inference read**: call evidence-only Noosphere retrieval on demand,
   preserve source fields as untrusted tool data, keep whole facts within
   a small memory budget. Conflicting/stale/insufficient results never
   become facts. No prompt instruction is sourced from retrieved text.
2. **Policy learning**: `RetrievalPolicy` is a small online hashed-feature
   logistic gate that learns *when* retrieval was useful from externally
   approved outcomes. This is real SGD on router weights, **not** fine-tuning
   the 80B backbone. Reviewer identity and verdict provenance are validated
   by the host. No private plaintext training corpus is retained.
3. **Optional future consolidation**: train licensed adapters from
   independently reviewed trajectories, without making Noosphere a
   self-reinforcing hallucination source. This is not implemented.

## Agent isolation

Two sessions have separate short histories and bounded peer inboxes.
Peer messages are untrusted data and are never automatically copied
to long-term memory. There is no background token generation, so an idle
agent uses no model inference cycles.

## Test/usage

```sh
python3 -m unittest discover -s tests -v
PYTHONPATH=. python3 tools/memory_demo.py
```

The demo uses **synthetic facts only**; it does not access production
Noosphere. CI uses only Python stdlib, requires no secrets, and tests:
contract validation, missing and conflicting evidence, provenance
boundary, historical timestamp validation, budgets, hostile memory
text, agent isolation, outage handling, and feedback-gated learning.

## Next implementation gates

- Host-side authenticated Noosphere MCP transport with rate limits,
  audit, capability discovery and real ACL; keep all keys outside GitHub.
- Add retrieval score/evidence benchmarks and supervised datasets
  without raw private content. Test held-out projects and time periods.
- Capture feedback only from independent reviewer or CI, not from either
  CYCLON twin's own unchecked claims.
- Integrate inference backend separately and measure Mikoshi memory,
  CPU/GPU usage and time/token before trying 80B.
