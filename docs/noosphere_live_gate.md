# CYCLON ↔ live Noosphere: authorization gate

Read-only operator diagnostics on 2026-11-21 reported a running Noosphere
backend version 0.4.0 with format `unified-tag-fabric-v1`, based on
SQLite FTS5 and embeddings. Aggregate counts: 963 memories, 959
`owner_only`, 4 `agent_shared`.

**It is not the same API as** public `noosphere-lite-mcp`, which uses an
encrypted NLV1 event journal and has an explicit
`memory_source_grounded_context` retrieval tool. The R0
`NoosphereReader` intentionally requires the latter response contract
and fails closed when a different tool is supplied.

The visible YodaBR API has `memory_search(query, limit)`, but it has no
client-supplied ACL selector. A client-side filter is not sufficient:
private data could already have crossed the model boundary before being
filtered. Therefore **do not** connect this broad search tool to
autonomous CYCLON until an authenticated backend supports an
`agent_shared`-only read capability.

A Noosphere 'scope' or project tag is never an access-control mechanism.

## Required next work

- Add backend-owned ACL enforcement and capability negotiation, with
  negative tests against mixed owner-only/shared memories.
- Add source identity, temporal validity and evidence status to results.
- If authorized, connect only a strictly scoped capability to the
  CYCLON host. Protect owner-only records, embeddings, ciphertext,
  service tokens, file paths and developer logs.
- Verify one read-only shared-memory retrieval before any weight training.
- Keep audit trails of which memory IDs were retrieved and for what
  authorized purpose, without logging private contents.

Tracking: [CyclonDex issue #3](https://github.com/Garyu-tt/cyclondex/issues/3).
