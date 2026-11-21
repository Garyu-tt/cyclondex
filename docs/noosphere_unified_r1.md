# CYCLON ↔ unified-tag-fabric-v1 (R1 trust boundary)

## Verified live host metadata, 21 Nov 2026

Read-only schema checks on the owner-authorized YodaBR Linux host:

- `unified_docs` has `id`, `title`, `kind`, `sha256`,
  `created_utc`, `visibility`, `object_path` and other columns.
- Visibility **on disk** is `owner` (959 rows) versus `agents` (4).
  The external health API calls these `owner_only` and `agent_shared`.
- FTS5 table `unified_fts` has title/content/tags/source; rows join
  to `unified_docs` through rowid. Raw private text was not queried.

## Implemented

`cyclondex.unified_store.SharedMemoryStore` is an experimental,
read-only, fixed-SQL provider that selects only
`unified_docs.visibility = 'agents'`, retrieves small excerpts from SHA-256-verified UTF-8 objects, and
never exposes `owner` rows or caller-supplied SQL. It does not return
object paths, private blob contents or credentials. Results are tagged
as **unreviewed**, not source-proven truth.

`UnifiedSharedReader` is the CYCLON-facing consumer. It accepts
only an explicitly authorized `unified.agent_shared.search` callback,
rejects any owner-visibility record, caps result size, and refuses
historical validity claims. It works with `TwinMemory`.

In an isolated test or trusted host fixture, one may wire:

```python
backend = SharedMemoryStore("/path/to/knowledge.sqlite3", "/path/to/objects")
def authorized_tool(action, args):
    if action != "unified.agent_shared.search":
        raise PermissionError(action)
    return backend.search(**args)

twins = TwinMemory(UnifiedSharedReader(authorized_tool))
result = twins.prepare("A", "Find previous CI fixes", "agent_shared")
```

## Production security gate NOT satisfied

This code is **not sufficient OS isolation** if the model process runs
as the same Unix account that can open the database. In production, the
trusted provider must run as a separate principal; CYCLON workers must
not have read access to the original SQLite DB, WAL or object storage.

The fixed-SQL visibility check is a defense-in-depth filter. It is not
a replacement for an authenticated provider capability, audit trail,
rate limit, tamper-proof principal and independent negative ACL tests.
Do not expose this provider on a public HTTP port; use approved internal
capabilities only.

Live read-only inspection found the FTS5 index is contentless. Exact
index matches require opening the content-addressed objects after the
visibility check. The gateway requires an operator-selected object root;
it validates the relative hash path, file size, SHA-256 and rejects
symlinks or changes during reading. The four currently shared objects
passed hash and size checks without exposing their contents.

No live Noosphere data is committed or used for training, and the
adapter does not change the Noosphere backend. Source hash alone does
not prove factual truth. No CLI is provided to export shared contents
into model checkpoints or training datasets.

## Acceptance gates for deployment

1. Run verified, least-privilege backend with an agent-shared-only
   capability and controlled local IPC.
2. Use a different OS account for model workers with no direct access
   to Noosphere DB, keys, archives, and unshared documents.
3. Run adversarial queries and mixed-ACL fixtures, verify only four
   current shared rows may be returned, without logging their contents.
4. Run one authorized read-only end-to-end retrieval.
5. Train retrieval policy only on independently verified and
   permitted, privacy-safe work traces, not private Noosphere text.
