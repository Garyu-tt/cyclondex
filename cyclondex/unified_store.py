"""Trusted, read-only gateway for the live unified-tag-fabric-v1 store.

Run in a backend principal; the model worker must NOT receive database access.
Only visibility='agents' documents are eligible for any result. This is not
a cryptographic proof of ACL: use separate OS accounts at deployment time.
"""
from pathlib import Path
import re
import sqlite3

SCHEMA = "cyclondex.unified-agent-shared.v1"
MAX_RESULTS = 12
MAX_EXCERPT = 1400


def _match_query(query):
    if not isinstance(query, str) or not 1 <= len(query) <= 1024:
        raise ValueError("invalid search query length")
    # Every token becomes a quoted FTS literal, never raw query syntax.
    terms = re.findall(r"\w{2,}", query.casefold(), re.UNICODE)[:8]
    if not terms:
        return None
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


class SharedMemoryStore:
    """No free-form SQL, no caller-provided visibility, no owner reads."""

    def __init__(self, database):
        self.database = Path(database).expanduser().resolve(strict=True)
        if not self.database.is_file():
            raise ValueError("database must be a regular file")
        self.uri = self.database.as_uri() + "?mode=ro"

    def search(self, query, limit=6):
        if type(limit) is not int or not 1 <= limit <= MAX_RESULTS:
            raise ValueError("invalid result limit")
        search = _match_query(query)
        if search is None:
            return {"schema": SCHEMA, "visibility": "agents",
                    "status": "no_matches", "records": []}
        with sqlite3.connect(self.uri, uri=True, timeout=3) as db:
            db.execute("PRAGMA query_only = ON")
            db.execute("PRAGMA busy_timeout = 2500")
            # Exact visibility filter lives in the trusted data process.
            rows = db.execute("""
                SELECT d.id, substr(d.title,1,200), d.kind, d.sha256,
                       d.created_utc,
                       substr(snippet(unified_fts, 1, '', '', ' … ', 20),1,?)
                FROM unified_fts
                JOIN unified_docs AS d ON d.id = unified_fts.rowid
                WHERE unified_fts MATCH ? AND d.visibility = 'agents'
                ORDER BY bm25(unified_fts) LIMIT ?
            """, (MAX_EXCERPT, search, limit)).fetchall()
        records = [
            {"id": str(r[0]), "title": r[1], "kind": r[2],
             "sha256": r[3], "created_utc": r[4], "excerpt": r[5],
             "visibility": "agents", "evidence_status": "unreviewed"}
            for r in rows
        ]
        return {"schema": SCHEMA, "visibility": "agents",
                "status": "agent_shared_unverified" if records else "no_matches",
                "records": records}
