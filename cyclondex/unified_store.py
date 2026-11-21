"""Trusted, read-only gateway for live unified-tag-fabric-v1 Noosphere.

This backend is intended to run in a separate, host-authorized process.
It only accesses rows marked visibility='agents' and verifies their objects
against the Noosphere content-addressed SHA-256 path and manifest.
"""
from pathlib import Path
import hashlib
import os
import re
import sqlite3
import stat

SCHEMA = "cyclondex.unified-agent-shared.v1"
MAX_RESULTS = 12
MAX_EXCERPT = 1400
MAX_OBJECT = 64 * 1024
HEX64 = re.compile(r"^[a-f0-9]{64}$")


def _match_query(query):
    if not isinstance(query, str) or not 1 <= len(query) <= 1024:
        raise ValueError("invalid search query length")
    terms = re.findall(r"\w{2,}", query.casefold(), re.UNICODE)[:8]
    if not terms:
        return None
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


class IntegrityError(RuntimeError):
    """Fail closed on untrusted path, size, SHA or changed object."""


class SharedMemoryStore:
    """Fixed SQL plus verified object IO; model worker gets neither DB nor path."""

    def __init__(self, database, object_root):
        db = Path(database).expanduser()
        root = Path(object_root).expanduser()
        if db.is_symlink() or not db.is_file() or root.is_symlink() or not root.is_dir():
            raise ValueError("trusted database and object root must exist")
        self.database = db.resolve(strict=True)
        self.object_root = root.resolve(strict=True)
        self.uri = self.database.as_uri() + "?mode=ro"

    def _read_object(self, path, expected_sha, expected_size):
        if (not isinstance(path, str) or not isinstance(expected_sha, str)
                or not HEX64.fullmatch(expected_sha) or type(expected_size) is not int
                or not 0 <= expected_size <= MAX_OBJECT):
            raise IntegrityError("invalid object identity")
        candidate = Path(path)
        if not candidate.is_absolute():
            raise IntegrityError("absolute object path required")
        try:
            rel = candidate.relative_to(self.object_root)
        except ValueError as exc:
            raise IntegrityError("object outside permitted root") from exc
        if (len(rel.parts) != 2 or rel.parts[0] != expected_sha[:2]
                or rel.parts[1] != expected_sha + ".txt"
                or candidate.is_symlink() or candidate.parent.is_symlink()):
            raise IntegrityError("unsafe object location")
        try:
            fd = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                before = os.fstat(fd)
                if (not stat.S_ISREG(before.st_mode)
                        or before.st_size != expected_size):
                    raise IntegrityError("object size mismatch")
                with os.fdopen(fd, "rb", closefd=False) as handle:
                    data = handle.read(MAX_OBJECT + 1)
                after = os.fstat(fd)
                fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
                if fields(before) != fields(after):
                    raise IntegrityError("object modified during reading")
            finally:
                os.close(fd)
        except OSError as exc:
            raise IntegrityError("object unavailable") from exc
        if (len(data) != expected_size
                or hashlib.sha256(data).hexdigest() != expected_sha):
            raise IntegrityError("object SHA-256 mismatch")
        try:
            return data.decode("utf-8")[:MAX_EXCERPT]
        except UnicodeDecodeError as exc:
            raise IntegrityError("object not UTF-8") from exc

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
            # Contentless FTS5 provides matching IDs only, never text.
            # Visibility filter is applied before fetching any object.
            rows = db.execute("""
                SELECT d.id, substr(d.title,1,200), d.kind, d.sha256,
                       d.created_utc, d.object_path, d.bytes
                FROM unified_fts
                JOIN unified_docs AS d ON d.id = unified_fts.rowid
                WHERE unified_fts MATCH ? AND d.visibility = 'agents'
                ORDER BY bm25(unified_fts) LIMIT ?
            """, (search, limit)).fetchall()
        records = [
            {"id": str(r[0]), "title": r[1], "kind": r[2],
             "sha256": r[3], "created_utc": r[4],
             "excerpt": self._read_object(r[5], r[3], r[6]),
             "visibility": "agents", "evidence_status": "unreviewed"}
            for r in rows
        ]
        return {"schema": SCHEMA, "visibility": "agents",
                "status": "agent_shared_unverified" if records else "no_matches",
                "records": records}
