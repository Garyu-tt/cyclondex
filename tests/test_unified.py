"""Adversarial ACL tests for an agent-shared-only read surface."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from cyclondex.memory import MemoryUnavailable
from cyclondex.twins import TwinMemory
from cyclondex.unified_reader import UnifiedSharedReader
from cyclondex.unified_store import SCHEMA, SharedMemoryStore


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        db_path = Path(self.tmp.name) / "knowledge.sqlite3"
        with sqlite3.connect(db_path) as c:
            c.execute("""CREATE TABLE unified_docs (
                id INTEGER PRIMARY KEY, title TEXT, kind TEXT,
                sha256 TEXT, created_utc TEXT, visibility TEXT)""")
            c.execute("CREATE VIRTUAL TABLE unified_fts USING fts5(title,content,tags,source)")
            samples = [
                (1, "owner secret", "PRIVATE_CANARY_DO_NOT_DISCLOSE fix",
                 "owner"),
                (2, "shared ci fix", "Run checks before merge",
                 "agents"),
                (3, "shared project plan", "Plan issues and review",
                 "agents"),
                (4, "owner fix", "Owner-only fix secret_42",
                 "owner"),
            ]
            for doc_id, title, content, visibility in samples:
                c.execute("""INSERT INTO unified_docs
                    (id,title,kind,sha256,created_utc,visibility)
                    VALUES (?,?,'note','sha','2026-11-21T00:00:00Z',?)""",
                          (doc_id,title,visibility))
                c.execute("""INSERT INTO unified_fts
                    (rowid,title,content,tags,source) VALUES (?,?,?,?,?)""",
                          (doc_id,title,content,"",""))
        self.store = SharedMemoryStore(db_path)

    def gateway(self, action, args):
        self.assertEqual(action, "unified.agent_shared.search")
        return self.store.search(**args)

    def test_visibility_sql_enforcement(self):
        for query in ("fix", "PRIVATE_CANARY_DO_NOT_DISCLOSE",
                      "secret_42", 'fix" OR secret*'):
            response = self.store.search(query)
            self.assertEqual(response["visibility"], "agents")
            self.assertTrue(all(r["visibility"] == "agents"
                                for r in response["records"]))
            self.assertNotIn("PRIVATE_CANARY", str(response))
            self.assertNotIn("secret_42", str(response))
            self.assertNotIn("owner secret", str(response))
        response = self.store.search("fix")
        self.assertEqual([r["id"] for r in response["records"]], ["2"])
        self.assertEqual(response["schema"], SCHEMA)

    def test_acl_not_client_filter(self):
        def malicious(action, args):
            return {"schema": SCHEMA, "visibility": "agents",
                    "status": "agent_shared_unverified",
                    "records": [{"id": "1", "visibility": "owner",
                                 "evidence_status": "unreviewed",
                                 "excerpt": "private"}]}
        with self.assertRaises(MemoryUnavailable):
            UnifiedSharedReader(malicious).recall("fix", "agent_shared")

    def test_unified_reader_and_twins(self):
        reader = UnifiedSharedReader(self.gateway)
        twins = TwinMemory(reader)
        twins.send_peer("A", "B", "Review the CI changes")
        a = twins.prepare("A", "fix", "agent_shared")
        b = twins.prepare("B", "plan", "agent_shared")
        self.assertEqual(a["memory_status"], "agent_shared_unverified")
        self.assertEqual(b["memory_status"], "agent_shared_unverified")
        memory = json.loads(a["memory_tool_data"])
        self.assertEqual(memory["trust"],
                         "unreviewed_agent_shared_not_verified_truth")
        self.assertEqual(memory["untrusted_facts"][0]["id"], "2")
        self.assertEqual(b["peer_messages_untrusted"][0]["sender"], "A")
        self.assertNotIn("PRIVATE_CANARY_DO_NOT_DISCLOSE", str((a,b)))

    def test_limits_and_time_and_scope(self):
        with self.assertRaises(PermissionError):
            UnifiedSharedReader(self.gateway).recall("fix", "owner")
        with self.assertRaises(ValueError):
            UnifiedSharedReader(self.gateway).recall(
                "fix", "agent_shared", as_of="2026-01-01T00:00:00Z")
        with self.assertRaises(ValueError):
            self.store.search("fix", limit=100)
        with self.assertRaises(ValueError):
            self.store.search("fix", limit=True)
        with self.assertRaises(ValueError):
            self.store.search("x" * 1025)
        self.assertEqual(self.store.search("a")["status"], "no_matches")

    def test_read_only(self):
        original = self.store.search("plan")
        with sqlite3.connect(self.store.database) as db:
            self.assertEqual(db.execute(
                "SELECT COUNT(*) FROM unified_docs").fetchone()[0], 4)
        self.assertEqual(self.store.search("plan"), original)

    def test_provider_missing_or_bad_contract(self):
        def offline(action, args):
            raise RuntimeError("offline")
        with self.assertRaises(MemoryUnavailable):
            UnifiedSharedReader(offline).recall("fix", "agent_shared")
        def mismatched(action, args):
            return {"schema": "unknown", "records": []}
        with self.assertRaises(MemoryUnavailable):
            UnifiedSharedReader(mismatched).recall("fix", "agent_shared")


if __name__ == "__main__":
    unittest.main()
