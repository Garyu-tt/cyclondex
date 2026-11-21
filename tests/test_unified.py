"""Adversarial ACL, content identity, and twin-session tests."""
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from cyclondex.memory import MemoryUnavailable
from cyclondex.twins import TwinMemory
from cyclondex.unified_reader import UnifiedSharedReader
from cyclondex.unified_store import SCHEMA, IntegrityError, SharedMemoryStore
from tools.noosphere_shared_probe import probe
from tools.noosphere_twins_probe import probe_twins


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.db_path = root / "knowledge.sqlite3"
        self.object_root = root / "objects"
        self.object_root.mkdir()
        self.objects = {}
        with sqlite3.connect(self.db_path) as c:
            c.execute("""CREATE TABLE unified_docs (
                id INTEGER PRIMARY KEY, title TEXT, kind TEXT, sha256 TEXT,
                created_utc TEXT, visibility TEXT, object_path TEXT, bytes INTEGER)""")
            c.execute("CREATE VIRTUAL TABLE unified_fts USING fts5(title,content,tags,source,content='')")
            samples = [
                (1, "owner secret", "PRIVATE_CANARY_DO_NOT_DISCLOSE fix", "owner"),
                (2, "shared ci fix", "Run checks before merge", "agents"),
                (3, "shared project plan", "Plan issues and review", "agents"),
                (4, "owner fix", "Owner-only fix secret_42", "owner"),
            ]
            for doc_id, title, data, visibility in samples:
                encoded = data.encode("utf-8")
                sha = hashlib.sha256(encoded).hexdigest()
                target = self.object_root / sha[:2] / (sha + ".txt")
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(encoded)
                self.objects[doc_id] = target
                c.execute("""INSERT INTO unified_docs
                    (id,title,kind,sha256,created_utc,visibility,object_path,bytes)
                    VALUES (?,?,'note',?,'2026-11-21T00:00:00Z',?,?,?)""",
                    (doc_id,title,sha,visibility,str(target),len(encoded)))
                c.execute("""INSERT INTO unified_fts
                    (rowid,title,content,tags,source) VALUES (?,?,?,?,?)""",
                    (doc_id,title,data,"",""))
        self.store = SharedMemoryStore(self.db_path, self.object_root)

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
        result = self.store.search("fix")
        self.assertEqual([r["id"] for r in result["records"]], ["2"])
        self.assertEqual(result["records"][0]["excerpt"], "Run checks before merge")
        self.assertEqual(result["schema"], SCHEMA)

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

    def test_limits_and_scope(self):
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

    def test_probe_never_logs_memory_content(self):
        result = probe(self.store, "fix")
        self.assertEqual(result["verified_records"], 1)
        self.assertFalse(result["contents_logged"])
        self.assertNotIn("Run checks before merge", str(result))
        self.assertNotIn("PRIVATE_CANARY", str(result))

    def test_two_agents_share_only_authorized_memory(self):
        summary = probe_twins(self.store, "fix")
        self.assertEqual(summary["agents"]["A"]["visible_records"], 1)
        self.assertEqual(summary["agents"]["B"]["visible_records"], 1)
        self.assertEqual(summary["agents"]["B"]["peer_messages"], 1)
        self.assertFalse(summary["contents_logged"])
        self.assertNotIn("Run checks", str(summary))
        self.assertNotIn("PRIVATE_CANARY", str(summary))

    def test_read_only(self):
        original = self.store.search("plan")
        with sqlite3.connect(self.store.database) as db:
            self.assertEqual(db.execute(
                "SELECT COUNT(*) FROM unified_docs").fetchone()[0], 4)
        self.assertEqual(self.store.search("plan"), original)

    def test_refuse_repointed_object_to_owner(self):
        with sqlite3.connect(self.db_path) as db:
            db.execute("UPDATE unified_docs SET object_path = ? WHERE id = 2",
                       (str(self.objects[1]),))
        with self.assertRaises(IntegrityError):
            self.store.search("fix")

    def test_refuse_corrupted_shared_object(self):
        self.objects[2].write_text("corrupted object", encoding="utf-8")
        with self.assertRaises(IntegrityError):
            self.store.search("fix")

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
