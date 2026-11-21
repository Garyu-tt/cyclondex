"""Read-only, content-silent two-agent CYCLON ↔ live Noosphere probe.

Designed for an owner-operated temporary test, not a production service.
Never passes owner records or writes to the underlying knowledge database.
"""
import argparse
import json

from cyclondex.twins import TwinMemory
from cyclondex.unified_reader import UnifiedSharedReader
from cyclondex.unified_store import SharedMemoryStore


def probe_twins(store, query):
    def trusted_gateway(action, args):
        if action != "unified.agent_shared.search":
            raise PermissionError("unsupported action")
        return store.search(**args)

    twins = TwinMemory(UnifiedSharedReader(trusted_gateway))
    twins.send_peer("A", "B", "A requested shared-memory review.")
    summary = {}
    for agent in ("A", "B"):
        context = twins.prepare(agent, query, "agent_shared", max_chars=3000)
        # Parse solely to count; never log, return or persist source text.
        memory = json.loads(context["memory_tool_data"])
        summary[agent] = {
            "status": context["memory_status"],
            "visible_records": len(memory["untrusted_facts"]),
            "peer_messages": len(context["peer_messages_untrusted"]),
        }
    return {"agents": summary, "source": "Noosphere agent_shared only",
            "contents_logged": False, "database_modified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    print(json.dumps(probe_twins(SharedMemoryStore(args.db, args.objects),
                                  args.query), sort_keys=True))


if __name__ == "__main__":
    main()
