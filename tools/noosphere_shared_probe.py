"""Owner-operated read-only Noosphere shared-object integrity probe.

Logs aggregate counts only, never record text, titles, document IDs or paths.
"""
import argparse
import json

from cyclondex.unified_store import SharedMemoryStore


def probe(store, query):
    result = store.search(query, limit=12)
    return {
        "schema": result["schema"],
        "visibility": result["visibility"],
        "status": result["status"],
        "verified_records": len(result["records"]),
        "contents_logged": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    store = SharedMemoryStore(args.db, args.objects)
    print(json.dumps(probe(store, args.query), sort_keys=True))


if __name__ == "__main__":
    main()
