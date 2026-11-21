"""Offline, synthetic two-agent demo; does not open a private Noosphere vault."""
from cyclondex.memory import NoosphereReader
from cyclondex.twins import TwinMemory


def fake_call(name, args):
    assert name == "memory_source_grounded_context"
    return {
        "mode": "source_grounded_only",
        "status": "source_matched_only", "truncated": False,
        "facts": [{"fact_key": "demo.ci",
                   "value": "Run tests before review",
                   "source_id": "example-only"}],
    }


def main():
    reader = NoosphereReader(fake_call, {"project:demo"})
    twins = TwinMemory(reader)
    twins.send_peer("A", "B", "Review the failing test summary")
    for name in ("A", "B"):
        result = twins.prepare(name, "How did we solve CI issues?",
                               "project:demo")
        print(name, result["memory_status"], result["memory_tool_data"])
    print("Diagnostics:", twins.diagnostics())


if __name__ == "__main__":
    main()
