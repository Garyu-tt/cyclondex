"""Two separately scoped CYCLON agent sessions using one memory client."""
from collections import deque
from dataclasses import dataclass, field
from .memory import MemoryUnavailable, NoosphereReader, Recall
from .policy import RetrievalPolicy


@dataclass
class Session:
    name: str
    history: deque = field(default_factory=lambda: deque(maxlen=12))
    inbox: deque = field(default_factory=lambda: deque(maxlen=8))
    calls: int = 0


class TwinMemory:
    def __init__(self, reader: NoosphereReader, policy=None):
        self.reader = reader
        self.policy = policy or RetrievalPolicy()
        self.agents = {n: Session(n) for n in ("A", "B")}

    def prepare(self, agent: str, task: str, scope: str,
                *, max_chars: int = 3000, as_of=None) -> dict:
        if agent not in self.agents:
            raise ValueError("unknown agent")
        if not isinstance(task, str) or not 1 <= len(task) <= 4096:
            raise ValueError("invalid task")
        s = self.agents[agent]
        if scope not in self.reader.allowed_scopes:
            raise PermissionError("scope not allowed")
        should = self.policy.should_search(task)
        if not should:
            recall = Recall(scope, "policy_skipped")
        else:
            try:
                recall = self.reader.recall(task, scope, max_chars=max_chars,
                                            as_of=as_of)
                s.calls += 1
            except MemoryUnavailable:
                recall = Recall(scope, "memory_unavailable")
        # Deliberately do not store source text, prompts or memory facts in logs.
        s.history.append({"task_length": len(task), "status": recall.status,
                          "facts": len(recall.facts)})
        incoming = list(s.inbox)
        s.inbox.clear()
        return {
            "agent": agent, "task": task, "scope": scope,
            "memory_tool_data": recall.tool_data(),
            "peer_messages_untrusted": incoming,
            "memory_status": recall.status,
        }

    def send_peer(self, sender: str, target: str, message: str) -> None:
        if (sender not in self.agents or target not in self.agents or
                sender == target or not isinstance(message, str)
                or len(message) > 500):
            raise ValueError("invalid bounded peer message")
        self.agents[target].inbox.append({
            "sender": sender, "message_untrusted": message,
        })

    def diagnostics(self):
        return {name: {"retrieval_calls": s.calls,
                       "events": len(s.history), "inbox": len(s.inbox)}
                for name, s in self.agents.items()}
