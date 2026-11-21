"""Tiny online policy: learn when retrieval helps, not model knowledge.

Weights stay in RAM; the host must independently authenticate reviewer identity.
"""
import hashlib
import math
import re

TOKENS = re.compile(r"[\w.-]{3,}", re.UNICODE)


def features(task: str) -> tuple[str, ...]:
    if not isinstance(task, str) or len(task) > 4096:
        raise ValueError("invalid task")
    # Hash words to avoid retaining plaintext task strings inside weight keys.
    unique = set(TOKENS.findall(task.casefold()))
    hashed = sorted(hashlib.sha256(word.encode()).hexdigest()[:12]
                    for word in unique)
    return ("bias", *hashed[:64])


class RetrievalPolicy:
    def __init__(self, learning_rate=.3):
        if not 0 < learning_rate <= 1:
            raise ValueError("invalid learning rate")
        self.weights = {"bias": 0.25}
        self.learning_rate = learning_rate
        self.verified_updates = 0

    def probability(self, task: str) -> float:
        score = sum(self.weights.get(key, 0.0) for key in features(task))
        return 1.0 / (1.0 + math.exp(-max(-20., min(20., score))))

    def should_search(self, task: str) -> bool:
        return self.probability(task) >= .5

    def learn(self, task: str, useful: bool, *,
              independently_reviewed: bool = False) -> float:
        """No self-reported rewards. Authenticated host verifies review."""
        if not independently_reviewed or type(useful) is not bool:
            raise PermissionError("independent signed-off review required")
        xs = features(task)
        prediction = self.probability(task)
        error = float(useful) - prediction
        step = self.learning_rate * error / (len(xs) ** .5)
        for key in xs:
            self.weights[key] = max(-8., min(8., self.weights.get(key, 0.) + step))
        self.verified_updates += 1
        return prediction
