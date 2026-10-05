"""Bounded prediction smoothing and deliberate text editing."""

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class StablePrediction:
    label: str | None
    confidence: float


class PredictionSmoother:
    def __init__(self, capacity: int = 7, required: int = 4):
        self.history: deque[tuple[str | None, float]] = deque(maxlen=capacity)
        self.required = required

    def update(self, label: str | None, confidence: float) -> StablePrediction:
        self.history.append((label, confidence))
        valid = [(name, score) for name, score in self.history if name is not None]
        if not valid:
            return StablePrediction(None, confidence)
        counts = {}
        for name, _ in valid:
            counts[name] = counts.get(name, 0) + 1
        label, count = max(counts.items(), key=lambda item: item[1])
        if count < self.required:
            return StablePrediction(None, confidence)
        scores = [score for name, score in valid if name == label]
        return StablePrediction(label, sum(scores) / len(scores))

    def reset(self) -> None:
        self.history.clear()


class TextBuffer:
    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self._letters: deque[str] = deque(maxlen=capacity)

    @property
    def text(self) -> str:
        return "".join(self._letters)

    def accept(self, letter: str) -> None:
        if len(letter) != 1 or not letter.isalpha():
            raise ValueError("Only single letters can be accepted")
        self._letters.append(letter.upper())

    def space(self) -> None:
        if self._letters and self._letters[-1] != " ":
            self._letters.append(" ")

    def backspace(self) -> None:
        if self._letters:
            self._letters.pop()

    def clear(self) -> None:
        self._letters.clear()
