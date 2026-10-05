"""NumPy centroid and exemplar classifiers with explicit rejection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Prediction:
    label: str | None
    confidence: float
    accepted: bool
    candidate: str | None = None
    distance: float = 0.0
    reason: str = ""


class CentroidClassifier:
    def __init__(self, labels: list[str], centroids: np.ndarray, threshold: float, confidence_threshold: float = 0.15):
        self.labels = labels
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self.threshold = float(threshold)
        self.confidence_threshold = float(confidence_threshold)
        if self.centroids.shape != (len(labels), 63):
            raise ValueError("Model centroids must have shape (class_count, 63)")

    def predict(self, features: np.ndarray) -> Prediction:
        x = np.asarray(features, dtype=np.float32).reshape(63)
        distances = np.linalg.norm(self.centroids - x[None, :], axis=1)
        order = np.argsort(distances)
        best = int(order[0])
        # Distance-based score calibrated by between-centroid separation.
        if len(order) > 1:
            gap = float(distances[order[1]] - distances[best])
            confidence = float(np.clip(gap / max(float(distances[order[1]]), 1e-6), 0, 1))
        else:
            confidence = 1.0
        accepted = float(distances[best]) < self.threshold and confidence >= self.confidence_threshold
        return Prediction(self.labels[best] if accepted else None, confidence, accepted)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump({"labels": self.labels, "centroids": self.centroids.tolist(),
                       "threshold": self.threshold, "confidence_threshold": self.confidence_threshold,
                       "format_version": 1}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CentroidClassifier":
        with Path(path).open(encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("format_version") == 2:
            return ExemplarClassifier(payload["labels"], payload["vectors"], payload["targets"], payload["threshold"], payload["confidence_threshold"])
        return cls(payload["labels"], np.asarray(payload["centroids"], dtype=np.float32), payload["threshold"],
                   payload.get("confidence_threshold", 0.15))


class ExemplarClassifier:
    """Small NumPy exemplar baseline, including unsupported examples.

    Match both horizontal reflections because public data lacks handedness.
    Keep vertical orientation and depth; never rotate the hand into a generic pose.
    """

    def __init__(self, labels, vectors, targets, threshold, confidence_threshold=0.05):
        self.labels = list(labels)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int32)
        self.threshold = float(threshold)
        self.confidence_threshold = float(confidence_threshold)
        if self.vectors.shape != (len(self.targets), 63) or not 0 < len(self.targets) <= 4096:
            raise ValueError("Expected 1–4096 finite 63-value exemplars")
        if not np.isfinite(self.vectors).all() or np.any(self.targets < 0) or np.any(self.targets > len(self.labels)):
            raise ValueError("Invalid exemplar values or label mapping")

    def predict(self, features):
        x = np.asarray(features, dtype=np.float32).reshape(63)
        if not np.isfinite(x).all():
            return Prediction(None, 0.0, False, reason="Invalid landmarks")
        mirrored = x.copy()
        mirrored[::3] *= -1
        distances = np.minimum(np.linalg.norm(self.vectors - x, axis=1), np.linalg.norm(self.vectors - mirrored, axis=1))
        per_class = np.array([distances[self.targets == i].min() if np.any(self.targets == i) else np.inf for i in range(len(self.labels) + 1)])
        order = np.argsort(per_class)
        best = int(order[0])
        distance = float(per_class[best])
        second = float(per_class[order[1]])
        confidence = float(np.clip((second - distance) / max(second, 1e-6), 0, 1)) if np.isfinite(second) else 1.0
        candidate = self.labels[best] if best < len(self.labels) else "UNSUPPORTED"
        reason = ("Unsupported shape" if best == len(self.labels) else
                  "Outside training examples" if distance > self.threshold else
                  "Ambiguous shape" if confidence < self.confidence_threshold else "")
        return Prediction(candidate if not reason else None, confidence, not reason, candidate, distance, reason)

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"format_version": 2, "labels": self.labels,
                                     "vectors": self.vectors.tolist(), "targets": self.targets.tolist(),
                                     "threshold": self.threshold, "confidence_threshold": self.confidence_threshold}), encoding="utf-8")
