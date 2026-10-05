"""Dataset loading, group splits, and compact classifier fitting."""

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from asl_app.classifier import CentroidClassifier, ExemplarClassifier
from asl_app.labels import SUPPORTED_LABELS, UNKNOWN_LABEL

SEED = 1729


def load_rows(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            # U was the collector's original keyboard-only rejection label.
            label = UNKNOWN_LABEL if row["label"] == "U" else row["label"]
            if label not in (*SUPPORTED_LABELS, UNKNOWN_LABEL):
                raise ValueError(f"Unsupported dataset label: {label}")
            vector = np.array([float(row[f"f{i}"]) for i in range(63)], dtype=np.float32)
            if not np.isfinite(vector).all() or not row["session_id"]:
                raise ValueError("Dataset rows require finite features and a nonempty group ID")
            rows.append((label, row["session_id"], vector))
    return rows


def group_split(rows, seed=SEED):
    by_session = defaultdict(list)
    for row in rows:
        by_session[row[1]].append(row)
    sessions = sorted(by_session)
    if len(sessions) < 2:
        raise ValueError("Need at least two distinct recording sessions for a leakage-resistant split")
    random.Random(seed).shuffle(sessions)
    test_sessions = set(sessions[:max(1, round(len(sessions) * 0.25))])
    train = [row for row in rows if row[1] not in test_sessions]
    test = [row for row in rows if row[1] in test_sessions]
    return train, test


def fit_centroid(rows):
    unknown = [row[2] for row in rows if row[0] == UNKNOWN_LABEL]
    rows = [row for row in rows if row[0] != UNKNOWN_LABEL]
    labels = sorted({row[0] for row in rows})
    centers = []
    within = []
    for label in labels:
        vectors = np.stack([row[2] for row in rows if row[0] == label])
        center = vectors.mean(axis=0)
        centers.append(center)
        within.extend(np.linalg.norm(vectors - center, axis=1).tolist())
    threshold = float(np.quantile(within, 0.95)) if within else 0.0
    if unknown:
        nearest = [np.linalg.norm(np.stack(centers) - sample[None, :], axis=1).min() for sample in unknown]
        threshold = min(threshold, float(np.quantile(nearest, 0.10)) * 0.90)
    return CentroidClassifier(labels, np.stack(centers), threshold)


def fit(rows):
    """Use source exemplars rather than averaging distinct signing styles."""
    if not rows or len(rows) > 4096:
        raise ValueError("Training requires 1–4096 rows; use a bounded representative dataset")
    labels = sorted({label for label, _, _ in rows if label != UNKNOWN_LABEL})
    vectors = np.stack([row[2] for row in rows])
    targets = np.array([labels.index(row[0]) if row[0] != UNKNOWN_LABEL else len(labels) for row in rows])
    # Estimate distance support using only training examples from different groups.
    nearest = []
    for label, group, vector in rows:
        if label == UNKNOWN_LABEL:
            continue
        others = [row[2] for row in rows if row[0] == label and row[1] != group]
        if others:
            bank = np.stack(others)
            mirrored = vector.copy()
            mirrored[::3] *= -1
            nearest.append(float(np.minimum(np.linalg.norm(bank - vector, axis=1), np.linalg.norm(bank - mirrored, axis=1)).min()))
    threshold = max(float(np.quantile(nearest, .95)) * 1.25, .15) if nearest else .5
    return ExemplarClassifier(labels, vectors, targets, threshold)
