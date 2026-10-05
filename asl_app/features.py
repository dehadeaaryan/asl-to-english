"""Compact, position- and scale-normalized MediaPipe hand features."""

from __future__ import annotations

import numpy as np

FEATURE_COUNT = 21 * 3


def normalize_landmarks(landmarks: object, handedness: str = "Right") -> np.ndarray:
    """Return a stable 63-value vector while retaining palm orientation.

    The wrist is the origin and the maximum wrist-to-landmark distance is the
    scale. Coordinates are not rotated, since hand orientation distinguishes
    some fingerspelling shapes. Left hands are reflected into a right-hand
    coordinate convention; callers pass the camera-corrected physical hand
    label returned by ``HandTracker``.
    """
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError(f"Expected 21 xyz landmarks, got shape {points.shape}")
    points = points - points[0]
    scale = float(np.linalg.norm(points, axis=1).max())
    if not np.isfinite(scale) or scale < 1e-6:
        raise ValueError("Landmarks have no usable spatial extent")
    points /= scale
    if handedness.lower() == "left":
        points[:, 0] *= -1
    elif handedness.lower() != "right":
        raise ValueError(f"Unknown handedness: {handedness}")
    return points.reshape(FEATURE_COUNT)
