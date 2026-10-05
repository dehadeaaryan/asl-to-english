"""MediaPipe VIDEO-mode hand tracking with one hand and monotonic timestamps."""

from __future__ import annotations

import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def correct_handedness(category_name: str, swap: bool = True) -> str:
    """Map MediaPipe's camera label to the physical hand label used by this app.

    The current M1/OpenCV camera path has been observed to report the reverse
    label for the mirrored preview. ``swap=False`` keeps MediaPipe's raw value
    for camera paths that already follow the expected selfie convention.
    """
    if not swap:
        return category_name
    return {"Left": "Right", "Right": "Left"}.get(category_name, category_name)


class HandTracker:
    def __init__(self, model_path: str = "models/hand_landmarker.task", max_hands: int = 1,
                 swap_handedness: bool = True):
        options = vision.HandLandmarkerOptions(
            # Tasks' Python GPU delegate is platform-limited and unnecessary
            # for this CPU-friendly prototype. Explicitly select CPU for macOS.
            base_options=python.BaseOptions(
                model_asset_path=model_path,
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1
        self._swap_handedness = swap_handedness

    def detect(self, rgb_frame):
        timestamp_ms = time.monotonic_ns() // 1_000_000
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._detector.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None, None
        points = result.hand_landmarks[0]
        handedness = correct_handedness(result.handedness[0][0].category_name, self._swap_handedness)
        xyz = [(point.x, point.y, point.z) for point in points]
        return xyz, handedness

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
