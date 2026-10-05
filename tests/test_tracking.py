import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from asl_app.tracking import HandTracker, correct_handedness


class HandTrackerTests(unittest.TestCase):
    def test_camera_handedness_correction_and_raw_override(self):
        self.assertEqual(correct_handedness("Left"), "Right")
        self.assertEqual(correct_handedness("Right"), "Left")
        self.assertEqual(correct_handedness("Left", swap=False), "Left")

    def test_video_mode_loads_and_returns_no_hand_for_blank_frame(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        try:
            tracker = HandTracker()
        except RuntimeError as error:
            if "NSOpenGLPixelFormat" in str(error) or "kGpuService" in str(error):
                self.skipTest(f"MediaPipe native graphics service unavailable in this session: {error}")
            raise
        with tracker:
            points, handedness = tracker.detect(frame)
            self.assertIsNone(points)
            self.assertIsNone(handedness)

    def test_timestamps_increase_even_when_clock_resolution_repeats(self):
        class Detector:
            def __init__(self):
                self.timestamps = []

            def detect_for_video(self, _image, timestamp):
                self.timestamps.append(timestamp)
                return SimpleNamespace(hand_landmarks=[], handedness=[])

        tracker = HandTracker.__new__(HandTracker)
        tracker._detector = Detector()
        tracker._last_timestamp_ms = -1
        with patch("asl_app.tracking.time.monotonic_ns", return_value=5_000_000):
            tracker.detect(np.zeros((2, 2, 3), dtype=np.uint8))
            tracker.detect(np.zeros((2, 2, 3), dtype=np.uint8))
        self.assertEqual(tracker._detector.timestamps, [5, 6])


if __name__ == "__main__":
    unittest.main()
