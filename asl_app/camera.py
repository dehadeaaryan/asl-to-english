"""Camera setup, local frame capture, and guaranteed release."""

from __future__ import annotations

import cv2


def open_camera(device: int = 0, width: int = 640, height: int = 480):
    capture = cv2.VideoCapture(device)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open camera {device}. Check macOS Camera permission in System Settings > Privacy & Security > Camera."
        )
    return capture


def read_mirrored_rgb(capture):
    ok, bgr = capture.read()
    if not ok:
        raise RuntimeError("Camera stopped returning frames")
    # Mirror the camera image for a familiar selfie preview. The tracker maps
    # MediaPipe's category to the observed physical hand label consistently.
    mirrored = cv2.flip(bgr, 1)
    return cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
