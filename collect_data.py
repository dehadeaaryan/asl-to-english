"""Collect labeled, local landmark features for A/B/C/L/V/Y.

Keys select a label; hold the pose and press Enter to record one sample. Press N
to change session ID, Q/Esc to quit. Only numeric landmarks are written.
"""

import argparse
import csv
import textwrap
import uuid
from datetime import datetime
from pathlib import Path

import cv2

from asl_app.camera import open_camera, read_mirrored_rgb
from asl_app.features import normalize_landmarks
from asl_app.labels import SUPPORTED_LABELS, UNKNOWN_LABEL
from asl_app.tracking import HandTracker

LABELS = "".join(SUPPORTED_LABELS) + "U"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/landmarks-v2.csv")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--label", choices=tuple(LABELS),
                        help="Start with this label selected, e.g. --label A (useful if window input is unavailable)")
    parser.add_argument("--keep-mediapipe-handedness", action="store_true",
                        help="Disable the observed left/right correction for cameras that report the expected labels")
    args = parser.parse_args()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists() and target.stat().st_size > 0
    counts = {label: 0 for label in LABELS}
    if exists:
        with target.open(newline="", encoding="utf-8") as prior:
            for row in csv.DictReader(prior):
                label = "U" if row.get("label") == UNKNOWN_LABEL else row.get("label")
                if label in counts:
                    counts[label] += 1
    session = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    ui = {
        "selected": args.label,
        "feedback": (f"Selected {args.label} from command line. Hold the pose, then press Enter."
                     if args.label else "Choose a letter, hold the pose, then press Enter to save."),
        "feedback_color": (80, 245, 100) if args.label else (240, 240, 240),
    }

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and 56 <= y <= 87 and 12 <= x < 12 + 48 * len(LABELS):
            index = (x - 12) // 48
            ui["selected"] = LABELS[index]
            ui["feedback"] = f"Selected {LABELS[index]}. Hold that pose, then press Enter."
            ui["feedback_color"] = (80, 245, 100)

    camera = None
    try:
        camera = open_camera(args.camera, args.width, args.height)
        with HandTracker(swap_handedness=not args.keep_mediapipe_handedness) as tracker, target.open("a", newline="", encoding="utf-8") as stream:
            window_name = "ASL landmark collection"
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, on_mouse)
            writer = csv.writer(stream)
            if not exists:
                writer.writerow(["label", "session_id", *[f"f{i}" for i in range(63)]])
            while True:
                rgb = read_mirrored_rgb(camera)
                points, handedness = tracker.detect(rgb)
                preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if points is not None:
                    frame_height, frame_width = preview.shape[:2]
                    for x, y, _z in points:
                        center = (int(x * frame_width), int(y * frame_height))
                        cv2.circle(preview, center, 7, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.circle(preview, center, 2, (255, 255, 255), -1, cv2.LINE_AA)
                feedback_lines = textwrap.wrap(ui["feedback"], width=55) or [""]
                panel_bottom = max(105, 103 + 24 * len(feedback_lines))
                cv2.rectangle(preview, (0, 0), (preview.shape[1], panel_bottom), (24, 24, 24), -1)
                selected = ui["selected"]
                status = f"SELECTED: {selected or 'choose a label'}    HAND: {handedness or 'NOT DETECTED'}    Session: {session[-10:]}"
                count_status = "Samples  " + "  ".join(f"{label}: {counts[label]}" for label in LABELS)
                cv2.putText(preview, status, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .49, (80, 245, 100), 1)
                cv2.putText(preview, count_status, (12, 51), cv2.FONT_HERSHEY_SIMPLEX, .46, (235, 235, 235), 1)
                for index, label in enumerate(LABELS):
                    left = 12 + index * 48
                    fill = (45, 145, 55) if label == selected else (68, 68, 68)
                    cv2.rectangle(preview, (left, 57), (left + 39, 86), fill, -1)
                    cv2.putText(preview, label, (left + 13, 78), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(preview, "Click a label or press its key. Yellow dots are tracked points. Enter saves.", (12, 101), cv2.FONT_HERSHEY_SIMPLEX, .39, (210, 210, 210), 1)
                # Keep action results visible and wrap them within the window.
                for index, line in enumerate(feedback_lines):
                    cv2.putText(preview, line, (12, 125 + 23 * index), cv2.FONT_HERSHEY_SIMPLEX, .48, ui["feedback_color"], 1, cv2.LINE_AA)
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                char = chr(key).upper() if key != 255 else ""
                if char in LABELS:
                    ui["selected"] = char
                    ui["feedback"] = f"Selected {char}. Hold that pose, then press Enter."
                    ui["feedback_color"] = (80, 245, 100)
                    selected = char
                elif key in (10, 13) and selected and points is not None:
                    writer.writerow([UNKNOWN_LABEL if selected == "U" else selected, session, *normalize_landmarks(points, handedness).tolist()])
                    stream.flush()
                    counts[selected] += 1
                    print(f"Saved {selected} sample")
                    ui["feedback"] = f"SAVED {selected} sample. This file now has {counts[selected]} {selected} samples."
                    ui["feedback_color"] = (80, 245, 100)
                elif key in (10, 13) and points is None:
                    ui["feedback"] = "NOT SAVED: no hand detected. Move into camera view and try again."
                    ui["feedback_color"] = (40, 80, 255)
                    print(ui["feedback"], flush=True)
                elif key in (10, 13) and not selected:
                    ui["feedback"] = "NOT SAVED: select A, B, C, L, V, Y, or U first."
                    ui["feedback_color"] = (40, 80, 255)
                    print(ui["feedback"], flush=True)
                elif char == "N":
                    session = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
                    print(f"New session: {session}")
                    ui["feedback"] = "New recording session started. Capture each label again in this session."
                    ui["feedback_color"] = (80, 245, 100)
                elif key in (ord("q"), ord("Q"), 27):
                    break
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
