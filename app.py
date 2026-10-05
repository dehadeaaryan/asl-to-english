"""Real-time local static fingerspelling demo (A, B, C, L, V, Y)."""

import argparse
import time

import cv2

from asl_app.camera import open_camera, read_mirrored_rgb
from asl_app.classifier import CentroidClassifier
from asl_app.features import normalize_landmarks
from asl_app.labels import SUPPORTED_LABELS
from asl_app.stabilizer import PredictionSmoother, TextBuffer
from asl_app.tracking import HandTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/asl_centroids.json")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--draw-landmarks", action="store_true")
    parser.add_argument("--keep-mediapipe-handedness", action="store_true",
                        help="Disable the observed left/right correction for cameras that report expected labels")
    args = parser.parse_args()
    try:
        classifier = CentroidClassifier.load(args.model)
    except FileNotFoundError:
        raise SystemExit(f"No trained letter model at {args.model}. Collect multiple sessions, then run python train.py.")
    if set(classifier.labels) != set(SUPPORTED_LABELS):
        raise SystemExit("The loaded model must contain all supported letters A/B/C/L/V/Y; collect missing classes and retrain.")
    smoother, text = PredictionSmoother(), TextBuffer()
    camera = None
    tracker = None
    try:
        tracker = HandTracker(swap_handedness=not args.keep_mediapipe_handedness)
        camera = open_camera(args.camera, args.width, args.height)
    except RuntimeError as error:
        if tracker is not None:
            tracker.close()
        raise SystemExit(f"Could not initialize local camera/tracker: {error}") from error
    ticks = 0
    start = time.perf_counter()
    try:
        while True:
            rgb = read_mirrored_rgb(camera)
            begin = time.perf_counter()
            points, handedness = tracker.detect(rgb)
            if points is None:
                smoother.reset()
                prediction = smoother.update(None, 0.0)
                display = "No hand / uncertain"
                confidence = 0.0
                detail = "Tracking: no hand detected"
            else:
                result = classifier.predict(normalize_landmarks(points, handedness))
                if not result.accepted:
                    smoother.reset()
                prediction = smoother.update(result.label if result.accepted else None, result.confidence)
                display = prediction.label if prediction.label else "Uncertain / unsupported"
                confidence = prediction.confidence
                detail = f"Closest: {result.candidate or result.label or '-'} | distance {result.distance:.2f} | {result.reason or ('Stable' if prediction.label else 'Hold pose briefly')}"
            latency_ms = (time.perf_counter() - begin) * 1000
            ticks += 1
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if args.draw_landmarks and points:
                height, width = frame.shape[:2]
                for x, y, _ in points:
                    cv2.circle(frame, (int(x * width), int(y * height)), 3, (40, 230, 40), -1)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 145), (24, 24, 24), -1)
            cv2.putText(frame, f"Prediction: {display}  score {confidence:.2f}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .60, (30, 240, 30), 2)
            cv2.putText(frame, f"Accepted text: {text.text}", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            cv2.putText(frame, f"{latency_ms:.1f} ms | {args.width}x{args.height} | handedness {handedness or '-'}", (16, 92), cv2.FONT_HERSHEY_SIMPLEX, .48, (230, 230, 230), 1)
            cv2.putText(frame, detail, (16, 123), cv2.FONT_HERSHEY_SIMPLEX, .43, (230, 230, 230), 1)
            cv2.putText(frame, "Enter accept | Space space | Backspace delete | C clear | Q/Esc quit", (16, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, .47, (230, 230, 230), 1)
            cv2.imshow("ASL Fingerspelling Prototype", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (10, 13) and prediction.label:
                text.accept(prediction.label)
            elif key == ord(" "):
                text.space()
            elif key in (8, 127):
                text.backspace()
            elif key in (ord("c"), ord("C")):
                text.clear()
    finally:
        tracker.close()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        elapsed = time.perf_counter() - start
        print(f"Frames: {ticks}; preview loop rate: {ticks / elapsed:.1f} FPS" if elapsed else "No frames processed")


if __name__ == "__main__":
    main()
