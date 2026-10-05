"""Measure webcam tracking latency, throughput, CPU time, and peak RSS."""

import argparse
import resource
import statistics
import sys
import time
from collections import deque

from asl_app.classifier import CentroidClassifier
from asl_app.training import load_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--model", help="Optional trained classifier JSON for separate classifier timing")
    parser.add_argument("--classification-only", action="store_true",
                        help="Measure NumPy classifier inference on stored feature vectors; does not open the camera")
    parser.add_argument("--features", default="data/aslnow-landmarks.csv",
                        help="Feature CSV used by --classification-only")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.classification_only:
        if not args.model:
            raise SystemExit("--classification-only requires --model")
        samples = [row[2] for row in load_rows(args.features)]
        if not samples:
            raise SystemExit(f"No feature vectors found in {args.features}")
        classifier = CentroidClassifier.load(args.model)
        latencies = deque(maxlen=10000)
        cpu_start, wall_start = time.process_time(), time.perf_counter()
        index = 0
        while time.perf_counter() - wall_start < args.seconds:
            start = time.perf_counter()
            classifier.predict(samples[index % len(samples)])
            latencies.append((time.perf_counter() - start) * 1000)
            index += 1
        wall = time.perf_counter() - wall_start
        cpu = time.process_time() - cpu_start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
        values = sorted(latencies)
        print(f"Classifier-only benchmark; vectors: {len(samples)}; predictions: {index}")
        print(f"Throughput: {index / wall:.2f} predictions/s")
        print(f"Inference latency: median {statistics.median(values):.4f} ms; p95 {values[int(.95 * (len(values) - 1))]:.4f} ms")
        print(f"Process CPU: {cpu / wall * 100:.1f}% of one core; peak RSS: {peak_mb:.1f} MiB")
        print("Camera capture and MediaPipe tracking are not included.")
        return
    from asl_app.camera import open_camera, read_mirrored_rgb
    from asl_app.features import normalize_landmarks
    from asl_app.tracking import HandTracker

    capture = None
    latencies = {name: deque(maxlen=10000) for name in ("capture", "tracking", "classifier", "total")}
    frames = 0
    classifier = CentroidClassifier.load(args.model) if args.model else None
    try:
        capture = open_camera(args.camera, args.width, args.height)
        actual_width = int(capture.get(3))
        actual_height = int(capture.get(4))
        cpu_start, wall_start = time.process_time(), time.perf_counter()
        with HandTracker() as tracker:
            while time.perf_counter() - wall_start < args.seconds:
                start = time.perf_counter()
                capture_start = time.perf_counter()
                rgb = read_mirrored_rgb(capture)
                tracking_start = time.perf_counter()
                latencies["capture"].append((tracking_start - capture_start) * 1000)
                points, handedness = tracker.detect(rgb)
                classifier_start = time.perf_counter()
                latencies["tracking"].append((classifier_start - tracking_start) * 1000)
                if classifier is not None and points is not None:
                    classifier.predict(normalize_landmarks(points, handedness))
                    latencies["classifier"].append((time.perf_counter() - classifier_start) * 1000)
                latencies["total"].append((time.perf_counter() - start) * 1000)
                frames += 1
        wall = time.perf_counter() - wall_start
        cpu = time.process_time() - cpu_start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
        print(f"Requested: {args.width}x{args.height}; camera actual: {actual_width}x{actual_height}")
        print(f"Frames: {frames}; throughput: {frames / wall:.2f} FPS")
        for name in ("capture", "tracking", "classifier", "total"):
            values = latencies[name]
            if values:
                p95 = sorted(values)[int(.95 * (len(values) - 1))]
                print(f"{name.title()} latency: median {statistics.median(values):.1f} ms; p95 {p95:.1f} ms; n={len(values)}")
            elif name == "classifier":
                print("Classifier latency: not measured (pass --model after training and ensure a hand is visible)")
        print(f"Process CPU: {cpu / wall * 100:.1f}% of one core; peak RSS: {peak_mb:.1f} MiB")
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    finally:
        if capture is not None:
            capture.release()


if __name__ == "__main__":
    main()
