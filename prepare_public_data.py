"""Download the small ASLNow! landmark set and convert it to project features.

The Hugging Face Parquet export omits the original folder labels. This script
reconstructs labels from the source's per-letter file counts and verifies each
letter boundary against a source JSON before accepting the order.
"""

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from asl_app.features import normalize_landmarks
from asl_app.labels import SUPPORTED_LABELS, UNKNOWN_LABEL

DATASET = "sid220/asl-now-fingerspelling"
REVISION = "9b3c96ae0adb7744a2c9fc72692842e6b3e25e33"
PARQUET_REVISION = "efbef8e2d3f74013d35e1c5c444f0ea102117043"
PARQUET_SHA256 = "c9fb5bc15f07c537353c41a28945dedca08a2f467422a4fef84e380414e8942c"
LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MOTION_LETTERS = {"J", "Z"}
PARQUET_URL = (
    f"https://huggingface.co/datasets/{DATASET}/resolve/{PARQUET_REVISION}/default/train/0000.parquet"
)
TREE_URL = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/{{letter}}"
SAMPLE_URL = (
    f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{{path}}"
)


def fetch(url):
    response = subprocess.run(
        ["curl", "-fsSL", "--retry", "2", "--max-time", "30", url],
        check=True,
        capture_output=True,
    )
    return response.stdout


def files_by_letter():
    result = {}
    for letter in LETTERS:
        entries = json.loads(fetch(TREE_URL.format(letter=letter)))
        paths = sorted(entry["path"] for entry in entries if entry["type"] == "file" and entry["path"].endswith(".json"))
        if not paths:
            raise RuntimeError(f"No landmark JSON files found for class {letter}")
        result[letter] = paths
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default="data/raw/asl-now-train.parquet")
    parser.add_argument("--output", default="data/aslnow-landmarks.csv")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if not parquet_path.is_file():
        parquet_path.write_bytes(fetch(PARQUET_URL))
    actual_hash = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    if actual_hash != PARQUET_SHA256:
        raise RuntimeError(f"Unexpected Parquet SHA-256 {actual_hash}; refusing to use unverified data")
    table = pq.read_table(parquet_path, columns=["x", "y", "z"])
    coords = [table.column(name).to_numpy(zero_copy_only=False) for name in ("x", "y", "z")]
    landmarks = np.column_stack(coords).astype(np.float32, copy=False)
    if landmarks.shape[1] != 3 or len(landmarks) % 21:
        raise RuntimeError(f"Expected complete 21-point samples, found {landmarks.shape}")

    paths = files_by_letter()
    expected = sum(len(items) for items in paths.values()) * 21
    if len(landmarks) != expected:
        raise RuntimeError(f"Parquet has {len(landmarks)} points; source file counts imply {expected}")

    # Check each alphabet boundary against the first source file in that class.
    point_offset = 0
    for letter in LETTERS:
        first = np.asarray(
            [[point[axis] for axis in ("x", "y", "z")] for point in json.loads(fetch(SAMPLE_URL.format(path=paths[letter][0])))],
            dtype=np.float32,
        )
        if first.shape != (21, 3) or not np.allclose(landmarks[point_offset:point_offset + 21], first, atol=1e-7):
            raise RuntimeError(f"Parquet order/format does not match the first sorted {letter} source sample; refusing label reconstruction")
        point_offset += len(paths[letter]) * 21

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "session_id", *[f"f{i}" for i in range(63)]])
        point_offset = 0
        for letter in LETTERS:
            for path in paths[letter]:
                points = landmarks[point_offset:point_offset + 21]
                point_offset += 21
                if points.shape != (21, 3):
                    raise RuntimeError(f"Malformed sample: {path}")
                label = letter if letter in SUPPORTED_LABELS else UNKNOWN_LABEL
                if letter in MOTION_LETTERS:
                    continue  # Static screenshots are not valid examples of motion letters.
                # The publisher does not expose handedness or signer IDs. Retain raw
                # camera orientation and use each source sample as its own split group.
                features = normalize_landmarks(points, handedness="Right")
                sample_id = Path(path).stem
                writer.writerow([label, f"aslnow:{letter}:{sample_id}", *features.tolist()])
                counts[label] += 1

    print(f"Wrote {sum(counts.values())} samples to {target}")
    print("Rows by training label:", dict(sorted(counts.items())))
    print("Source split: file/sample-level only; signer identity is unavailable.")


if __name__ == "__main__":
    main()
