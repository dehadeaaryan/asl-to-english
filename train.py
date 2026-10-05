"""Fit the deterministic exemplar classifier on public or collected features."""

import argparse
from pathlib import Path

import numpy as np

from asl_app.training import fit, load_rows
from asl_app.labels import SUPPORTED_LABELS, UNKNOWN_LABEL

SEED = 1729


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="append", dest="data_paths",
                        help="Input CSV; repeat to combine ASLNow! and one or more local collections. Default: data/aslnow-landmarks.csv")
    parser.add_argument("--output", default="models/asl_centroids.json")
    args = parser.parse_args()
    np.random.seed(SEED)
    data_paths = args.data_paths or ["data/aslnow-landmarks.csv"]
    missing_paths = [path for path in data_paths if not Path(path).is_file()]
    if missing_paths:
        raise SystemExit(f"Landmark dataset not found: {', '.join(missing_paths)}")
    rows = [row for path in data_paths for row in load_rows(path)]
    if not rows:
        raise SystemExit(f"No labeled landmark rows found in {', '.join(data_paths)}. Collect samples before training.")
    labels = {row[0] for row in rows if row[0] != UNKNOWN_LABEL}
    missing = set(SUPPORTED_LABELS) - labels
    if missing:
        raise SystemExit(f"Collect samples for every supported letter first; missing: {', '.join(sorted(missing))}.")
    for label in SUPPORTED_LABELS:
        if len({row[1] for row in rows if row[0] == label}) < 2:
            raise SystemExit(f"Class {label} needs examples from at least two independent groups (source samples or recording sessions).")
    model = fit(rows)
    model.save(args.output)
    print(f"Saved {args.output}: {len(model.labels)} classes, {len(rows)} feature rows, seed {SEED}")


if __name__ == "__main__":
    main()
