"""Evaluate a session-held-out split before trusting or comparing models."""

import argparse
from pathlib import Path

from asl_app.evaluation import evaluate, print_report
from asl_app.training import SEED, fit, fit_centroid, group_split, load_rows
from asl_app.labels import SUPPORTED_LABELS, UNKNOWN_LABEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", choices=("exemplar", "centroid"), default="exemplar",
                        help="Compare the current exemplar classifier with the original centroid baseline")
    parser.add_argument("--data", action="append", dest="data_paths",
                        help="Input CSV; repeat to combine ASLNow! and one or more local collections. Default: data/aslnow-landmarks.csv")
    args = parser.parse_args()
    data_paths = args.data_paths or ["data/aslnow-landmarks.csv"]
    missing_paths = [path for path in data_paths if not Path(path).is_file()]
    if missing_paths:
        raise SystemExit(f"Landmark dataset not found: {', '.join(missing_paths)}")
    rows = [row for path in data_paths for row in load_rows(path)]
    if not rows:
        raise SystemExit(f"No labeled landmark rows found in {', '.join(data_paths)}. Collect samples before evaluation.")
    labels = {row[0] for row in rows if row[0] != UNKNOWN_LABEL}
    missing = set(SUPPORTED_LABELS) - labels
    if missing:
        raise SystemExit(f"Collect samples for every supported letter first; missing: {', '.join(sorted(missing))}.")
    try:
        train_rows, test_rows = group_split(rows, SEED)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if {row[0] for row in train_rows} & labels != labels or {row[0] for row in test_rows} & labels != labels:
        raise SystemExit("The deterministic session split must include every supported class on both sides; collect another session.")
    model = (fit if args.classifier == "exemplar" else fit_centroid)(train_rows)
    print_report(model, evaluate(model, test_rows))
    groups = {row[1] for row in rows}
    group_name = "source samples" if all(group.startswith("aslnow:") for group in groups) else "recording sessions/groups"
    print(f"Seed: {SEED}; train {group_name}: {len({r[1] for r in train_rows})}; test {group_name}: {len({r[1] for r in test_rows})}")


if __name__ == "__main__":
    main()
