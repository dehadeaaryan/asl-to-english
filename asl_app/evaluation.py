"""Held-out classification and open-set rejection metrics."""

import numpy as np

from asl_app.labels import UNKNOWN_LABEL


def evaluate(model, rows):
    labels = model.labels
    columns = labels + ["REJECT"]
    matrix = np.zeros((len(labels) + 1, len(columns)), dtype=int)
    per_class = {}
    truths, predictions = [], []
    for true, _session, vector in rows:
        result = model.predict(vector)
        predicted = result.label if result.accepted else "REJECT"
        if true == UNKNOWN_LABEL:
            matrix[len(labels), columns.index(predicted)] += 1
        else:
            row_index = labels.index(true)
            matrix[row_index, columns.index(predicted)] += 1
        truths.append(true)
        predictions.append(predicted)
    for i, label in enumerate(labels):
        tp = matrix[i, i]
        support = int(matrix[i].sum())
        predicted_count = int(matrix[:, i].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    known = [(t, p) for t, p in zip(truths, predictions) if t != UNKNOWN_LABEL]
    accuracy = sum(t == p for t, p in known) / len(known) if known else 0.0
    macro_f1 = sum(item["f1"] for item in per_class.values()) / len(per_class) if per_class else 0.0
    unknown_support = int(matrix[len(labels)].sum())
    rejection = int(matrix[len(labels), columns.index("REJECT")]) / unknown_support if unknown_support else None
    return accuracy, macro_f1, per_class, columns, matrix, rejection


def print_report(model, metrics):
    accuracy, macro_f1, per_class, columns, matrix, rejection = metrics
    print("Held-out evaluation (groups kept separate):")
    print(f"Supported-letter accuracy: {accuracy:.3f}; macro F1: {macro_f1:.3f}")
    for label, values in per_class.items():
        print(f"  {label}: P={values['precision']:.3f} R={values['recall']:.3f} F1={values['f1']:.3f} n={values['support']}")
    print("Confusion matrix (rows=true class, columns=" + ", ".join(columns) + "):")
    for label, row in zip(model.labels + [UNKNOWN_LABEL], matrix):
        print(f"  {label}: {row.tolist()}")
    print(f"Unsupported-pose rejection: {rejection if rejection is not None else 'not measured (no unsupported samples)'}")
