import tempfile
import csv
import unittest
from pathlib import Path

import numpy as np

from asl_app.classifier import CentroidClassifier, ExemplarClassifier
from asl_app.features import normalize_landmarks
from asl_app.labels import SUPPORTED_LABELS
from asl_app.stabilizer import PredictionSmoother, TextBuffer
from asl_app.evaluation import evaluate
from asl_app.training import fit_centroid, group_split, load_rows


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.points = np.arange(63, dtype=np.float32).reshape(21, 3)
        self.points[:, 0] += 20
        self.points[:, 1] += 30

    def test_translation_and_scale_normalized(self):
        original = normalize_landmarks(self.points)
        changed = normalize_landmarks(self.points * 4 + np.array([50, -12, 4]))
        np.testing.assert_allclose(original, changed, atol=1e-6)

    def test_left_handedness_reflects_x_only(self):
        right = normalize_landmarks(self.points, "Right").reshape(21, 3)
        left = normalize_landmarks(self.points, "Left").reshape(21, 3)
        np.testing.assert_allclose(left[:, 0], -right[:, 0])
        np.testing.assert_allclose(left[:, 1:], right[:, 1:])


class RecognitionTests(unittest.TestCase):
    def test_supported_label_mapping_is_fixed(self):
        self.assertEqual(SUPPORTED_LABELS, ("A", "B", "C", "L", "V", "Y"))

    def test_fixed_letter_mapping_and_no_hand_abstains(self):
        labels = ["A", "B"]
        centers = np.zeros((2, 63), dtype=np.float32)
        centers[1, 0] = 2
        classifier = CentroidClassifier(labels, centers, 0.25)
        self.assertEqual(classifier.predict(centers[0]).label, "A")
        smoother = PredictionSmoother()
        for _ in range(4):
            smoother.update("A", .9)
        self.assertEqual(smoother.update("A", .9).label, "A")
        smoother.reset()
        for _ in range(8):
            stable = smoother.update(None, 0)
        self.assertIsNone(stable.label)

    def test_distant_shape_is_rejected(self):
        classifier = CentroidClassifier(["A"], np.zeros((1, 63)), .1)
        result = classifier.predict(np.ones(63))
        self.assertFalse(result.accepted)
        self.assertIsNone(result.label)

    def test_low_margin_is_rejected(self):
        centers = np.zeros((2, 63), dtype=np.float32)
        centers[1, 0] = 2
        classifier = CentroidClassifier(["A", "B"], centers, 2.0, confidence_threshold=.15)
        result = classifier.predict(np.array([1.] + [0.] * 62, dtype=np.float32))
        self.assertLess(result.confidence, .15)
        self.assertFalse(result.accepted)

    def test_model_round_trip(self):
        model = CentroidClassifier(["A"], np.zeros((1, 63)), .5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            self.assertEqual(CentroidClassifier.load(path).predict(np.zeros(63)).label, "A")

    def test_training_metrics_include_supported_and_unknown_rejection(self):
        a, b, unknown = np.zeros(63), np.zeros(63), np.zeros(63)
        a_low, a_high, b_low, b_high = (np.zeros(63) for _ in range(4))
        a_low[0], a_high[0], b[0], b_low[0], b_high[0], unknown[0] = -.1, .1, 2, 1.9, 2.1, .01
        training = [("A", "s1", a_low), ("A", "s1", a_high),
                    ("B", "s1", b_low), ("B", "s1", b_high),
                    ("UNSUPPORTED", "s1", unknown)]
        model = fit_centroid(training)
        metrics = evaluate(model, [("A", "test", a), ("B", "test", b),
                                   ("UNSUPPORTED", "test", unknown)])
        accuracy, macro_f1, per_class, columns, matrix, rejection = metrics
        self.assertEqual(accuracy, 1.0)
        self.assertEqual(macro_f1, 1.0)
        self.assertEqual(columns, ["A", "B", "REJECT"])
        self.assertEqual(matrix.tolist(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(rejection, 1.0)
        self.assertEqual(set(per_class), {"A", "B"})

    def test_exemplars_mirror_rejection_and_loaded_inference(self):
        vectors = np.zeros((3, 63), dtype=np.float32)
        vectors[0, 0] = .4
        vectors[1, 1] = 2
        vectors[2, 1] = 4
        model = ExemplarClassifier(["A", "B"], vectors, [0, 1, 2], .5)
        mirrored = vectors[0].copy()
        mirrored[::3] *= -1
        self.assertEqual(model.predict(mirrored).label, "A")
        self.assertFalse(model.predict(vectors[2]).accepted)
        self.assertEqual(model.predict(vectors[2]).reason, "Unsupported shape")
        self.assertFalse(model.predict(np.ones(63) * 10).accepted)
        self.assertFalse(model.predict(np.full(63, np.nan)).accepted)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exemplars.json"
            model.save(path)
            loaded = CentroidClassifier.load(path)
            self.assertEqual(loaded.predict(mirrored).label, "A")
            self.assertEqual(loaded.predict(vectors[1]).label, "B")

    def test_exemplars_reject_ambiguous_shapes(self):
        vectors = np.zeros((2, 63), dtype=np.float32)
        vectors[1, 1] = 2
        model = ExemplarClassifier(["A", "B"], vectors, [0, 1], 2)
        midpoint = np.zeros(63, dtype=np.float32)
        midpoint[1] = 1
        self.assertEqual(model.predict(midpoint).reason, "Ambiguous shape")

    def test_repeated_letters_spaces_editing_and_capacity(self):
        text = TextBuffer(capacity=3)
        text.accept("a")
        text.accept("a")
        text.space()
        text.space()
        self.assertEqual(text.text, "AA ")
        text.backspace()
        text.accept("b")
        self.assertEqual(text.text, "AAB")
        text.accept("c")
        text.accept("d")
        self.assertEqual(text.text, "BCD")
        text.clear()
        self.assertEqual(text.text, "")

    def test_prediction_history_is_bounded_and_requires_consensus(self):
        smoother = PredictionSmoother(capacity=5, required=3)
        self.assertIsNone(smoother.update("A", .9).label)
        smoother.update("A", .9)
        self.assertEqual(smoother.update("A", .9).label, "A")
        for _ in range(20):
            smoother.update(None, 0)
        self.assertLessEqual(len(smoother.history), 5)
        self.assertIsNone(smoother.update(None, 0).label)


class SplitTests(unittest.TestCase):
    def test_collection_unknown_labels_and_invalid_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collection.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["label", "session_id", *[f"f{i}" for i in range(63)]])
                writer.writerow(["U", "session", *np.zeros(63)])
                writer.writerow(["UNSUPPORTED", "session", *np.zeros(63)])
            self.assertEqual([row[0] for row in load_rows(path)], ["UNSUPPORTED", "UNSUPPORTED"])
            with path.open("a", newline="") as stream:
                csv.writer(stream).writerow(["A", "session", *np.full(63, np.nan)])
            with self.assertRaisesRegex(ValueError, "finite features"):
                load_rows(path)

    def test_recording_sessions_never_leak_across_split(self):
        rows = [(label, f"session-{session}", np.zeros(63))
                for session in range(8) for label in ["A", "B"]]
        train, test = group_split(rows, seed=1729)
        train_sessions = {row[1] for row in train}
        test_sessions = {row[1] for row in test}
        self.assertTrue(train_sessions)
        self.assertTrue(test_sessions)
        self.assertFalse(train_sessions & test_sessions)


if __name__ == "__main__":
    unittest.main()
