# Technical report: static ASL fingerspelling prototype

## Research question

Can a small CPU-friendly model classify a limited vocabulary of static ASL fingerspelling shapes from webcam hand landmarks, while abstaining when the hand is absent or a pose is outside its supported classes?

## Scope and method

The first vocabulary is A, B, C, L, V, and Y. J and Z require motion and are excluded. The app requests 640×480 webcam frames, mirrors the preview, and tracks one hand with MediaPipe Hand Landmarker in VIDEO mode. Monotonic millisecond timestamps enable video tracking. MediaPipe supplies the 21-point landmark estimate and handedness; this is pretrained functionality, not a model trained by this project. The M1/OpenCV path swaps the category label because the raw label was observed reversed against the physical hand; a flag disables this camera-specific correction.

Project-authored preprocessing subtracts the wrist coordinate, scales by the maximum wrist-to-landmark distance, and reflects left hands into the right-hand x convention. The current NumPy exemplar classifier matches supported and unsupported training examples, checking both horizontal reflections because the public data lacks handedness. Vertical orientation and depth are retained. Distance and match-margin cutoffs allow rejection. A 4-of-7 bounded consensus stabilizes output; rejected observations clear the history. The user deliberately accepts letters, including repeated letters, and can add spaces, delete, or clear. The original nearest-centroid baseline remains available for comparison.

Both classifiers are deterministic and use NumPy without a neural network framework. The exemplar baseline improved the available sample-level results, so a neural network has not yet been justified. Landmark-only inputs discard appearance and some orientation/depth evidence, so similar poses may remain inseparable.

## Data and split

The starter training set is ASLNow! (`sid220/asl-now-fingerspelling`), whose card reports MIT terms, 44,562 individual landmark rows, and 3.28 MB of source files. `prepare_public_data.py` uses the 2.85 MB pinned Hugging Face Parquet export, verifies its SHA-256, reconstructs labels from alphabetically sorted source-folder counts, and checks each letter boundary against the source JSON. It converts complete 21-point samples to wrist-centered, scale-normalized vectors. A/B/C/L/V/Y retain their labels, other static letters are mapped to `UNSUPPORTED`, and J/Z are excluded. The publisher gives no signer IDs or handedness labels; the importer assumes a right-hand feature convention. Therefore the fixed-seed 75/25 split groups each individual source file only, not its signer, and cannot establish signer-independent accuracy. This is a material data limitation.

The separate local collector stores normalized vectors, labels, and recording session IDs without retaining raw camera frames. The same group splitter keeps complete local sessions together. Collecting more sessions from distinct signers is needed for credible generalization evidence.

Dataset notes, sizes, and terms are in [README.md](README.md#data-choices). ASLNow! was downloaded locally for the experiments; datasets and generated classifiers are excluded from Git. A fresh checkout must run the documented preparation and training commands.

## Evaluation protocol

Run `python evaluate.py` to evaluate the public starter data, then `python train.py` to fit the final model. The held-out rows are samples from the same published pool, not held-out people. The evaluator reports supported-letter accuracy, macro F1, per-class precision/recall/F1, a confusion matrix with a reject column, and unsupported-letter rejection. No-hand rejection is deterministic at the application boundary: no detected hand does not enter the classifier.

Run `python benchmark.py --seconds 30 --width 640 --height 480 --model models/asl_centroids.json` and repeat at 320×240. The script reports separate capture, tracking, classifier, and full-loop median/p95 latency, throughput, process CPU percentage, peak process RSS, and actual camera dimensions.

## Initial centroid baseline

The initial sample-held-out public-data run (seed 1729; 75/25 split; 105 supported samples and 363 unsupported samples in the test fold) produced **18.1% supported-letter accuracy** and **0.210 macro F1**. Per-class F1: A 0.480, B 0.667, C 0.000, L 0.000, V 0.111, Y 0.000. Unsupported-pose rejection was 93.1%. Reproduce it with `python evaluate.py --classifier centroid`. The confusion matrix is:

| True class | A | B | C | L | V | Y | Reject |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 6 | 0 | 0 | 0 | 0 | 0 | 13 |
| B | 0 | 11 | 0 | 0 | 0 | 0 | 8 |
| C | 0 | 0 | 0 | 0 | 0 | 0 | 13 |
| L | 0 | 0 | 0 | 0 | 0 | 0 | 15 |
| V | 0 | 0 | 0 | 0 | 2 | 0 | 13 |
| Y | 0 | 0 | 0 | 0 | 0 | 0 | 24 |
| Unsupported | 0 | 3 | 0 | 3 | 19 | 0 | 338 |

Most C/L/Y examples were rejected. A diagnostic threshold sweep showed the expected tradeoff: at a distance threshold of 1.30, supported coverage rose to 80.0% and accuracy to 75.2%, while unsupported rejection fell to 50.1%. These diagnostic settings were examined on the test split and are not independent validation results. The baseline's strict default rejected too many supported signs and was replaced by the exemplar classifier below.

Automated checks: 13 passed; one MediaPipe native task startup test was skipped because the managed session could not create its `NSOpenGLPixelFormat`/`kGpuService`, even with the CPU delegate selected. A 640×480 benchmark was attempted, but macOS denied camera access to the command-line process. Camera latency, throughput, CPU usage, and peak memory are therefore still not measured. The public classifier metrics do not establish signer generalization or live webcam performance.

A separate 2-second NumPy-only classifier run over the 1,874 saved feature vectors measured 79,979 predictions/s, 0.0120 ms median latency, 0.0123 ms p95 latency, 124.3% process CPU (about 1.24 cores), and 45.1 MiB peak RSS. This excludes camera capture and MediaPipe, and measures the inference loop only. It does not substitute for the pending full 640×480/320×240 end-to-end measurement.

## Exemplar correction after live rejection failures

The live centroid model rejected every pose reported by the user. Its single global distance cutoff was too strict, and averaging each class erased distinct signing styles. The replacement stores a bounded bank of supported and unsupported training examples and compares each query to both horizontal reflections. This removes reliance on unknown public-data handedness while retaining vertical orientation and depth. Unsupported nearest matches, a distance beyond the training-derived support limit, and a relative distance margin below 0.05 are rejected. The displayed score is a match margin rather than a probability. A rejected observation now clears stabilization immediately, preventing acceptance of a stale letter. The live UI displays the closest class, distance, and rejection reason.

On the identical sample split, the exemplar model measured **93.3% supported accuracy, 0.958 macro F1, and 99.4% unsupported rejection**. There are 105 supported and 363 unsupported test samples. Per-class F1: A 0.857, B 0.974, C 0.960, L 1.000, V 1.000, Y 0.957. The confusion matrix follows:

| True class | A | B | C | L | V | Y | Reject |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 15 | 0 | 0 | 0 | 0 | 0 | 4 |
| B | 0 | 19 | 0 | 0 | 0 | 0 | 0 |
| C | 0 | 0 | 12 | 0 | 0 | 0 | 1 |
| L | 0 | 0 | 0 | 15 | 0 | 0 | 0 |
| V | 0 | 0 | 0 | 0 | 15 | 0 | 0 |
| Y | 0 | 0 | 0 | 0 | 0 | 22 | 2 |
| Unsupported | 1 | 1 | 0 | 0 | 0 | 0 | 361 |

An audit found zero identical normalized vectors across partitions (rounded to six decimals), and zero test vectors within Euclidean distance 0.01 of any training vector. This does not exclude shared signers or recording sessions, which the publisher does not identify. Both model development and comparison used this public sample split; an independent signer test is still required, and live accuracy remains unverified.

The replacement's 2-second classifier-only benchmark measured 3,155 predictions/s, 0.3228 ms median, 0.3753 ms p95, 99.9% process CPU while running continuously, and 51.3 MiB peak process RSS. The bank stores 1,874 × 63 float32 values (about 0.45 MiB). These numbers exclude tracking and camera capture. Final automated checks: all 17 passed outside the restricted sandbox, including real VIDEO-mode tracker startup and blank-frame no-hand behavior. The collector now maps its U control to UNSUPPORTED, with backward compatibility for older U rows. Generated data and trained model files remain excluded from Git. The restarted live app initialized successfully; its recognition output still needs user review.

## Limitations and next steps

The prototype recognizes a static hand shape, not ASL words or sentences. It does not model non-manual grammar, face/body context, two-handed signs, coarticulation, or sign motion. Camera lighting, clutter, distance, hand occlusion, camera mirroring, and signer variation need explicit evaluation. Six labels are the only supported letters; all other letters, including J/Z, must be treated as unsupported.

Next: collect at least two sessions per class across multiple signers; include realistic unsupported poses; report a signer-held-out test; inspect per-class confusion and failure cases; measure both camera settings on the M1 Mac; only then consider a small neural classifier or temporal J/Z model.
