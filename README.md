# Local ASL Fingerspelling Prototype

A lightweight, local webcam prototype for recognizing a small set of **static ASL fingerspelling hand shapes**. This is not ASL-to-English translation and does not recognize words or conversational ASL.

## Scope

The initial supported vocabulary is **A, B, C, L, V, Y**. Motion-dependent **J and Z are unsupported**. Every other letter is also unsupported in this first milestone. The model may report **uncertain / unsupported** instead of forcing a letter. Prepare the small ASLNow! public dataset and train the starter model using the commands below. Generated models and datasets are kept local and are not bundled in Git.

## Setup (macOS, native Python)

Development target: Apple Silicon Mac with macOS 12 or later and Python 3.11 or 3.12. Python 3.12.2 and the pinned package versions were used for the reported checks. Native webcam startup succeeded on the M1; live recognition accuracy and full camera benchmarks remain unverified. The supplied pins are for CPU inference and do not install a neural network training framework.

```bash
git clone https://github.com/dehadeaaryan/asl-to-english.git
cd asl-to-english
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The checked-in MediaPipe hand model is about 7.5 MB. Installing the Python packages downloads more than the model itself; exact wheel sizes vary by macOS and Python version. The prepared public landmark dataset is downloaded separately and is about 2.85 MB. If macOS blocks camera access, allow the terminal or IDE under **System Settings → Privacy & Security → Camera**.

## Collecting local training data

The collection tool writes only normalized numeric landmarks and labels to `data/landmarks-v2.csv`. It does not save frames or upload data. Yellow circles show the 21 points being tracked. Use consistent lighting first, then vary distance, background, and lighting. Collect each supported letter in at least two separate recording sessions; more examples and signers improve the evaluation. Click a letter button in the preview (or press its key), verify the selected label stays highlighted, hold the pose, then press Enter to save a sample. The preview confirms each save and shows per-letter totals; it also reports when no hand was detected and a sample was not saved. Press `N` between sessions, and choose `U` before Enter to record an out-of-vocabulary pose for rejection calibration. Use realistic unsupported shapes rather than arbitrary hand poses.

```bash
python collect_data.py
```

If selecting a label in the preview does not work, start with `python collect_data.py --label A`. The selected label persists until changed; Enter saves one sample immediately, with no countdown. `U` is a collection control for unsupported poses, not a supported letter.

Raw recordings are not retained. CSV contains coordinates, labels, and generated session IDs. This avoids keeping identifiable video by default. Do not commit collected data; `data/` is ignored by Git.

## Prepare the public data, train, and evaluate

ASLNow! is a small, MIT-licensed dataset of ASL letter landmarks collected from multiple participants. Its Hugging Face card reports 44,562 landmarks (21 per hand sample) and 3.28 MB total source files. The conversion stores 1,874 usable samples: A/B/C/L/V/Y keep their labels, other static letters become `UNSUPPORTED`, and J/Z are excluded because static landmark snapshots do not represent their motion. The importer pins the Parquet revision and SHA-256, checks each letter boundary against the first source example in that class, and refuses a mismatched row order. It retains only numeric normalized landmarks and labels; raw samples and converted CSV are under ignored `data/`.

The publisher provides no signer IDs and no handedness metadata. Evaluation therefore holds out individual source samples, not signers. The importer preserves source orientation, and the exemplar classifier checks both horizontal reflections to handle that missing metadata. These metrics are preliminary, not evidence of signer-independent accuracy. Collect your own data across multiple sessions and signers for a stronger test. The data card describes source collection and coordinate convention, and marks this dataset MIT: [ASLNow! dataset card](https://huggingface.co/datasets/sid220/asl-now-fingerspelling). The dataset itself is separate from the bundled MediaPipe model; its license does not cover that model.

The runtime dependencies do not include a training framework. Only the Parquet conversion tool uses optional PyArrow:

```bash
python -m pip install -r requirements-train.txt
python prepare_public_data.py
python evaluate.py
python train.py
```

The default `train.py`/`evaluate.py` input is `data/aslnow-landmarks.csv`. You can pass `--data` more than once to combine the public set with your own collected CSV, for example:

```bash
python train.py --data data/aslnow-landmarks.csv --data data/landmarks-v2.csv
```

Evaluate only the local collection with `python evaluate.py --data data/landmarks-v2.csv`; every supported letter must occur in both train and test groups.

`evaluate.py` uses fixed seed `1729` and keeps each source sample entirely in either the 75% training or 25% test partition. It reports accuracy, macro F1, per-class precision/recall/F1, a confusion matrix, and unsupported-pose rejection. The current classifier matches individual examples of supported and unsupported shapes using NumPy, checking both horizontal reflections because the public data lacks handedness. It preserves vertical orientation and depth. Training caps the model at 4,096 examples and estimates its distance limit from other training groups; an ambiguous match or distant input is rejected. The displayed match score is a distance margin, not a calibrated probability. Run `python evaluate.py --classifier centroid` to reproduce the original baseline. `train.py` saves the current exemplar model at the legacy filename `models/asl_centroids.json`. No accuracy claim should be generalized beyond this sample-level evaluation.

Session separation only tests variation between recording sessions by the same or different signers. For a signer-independent result, collect held-out sessions from people who contributed no training examples and report that separately. Do not split neighboring frames from one take across train and test.

## Run the webcam app

```bash
python app.py
```

The default capture request is 640×480 and one hand. The live prediction requires four matching observations in a bounded seven-frame history. Press **Enter** to accept the current stable letter (press again for a repeated letter), **Space** to insert a space, **Backspace** to delete, **C** to clear, and **Q** or **Esc** to exit. Camera and model resources are released on exit. Add `--draw-landmarks` to draw the points; drawing is off by default. The M1/OpenCV path now swaps MediaPipe's left/right labels to match the observed physical hand; use `--keep-mediapipe-handedness` if a different camera reports the expected convention. Frames stay on the machine.

If the model file does not exist, prepare the public dataset and train it first. Missing hands, unsupported nearest matches, distant landmarks, and ambiguous matches produce uncertainty and cannot be accepted into text. The preview shows the closest class, match distance, and rejection reason.

## Benchmark and checks

```bash
python benchmark.py --seconds 30 --width 640 --height 480 --model models/asl_centroids.json
python benchmark.py --seconds 30 --width 320 --height 240 --model models/asl_centroids.json
python benchmark.py --classification-only --seconds 10 --model models/asl_centroids.json
python -m unittest discover -s tests -v
```

The camera benchmark reports actual camera dimensions, capture, MediaPipe tracking, optional classifier, and total median/p95 latency, throughput, process CPU time as a percentage of one core, and peak process RSS. Latency samples are bounded to the latest 10,000 observations. Compare 640×480 and 320×240 runs on the target Mac; lower resolution may improve speed but can make tracking harder at distance. The separate classification-only command uses saved landmark vectors and does not import MediaPipe or open the camera. All 17 automated checks passed, including real MediaPipe VIDEO-mode startup on a blank frame outside the graphics-restricted sandbox. Checks cover normalization, handedness, unknown label mapping, invalid features, confidence rejection, bounded buffers, model loading, and group split integrity. Live recognition accuracy under different lighting, backgrounds, distances, and motion remains to be measured.

## Data choices

The app uses ASLNow! for its starter training model. Other dataset choices have these fit or size limits:

| Dataset | Terms and size | Fit for this project |
| --- | --- | --- |
| [ASLNow! fingerspelling](https://huggingface.co/datasets/sid220/asl-now-fingerspelling) | MIT; 44,562 landmarks, 3.28 MB across source files (Parquet export is 2.85 MB) | Matches MediaPipe's 21-point representation; no signer or handedness metadata, so evaluation cannot establish signer generalization. |
| [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) | CC0; about 63 MB compressed, roughly 100–150 MB extracted depending on CSV packaging; 27,455 train and 7,172 test rows, each 28×28 grayscale | Compact and excludes J/Z, but tiny image crops do not reliably support MediaPipe landmark extraction. Signer IDs are not provided for a person-wise split. |
| [ASL Alphabet on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) | GPL-2.0; 1.11 GB | 87,000 images and 29 labels, but large for this machine and no reliable signer-level split is documented. |
| [ASL Citizen](https://www.microsoft.com/en-us/download/details.aspx?id=105253) | Microsoft download lists 42.8 GB for the main archive | Far beyond the available storage and oriented to isolated signs, not this first static alphabet milestone. |

The local collection route avoids those downloads, enables varied webcam conditions, and makes session-level splits explicit. Its limitation is that useful signer-independent evidence requires multiple volunteers and consent.

## What is implemented vs. pretrained

Google MediaPipe's pretrained Hand Landmarker produces 21 landmarks and a handedness label. It is run in VIDEO mode with monotonic timestamps and a one-hand limit. The M1 camera path applies a label swap based on the physical-hand mismatch observed during collection; the command line can disable it for other camera conventions. The task model file is checked into this repository. The rest of this prototype—feature normalization, the centroid baseline, rejection threshold, smoothing, text controls, collection workflow, training, and evaluation—is project code. The generic `gesture_recognizer.task` is retained from the original repository but unused; its labels are generic gestures, not ASL letters.

The official MediaPipe Hand Landmarker documentation describes its outputs and tracking modes: [Google AI Edge guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker). The bundled model's precise source revision and redistribution terms have not been independently verified from its binary metadata; check upstream attribution and model terms before redistributing it. No license is declared here for third-party model files, collected data, or this project.

## Limitations and next steps

Landmarks omit image appearance, skin/background separation, some depth cues, facial grammar, body pose, and language context. Small landmark changes, occlusion, camera orientation, individual signing variation, and two-hand signs can make letters ambiguous. A rejected or incorrect static letter does not imply recognition of a word or ASL sentence. The next step is to collect multiple sessions and signers, measure per-class and unknown rejection performance, inspect errors, then decide whether a more expressive small classifier or temporal model is justified. J and Z need a motion-aware extension.

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for the research question, method, evaluation protocol, and results status.
