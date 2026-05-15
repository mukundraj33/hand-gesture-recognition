---
title: Hand Gesture Recognition
sdk: streamlit
app_file: app.py
python_version: "3.13"
---

# Hand Gesture Recognition

A professional Streamlit web application for real-time hand gesture recognition. The app uses browser webcam input, MediaPipe Hands landmark detection, and the existing TensorFlow/Keras keypoint classifier to identify eight trained gestures.

## Features

- Browser webcam support through `streamlit-webrtc`
- Real-time MediaPipe hand landmark detection
- 21-point hand skeleton drawing on processed frames
- Existing preprocessing pipeline preserved exactly
- TensorFlow/Keras model inference without retraining
- Gesture label and confidence display
- Snapshot mode for image upload or browser camera capture
- Streamlit Community Cloud ready configuration

## Supported Gestures

| Class | Gesture |
| --- | --- |
| 0 | Open hand |
| 1 | Close hand |
| 2 | Pointer |
| 3 | OK |
| 4 | Rock |
| 5 | Good luck |
| 6 | Dislike |
| 7 | Like |

## Architecture

```text
Browser webcam frame
  -> streamlit-webrtc frame processor
  -> RGB conversion
  -> MediaPipe Hands detection
  -> 21 landmark coordinate extraction
  -> wrist-relative landmark normalization
  -> TensorFlow/Keras keypoint classifier
  -> gesture label, confidence, and annotated frame
```

The ML behavior is implemented in reusable modules:

- `utils/mediapipe_detector.py`: MediaPipe Hands wrapper with Python 3.13 MediaPipe Tasks support
- `utils/preprocessing.py`: bounding box, landmark extraction, and landmark normalization
- `utils/pipeline.py`: end-to-end prediction pipeline
- `model/keypoint_classifier/keypoint_classifier.py`: Keras model loading and inference
- `model/keypoint_classifier/keypoint_classifier_label.csv`: label mapping
- `assets/hand_landmarker.task`: MediaPipe hand landmark detector asset for Python 3.13

## Project Structure

```text
hand-gesture-recognition/
|-- app.py
|-- README.md
|-- requirements.txt
|-- runtime.txt
|-- .streamlit/
|   `-- config.toml
|-- assets/
|   `-- hand_landmarker.task
|-- model/
|   |-- __init__.py
|   `-- keypoint_classifier/
|       |-- keypoint_classifier.py
|       |-- keypoint_classifier.keras
|       |-- keypoint_classifier_label.csv
|       `-- keypoint.csv
`-- utils/
    |-- config.py
    |-- drawing.py
    |-- labels.py
    |-- mediapipe_detector.py
    |-- pipeline.py
    `-- preprocessing.py
```

## Local Installation

Use Python 3.13.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS or Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit, allow browser camera access, and start the live recognition stream.

## Streamlit Community Cloud Deployment

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud.
3. Select **New app**.
4. Choose the repository and branch.
5. Set the main file path to `app.py`.
6. Confirm Python 3.13 is selected through `runtime.txt`.
7. Deploy.

No secrets are required for this project.

## Deployment Notes

- The app does not use `cv2.VideoCapture(0)`.
- Webcam access is handled by the browser through WebRTC.
- The classifier model file must remain at `model/keypoint_classifier/keypoint_classifier.keras`.
- Labels must remain at `model/keypoint_classifier/keypoint_classifier_label.csv`.
- Python 3.13 uses MediaPipe Tasks with `assets/hand_landmarker.task`.
- If webcam permission is blocked, reset site permissions in the browser and reload the app.
- If MediaPipe fails to install, confirm the deployment runtime is Python 3.13.

## Screenshots

Add screenshots after deployment:

- `assets/live-recognition.png`
- `assets/snapshot-inference.png`
- `assets/confidence-scores.png`

## Tech Stack

- Streamlit
- streamlit-webrtc
- MediaPipe Hands / MediaPipe Tasks
- OpenCV
- TensorFlow/Keras
- NumPy

## Future Improvements

- Add temporal smoothing across frame predictions
- Add optional confidence threshold filtering
- Add gesture history charts
- Add mobile-specific layout tuning
- Add model cards and dataset documentation
