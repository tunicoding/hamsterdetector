# Hamster Detector

A webcam demo that changes the hamster image in response to hand gestures and facial expressions.

## Run locally

Use Python 3.11 and a webcam. From the repository root:

```sh
python -m venv .venv
```

Activate the environment (`source .venv/bin/activate` on macOS/Linux, or `.venv\Scripts\Activate.ps1` in Windows PowerShell), then run:

```sh
python -m pip install mediapipe==0.10.21
python hamsterdetector/hamster.py
```

This MediaPipe release provides the legacy `mp.solutions` API and installs OpenCV as a dependency. Allow camera access if prompted. Press **Q** while the preview window is focused to quit.
