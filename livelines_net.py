#!/usr/bin/env python3
"""
Robust liveliness_net script.
Usage: python liveliness_net.py          # default camera 0
       python liveliness_net.py --cam 1  # camera index 1
"""

import os
import argparse
import sys
import time
import traceback

import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

def eprint(*a, **k):
    print(*a, **k, file=sys.stderr)

def load_model_json_weights(json_path, weights_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Model JSON not found: {json_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    with open(json_path, "r") as f:
        model_json = f.read()

    model = model_from_json(model_json)
    model.load_weights(weights_path)
    return model

def main(args):
    root = os.getcwd()
    face_xml = os.path.join(root, "models", "haarcascade_frontalface_default.xml")
    json_path = os.path.join(root, "antispoofing_models", "antispoofing_model.json")
    weights_path = os.path.join(root, "antispoofing_models", "antispoofing_model.h5")

    # File checks
    for p in (face_xml, json_path, weights_path):
        if not os.path.exists(p):
            eprint("[ERROR] Missing file:", p)
            return 2

    # Load face detector
    face_cascade = cv2.CascadeClassifier(face_xml)
    if face_cascade.empty():
        eprint("[ERROR] Failed to load Haar cascade from:", face_xml)
        return 3

    # Load model
    try:
        model = load_model_json_weights(json_path, weights_path)
    except Exception as ex:
        eprint("[ERROR] Loading model failed:", ex)
        traceback.print_exc()
        return 4

    # Print model input shape for debugging
    try:
        inp_shape = model.input_shape
        eprint("[INFO] Model input_shape:", inp_shape)
    except Exception:
        eprint("[INFO] Could not read model.input_shape")

    model.summary(print_fn=lambda s: eprint("[MODEL]", s))

    # Open camera
    vid = cv2.VideoCapture(args.cam)
    if not vid.isOpened():
        eprint(f"[ERROR] Cannot open camera index {args.cam}. Try other index or check permissions.")
        return 5

    eprint("[INFO] Camera opened. Press 'q' to quit.")
    time.sleep(0.5)

    while True:
        ret, frame = vid.read()
        if not ret:
            eprint("[WARN] Frame not received from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            y1, y2 = max(0, y - 5), min(frame.shape[0], y + h + 5)
            x1, x2 = max(0, x - 5), min(frame.shape[1], x + w + 5)
            face = frame[y1:y2, x1:x2]

            try:
                face_resized = cv2.resize(face, (160, 160))
            except Exception as ex:
                eprint("[WARN] resize failed:", ex)
                continue

            face_resized = face_resized.astype("float32") / 255.0
            face_batch = np.expand_dims(face_resized, axis=0)

            preds = model.predict(face_batch)
            spoof_score = None
            if preds is None:
                eprint("[WARN] model.predict returned None")
                continue
            preds = np.array(preds)
            if preds.ndim == 2 and preds.shape[1] == 1:
                spoof_score = float(preds[0, 0])
            elif preds.ndim == 2 and preds.shape[1] == 2:
                spoof_score = float(preds[0, 1])
            elif preds.ndim == 1:
                spoof_score = float(preds[0])
            else:
                spoof_score = float(preds.ravel().mean())

            label = "Spoof" if spoof_score > args.threshold else "Real"
            color = (0, 0, 255) if label == "Spoof" else (0, 255, 0)

            cv2.putText(frame, f"{label} {spoof_score:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("Face Antispoofing", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0, help="camera index")
    parser.add_argument("--threshold", type=float, default=0.5, help="spoof score threshold (0-1)")
    args = parser.parse_args()
    sys.exit(main(args))
