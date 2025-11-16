# version 1

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# -------------------------------
# Load Anti-Spoofing Model
# -------------------------------
@st.cache_resource
def load_antispoof_model():
    with open("antispoofing_models/antispoofing_model.json", "r") as f:
        model_json = f.read()

    model = model_from_json(model_json)
    model.load_weights("antispoofing_models/antispoofing_model.h5")
    return model

model = load_antispoof_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Face Anti-Spoofing System", layout="centered")

st.title("Face Anti-Spoofing System")
st.write("Click **Start Camera** to check Real vs Spoof.")

threshold = 0.5  # You can modify in sidebar if needed

# -------------------------------
# Camera Control
# -------------------------------
start = st.button("Start Camera")
stop = st.button("Stop Camera")

# Placeholder for video frames
frame_window = st.empty()

if start:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Camera not available.")
            break

        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face region (simple center crop)
        h, w, _ = rgb.shape
        face = rgb[h//4 : h - h//4, w//4 : w - w//4]  # rough face area

        try:
            resized = cv2.resize(face, (160, 160))
        except:
            resized = cv2.resize(rgb, (160, 160))

        resized = resized.astype("float32") / 255.0
        input_batch = np.expand_dims(resized, axis=0)

        # Predict
        pred = model.predict(input_batch)[0][0]

        label = "Real" if pred < threshold else "Spoof"
        color = (0, 255, 0) if label == "Real" else (255, 0, 0)

        # Draw result on frame
        cv2.putText(
            rgb,
            f"{label}  Score: {pred:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

        frame_window.image(rgb)

        # Stop button logic
        if stop:
            cap.release()
            break

    cap.release()
