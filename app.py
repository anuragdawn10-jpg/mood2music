"""
Mood2Music — Streamlit frontend

Detects the user's mood from a photo (CNN trained on FER-2013), then
recommends Spotify tracks whose audio features best match that mood.

Expected repo layout (paths below are relative to this file):
    app.py
    models/best_fer2013_model.keras
    models/spotify_scaler.pkl
    data/spotify_moods.csv   <- spotify dataframe with "Mood" column already assigned
"""

import os

import cv2

# --- TEMPORARY DIAGNOSTIC: remove once cv2.CascadeClassifier issue is resolved ---
import sys
print("cv2 loaded from:", getattr(cv2, "__file__", "NO __file__ ATTRIBUTE"), file=sys.stderr)
print("cv2 version:", getattr(cv2, "__version__", "NO __version__ ATTRIBUTE"), file=sys.stderr)
print("cv2 has CascadeClassifier:", hasattr(cv2, "CascadeClassifier"), file=sys.stderr)
print("cv2 dir sample:", [a for a in dir(cv2) if not a.startswith("_")][:20], file=sys.stderr)
# --- END TEMPORARY DIAGNOSTIC ---
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

MODEL_PATH = "models/best_fer2013_model.keras"
SCALER_PATH = "models/spotify_scaler.pkl"
DATA_PATH = "data/spotify_moods.csv"

# Must match the order the CNN's output layer was trained on (FER-2013 folder
# order) -- the "suprise" spelling is intentional, it matches the training data.
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "suprise"]

# Maps every class the CNN can output to one of the 7 mood profiles below.
EMOTION_TO_MOOD = {
    "angry": "Angry",
    "disgust": "Angry",   # negative valence + raised arousal, closest to Angry
    "fear": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "suprise": "Surprise",
}

FEATURES = [
    "Danceability", "Energy", "Loudness", "Speechiness", "Acousticness",
    "Instrumentalness", "Liveness", "Valence", "Tempo",
]

MOOD_PROFILES = {
    "Happy":    {"Danceability": 0.75, "Energy": 0.80, "Loudness": -5.0,  "Speechiness": 0.08, "Acousticness": 0.20, "Instrumentalness": 0.05, "Liveness": 0.15, "Valence": 0.85, "Tempo": 120},
    "Sad":      {"Danceability": 0.45, "Energy": 0.35, "Loudness": -10.0, "Speechiness": 0.05, "Acousticness": 0.65, "Instrumentalness": 0.10, "Liveness": 0.12, "Valence": 0.25, "Tempo": 80},
    "Angry":    {"Danceability": 0.55, "Energy": 0.90, "Loudness": -4.0,  "Speechiness": 0.12, "Acousticness": 0.10, "Instrumentalness": 0.10, "Liveness": 0.20, "Valence": 0.30, "Tempo": 135},
    "Calm":     {"Danceability": 0.40, "Energy": 0.25, "Loudness": -12.0, "Speechiness": 0.04, "Acousticness": 0.75, "Instrumentalness": 0.20, "Liveness": 0.10, "Valence": 0.50, "Tempo": 70},
    "Fear":     {"Danceability": 0.35, "Energy": 0.55, "Loudness": -8.0,  "Speechiness": 0.06, "Acousticness": 0.45, "Instrumentalness": 0.15, "Liveness": 0.15, "Valence": 0.20, "Tempo": 100},
    "Surprise": {"Danceability": 0.80, "Energy": 0.85, "Loudness": -5.0,  "Speechiness": 0.10, "Acousticness": 0.20, "Instrumentalness": 0.05, "Liveness": 0.20, "Valence": 0.75, "Tempo": 130},
    "Neutral":  {"Danceability": 0.55, "Energy": 0.55, "Loudness": -7.0,  "Speechiness": 0.06, "Acousticness": 0.40, "Instrumentalness": 0.10, "Liveness": 0.15, "Valence": 0.50, "Tempo": 100},
}

MOOD_EMOJI = {
    "Happy": "\U0001F604", "Sad": "\U0001F622", "Angry": "\U0001F620", "Calm": "\U0001F60C",
    "Fear": "\U0001F628", "Surprise": "\U0001F632", "Neutral": "\U0001F610",
}

# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_resource
def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


@st.cache_data
def load_song_data():
    return pd.read_csv(DATA_PATH)


# ----------------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------------

def predict_emotion(bgr_image, model, face_detector):
    """Detect a face in the image and classify its emotion.

    Returns (emotion, confidence, bounding_box_or_None). Falls back to
    running the whole frame through the model if no face is found.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    box = None
    if len(faces) > 0:
        # Use the largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_gray = gray[y:y + h, x:x + w]
        box = (x, y, w, h)
    else:
        face_gray = gray

    face = cv2.resize(face_gray, (48, 48)).astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))  # -> (1, 48, 48, 1)

    prediction = model.predict(face, verbose=0)[0]
    emotion_index = int(np.argmax(prediction))

    return CLASS_NAMES[emotion_index], float(prediction[emotion_index]), box


def recommend_by_emotion(detected_emotion, spotify, scaler, top_n=10):
    """Recommend tracks for a detected emotion: 60% mood-profile similarity,
    40% popularity, restricted to songs already assigned that mood."""
    if detected_emotion not in EMOTION_TO_MOOD:
        raise ValueError(f"Unknown emotion: {detected_emotion}")

    mood = EMOTION_TO_MOOD[detected_emotion]
    mood_songs = spotify[spotify["Mood"] == mood].copy()

    if len(mood_songs) == 0:
        return mood, pd.DataFrame()

    mood_vector = np.array([[MOOD_PROFILES[mood][f] for f in FEATURES]])
    song_features = mood_songs[FEATURES].to_numpy()

    song_scaled = scaler.transform(song_features)
    mood_scaled = scaler.transform(mood_vector)

    mood_songs["Similarity"] = cosine_similarity(mood_scaled, song_scaled)[0]
    mood_songs["Popularity_Score"] = mood_songs["Popularity"] / 100
    mood_songs["Recommendation_Score"] = (
        0.6 * mood_songs["Similarity"] + 0.4 * mood_songs["Popularity_Score"]
    )

    recommendations = mood_songs.sort_values(
        "Recommendation_Score", ascending=False
    ).head(top_n)

    return mood, recommendations[
        ["Track Name", "Artist Name(s)", "Popularity", "Recommendation_Score"]
    ].reset_index(drop=True)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Mood2Music", page_icon="\U0001F3B5", layout="centered")
st.title("\U0001F3B5 Mood2Music")
st.caption("Take a photo, and get song recommendations that match your mood.")

missing = [p for p in (MODEL_PATH, SCALER_PATH, DATA_PATH) if not os.path.exists(p)]
if missing:
    st.error(
        "Missing required file(s): " + ", ".join(missing) +
        ". Make sure the model, scaler, and dataset are in the repo at the "
        "paths shown at the top of app.py."
    )
    st.stop()

model = load_cnn_model()
scaler = load_scaler()
face_detector = load_face_detector()
spotify = load_song_data()

top_n = st.slider("Number of songs to recommend", min_value=5, max_value=25, value=10)

source = st.radio("Photo source", ["Camera", "Upload a photo"], horizontal=True)
image_file = (
    st.camera_input("Take a photo")
    if source == "Camera"
    else st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
)

if image_file is not None:
    pil_image = Image.open(image_file).convert("RGB")
    bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    emotion, confidence, box = predict_emotion(bgr_image, model, face_detector)

    if box is None:
        st.warning("No face detected -- classified the full photo instead.")
    else:
        x, y, w, h = box
        cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        st.image(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), caption="Detected face")

    mood = EMOTION_TO_MOOD[emotion]
    st.subheader(f"{MOOD_EMOJI.get(mood, '')} Detected mood: {mood} ({confidence:.0%} confidence)")

    _, recommendations = recommend_by_emotion(emotion, spotify, scaler, top_n)

    if recommendations.empty:
        st.warning("No songs found for this mood in the dataset.")
    else:
        st.subheader("Recommended songs")
        st.dataframe(
            recommendations.rename(columns={
                "Track Name": "Track",
                "Artist Name(s)": "Artist",
                "Recommendation_Score": "Match Score",
            }),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("Take or upload a photo to get started.")
