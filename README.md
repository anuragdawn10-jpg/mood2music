# Mood2Music

Detects a person's mood from a photo (CNN trained on FER-2013) and recommends
Spotify tracks whose audio features match that mood.

## Repo structure

```
mood2music/
├── app.py                     # Streamlit frontend
├── requirements.txt           # Python dependencies
├── packages.txt               # System (apt) dependencies for OpenCV on Streamlit Cloud
├── models/
│   └── best_fer2013_model.keras
├── data/
│   └── top_10000_1950-now.csv # raw Spotify dataset
└── notebooks/
    └── mood2music_ipynb.ipynb # training / development notebook
```

Only two artifacts need to be shipped: the trained CNN and the raw Spotify
CSV. `app.py` cleans the CSV, fits a scaler, and assigns each song a mood by
nearest mood-profile similarity itself at startup (cached), so there's no
separate scaler or pre-labeled CSV to keep in sync with the app code.

## Generating the model file

`models/` isn't included in the notebook export — running the notebook
(through the CNN training section) produces `best_fer2013_model.keras` in
its own working directory. After running it:

```bash
mkdir -p models data
mv best_fer2013_model.keras models/
cp /path/to/top_10000_1950-now.csv data/
```

Then commit `models/` and `data/` along with the code.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

(`packages.txt` only matters for Streamlit Community Cloud — it lists apt
packages, so it's ignored when running locally. If OpenCV complains about a
missing system library on your machine, install the equivalent package for
your OS.)

## Deploying on Streamlit Community Cloud

1. Push this repo to GitHub, including `models/` and `data/`.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** → deploy from GitHub → select this repo, the branch,
   and `app.py` as the main file.
4. In **Advanced settings**, set the Python version to **3.11**.
5. Deploy, and watch the build log for `packages.txt` (apt) then
   `requirements.txt` (pip) installing.

## Notes

- `emotion_to_mood` in `app.py` maps the CNN's raw output classes
  (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `suprise` — note
  the intentional spelling, it matches the FER-2013 training labels) to one
  of 7 mood profiles used to filter and rank songs.
- `opencv-python-headless` is pinned to `4.13.0.92` in `requirements.txt`.
  OpenCV 5.0 moved `CascadeClassifier` (used for face detection) out of the
  base package, so don't remove this pin without switching the detector.
- The mood-assignment logic in `app.py`'s `load_and_prepare_song_data()`
  mirrors the notebook's profile-based approach — if you tune `MOOD_PROFILES`
  in the notebook, update the copy in `app.py` to match.
