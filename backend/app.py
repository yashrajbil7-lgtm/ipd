"""
app.py  -  FastAPI backend for the Ornament Detection demo

Run:   python app.py
Open:  http://localhost:8000

This one server does two jobs:
  1. POST /predict   -> receives the audio, runs preprocessing + CNN, returns JSON
  2. Serves the website (the ../frontend folder), so no second server is needed

Adding Kanswar / Murki later: put  kanswar_CNN.pth + kanswar_config.json
into the models/ folder and restart. Every *_config.json is loaded automatically.
"""

import os
import json
import glob
import shutil
import tempfile

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import preprocessing as prep
from model import load_ornament_model

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
FRONTEND_DIR = os.path.join(BACKEND_DIR, "..", "frontend")
ALLOWED_EXTENSIONS = [".wav", ".mp3", ".m4a", ".ogg"]


# ---------------- Load every ornament model once, at start-up ----------------
ORNAMENT_MODELS = []   # list of (display_name, model, config)

for config_path in sorted(glob.glob(os.path.join(MODELS_DIR, "*_config.json"))):
    with open(config_path) as f:
        model_file = json.load(f)["model_file"]
    model, config = load_ornament_model(config_path, os.path.join(MODELS_DIR, model_file))
    name = config["ornament"].capitalize()          # "meend" -> "Meend"
    ORNAMENT_MODELS.append((name, model, config))
    print(f"Loaded model: {name}")

if not ORNAMENT_MODELS:
    raise SystemExit("No models found in backend/models/. Add <name>_config.json and the .pth file.")


# ---------------- FastAPI app ----------------
app = FastAPI(title="Ornament Detection API")

# CORS lets the page call the API even if the page is opened from another port / file
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/predict")
def predict(audio: UploadFile = File(...)):
    # ---- 1. Check and save the uploaded file to a temporary location ----
    extension = os.path.splitext(audio.filename or "")[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Unsupported file type. Please upload WAV, MP3, M4A or OGG.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        temp_path = tmp.name

    try:
        # ---- 2. Load audio ----
        try:
            waveform = prep.load_audio(temp_path)
        except Exception:
            raise HTTPException(400, "Could not read this audio file. It may be damaged or in an unsupported format.")

        duration = len(waveform) / prep.SAMPLE_RATE
        if duration < 1.0:
            raise HTTPException(400, "Audio is too short. Please upload at least 1 second.")
        if duration > prep.MAX_DURATION_SECONDS:
            raise HTTPException(400, f"Audio is too long ({duration:.0f} s). Please upload at most {prep.MAX_DURATION_SECONDS} s.")

        # ---- 3. CREPE pitch + voiced / unvoiced ----
        times, f0, confidence = prep.extract_pitch(waveform)
        voiced = confidence >= prep.CONFIDENCE_THRESHOLD
        if voiced.sum() < 2:
            raise HTTPException(400, "No singing / pitch was detected in this recording.")
        f0 = prep.fill_unvoiced(f0, voiced)

        # ---- 4. Features: F0, velocity, acceleration ----
        features = prep.compute_features(times, f0)

        # ---- 5 + 6. One prediction per ornament model ----
        segments = None
        standardisation_mode = ""
        for name, model, config in ORNAMENT_MODELS:
            standardised, standardisation_mode = prep.standardise(features, voiced, config)
            frames, starts, voiced_fractions = prep.make_frames(standardised, voiced)

            with torch.no_grad():
                logits = model(torch.from_numpy(frames)).squeeze(1).numpy()

            # The model's own yes / no answer:
            #   logit > 0  -> the model says "Meend"
            #   logit <= 0 -> the model says "No Meend"
            predictions = logits > 0

            # Build the segment list once (same frames for every ornament)
            if segments is None:
                segments = [{
                    "start": round(start, 2),
                    "end": round(start + prep.FRAME_SIZE * prep.HOP_SECONDS, 2),
                    "voiced_fraction": round(vf, 2),
                    "silent": vf < prep.MIN_VOICED_FRACTION,   # mostly silence -> not analysed
                    "ornaments": {},
                } for start, vf in zip(starts, voiced_fractions)]

            # Silent frames are never marked as present
            for segment, is_present in zip(segments, predictions):
                segment["ornaments"][name] = {
                    "present": bool(is_present) and not segment["silent"],
                }

        # ---- 7. JSON response ----
        return {
            "duration": round(duration, 2),
            "standardisation": standardisation_mode,
            "results": segments,
        }

    finally:
        os.remove(temp_path)   # delete the temporary upload


# ---------------- Serve the website ----------------
@app.get("/ornaments")
def ornaments_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "ornaments.html"))

# Everything else (index.html, css, js, assets) comes from the frontend folder
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    print("\nOpen the website at:  http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
