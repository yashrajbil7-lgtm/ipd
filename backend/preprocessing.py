"""
preprocessing.py  -  Turns an audio file into CNN input frames
(Report, Sections 4.2.1 - 4.2.4)

    1. Librosa      : load audio, convert to mono, resample to 16 kHz
    2. CREPE        : fundamental frequency (F0) + confidence every 10 ms
    3. Unvoiced     : samples with confidence < 0.5 are treated as unvoiced
    4. NumPy        : velocity = dF0/dt , acceleration = dv/dt  (np.gradient)
    5. Standardise  : z = (x - mean) / std   for each of the 3 channels
    6. Framing      : frames of 100 steps (1 s) with stride 50 (50 % overlap)

CREPE is run with "torchcrepe", the PyTorch version of the same pretrained
CREPE network ("full" model, Viterbi smoothing). This avoids installing
TensorFlow next to PyTorch.
"""

import os
import shutil
import subprocess

import numpy as np
import librosa
import torch
import torchcrepe

# ---------------- Settings (change here if your training used other values) ----------------
SAMPLE_RATE = 16000          # CREPE works at 16 kHz
HOP_SECONDS = 0.01           # one F0 value every 10 ms
CONFIDENCE_THRESHOLD = 0.5   # below this a sample is "unvoiced" (silence / tanpura)
FRAME_SIZE = 100             # 100 steps = 1 second
FRAME_STRIDE = 50            # 50 steps = 0.5 second  ->  50 % overlap
MIN_VOICED_FRACTION = 0.5    # frames with less singing than this are skipped as "silence"
MAX_DURATION_SECONDS = 300   # longer files are rejected (CREPE on CPU is slow)


def load_audio(file_path):
    """Step 1: load any audio file as a mono 16 kHz waveform.
    WAV, MP3 and OGG are read directly. M4A cannot be read by the Python audio
    library, so it is first converted to WAV with the free "ffmpeg" program."""
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        return audio
    except Exception:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("This format needs ffmpeg. Install ffmpeg or upload WAV / MP3.")
        wav_path = file_path + ".converted.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", file_path, wav_path], check=True)
        try:
            audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        return audio


def extract_pitch(audio):
    """Step 2: CREPE pitch. Returns time (s), F0 (Hz) and confidence, one value per 10 ms."""
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    hop_length = int(SAMPLE_RATE * HOP_SECONDS)   # 160 samples

    f0, confidence = torchcrepe.predict(
        audio_tensor, SAMPLE_RATE, hop_length,
        model="full",
        decoder=torchcrepe.decode.viterbi,   # Viterbi smoothing, as in the report
        return_periodicity=True,
        batch_size=512,
        device="cpu",
    )
    f0 = f0.squeeze(0).numpy()
    confidence = confidence.squeeze(0).numpy()
    times = np.arange(len(f0)) * HOP_SECONDS
    return times, f0, confidence


def fill_unvoiced(f0, voiced):
    """Step 3: replace unvoiced F0 values by linear interpolation between voiced ones,
    so that silence does not create huge fake jumps in velocity / acceleration."""
    f0 = f0.copy()
    voiced_idx = np.where(voiced)[0]
    unvoiced_idx = np.where(~voiced)[0]
    f0[unvoiced_idx] = np.interp(unvoiced_idx, voiced_idx, f0[voiced_idx])
    return f0


def compute_features(times, f0):
    """Step 4: stack [F0, velocity, acceleration] -> array of shape (3, n)."""
    velocity = np.gradient(f0, times)            # Hz / s
    acceleration = np.gradient(velocity, times)  # Hz / s^2
    return np.stack([f0, velocity, acceleration])


def standardise(features, voiced, config):
    """Step 5: z = (x - mean) / std for each channel.

    Best:  the TRAINING-SET mean and std. Add them to the model config as
           "feature_mean": [m_f0, m_v, m_a]  and  "feature_std": [s_f0, s_v, s_a]
    If they are missing, the mean/std of this recording's singing is used
    ("per-recording" standardisation, the default in the report's prototype)."""
    if "feature_mean" in config and "feature_std" in config:
        mean = np.array(config["feature_mean"]).reshape(3, 1)
        std = np.array(config["feature_std"]).reshape(3, 1)
        mode = "training-set statistics"
    else:
        singing = features[:, voiced]
        mean = singing.mean(axis=1, keepdims=True)
        std = singing.std(axis=1, keepdims=True)
        mode = "per-recording statistics"

    std[std == 0] = 1.0   # avoid division by zero
    return (features - mean) / std, mode


def make_frames(features, voiced):
    """Step 6: cut into 100-step frames with 50-step stride.
    Frame k covers steps 50k ... 50k+99  ->  time 0.5k s ... 0.5k+1 s."""
    frames, starts, voiced_fractions = [], [], []
    n = features.shape[1]
    for start in range(0, n - FRAME_SIZE + 1, FRAME_STRIDE):
        frames.append(features[:, start:start + FRAME_SIZE])
        starts.append(start * HOP_SECONDS)
        voiced_fractions.append(float(voiced[start:start + FRAME_SIZE].mean()))
    return np.array(frames, dtype=np.float32), starts, voiced_fractions
