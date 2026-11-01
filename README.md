# Indian Classical Music Ornament Detection – Complete Application

Website (HTML / CSS / JavaScript) + FastAPI backend running the trained Meend 1D CNN.

```
ornament-detection-app/
├── backend/
│   ├── app.py              FastAPI server: POST /predict + serves the website
│   ├── preprocessing.py    Librosa → CREPE → velocity/acceleration → standardise → frames
│   ├── model.py            The 1D CNN (same layers as the report, 9,153 parameters)
│   ├── requirements.txt    Python packages
│   └── models/
│       ├── meend_CNN.pth       trained weights
│       └── meend_config.json   threshold 0.78, architecture, training settings
└── frontend/
    ├── index.html          Model demo (upload → prediction)
    ├── ornaments.html      Ornament audio demo
    ├── css/style.css
    ├── js/app.js           API_URL is at the top of this file
    └── assets/
        ├── audio/          meend.mp3 (add kanswar.mp3, murki.mp3)
        ├── reports/        preprocessing_report.pdf, model_report.pdf
        └── model_config.json
```

---

## How to run (one server does everything)

### 1. Install Python
Python 3.10, 3.11 or 3.12 from https://www.python.org
(on Windows tick **"Add Python to PATH"** during installation).

### 2. Open a terminal in the `backend` folder

```bash
cd ornament-detection-app/backend
```

### 3. Create a virtual environment (first time only)

Windows:
```bash
python -m venv venv
```
```bash
venv\Scripts\activate
```

Mac / Linux:
```bash
python3 -m venv venv
```
```bash
source venv/bin/activate
```

### 4. Install the packages (first time only, needs internet)

First the smaller CPU-only PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Then the rest:
```bash
pip install -r requirements.txt
```

### 5. Start the application

```bash
python app.py
```

You should see:
```
Loaded model: Meend  (threshold 0.78)
Open the website at:  http://localhost:8000
```

### 6. Open http://localhost:8000 in Chrome

- Choose an audio file → **Analyze Audio** → wait 10–60 s (CREPE pitch extraction on the CPU).
- The demo page is http://localhost:8000/ornaments

Stop the server with **Ctrl + C**. Next time only steps 2, 3 (activate line), 5 and 6 are needed.

**M4A files** need the free program ffmpeg (https://ffmpeg.org). WAV, MP3 and OGG work without it.

---

## What happens when you click "Analyze Audio"

```
Browser (js/app.js)
   FormData { audio: file }  ──POST /predict──►  backend/app.py
                                                    │
     1. Librosa: load, mono, 16 kHz                 │ preprocessing.py
     2. CREPE (full, Viterbi): F0 every 10 ms       │
     3. confidence < 0.5 → unvoiced (gaps filled)   │
     4. np.gradient → velocity, acceleration        │
     5. standardise each channel                    │
     6. frames 3 × 100, stride 50 (1 s, 50 %)       │
     7. 1D CNN → sigmoid → probability              │ model.py
     8. present = probability ≥ 0.78                │
     9. frames with < 50 % singing → "silence"      │
   JSON  ◄──────────────────────────────────────────┘
   Summary + table in the page
```

Response example:
```json
{
  "duration": 6.0,
  "standardisation": "per-recording statistics",
  "thresholds": { "Meend": 0.78 },
  "results": [
    { "start": 0.5, "end": 1.5, "voiced_fraction": 0.95, "silent": false,
      "ornaments": { "Meend": { "present": false, "confidence": 0.029 } } }
  ]
}
```

---

## Important: training-set mean and standard deviation

The report (Section 4.2.4 and 4.4.4) says the features must be standardised with the
**training-set** mean and std. These numbers are not stored in `meend_config.json`, so
the backend currently uses **per-recording** standardisation (mean/std of the uploaded
recording's singing), the same default as the report's prototype. The page shows which
mode was used.

To use the training statistics, compute them in your training notebook
(`X_train` shape = frames × 3 × 100, before standardisation):

```python
print(X_train.mean(axis=(0, 2)).tolist())   # feature_mean
print(X_train.std(axis=(0, 2)).tolist())    # feature_std
```

and add them to `backend/models/meend_config.json`:

```json
"feature_mean": [250.1, 0.3, 1.2],
"feature_std":  [80.5, 150.2, 9000.0],
```

(the numbers above are only an example). Restart `python app.py` – the page will then say
"Standardisation: training-set statistics".

Other settings that must match your training code are at the top of
`backend/preprocessing.py` (confidence threshold 0.5, how unvoiced samples are filled,
minimum voiced fraction).

---

## Adding Kanswar and Murki models later

Put the files in `backend/models/` using the same naming:

```
kanswar_CNN.pth   +  kanswar_config.json   ("ornament": "kanswar", "model_file": "kanswar_CNN.pth")
murki_CNN.pth     +  murki_config.json
```

Restart the server. A new column appears in the result table automatically.

## Other files

| What                   | Where                                           |
|------------------------|-------------------------------------------------|
| Preprocessing report   | `frontend/assets/reports/preprocessing_report.pdf` |
| Model report           | `frontend/assets/reports/model_report.pdf`      |
| Ornament example audio | `frontend/assets/audio/meend.mp3`, `kanswar.mp3`, `murki.mp3` |
| Backend URL            | `frontend/js/app.js`, line `const API_URL = ...` |

## Troubleshooting

| Problem | Fix |
|---|---|
| "Unable to connect to the ML model" | The terminal with `python app.py` must stay open. |
| `pip` / `python` not found | Reinstall Python with "Add to PATH" ticked. |
| Port 8000 already in use | Change `port=8000` at the bottom of `app.py` **and** `API_URL` in `js/app.js`. |
| "This format needs ffmpeg" | Install ffmpeg, or upload WAV / MP3. |
| Analysis is slow | Normal on CPU – about 10–30 s per minute of audio. |
