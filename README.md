---
title: Prior Aware Piano Transcriber
emoji: 🎹
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Prior-Aware Piano Transcription

Research scaffold and deployment stack for MAESTRO piano transcription using a CNN-BiGRU acoustic model coupled with **generative frame-wise priors** (**RBM**, **NADE**, and **MADE**) to resolve polyphonic chord ambiguity and suppress spurious note frames.

---

## 1. Quick Start

### Setup Environment
```bash
python -m venv .venv
# Activate environment:
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-deploy.txt
pip install -e .
```

### Run Tests
```bash
# Runs contract tests, FastAPI integration tests, and end-to-end smoke tests
pytest -v
```

---

## 2. Deployment (Antigravity Deliverables)

### D1: FastAPI Backend Service
The backend provides high-performance audio validation, tempfile cleanup, size limiting, and transcription endpoints:
```bash
# Run with test stub (before model training finishes)
export PIANO_TRANSCRIBER_STUB=1   # Linux/macOS
# $env:PIANO_TRANSCRIBER_STUB="1"  # PowerShell

uvicorn deployment.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation is live at `http://localhost:8000/docs`.

**Endpoints:**
- `GET /health`: Health status, stub mode, supported priors, and size limits.
- `POST /transcribe`: Multipart audio upload with `prior` (`none`, `rbm`, `nade`, `made`) and `lam` ($\lambda \in [0.0, 5.0]$). Returns detected note events, $T \times 88$ piano roll matrix, base64-encoded MIDI, and performance metadata.
- `POST /transcribe/midi`: Direct binary stream returning downloadable `.mid` file.
- `POST /export/lilypond`: Generates LilyPond notation score text (.ly) split across treble and bass staves.

---

### D2: Streamlit Interactive Frontend
The frontend features:
- File upload (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`) and live microphone recording.
- **With-Prior vs Without-Prior toggle** and comparative side-by-side mode.
- Interactive Plotly Piano-Roll visualizer with hover info (musical note names, onset/offset times, duration, velocity).
- **Pure-NumPy audio synthesizer** for instant in-browser playback of transcribed notes.
- MIDI file download button (`.mid`).
- LilyPond sheet music generation and score preview (Stretch D6).
- Dual execution: directly in-process or communicating over HTTP with FastAPI.

To launch:
```bash
streamlit run app.py
```
Open `http://localhost:8501` (or `http://localhost:7860`).

---

### D3: Containerization (Docker)
The provided multi-stage, CPU-only `Dockerfile` installs system audio libraries (`libsndfile1`, `ffmpeg`), sets up a non-root user, and pins dependencies:

```bash
# Build Docker image
docker build -t piano-transcriber:latest .

# Run Streamlit frontend (default port 7860)
docker run -p 7860:7860 -e PIANO_TRANSCRIBER_STUB=1 piano-transcriber:latest

# Or run FastAPI backend on port 8000
docker run -p 8000:8000 -e PIANO_TRANSCRIBER_STUB=1 piano-transcriber:latest \
    uvicorn deployment.api:app --host 0.0.0.0 --port 8000
```

---

### D4: Hugging Face Spaces Deployment
The repository is directly formatted for 1-click deployment on [Hugging Face Spaces](https://huggingface.co/spaces):
1. Create a new Space on Hugging Face:
   - SDK: **Streamlit**
   - Hardware: **CPU Basic** (Free tier)
2. Add your repository remote and push:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
3. In your Space **Settings -> Variables and secrets**:
   - Set `PIANO_TRANSCRIBER_STUB = 1` while training checkpoints are in progress.
   - Once trained, add checkpoints under `models/acoustic.pt`, `models/rbm.pt`, `models/nade.pt`, `models/made.pt` (using Git LFS for large binaries).

---

### D5: Test Suite
The automated test suite verifies:
- `tests/test_contract.py`: Shape and contract compliance of `transcribe()`.
- `tests/test_api.py`: FastAPI endpoints, parameter validation, file formats, size rejection (413), and binary MIDI streaming.
- `tests/test_smoke.py`: End-to-end smoke test on a synthesized audio sample, audio playback synthesis, Plotly roll rendering, and LilyPond generation.

Execute with:
```bash
pytest -v
```

---

## 3. The Contract

```python
from piano_transcriber.transcriber import transcribe

result = transcribe(audio_path="piano.wav", prior="rbm", lam=0.5)
# Returns:
# {
#   "notes": [{"pitch": 60, "onset": 1.23, "offset": 1.80, "velocity": 80}, ...],
#   "frame_roll": [[...88 values...], ...],  # T x 88 binary matrix
#   "frame_rate_hz": 31.25,
#   "midi_bytes": b"...",
#   "meta": {"prior": "rbm", "lam": 0.5, "duration_s": 28.4, "latency_s": 2.1}
# }
```

---

## 4. Training Deliverables (Model Research)
- N1–N7: Notebook entry points (`01_data_eda.ipynb` through `07_error_analysis.ipynb`).
- S1: `src/piano_transcriber/` reusable package.
- S2: Inference contract wrapper in `transcriber.py`.
- R1: Reports and evaluation logs.
