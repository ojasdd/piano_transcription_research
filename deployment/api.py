"""FastAPI backend service for Prior-Aware Piano Transcription."""
import base64
import os
import shutil
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root and src/ are on sys.path
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from piano_transcriber.transcriber import transcribe
from deployment.sheet_music import notes_to_lilypond

# Configuration constants
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aif", ".aiff"}
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_AUDIO_SIZE_BYTES", 25 * 1024 * 1024))  # 25 MB default
VALID_PRIORS = {"none", "rbm", "nade", "made"}
HTTP_413_CODE = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup checks."""
    is_stub = os.getenv("PIANO_TRANSCRIBER_STUB") == "1"
    print(f"[STARTUP] Prior-Aware Piano Transcriber API starting up. STUB_MODE={is_stub}")
    yield
    print("[SHUTDOWN] Prior-Aware Piano Transcriber API shutting down.")


app = FastAPI(
    title="Prior-Aware Piano Transcriber API",
    description="Acoustic CNN-BiGRU inference coupled with RBM/NADE/MADE frame-wise priors.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Status"])
def health() -> Dict[str, Any]:
    """Health check endpoint reporting backend readiness and configuration."""
    is_stub = os.getenv("PIANO_TRANSCRIBER_STUB") == "1"
    config_path = os.getenv("PIANO_TRANSCRIBER_CONFIG", "config.yaml")
    return {
        "status": "ok",
        "stub_mode": is_stub,
        "config_path": config_path,
        "max_upload_size_mb": MAX_FILE_SIZE_BYTES / (1024 * 1024),
        "supported_priors": list(VALID_PRIORS),
        "timestamp": time.time(),
    }


def _validate_request(file: UploadFile, prior: str, lam: float) -> str:
    """Validate incoming transcription arguments and file metadata."""
    if prior not in VALID_PRIORS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid prior '{prior}'. Must be one of: {sorted(list(VALID_PRIORS))}",
        )
    if not (0.0 <= float(lam) <= 5.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid lambda '{lam}'. Must be between 0.0 and 5.0.",
        )
    
    filename = file.filename or "audio.wav"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{suffix}'. Allowed formats: {sorted(list(ALLOWED_EXTENSIONS))}",
        )
    return suffix


def _save_upload_to_temp(file: UploadFile, suffix: str) -> str:
    """Safely stream uploaded content to a temporary file while enforcing max size limits."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    bytes_read = 0
    try:
        while chunk := file.file.read(1024 * 64):  # 64 KB chunks
            bytes_read += len(chunk)
            if bytes_read > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=HTTP_413_CODE,
                    detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES / (1024 * 1024):.1f} MB",
                )
            temp_file.write(chunk)
        temp_file.flush()
        return temp_file.name
    except Exception:
        temp_file.close()
        if os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass
        raise
    finally:
        temp_file.close()


@app.post("/transcribe", tags=["Transcription"])
async def api_transcribe(
    file: UploadFile = File(..., description="Audio file (.wav, .mp3, .flac, etc.)"),
    prior: str = Form("none", description="Prior model to use: none, rbm, nade, made"),
    lam: float = Form(0.5, description="Prior coupling strength lambda in [0.0, 5.0]"),
) -> Dict[str, Any]:
    """Transcribe an audio file into MIDI note events and frame piano roll.
    
    Returns:
    - notes: List of detected notes (pitch, onset, offset, velocity)
    - frame_roll: Binary T x 88 piano roll matrix for visualization
    - frame_rate_hz: Frame rate in Hz (e.g. 31.25)
    - midi_base64: Base64-encoded standard MIDI file bytes
    - meta: Duration, inference latency, prior settings
    """
    suffix = _validate_request(file, prior, lam)
    temp_path = _save_upload_to_temp(file, suffix)

    try:
        result = transcribe(temp_path, prior=prior, lam=float(lam))
        midi_b64 = base64.b64encode(result["midi_bytes"]).decode("ascii")

        return {
            "notes": result["notes"],
            "frame_roll": result["frame_roll"],
            "frame_rate_hz": result["frame_rate_hz"],
            "midi_base64": midi_b64,
            "meta": result["meta"],
        }
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Required model checkpoint missing on server: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}",
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@app.post("/transcribe/midi", tags=["Transcription"])
async def api_transcribe_midi(
    file: UploadFile = File(..., description="Audio file (.wav, .mp3, .flac, etc.)"),
    prior: str = Form("none", description="Prior model to use: none, rbm, nade, made"),
    lam: float = Form(0.5, description="Prior coupling strength lambda in [0.0, 5.0]"),
):
    """Transcribe and return raw MIDI file stream directly for download."""
    suffix = _validate_request(file, prior, lam)
    temp_path = _save_upload_to_temp(file, suffix)

    try:
        result = transcribe(temp_path, prior=prior, lam=float(lam))
        midi_bytes = result["midi_bytes"]
        return Response(
            content=midi_bytes,
            media_type="audio/midi",
            headers={"Content-Disposition": 'attachment; filename="transcription.mid"'},
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@app.post("/export/lilypond", tags=["Sheet Music"])
async def api_export_lilypond(
    notes: List[Dict[str, Any]],
    title: Optional[str] = "Piano Transcription",
) -> Dict[str, str]:
    """Convert note list to LilyPond sheet music score text."""
    score = notes_to_lilypond(notes, title=title or "Piano Transcription")
    return {"lilypond": score}
