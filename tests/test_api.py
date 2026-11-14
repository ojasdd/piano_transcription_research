"""Unit and integration tests for FastAPI backend service."""
import base64
import io
import os
import wave
import pytest
from fastapi.testclient import TestClient

# Set stub mode for tests
os.environ["PIANO_TRANSCRIBER_STUB"] = "1"

from deployment.api import app, MAX_FILE_SIZE_BYTES


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def dummy_wav_bytes():
    """Generate in-memory valid 16-bit PCM WAV audio bytes (0.1s tone)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 1600 samples of silence/zero
        wf.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


def test_health_endpoint(client):
    """Test /health status check."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["stub_mode"] is True
    assert "rbm" in data["supported_priors"]
    assert "none" in data["supported_priors"]


def test_transcribe_stub_success(client, dummy_wav_bytes):
    """Test /transcribe happy path against contract stub."""
    files = {"file": ("test_clip.wav", dummy_wav_bytes, "audio/wav")}
    data = {"prior": "rbm", "lam": "0.75"}
    
    resp = client.post("/transcribe", files=files, data=data)
    assert resp.status_code == 200
    payload = resp.json()
    
    assert "notes" in payload
    assert "frame_roll" in payload
    assert "frame_rate_hz" in payload
    assert "midi_base64" in payload
    assert "meta" in payload
    
    # Contract validation
    assert len(payload["frame_roll"][0]) == 88
    assert payload["meta"]["prior"] == "rbm"
    assert payload["meta"]["lam"] == 0.75
    
    # Check MIDI decodable
    midi_raw = base64.b64decode(payload["midi_base64"])
    assert len(midi_raw) > 0
    assert midi_raw.startswith(b"MThd")  # Standard MIDI header


@pytest.mark.parametrize("prior", ["none", "rbm", "nade", "made"])
def test_transcribe_all_priors(client, dummy_wav_bytes, prior):
    """Verify all allowed priors pass parameter validation."""
    files = {"file": ("test.wav", dummy_wav_bytes, "audio/wav")}
    resp = client.post("/transcribe", files=files, data={"prior": prior, "lam": "0.5"})
    assert resp.status_code == 200
    assert resp.json()["meta"]["prior"] == prior


def test_transcribe_invalid_prior(client, dummy_wav_bytes):
    """Verify unknown prior is rejected with 400 Bad Request."""
    files = {"file": ("test.wav", dummy_wav_bytes, "audio/wav")}
    resp = client.post("/transcribe", files=files, data={"prior": "transformer", "lam": "0.5"})
    assert resp.status_code == 400
    assert "Invalid prior" in resp.json()["detail"]


def test_transcribe_invalid_lambda(client, dummy_wav_bytes):
    """Verify lambda outside [0.0, 5.0] is rejected with 400 Bad Request."""
    files = {"file": ("test.wav", dummy_wav_bytes, "audio/wav")}
    resp = client.post("/transcribe", files=files, data={"prior": "rbm", "lam": "8.5"})
    assert resp.status_code == 400
    assert "Invalid lambda" in resp.json()["detail"]


def test_transcribe_unsupported_extension(client):
    """Verify unsupported file extensions are rejected with 400 Bad Request."""
    files = {"file": ("notes.txt", b"not audio data", "text/plain")}
    resp = client.post("/transcribe", files=files, data={"prior": "none", "lam": "0.0"})
    assert resp.status_code == 400
    assert "Unsupported file format" in resp.json()["detail"]


def test_transcribe_midi_stream_endpoint(client, dummy_wav_bytes):
    """Test /transcribe/midi endpoint returns raw downloadable MIDI binary."""
    files = {"file": ("test.wav", dummy_wav_bytes, "audio/wav")}
    resp = client.post("/transcribe/midi", files=files, data={"prior": "none", "lam": "0.0"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/midi")
    assert 'attachment; filename="transcription.mid"' in resp.headers["content-disposition"]
    assert resp.content.startswith(b"MThd")


def test_export_lilypond_endpoint(client):
    """Test /export/lilypond converts note list to LilyPond notation score."""
    sample_notes = [
        {"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80},
        {"pitch": 48, "onset": 0.0, "offset": 1.0, "velocity": 75},
    ]
    resp = client.post("/export/lilypond", json=sample_notes)
    assert resp.status_code == 200
    ly = resp.json()["lilypond"]
    assert '\\version "2.22.0"' in ly
    assert "PianoStaff" in ly
    assert "c'4" in ly  # pitch 60 -> c' in treble clef


def test_transcribe_size_limit_rejection(client, monkeypatch):
    """Verify files exceeding MAX_FILE_SIZE_BYTES are rejected with 413 Payload Too Large."""
    import deployment.api as api_mod
    monkeypatch.setattr(api_mod, "MAX_FILE_SIZE_BYTES", 100)  # artificially low 100 bytes limit
    
    big_data = b"0" * 500
    files = {"file": ("big.wav", big_data, "audio/wav")}
    resp = client.post("/transcribe", files=files, data={"prior": "none", "lam": "0.0"})
    assert resp.status_code == 413
    assert "exceeds maximum allowed size" in resp.json()["detail"]
