"""Smoke test exercising end-to-end transcription pipeline and UI components."""
import io
import os
import tempfile
import wave
import numpy as np
import pytest

os.environ["PIANO_TRANSCRIBER_STUB"] = "1"

from piano_transcriber.transcriber import transcribe
from deployment.frontend import synthesize_notes_audio, plot_piano_roll, midi_pitch_to_name
from deployment.sheet_music import notes_to_lilypond


def create_sample_wav(duration_s: float = 0.5, freq: float = 440.0, sr: int = 16000) -> str:
    """Generate a clean temporary WAV file containing a sine wave."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(sine.tobytes())
    tmp.close()
    return tmp.name


def test_smoke_transcribe_clip():
    """Smoke test running transcription against a temporary sample audio file."""
    wav_path = create_sample_wav(0.5, 440.0)
    try:
        res = transcribe(wav_path, prior="rbm", lam=0.5)
        
        # Verify contract schema
        assert isinstance(res, dict)
        assert set(res.keys()) == {"notes", "frame_roll", "frame_rate_hz", "midi_bytes", "meta"}
        
        # Verify notes
        notes = res["notes"]
        assert len(notes) > 0
        for n in notes:
            assert "pitch" in n and 21 <= n["pitch"] <= 108
            assert "onset" in n and n["onset"] >= 0
            assert "offset" in n and n["offset"] >= n["onset"]
            assert "velocity" in n and 0 <= n["velocity"] <= 127
            
        # Verify frame roll shape (T x 88)
        roll = res["frame_roll"]
        assert isinstance(roll, list)
        assert len(roll) > 0
        assert len(roll[0]) == 88
        
        # Verify MIDI bytes
        midi_bytes = res["midi_bytes"]
        assert isinstance(midi_bytes, bytes)
        assert len(midi_bytes) > 0
        assert midi_bytes.startswith(b"MThd")
        
        # Verify metadata
        meta = res["meta"]
        assert meta["prior"] == "rbm"
        assert meta["lam"] == 0.5
        assert meta["stub"] is True
        
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def test_smoke_synthesize_audio_playback():
    """Smoke test audio synthesizer generating valid WAV bytes from notes."""
    notes = [
        {"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 90},
        {"pitch": 64, "onset": 0.2, "offset": 0.7, "velocity": 85},
        {"pitch": 67, "onset": 0.4, "offset": 0.9, "velocity": 80},
    ]
    wav_bytes = synthesize_notes_audio(notes, sr=16000)
    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 1000
    assert wav_bytes.startswith(b"RIFF")


def test_smoke_plot_piano_roll():
    """Smoke test Plotly piano roll generation."""
    notes = [
        {"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 90},
    ]
    fig = plot_piano_roll(notes, title="Smoke Test Roll")
    assert fig is not None
    assert len(fig.data) == 1
    
    # Test empty notes plot
    empty_fig = plot_piano_roll([], title="Empty Roll")
    assert empty_fig is not None


def test_smoke_lilypond_export():
    """Smoke test LilyPond sheet music generation."""
    notes = [
        {"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80},  # Middle C -> upper staff
        {"pitch": 48, "onset": 0.0, "offset": 1.0, "velocity": 80},  # C3 -> lower staff
    ]
    ly_str = notes_to_lilypond(notes, title="Smoke Score")
    assert '\\new Staff = "upper"' in ly_str
    assert '\\new Staff = "lower"' in ly_str
    assert "c'4" in ly_str
    assert "c2" in ly_str


def test_smoke_polyphonic_bach_prelude():
    """Verify polyphonic note detection and duration on bach_prelude_pattern.wav."""
    audio_path = "test_audios/bach_prelude_pattern.wav"
    if not os.path.exists(audio_path):
        pytest.skip("bach_prelude_pattern.wav not found")
    res = transcribe(audio_path, prior="none")
    assert res["meta"]["duration_s"] > 3.0
    notes = res["notes"]
    assert len(notes) >= 8  # Multiple polyphonic notes detected
    pitches = set(n["pitch"] for n in notes)
    assert len(pitches) >= 4  # Multiple distinct pitches detected across octave ranges
