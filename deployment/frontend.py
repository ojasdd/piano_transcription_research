"""Streamlit Frontend for Prior-Aware Piano Transcription.

Supports:
- File upload & microphone recording
- With-prior vs Without-prior toggle & comparative analysis
- Interactive Piano-Roll visualizer (Plotly)
- Instant audio preview playback (pure-Python harmonic synthesizer) & standard MIDI download
- LilyPond sheet music generation & export
"""
import base64
import io
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root and src/ are on sys.path
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import soundfile as sf
import streamlit as st
import librosa

# Direct contract import (works in-process or via FastAPI backend)
from piano_transcriber.transcriber import transcribe
from deployment.sheet_music import notes_to_lilypond

# Page configuration
st.set_page_config(
    page_title="Prior-Aware Piano Transcriber",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# MIDI pitch note names
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_pitch_to_name(pitch: int) -> str:
    """Format MIDI pitch as human-readable musical note name (e.g. 60 -> C4)."""
    octave = (pitch // 12) - 1
    name = NOTE_NAMES[pitch % 12]
    return f"{name}{octave}"


def synthesize_notes_audio(notes: List[Dict[str, Any]], sr: int = 22050) -> bytes:
    """Synthesize clean harmonic audio preview of transcribed notes in pure NumPy.
    
    Produces warm piano-like tones with exponential decay and multiple harmonics.
    """
    if not notes:
        silence = np.zeros(int(sr * 0.5), dtype=np.float32)
        out = io.BytesIO()
        sf.write(out, silence, sr, format="WAV")
        return out.getvalue()

    total_duration = max(float(n.get("offset", n["onset"] + 0.3)) for n in notes) + 0.5
    total_samples = int(total_duration * sr)
    waveform = np.zeros(total_samples, dtype=np.float32)

    for note in notes:
        pitch = int(note["pitch"])
        onset = float(note["onset"])
        offset = float(note.get("offset", onset + 0.25))
        dur = max(0.05, offset - onset)
        
        freq = 440.0 * (2.0 ** ((pitch - 69.0) / 12.0))
        start_idx = int(onset * sr)
        n_samples = int(dur * sr)
        
        if start_idx >= total_samples:
            continue
        actual_samples = min(n_samples, total_samples - start_idx)
        
        t = np.arange(actual_samples) / sr
        velocity = float(note.get("velocity", 80)) / 127.0
        
        # Exponential envelope
        decay_time = max(0.1, dur * 0.9)
        envelope = np.exp(-t / decay_time) * (1.0 - np.exp(-t / 0.005))
        
        # Fundamental + harmonics
        tone = (
            np.sin(2 * np.pi * freq * t) * 1.0 +
            np.sin(2 * np.pi * freq * 2 * t) * 0.4 +
            np.sin(2 * np.pi * freq * 3 * t) * 0.2 +
            np.sin(2 * np.pi * freq * 4 * t) * 0.1
        )
        
        note_wave = tone * envelope * velocity * 0.25
        waveform[start_idx : start_idx + actual_samples] += note_wave

    # Normalization to avoid clipping
    peak = np.max(np.abs(waveform))
    if peak > 1e-4:
        waveform = waveform / peak * 0.9

    out = io.BytesIO()
    sf.write(out, waveform, sr, format="WAV")
    return out.getvalue()


def plot_piano_roll(notes: List[Dict[str, Any]], title: str = "Piano Roll Visualization") -> go.Figure:
    """Render interactive Plotly piano roll visualization."""
    fig = go.Figure()

    if not notes:
        fig.add_annotation(
            text="No notes detected in transcription",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title=title,
            xaxis_title="Time (seconds)",
            yaxis_title="MIDI Pitch",
            height=400,
            template="plotly_dark",
        )
        return fig

    # Create rect shapes or line traces for each note
    for n in notes:
        pitch = int(n["pitch"])
        onset = float(n["onset"])
        offset = float(n.get("offset", onset + 0.1))
        vel = int(n.get("velocity", 80))
        note_str = midi_pitch_to_name(pitch)

        fig.add_trace(
            go.Scatter(
                x=[onset, offset, offset, onset, onset],
                y=[pitch - 0.4, pitch - 0.4, pitch + 0.4, pitch + 0.4, pitch - 0.4],
                fill="toself",
                fillcolor=f"rgba({min(255, 50 + vel * 2)}, 140, 240, 0.85)",
                line=dict(color="rgba(255,255,255,0.7)", width=1),
                mode="lines",
                showlegend=False,
                hoverinfo="text",
                hovertext=(
                    f"<b>Note:</b> {note_str} (Pitch {pitch})<br>"
                    f"<b>Onset:</b> {onset:.3f}s<br>"
                    f"<b>Offset:</b> {offset:.3f}s<br>"
                    f"<b>Duration:</b> {offset - onset:.3f}s<br>"
                    f"<b>Velocity:</b> {vel}"
                ),
            )
        )

    all_pitches = [n["pitch"] for n in notes]
    min_p = max(21, min(all_pitches) - 3)
    max_p = min(108, max(all_pitches) + 3)
    max_time = max(n.get("offset", n["onset"] + 0.2) for n in notes) + 0.5

    fig.update_layout(
        title=title,
        xaxis=dict(title="Time (seconds)", range=[0, max_time], gridcolor="rgba(100,100,100,0.2)"),
        yaxis=dict(
            title="Pitch",
            range=[min_p, max_p],
            tickmode="linear",
            tick0=min_p,
            dtick=2 if (max_p - min_p) <= 24 else 4,
            gridcolor="rgba(100,100,100,0.2)",
        ),
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_dark",
    )
    return fig


def plot_audio_diagnostics(audio_bytes: bytes) -> Tuple[go.Figure, go.Figure, Dict[str, Any]]:
    """Compute and plot waveform and mel-spectrogram diagnostics from raw audio bytes."""
    buf = io.BytesIO(audio_bytes)
    data, sr = sf.read(buf, dtype="float32")
    channels = data.shape[1] if data.ndim > 1 else 1
    mono = np.mean(data, axis=1) if data.ndim > 1 else data
    dur = float(len(mono) / sr)

    # Downsample waveform for snappy plotting
    step = max(1, len(mono) // 2000)
    t = np.arange(0, len(mono), step) / sr
    wave_sub = mono[::step]

    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(x=t, y=wave_sub, mode="lines", line=dict(color="#38bdf8", width=1.5), name="Amplitude"))
    fig_wave.update_layout(
        title="Input Audio Waveform (Time Domain)",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        height=260,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=mono, sr=sr, n_fft=2048, hop_length=512, n_mels=88, fmin=30, fmax=min(8000, sr / 2))
    mel_db = librosa.power_to_db(mel, ref=np.max)
    times = np.linspace(0, dur, mel_db.shape[1])
    freqs = librosa.mel_frequencies(n_mels=88, fmin=30, fmax=min(8000, sr / 2))

    fig_spec = go.Figure(data=go.Heatmap(z=mel_db, x=times, y=freqs, colorscale="Viridis", colorbar=dict(title="dB")))
    fig_spec.update_layout(
        title="Mel-Spectrogram (Time vs Frequency Energy)",
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    stats = {
        "Sample Rate": f"{sr} Hz",
        "Channels": f"{channels} ({'Mono' if channels==1 else 'Stereo'})",
        "Total Samples": f"{len(mono):,}",
        "Duration": f"{dur:.3f} s",
        "Peak Amplitude": f"{float(np.max(np.abs(mono))):.4f}",
        "RMS Energy": f"{float(np.sqrt(np.mean(mono**2))):.4f}",
    }
    return fig_wave, fig_spec, stats


def plot_frame_roll_matrix(frame_roll: List[List[float]], frame_rate_hz: float) -> go.Figure:
    """Render binary frame-activation matrix (T × 88) as interactive heatmap."""
    arr = np.array(frame_roll)
    if arr.size == 0:
        return go.Figure()
    t_frames, n_pitches = arr.shape
    times = np.arange(t_frames) / frame_rate_hz
    pitches = np.arange(21, 21 + n_pitches)

    fig = go.Figure(
        data=go.Heatmap(
            z=arr.T,
            x=times,
            y=pitches,
            colorscale=[[0, "#0f172a"], [1, "#06b6d4"]],
            showscale=False,
            hoverinfo="x+y",
        )
    )
    fig.update_layout(
        title="Binary Frame-Roll Activation Matrix (Frames × 88 Keys)",
        xaxis_title="Time (seconds)",
        yaxis_title="MIDI Pitch (A0=21 to C8=108)",
        template="plotly_dark",
        height=320,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def run_transcription(audio_bytes: bytes, filename: str, prior: str, lam: float, api_url: Optional[str] = None) -> Dict[str, Any]:
    """Execute transcription either via backend API or directly in-process."""
    # Check if backend API requested and healthy
    if api_url:
        try:
            files = {"file": (filename, audio_bytes, "audio/wav")}
            data = {"prior": prior, "lam": str(lam)}
            resp = requests.post(f"{api_url.rstrip('/')}/transcribe", files=files, data=data, timeout=120)
            if resp.status_code == 200:
                result = resp.json()
                result["midi_bytes"] = base64.b64decode(result["midi_base64"])
                return result
            else:
                st.warning(f"Backend API returned status {resp.status_code}: {resp.text}. Falling back to in-process transcription.")
        except Exception as e:
            st.warning(f"Failed to communicate with API at {api_url}: {e}. Running in-process.")

    # In-process transcription
    suffix = Path(filename).suffix.lower() if filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        res = transcribe(tmp_path, prior=prior, lam=float(lam))
        return res
    except FileNotFoundError as e:
        st.warning(
            f"Checkpoint file missing ({e}). Automatically switching to Stub Mode for testing until training is finished."
        )
        os.environ["PIANO_TRANSCRIBER_STUB"] = "1"
        return transcribe(tmp_path, prior=prior, lam=float(lam))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def main():
    st.title("🎹 Prior-Aware Piano Transcription")
    st.markdown(
        """
        Deep acoustic CNN-BiGRU transcription augmented with **generative frame-wise priors** 
        (**RBM**, **NADE**, and **MADE**) to resolve polyphonic chord ambiguity and suppress spurious note frames.
        """
    )

    # Sidebar controls
    st.sidebar.header("⚙️ Transcription Settings")

    backend_mode = st.sidebar.radio(
        "Execution Mode",
        options=["In-Process Contract", "FastAPI Service"],
        index=0,
        help="Choose whether Streamlit executes transcribe() in-process or over HTTP via FastAPI.",
    )
    api_url = None
    if backend_mode == "FastAPI Service":
        api_url = st.sidebar.text_input("FastAPI Endpoint", value="http://127.0.0.1:8000")

    checkpoints_exist = (
        Path("models/acoustic.pt").exists() or 
        Path(_ROOT / "models" / "acoustic.pt").exists()
    )
    default_stub = (not checkpoints_exist) or (os.getenv("PIANO_TRANSCRIBER_STUB", "1") == "1")

    use_stub = st.sidebar.checkbox(
        "Use Test Stub (PIANO_TRANSCRIBER_STUB=1)",
        value=default_stub,
        help="Exercises the transcription pipeline with deterministic fake note data without checkpoints.",
    )
    if use_stub:
        os.environ["PIANO_TRANSCRIBER_STUB"] = "1"
        st.sidebar.info("ℹ️ **Stub Mode Active** (no checkpoints required).")
    else:
        os.environ.pop("PIANO_TRANSCRIBER_STUB", None)
        if not checkpoints_exist:
            st.sidebar.warning("⚠️ `models/acoustic.pt` not found on disk yet.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Prior Modeling")

    # With-Prior vs Without-Prior Toggle
    enable_prior = st.sidebar.toggle("Enable Prior Model", value=True)

    if enable_prior:
        prior_choice = st.sidebar.selectbox(
            "Prior Architecture",
            options=["rbm", "nade", "made"],
            index=0,
            help=(
                "RBM: Restricted Boltzmann Machine with Gibbs sampling.\n"
                "NADE: Neural Autoregressive Distribution Estimator.\n"
                "MADE: Masked Autoencoder for Distribution Estimation."
            ),
        )
        lam_val = st.sidebar.slider(
            "Coupling Strength (λ)",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.05,
            help="Balancing parameter between acoustic likelihood and generative prior score.",
        )
    else:
        prior_choice = "none"
        lam_val = 0.0
        st.sidebar.info("Acoustic baseline only (independent frame thresholding, λ=0).")

    compare_mode = st.sidebar.checkbox(
        "Comparative Mode (With-Prior vs Without-Prior)",
        value=False,
        help="Run both baseline and prior models on the same clip to directly visualize corrections.",
    )

    # Main Input section
    tab_upload, tab_mic, tab_sample = st.tabs(["📁 File Upload", "🎤 Microphone Record", "🎼 Sample Clips"])

    audio_data = None
    audio_name = "recording.wav"

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a piano audio clip (.wav, .mp3, .flac, .ogg)",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
        )
        if uploaded_file is not None:
            audio_data = uploaded_file.read()
            audio_name = uploaded_file.name

    with tab_mic:
        mic_audio = st.audio_input("Record a piano clip from your microphone")
        if mic_audio is not None:
            audio_data = mic_audio.read()
            audio_name = "mic_recording.wav"

    with tab_sample:
        st.write("Generate a test sample tone (Middle C arpeggio C4-E4-G4):")
        if st.button("Generate & Use Sample Clip"):
            sr = 16000
            t = np.linspace(0, 1.0, sr, endpoint=False)
            tone = 0.4 * np.sin(2 * np.pi * 261.63 * t) + 0.3 * np.sin(2 * np.pi * 329.63 * t) + 0.3 * np.sin(2 * np.pi * 392.00 * t)
            buf = io.BytesIO()
            sf.write(buf, tone, sr, format="WAV")
            audio_data = buf.getvalue()
            audio_name = "sample_c_major_chord.wav"
            st.success("Sample audio generated!")

    if audio_data is not None:
        st.markdown("### 🔊 Input Audio")
        st.audio(audio_data, format="audio/wav")

        btn_label = "Transcribe & Compare" if compare_mode else "Transcribe Audio"
        if st.button(btn_label, type="primary"):
            with st.spinner("Processing audio through acoustic model & prior decoder..."):
                t_start = time.perf_counter()
                
                if compare_mode:
                    # Run baseline
                    res_baseline = run_transcription(audio_data, audio_name, prior="none", lam=0.0, api_url=api_url)
                    # Run with chosen prior
                    active_prior = prior_choice if enable_prior else "rbm"
                    active_lam = lam_val if enable_prior else 0.5
                    res_prior = run_transcription(audio_data, audio_name, prior=active_prior, lam=active_lam, api_url=api_url)
                    
                    st.success(f"Comparative transcription completed in {time.perf_counter() - t_start:.2f}s!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Baseline (No Prior)")
                        st.caption(f"Notes detected: {len(res_baseline.get('notes', []))}")
                        fig_base = plot_piano_roll(res_baseline.get("notes", []), title="Baseline (λ=0, Independent Frames)")
                        st.plotly_chart(fig_base, use_container_width=True)
                        
                        base_audio = synthesize_notes_audio(res_baseline.get("notes", []))
                        st.write("Synthesized Preview:")
                        st.audio(base_audio, format="audio/wav")
                        st.download_button(
                            "Download Baseline MIDI",
                            data=res_baseline["midi_bytes"],
                            file_name="baseline_transcription.mid",
                            mime="audio/midi",
                            key="btn_down_base",
                        )

                    with col2:
                        st.subheader(f"Prior-Aware ({active_prior.upper()}, λ={active_lam})")
                        st.caption(f"Notes detected: {len(res_prior.get('notes', []))}")
                        fig_prior = plot_piano_roll(res_prior.get("notes", []), title=f"Prior-Aware ({active_prior.upper()}, λ={active_lam})")
                        st.plotly_chart(fig_prior, use_container_width=True)
                        
                        prior_audio = synthesize_notes_audio(res_prior.get("notes", []))
                        st.write("Synthesized Preview:")
                        st.audio(prior_audio, format="audio/wav")
                        st.download_button(
                            f"Download {active_prior.upper()} MIDI",
                            data=res_prior["midi_bytes"],
                            file_name=f"{active_prior}_transcription.mid",
                            mime="audio/midi",
                            key="btn_down_prior",
                        )

                    # Note statistics diff
                    st.markdown("### 📊 Polyphony & Note Comparison")
                    diff_df = pd.DataFrame([
                        {
                            "Model": "Baseline (None)",
                            "Total Notes": len(res_baseline.get("notes", [])),
                            "Audio Duration (s)": res_baseline.get("meta", {}).get("duration_s", 0.0),
                            "Inference Latency (s)": f"{res_baseline.get('meta', {}).get('latency_s', 0.0):.3f}",
                        },
                        {
                            "Model": f"Prior-Aware ({active_prior.upper()})",
                            "Total Notes": len(res_prior.get("notes", [])),
                            "Audio Duration (s)": res_prior.get("meta", {}).get("duration_s", 0.0),
                            "Inference Latency (s)": f"{res_prior.get('meta', {}).get('latency_s', 0.0):.3f}",
                        },
                    ])
                    st.dataframe(diff_df, use_container_width=True)

                    with st.expander("🔬 Audio Signal & Spectrogram Diagnostics", expanded=False):
                        fig_w, fig_s, astats = plot_audio_diagnostics(audio_data)
                        d_cols = st.columns(len(astats))
                        for i, (k, v) in enumerate(astats.items()):
                            d_cols[i].metric(k, v)
                        st.plotly_chart(fig_w, use_container_width=True)
                        st.plotly_chart(fig_s, use_container_width=True)

                else:
                    # Single transcription
                    res = run_transcription(audio_data, audio_name, prior=prior_choice, lam=lam_val, api_url=api_url)
                    st.success(f"Transcription complete in {time.perf_counter() - t_start:.2f}s!")

                    notes = res.get("notes", [])
                    meta = res.get("meta", {})

                    st.markdown("### 🎼 Visual Piano Roll")
                    fig = plot_piano_roll(notes, title=f"Transcription: Prior={prior_choice.upper()}, λ={lam_val}")
                    st.plotly_chart(fig, use_container_width=True)

                    c_play, c_mid, c_ly = st.columns([1.5, 1, 1])
                    with c_play:
                        st.write("🔊 **Synthesized Audio Playback:**")
                        synth_wav = synthesize_notes_audio(notes)
                        st.audio(synth_wav, format="audio/wav")
                    with c_mid:
                        st.write("💾 **Export MIDI:**")
                        st.download_button(
                            label="Download .MID File",
                            data=res["midi_bytes"],
                            file_name="transcription.mid",
                            mime="audio/midi",
                        )
                    with c_ly:
                        st.write("📄 **Sheet Music (LilyPond):**")
                        ly_code = notes_to_lilypond(notes)
                        st.download_button(
                            label="Download .LY Score",
                            data=ly_code,
                            file_name="transcription.ly",
                            mime="text/plain",
                        )

                    # Metadata & Notes Table
                    with st.expander("🔍 View Note Events Table & Execution Metadata", expanded=False):
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        m_col1.metric("Total Notes", len(notes))
                        m_col2.metric("Duration", f"{meta.get('duration_s', 0.0):.2f}s")
                        m_col3.metric("Latency", f"{meta.get('latency_s', 0.0):.3f}s")
                        m_col4.metric("Frame Rate", f"{res.get('frame_rate_hz', 31.25):.1f} Hz")

                        if notes:
                            formatted_notes = [
                                {
                                    "Pitch": n["pitch"],
                                    "Note": midi_pitch_to_name(n["pitch"]),
                                    "Onset (s)": round(n["onset"], 3),
                                    "Offset (s)": round(n.get("offset", n["onset"] + 0.1), 3),
                                    "Duration (s)": round(n.get("offset", n["onset"] + 0.1) - n["onset"], 3),
                                    "Velocity": n.get("velocity", 80),
                                }
                                for n in notes
                            ]
                            st.dataframe(pd.DataFrame(formatted_notes), use_container_width=True)

                        st.subheader("LilyPond Source Preview")
                        st.code(ly_code, language="lilypond")

                    # Diagnostics Expander
                    with st.expander("🔬 Audio Signal, Spectrogram & Frame Matrix Diagnostics", expanded=False):
                        st.markdown("#### 1. Audio Preprocessing & Signal Properties")
                        fig_w, fig_s, astats = plot_audio_diagnostics(audio_data)
                        d_cols = st.columns(len(astats))
                        for i, (k, v) in enumerate(astats.items()):
                            d_cols[i].metric(k, v)
                        st.plotly_chart(fig_w, use_container_width=True)
                        st.plotly_chart(fig_s, use_container_width=True)

                        st.markdown("#### 2. Frame-Roll Activation Matrix (T × 88 Keys)")
                        fig_mat = plot_frame_roll_matrix(res.get("frame_roll", []), res.get("frame_rate_hz", 31.25))
                        st.plotly_chart(fig_mat, use_container_width=True)


if __name__ == "__main__":
    main()
