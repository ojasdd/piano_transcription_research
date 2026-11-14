# Antigravity deployment brief
Own deployment/UI only. Import only `from piano_transcriber.transcriber import transcribe`; do not import training code.

Contract: `transcribe(audio_path: str, prior: str = "none", lam: float = 0.5) -> dict`.
Return `notes` (pitch/onset/offset/velocity), `frame_roll` (T x 88), `frame_rate_hz`, `midi_bytes`, `meta`.
Allowed prior values: none/rbm/nade/made. Set `PIANO_TRANSCRIBER_STUB=1` to exercise the fixed fake-note stub before checkpoints exist. Real inference is CPU-only; chunking is inside the wrapper.

Build FastAPI `/health` and `/transcribe`, validation/size limits/temp cleanup, frontend upload+mic, piano-roll, MIDI playback/download, prior and lambda controls, CPU Docker image, HF Spaces deployment and tests. Keep binary MIDI separate or base64-encode for JSON. Do not alter the Python contract without agreement.
