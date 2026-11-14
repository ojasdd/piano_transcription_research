import os
os.environ["PIANO_TRANSCRIBER_STUB"]="1"
from piano_transcriber.transcriber import transcribe
def test_stub_shape():
    r=transcribe("unused.wav",prior="rbm")
    assert set(r)=={"notes","frame_roll","frame_rate_hz","midi_bytes","meta"}
    assert len(r["frame_roll"][0])==88 and isinstance(r["midi_bytes"],bytes)
