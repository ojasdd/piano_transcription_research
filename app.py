"""Hugging Face Spaces & Streamlit Root Entrypoint."""
import os
import sys
from pathlib import Path

# Ensure 'src' and root directories are on pythonpath
ROOT = Path(__file__).resolve().parent
src_dir = ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Auto-enable stub mode if model checkpoints are not trained yet
if not (ROOT / "models" / "acoustic.pt").exists():
    os.environ.setdefault("PIANO_TRANSCRIBER_STUB", "1")

from deployment.frontend import main

if __name__ == "__main__":
    main()
