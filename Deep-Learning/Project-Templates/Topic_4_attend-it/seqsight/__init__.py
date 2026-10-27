"""Seq-Sight model code: hand-written attention and the two LSTM classifiers.

Imported by the offline training script and by the FastAPI tier; never by
Streamlit. Tokenization lives in ``shared.text`` so every tier encodes text the
same way.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
