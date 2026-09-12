"""sgl-eval: one-click accuracy evaluation harness for SGLang."""

from pathlib import Path

__version__ = "0.1.1"

# Shared root for dataset loading and run provenance.
VENDORED_NS_ROOT = Path(__file__).resolve().parent / "_vendored" / "nemo_skills"
