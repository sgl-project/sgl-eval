"""sgl-eval: one-click accuracy evaluation harness for SGLang."""

from pathlib import Path

__version__ = "0.0.1"

# Single source of truth for the vendored NeMo-Skills slice path. Used by
# the dataset loader, the CLI's run-meta provenance reader, and any future
# consumer that needs to reach into ``_vendored/nemo_skills/``.
VENDORED_NS_ROOT = Path(__file__).resolve().parent / "_vendored" / "nemo_skills"
