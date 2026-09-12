"""Check translated imports in the vendored NeMo-Skills slice."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def test_no_untranslated_nemo_skills_imports():
    from audit_vendored import find_untranslated

    bad = find_untranslated()
    assert not bad, f"Untranslated imports under _vendored: {bad}"
