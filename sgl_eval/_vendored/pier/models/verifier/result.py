# Vendored from datacurve-ai/pier@0c802fc067a425345b24d1c69411aa98acf61a1d
# Source: src/pier/models/verifier/result.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

from pydantic import BaseModel


class VerifierResult(BaseModel):
    rewards: dict[str, float | int] | None = None
