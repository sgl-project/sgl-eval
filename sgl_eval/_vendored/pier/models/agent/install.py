# Vendored from datacurve-ai/pier@0c802fc067a425345b24d1c69411aa98acf61a1d
# Source: src/pier/models/agent/install.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class InstallStep(BaseModel):
    """One install phase (maps to Dockerfile ``USER`` + ``RUN bash -c ...``)."""

    run: str
    user: Literal["root", "agent"] = "agent"
    env: dict[str, str] | None = None


class AgentInstallSpec(BaseModel):
    agent_name: str
    version: str | None = None
    steps: list[InstallStep]
    verification_command: str | None = None
    cache_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _non_empty(self) -> "AgentInstallSpec":
        if not self.steps:
            raise ValueError("AgentInstallSpec requires at least one step")
        return self

    def fingerprint(self) -> str:
        if self.cache_key:
            return self.cache_key
        steps_json = json.dumps(
            [s.model_dump(mode="python", exclude_none=True) for s in self.steps],
            sort_keys=True,
        )
        payload = "\n".join(
            [
                self.agent_name,
                self.version or "",
                steps_json,
                self.verification_command or "",
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
