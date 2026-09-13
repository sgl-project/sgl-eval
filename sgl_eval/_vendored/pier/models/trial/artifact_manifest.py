# Vendored from datacurve-ai/pier@0c802fc067a425345b24d1c69411aa98acf61a1d
# Source: src/pier/models/trial/artifact_manifest.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

from typing import Any, Literal

from pydantic import BaseModel, Field


class ArtifactManifestEntry(BaseModel):
    source: str
    destination: str
    type: Literal["file", "directory"]
    status: Literal["ok", "failed", "empty"]


class ArtifactManifest(BaseModel):
    entries: list[ArtifactManifestEntry] = Field(default_factory=list)

    def to_json_data(self) -> list[dict[str, Any]]:
        return [entry.model_dump(mode="json") for entry in self.entries]
