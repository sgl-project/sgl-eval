# Vendored from datacurve-ai/pier@0c802fc067a425345b24d1c69411aa98acf61a1d
# Source: src/pier/models/agent/network.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class NetworkAllowlist(BaseModel):
    domains: list[str] = Field(
        default_factory=list,
        description=(
            "Exact domain or leading-dot suffix allowed by network policy, "
            "for example 'api.anthropic.com' or '.anthropic.com'."
        ),
    )

    @field_validator("domains")
    @classmethod
    def normalize_domains(cls, domains: list[str]) -> list[str]:
        normalized: set[str] = set()
        for value in domains:
            domain = value.strip().lower().rstrip(".")
            if not domain:
                raise ValueError("domain cannot be empty")
            if any(char in domain for char in "/:*"):
                raise ValueError("domain must be an exact domain or leading-dot suffix")
            normalized.add(domain)
        return sorted(normalized)
