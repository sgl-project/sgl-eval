# Vendored from datacurve-ai/pier@0c802fc067a425345b24d1c69411aa98acf61a1d
# Source: src/pier/models/agent/name.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

from enum import Enum


class AgentName(str, Enum):
    ORACLE = "oracle"
    NOP = "nop"
    CLAUDE_CODE = "claude-code"
    ANTIGRAVITY_SDK = "antigravity-sdk"
    CODEX = "codex"
    CURSOR_CLI = "cursor-cli"
    GEMINI_CLI = "gemini-cli"
    MINI_SWE_AGENT = "mini-swe-agent"
    SWE_AGENT = "swe-agent"
    OPENCODE = "opencode"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}
