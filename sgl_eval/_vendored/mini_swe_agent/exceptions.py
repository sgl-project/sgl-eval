# Vendored from SWE-agent/mini-swe-agent@a83fcae82d2a08f0ee0c688f9d137b3566c097f8
# Source: src/minisweagent/exceptions.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

class InterruptAgentFlow(Exception):
    """Raised to interrupt the agent flow and add messages."""

    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()


class Submitted(InterruptAgentFlow):
    """Raised when the agent has completed its task."""


class LimitsExceeded(InterruptAgentFlow):
    """Raised when the agent has exceeded its cost or step limit."""


class TimeExceeded(LimitsExceeded):
    """Raised when the agent has exceeded its wall-clock time limit."""


class UserInterruption(InterruptAgentFlow):
    """Raised when the user interrupts the agent."""


class FormatError(InterruptAgentFlow):
    """Raised when the LM's output is not in the expected format."""
