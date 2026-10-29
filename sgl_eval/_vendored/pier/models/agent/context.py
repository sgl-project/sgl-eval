# Vendored from datacurve-ai/pier@0c802fc067a425345b24d1c69411aa98acf61a1d
# Source: src/pier/models/agent/context.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

from typing import Any

from pydantic import BaseModel, Field

from sgl_eval._vendored.pier.models.agent.rollout_detail import RolloutDetail


class AgentContext(BaseModel):
    n_input_tokens: int | None = Field(
        default=None, description="The number of input tokens used including cache."
    )
    n_cache_tokens: int | None = Field(
        default=None, description="The number of cache tokens used."
    )
    n_output_tokens: int | None = Field(
        default=None, description="The number of output tokens used."
    )
    cost_usd: float | None = Field(
        default=None, description="The cost in USD for the agent execution."
    )
    peak_context_tokens: int | None = Field(
        default=None,
        description="The largest input context observed for a single agent LLM call.",
    )
    summarization_count: int | None = Field(
        default=None,
        description="The number of detected context summarization or compaction events.",
    )
    n_agent_steps: int | None = Field(
        default=None,
        description="The number of agent-authored trajectory steps.",
    )
    rollout_details: list[RolloutDetail] | None = Field(
        default=None,
        description=(
            "Detailed information about each rollout trajectory including token IDs, "
            "loss masks, and logprobs. Each element represents one trajectory. For a "
            "linear chat history, there is only one rollout trajectory."
        ),
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata about the agent execution."
    )

    def is_empty(self) -> bool:
        return all(value is None for value in self.model_dump().values())
