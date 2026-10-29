# Vendored from datacurve-ai/pier@0c802fc067a425345b24d1c69411aa98acf61a1d
# Source: src/pier/agents/nop.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

from sgl_eval._vendored.pier.agents.base import BaseAgent
from sgl_eval._vendored.pier.environments.base import BaseEnvironment
from sgl_eval._vendored.pier.models.agent.context import AgentContext
from sgl_eval._vendored.pier.models.agent.name import AgentName


class NopAgent(BaseAgent):
    SUPPORTS_WINDOWS: bool = True

    @staticmethod
    def name() -> str:
        return AgentName.NOP.value

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        pass
