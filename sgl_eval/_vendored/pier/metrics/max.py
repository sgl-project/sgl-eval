# Vendored from datacurve-ai/pier@0c802fc067a425345b24d1c69411aa98acf61a1d
# Source: src/pier/metrics/max.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

from sgl_eval._vendored.pier.metrics.base import BaseMetric, RewardDict, aggregate_reward_dicts


class Max(BaseMetric[RewardDict]):
    def compute(self, rewards: list[RewardDict | None]) -> RewardDict:
        return aggregate_reward_dicts(rewards, metric_name="max", aggregate=max)
