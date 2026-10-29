# Vendored from datacurve-ai/pier@0c802fc067a425345b24d1c69411aa98acf61a1d
# Source: src/pier/models/metric/config.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

from typing import Any

from pydantic import BaseModel, Field

from sgl_eval._vendored.pier.models.metric.type import MetricType


class MetricConfig(BaseModel):
    type: MetricType = Field(default=MetricType.MEAN)
    kwargs: dict[str, Any] = Field(default_factory=dict)
