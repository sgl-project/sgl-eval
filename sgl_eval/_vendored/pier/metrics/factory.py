# Vendored from datacurve-ai/pier@0c802fc067a425345b24d1c69411aa98acf61a1d
# Source: src/pier/metrics/factory.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

from sgl_eval._vendored.pier.metrics.base import BaseMetric
from sgl_eval._vendored.pier.metrics.max import Max
from sgl_eval._vendored.pier.metrics.mean import Mean
from sgl_eval._vendored.pier.metrics.min import Min
from sgl_eval._vendored.pier.metrics.sum import Sum
from sgl_eval._vendored.pier.models.metric.type import MetricType


class MetricFactory:
    _METRICS: list[type[BaseMetric]] = [
        Sum,
        Min,
        Max,
        Mean,
    ]
    _METRIC_MAP: dict[MetricType, type[BaseMetric]] = {
        MetricType.SUM: Sum,
        MetricType.MIN: Min,
        MetricType.MAX: Max,
        MetricType.MEAN: Mean,
    }

    @classmethod
    def create_metric(
        cls,
        metric_type: MetricType,
        **kwargs,
    ) -> BaseMetric:
        """
        Create a metric from a metric type.

        Args:
            metric_type (MetricType): The type of the metric.
            **kwargs: Additional keyword arguments to pass to the metric constructor.

        Returns:
            BaseMetric: The created metric.

        Raises:
            ValueError: If the metric type is invalid or required parameters are missing.
        """
        if metric_type not in cls._METRIC_MAP:
            raise ValueError(
                f"Unsupported metric type: {metric_type}. This could be because the "
                "metric is not registered in the MetricFactory or because the metric "
                "type is invalid."
            )

        return cls._METRIC_MAP[metric_type](**kwargs)
