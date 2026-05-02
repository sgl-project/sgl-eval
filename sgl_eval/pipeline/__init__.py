"""``sgl-eval run`` pipeline:
Stage 1 ``setup``    : resolve inputs, build sampler, mkdir, install sigint
Stage 2 ``spec.run`` : live sampling + scoring + aggregation -> RunResult
Stage 3 ``report``   : stdout summary + metrics.json + footer + exit code
"""

from sgl_eval.pipeline.refresh import cmd_refresh
from sgl_eval.pipeline.run import cmd_run

__all__ = ["cmd_refresh", "cmd_run"]
