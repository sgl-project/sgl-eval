"""Stage 1+2+3 orchestration for ``sgl-eval run``.

Pipeline stages (per ``cmd_run``):

  Stage 1 ``setup``  : resolve inputs, build sampler, mkdir, install sigint
  Stage 2 ``spec.run``: live sampling + scoring + aggregation (-> RunResult)
  Stage 3 ``report`` : stdout summary + metrics.json + footer + exit code

``cmd_run`` itself is the thin glue that walks them; cli.py imports it.
"""

from sgl_eval.runtime.run import cmd_run

__all__ = ["cmd_run"]
