"""Render a vendored NeMo-Skills prompt yaml into a final user message.

Mirrors upstream's prompt subsystem for the simple case used by ``generic/math``
and ``eval/aai/mcq-4choices*`` configs: a top-level ``user`` template with
optional ``few_shot_examples`` block. We do not replicate hydra's full prompt
machinery -- benchmarks beyond math/mcq may need richer rendering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

_VENDORED_PROMPT_DIR = (
    Path(__file__).resolve().parent.parent / "_vendored" / "nemo_skills" / "prompts"
)
_SE_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def load_few_shot_examples(spec: str) -> list:
    """Resolve ``"<module>:<name>"`` (relative to the vendored
    ``nemo_skills.prompt.few_shot_examples`` package) to its list of
    ``{"problem", "solution"}`` dicts."""
    import importlib

    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"few_shot_examples spec must be '<module>:<name>', got {spec!r}")
    module = importlib.import_module(
        f"sgl_eval._vendored.nemo_skills.prompt.few_shot_examples.{module_name}"
    )
    return list(getattr(module, attr))


def vendored_prompt(name: str) -> Path:
    """Return the path to a vendored prompt yaml by basename (no extension)."""
    return _VENDORED_PROMPT_DIR / f"{name}.yaml"


def resolve_prompt(spec: str) -> Path:
    """Resolve a prompt spec to a yaml path.

    The path branch exists so a caller can point at a prompt this repo does not
    ship. Existence is deliberately not checked: registration resolves every
    benchmark's default at import time, so a raise here would fail at import.
    """
    if "/" in spec or spec.endswith(".yaml"):
        return Path(spec).expanduser()
    vendored = vendored_prompt(spec)
    if vendored.exists():
        return vendored
    return _SE_PROMPT_DIR / f"{spec}.yaml"


def prompt_media_config(yaml_path: Path) -> dict:
    """Media placement keys from a VLM prompt yaml (``image_position``).

    ``render_prompt`` consumes only the ``user`` template, so multimodal callers
    read placement here. Upstream's sibling ``image_field`` stays unexposed --
    the loader has already resolved it into ``Example.media``.
    """
    cfg = yaml.safe_load(yaml_path.read_text())
    return {"image_position": cfg.get("image_position", "after")}


def render_prompt(
    yaml_path: Path,
    problem: str,
    few_shot_examples: Optional[list] = None,
    **extra_fields: Any,
) -> str:
    cfg = yaml.safe_load(yaml_path.read_text())
    user_template: str = cfg["user"]

    fmt: dict = {"problem": problem, **extra_fields}
    if "{examples}" in user_template:
        examples_text = ""
        if few_shot_examples and cfg.get("few_shot_examples"):
            fs = cfg["few_shot_examples"]
            rendered = "".join(fs["template"].format(**ex) for ex in few_shot_examples)
            examples_text = fs["prefix"] + rendered + fs["suffix"]
        fmt["examples"] = examples_text

    return user_template.format(**fmt)
