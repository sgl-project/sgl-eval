# Vendored from datacurve-ai/pier@0c802fc067a425345b24d1c69411aa98acf61a1d
# Source: src/pier/constants.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

from pathlib import Path

PYPI_PACKAGE_NAME = "datacurve-pier"
CACHE_DIR = Path("~/.cache/pier").expanduser()
TASK_CACHE_DIR = CACHE_DIR / "tasks"
PACKAGE_CACHE_DIR = CACHE_DIR / "tasks" / "packages"
ORG_NAME_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$"
