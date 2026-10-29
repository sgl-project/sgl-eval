# Vendored from datacurve-ai/pier@0c802fc067a425345b24d1c69411aa98acf61a1d
# Source: src/pier/models/environment_type.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

from enum import Enum


class EnvironmentType(str, Enum):
    DOCKER = "docker"
    MODAL = "modal"
    DAYTONA = "daytona"
