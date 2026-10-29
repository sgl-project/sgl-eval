# Vendored from datacurve-ai/pier@0c802fc067a425345b24d1c69411aa98acf61a1d
# Source: src/pier/utils/logger.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

import logging


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger


logger = setup_logger(__name__)
