# Vendored from SWE-agent/mini-swe-agent@a83fcae82d2a08f0ee0c688f9d137b3566c097f8
# Source: src/minisweagent/models/utils/retry.py
# DO NOT EDIT directly. To upgrade, edit SOURCES.yaml and rerun
# `python scripts/sync_vendored.py`.

"""Retry utility for model queries."""

import logging
import os

from tenacity import Retrying, before_sleep_log, retry_if_not_exception_type, stop_after_attempt, wait_exponential


def retry(*, logger: logging.Logger, abort_exceptions: list[type[Exception]]) -> Retrying:
    """Thin wrapper around tenacity.Retrying to make use of global config etc.

    Args:
        logger: Logger to use for reporting retries
        abort_exceptions: Exceptions to abort on.

    Returns:
        A tenacity.Retrying object.
    """
    return Retrying(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(tuple(abort_exceptions)),
    )
