"""Make the vendored `prepare.py` downloads survive a flaky origin.

Upstream's prepare scripts call `urllib.request.urlretrieve(URL, path)` with a
hard-coded URL, no timeout and no retry, so one unreachable host ends the run
with a bare `Errno 110`. Those files are synced verbatim from NeMo-Skills and
carry a DO-NOT-EDIT banner, so the retry/mirror policy lives here and is
installed around the call instead.

MMLU is the case that actually bites: its archive is served from a personal
faculty web space that has had repeated outages (sgl-project/sgl-eval#33), while
every other benchmark here loads from HuggingFace or GitHub.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Dict, List

# Same archive as the primary, pinned to a commit so the bytes cannot move under
# us. Verified to produce the expected 14042 MMLU test rows.
_MMLU_PRIMARY = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
_MMLU_HF_MIRROR = (
    "https://huggingface.co/datasets/cais/mmlu/resolve/"
    "c30699e8356da336a370243923dbaf21066bb9fe/data.tar"
)

MIRRORS: Dict[str, List[str]] = {_MMLU_PRIMARY: [_MMLU_HF_MIRROR]}

# Override the whole chain for one dataset, e.g. an internal mirror or a
# file:// path in an air-gapped environment.
_OVERRIDE_ENV = {_MMLU_PRIMARY: "SGL_EVAL_MMLU_URL"}

# Two tries, not more: a host that refuses twice with a 30s socket timeout is
# down, not busy, and the mirror is one hop away. Worst case before falling
# through is 2 * 30s + one backoff, which a cold-cache CI job can absorb.
_ATTEMPTS_PER_URL = 2
_BACKOFF_SECONDS = 2.0
# Per socket operation, not per download -- a 166 MB body reads in many chunks,
# each with its own budget, so this does not cap a slow but working transfer.
_TIMEOUT_SECONDS = 30.0


def _candidates(url: str) -> List[str]:
    override = os.environ.get(_OVERRIDE_ENV.get(url, ""))
    if override:
        return [override]
    return [url, *MIRRORS.get(url, [])]


def _download(url: str, filename: str) -> None:
    """One attempt, with a socket timeout urlretrieve does not offer."""
    with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:
        with open(filename, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)


def retrieve(url: str, filename: str) -> str:
    """Try each candidate `_ATTEMPTS_PER_URL` times before moving to the next."""
    errors = []
    for candidate in _candidates(url):
        for attempt in range(_ATTEMPTS_PER_URL):
            try:
                _download(candidate, filename)
                return candidate
            except (urllib.error.URLError, OSError) as exc:
                errors.append(f"{candidate}: {exc}")
                if attempt + 1 < _ATTEMPTS_PER_URL:
                    time.sleep(_BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError("could not download " + url + " from any source:\n  " + "\n  ".join(errors))


@contextmanager
def resilient_downloads():
    """Point `urllib.request.urlretrieve` at `retrieve` for the duration.

    Patching the module the vendored script imports is what keeps the policy out
    of the vendored file; the original is restored on the way out so nothing
    else in the process inherits it.
    """
    original = urllib.request.urlretrieve

    def patched(url, filename=None, reporthook=None, data=None):
        if filename is None:  # only the caching form is used by prepare scripts
            return original(url, filename, reporthook, data)
        retrieve(url, filename)
        return filename, None

    urllib.request.urlretrieve = patched
    try:
        yield
    finally:
        urllib.request.urlretrieve = original
