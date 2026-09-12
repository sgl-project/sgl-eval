"""tqdm bars + background refresher so elapsed/ETA stay live between ticks."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from tqdm import tqdm

TickFn = Callable[[int, float, Optional[str]], None]


class _ProgressBar(tqdm):
    @property
    def format_dict(self):
        values = super().format_dict
        if values["ncols"] and values["ncols"] < 100:
            # Keep statistics and elapsed/ETA ahead of the bar on narrow terminals.
            values["bar_format"] = (
                "{desc}: {n_fmt}/{total_fmt}{postfix} [{elapsed}<{remaining}] {bar}"
            )
        return values


@dataclass
class _ProgressCounts:
    correct: float = 0.0
    total: int = 0
    known_reasons: int = 0
    truncated: int = 0

    def update(self, score: float, finish_reason: Optional[str]) -> dict[str, str]:
        self.correct += float(score)
        self.total += 1
        self.known_reasons += bool(finish_reason)
        self.truncated += finish_reason == "length"
        rate = f"{self.truncated / self.known_reasons:.2%}" if self.known_reasons else "N/A"
        return {
            "acc": f"{self.correct / self.total:.2%}",
            "truncated": f"{self.truncated}/{self.known_reasons} ({rate})",
        }


def _build_progress(
    name: str, num_examples: int, n_repeats: int, *, enabled: bool
) -> Tuple[List[tqdm], TickFn]:
    """Per-repeat bars + an overall bar; ``tick`` updates the running
    accuracy and truncation count/rate postfixes. Truncation rates exclude
    unknown finish reasons, matching the final ``truncated_rate`` metric."""
    if not enabled:
        return [], lambda _idx, _score, _finish_reason: None

    if n_repeats <= 1:
        bar = _ProgressBar(total=num_examples, desc=name, dynamic_ncols=True)
        counts = _ProgressCounts()

        def tick(_rep_idx: int, score: float, finish_reason: Optional[str]) -> None:
            bar.set_postfix(counts.update(score, finish_reason), refresh=False)
            bar.update(1)

        return [bar], tick

    width = len(str(n_repeats))
    prefix_len = len(f"rep {n_repeats}/{n_repeats}")
    rep_bars = [
        _ProgressBar(
            total=num_examples,
            desc=f"{name} rep {i + 1:>{width}}/{n_repeats}",
            position=i,
            leave=True,
            dynamic_ncols=True,
        )
        for i in range(n_repeats)
    ]
    overall_label = "overall".ljust(prefix_len)
    overall_bar = _ProgressBar(
        total=num_examples * n_repeats,
        desc=f"{name} {overall_label}",
        position=n_repeats,
        leave=True,
        dynamic_ncols=True,
    )

    rep_counts = [_ProgressCounts() for _ in range(n_repeats)]
    overall_counts = _ProgressCounts()

    def tick(rep_idx: int, score: float, finish_reason: Optional[str]) -> None:
        if 0 <= rep_idx < len(rep_bars):
            rep_bars[rep_idx].set_postfix(
                rep_counts[rep_idx].update(score, finish_reason),
                refresh=False,
            )
            rep_bars[rep_idx].update(1)
        overall_bar.set_postfix(
            overall_counts.update(score, finish_reason),
            refresh=False,
        )
        overall_bar.update(1)

    return rep_bars + [overall_bar], tick


def _start_bar_refresher(
    bars: List[tqdm], interval: float = 0.5
) -> Tuple[threading.Event, threading.Thread]:
    """Daemon that calls ``bar.refresh()`` on every bar each ``interval``.
    Returns ``(stop_event, thread)`` -- caller does ``stop.set()`` then
    ``thread.join()`` BEFORE closing bars to avoid a teardown race."""
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(interval):
            for bar in bars:
                try:
                    bar.refresh()
                except Exception:
                    pass

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return stop, thread
