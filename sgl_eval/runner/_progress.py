"""tqdm bars + background refresher so elapsed/ETA stay live between ticks."""

from __future__ import annotations

import threading
from typing import Callable, List, Tuple

from tqdm import tqdm

TickFn = Callable[[int, float], None]


def _build_progress(
    name: str, num_examples: int, n_repeats: int, *, enabled: bool
) -> Tuple[List[tqdm], TickFn]:
    """Per-repeat bars + an overall bar; ``tick`` updates the running
    ``acc`` postfix."""
    if not enabled:
        return [], lambda _idx, _score: None

    if n_repeats <= 1:
        bar = tqdm(total=num_examples, desc=name, dynamic_ncols=True)
        cum = {"correct": 0.0, "total": 0}

        def tick(_rep_idx: int, score: float) -> None:
            cum["correct"] += float(score)
            cum["total"] += 1
            bar.set_postfix({"acc": f"{cum['correct'] / cum['total']:.2%}"}, refresh=False)
            bar.update(1)

        return [bar], tick

    width = len(str(n_repeats))
    prefix_len = len(f"rep {n_repeats}/{n_repeats}")
    rep_bars = [
        tqdm(
            total=num_examples,
            desc=f"{name} rep {i + 1:>{width}}/{n_repeats}",
            position=i,
            leave=True,
            dynamic_ncols=True,
        )
        for i in range(n_repeats)
    ]
    overall_label = "overall".ljust(prefix_len)
    overall_bar = tqdm(
        total=num_examples * n_repeats,
        desc=f"{name} {overall_label}",
        position=n_repeats,
        leave=True,
        dynamic_ncols=True,
    )

    rep_correct = [0.0] * n_repeats
    rep_total = [0] * n_repeats
    overall_correct = [0.0]
    overall_total = [0]

    def tick(rep_idx: int, score: float) -> None:
        if 0 <= rep_idx < len(rep_bars):
            rep_correct[rep_idx] += float(score)
            rep_total[rep_idx] += 1
            rep_bars[rep_idx].set_postfix(
                {"acc": f"{rep_correct[rep_idx] / rep_total[rep_idx]:.2%}"},
                refresh=False,
            )
            rep_bars[rep_idx].update(1)
        overall_correct[0] += float(score)
        overall_total[0] += 1
        overall_bar.set_postfix(
            {"acc": f"{overall_correct[0] / overall_total[0]:.2%}"},
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
