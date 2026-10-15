"""Video-MME video-understanding benchmark (sgl-eval-own; NeMo-Skills has no Video-MME).

Dataset: ``lmms-lab/Video-MME`` -- 2,700 multiple-choice questions (A-D) over 900
videos, 300 videos per duration bucket (``short`` < 2 min, ``medium`` 4-15 min,
``long`` 30-60 min), plus ``.srt`` subtitles for 744 of them. Each question is
sent as one ``video_url`` content block followed by the official prompt, so the
served model does its own frame sampling -- there is no client-side frame
extraction. Scoring is the official ``eval_your_results.py`` (MME-Benchmarks/
Video-MME @ 4e7566d36f, the last commit that shipped it): ``extract_characters_regex``
ported verbatim, letter == ``answer``, and accuracy over *answered* questions
(the official script drops unparsable responses from the denominator; the
stricter all-questions accuracy is reported alongside as ``score_all``).

Video transport. The videos are 2 MB - 915 MB, so inlining them as base64 is
only practical for small deployments. ``--video-mme-video-url TEMPLATE`` tells
the benchmark how the *server* reaches each video instead: a template with
``{videoID}`` (and optionally ``{filename}``), e.g.
``file:///mnt/video-mme/data/{videoID}.mp4`` for a co-located server or
``https://files.example.org/video-mme/{filename}`` for a hosted copy. Without
it, each request inlines the file as a ``data:video/mp4;base64,...`` URL, read
at request time (never all at once).

Data. ``$SGL_EVAL_VIDEO_MME_DIR`` may point at a directory holding
``data/<videoID>.mp4`` (and optionally ``subtitle/<videoID>.srt``); otherwise
the parquet, ``subtitle.zip`` and the 20 ``videos_chunked_*.zip`` (~101 GB) are
fetched from the Hub and extracted once into ``~/.cache/sgl_eval/video_mme/``.
"""

from __future__ import annotations

import base64
import os
import re
import sys
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sgl_eval.evals._vision import build_user_content
from sgl_eval.predictions import PredictionsWriter
from sgl_eval.registry import EvalSpec, register
from sgl_eval.runner import SampleFn, ScoreOneFn, run_examples
from sgl_eval.sampler import ChatCompletionSampler
from sgl_eval.types import Example, ExampleResult, GenConfig, MediaItem, RunResult, Sample

DATASET_REPO = "lmms-lab/Video-MME"
PARQUET_FILE = "videomme/test-00000-of-00001.parquet"
SUBTITLE_ZIP = "subtitle.zip"
VIDEO_ZIP_GLOB = "videos_chunked_*.zip"
_CACHE_DIR = Path.home() / ".cache" / "sgl_eval" / "video_mme"
_ENV_DIR = "SGL_EVAL_VIDEO_MME_DIR"

DURATIONS = ("short", "medium", "long")
CHOICES = ("A", "B", "C", "D")

# Official prompt (Video-MME README, "Evaluation").
PROMPT_INSTRUCTION = (
    "Select the best answer to the following multiple-choice question based on the video.\n"
    "Respond with only the letter (A, B, C, or D) of the correct option."
)
PROMPT_SUBTITLE_HEADER = "This video's subtitles are listed below:"
PROMPT_SUFFIX = "The best answer is:"
_SRT_TEXT_RE = re.compile(r'<font color="white" size=".72c">(.*?)</font>')


# ---------- official scoring (ported verbatim from evaluation/eval_your_results.py) ----------


def extract_characters_regex(s: str) -> str:
    """Official Video-MME extractor. Returns the first A-D letter left after
    stripping the answer prefixes, or ``""``. (The prefix list is reproduced
    as published, including its two missing commas.)"""
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        # (the two missing commas are in the published script; the parentheses
        # keep the identical concatenated strings while satisfying the linter)
        ("The best option is" "The correct option is"),
        ("Best answer:" "Best option:"),
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


# ---------- prompt ----------


def subtitle_text(srt: str) -> str:
    """Official subtitle cleaning: keep the caption text inside the white
    ``<font>`` spans, one caption per line."""
    return "\n".join(m.strip() for m in _SRT_TEXT_RE.findall(srt) if m.strip())


def build_prompt(question: str, options: list[str], subtitles: str | None = None) -> str:
    parts: list[str] = []
    if subtitles:
        parts += [PROMPT_SUBTITLE_HEADER, subtitles]
    parts += [PROMPT_INSTRUCTION, question, *options, PROMPT_SUFFIX]
    return "\n".join(parts)


# ---------- dataset ----------


def _has_videos(root: Path) -> bool:
    return (root / "data").is_dir() and any((root / "data").glob("*.mp4"))


def _data_root() -> Path:
    override = os.environ.get(_ENV_DIR)
    if override:
        root = Path(override).expanduser()
        if not _has_videos(root):
            sys.exit(f"error: {_ENV_DIR}={override} has no data/*.mp4")
        return root
    if _has_videos(_CACHE_DIR):
        return _CACHE_DIR
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover - datasets pulls it in
        raise SystemExit(f"error: video_mme needs huggingface_hub ({e})")
    warnings.warn(
        f"Video-MME videos not found; downloading ~101 GB of {VIDEO_ZIP_GLOB} from {DATASET_REPO} "
        f"into {_CACHE_DIR} (set {_ENV_DIR} to reuse an existing copy)"
    )
    snap = Path(
        snapshot_download(
            DATASET_REPO, repo_type="dataset", allow_patterns=[VIDEO_ZIP_GLOB, SUBTITLE_ZIP]
        )
    )
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for zpath in sorted(snap.glob(VIDEO_ZIP_GLOB)) + [snap / SUBTITLE_ZIP]:
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(_CACHE_DIR)  # zips carry data/<videoID>.mp4 and subtitle/<videoID>.srt
    if not _has_videos(_CACHE_DIR):
        sys.exit(f"error: extracted {VIDEO_ZIP_GLOB} but found no data/*.mp4 under {_CACHE_DIR}")
    return _CACHE_DIR


def _read_rows() -> list[dict[str, Any]]:
    from datasets import load_dataset  # lazy: heavy import

    ds = load_dataset(DATASET_REPO, "videomme", split="test")
    return [dict(row) for row in ds]


def _interleave_rows_by_duration(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Cap the row count while keeping duration buckets balanced: the parquet
    is ordered short -> medium -> long, so a head-slice would be all short."""
    queues: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        queues.setdefault(row["duration"], []).append(row)
    picked: list[dict[str, Any]] = []
    depth = 0
    dq = list(queues.values())
    while len(picked) < limit and any(depth < len(q) for q in dq):
        for q in dq:
            if depth < len(q):
                picked.append(q[depth])
                if len(picked) >= limit:
                    break
        depth += 1
    return picked


def _load_subtitle(root: Path, video_id: str) -> str | None:
    path = root / "subtitle" / f"{video_id}.srt"
    if not path.exists():
        return None
    text = subtitle_text(path.read_text(encoding="utf-8", errors="replace"))
    return text or None


def _build_example(root: Path, row: dict[str, Any], use_subtitles: bool) -> Example:
    video_id = str(row["videoID"])
    video_path = root / "data" / f"{video_id}.mp4"
    if not video_path.exists():
        raise ValueError(f"missing video data/{video_id}.mp4")
    subtitles = _load_subtitle(root, video_id) if use_subtitles else None
    options = [str(o) for o in row["options"]]
    return Example(
        id=str(row["question_id"]),
        inputs={"problem": build_prompt(str(row["question"]), options, subtitles)},
        target=str(row["answer"]).strip().upper(),
        meta={
            "video_id": video_id,
            "duration": row["duration"],
            "domain": row.get("domain"),
            "sub_category": row.get("sub_category"),
            "task_type": row.get("task_type"),
            "has_subtitles": subtitles is not None,
        },
        # ``url`` holds the local path; the sample fn turns it into the URL the
        # server will actually fetch (template) or an inline data URL.
        media=[MediaItem(kind="video", url=str(video_path), mime="video/mp4")],
    )


def load_video_mme_examples(
    num_examples: int | None, use_subtitles: bool = False, duration: str | None = None
) -> list[Example]:
    root = _data_root()
    rows = _read_rows()
    if duration:
        # Official ``eval_your_results.py --video_duration_type``: score one
        # bucket (900 questions: 300 videos x 3) on its own.
        rows = [row for row in rows if row["duration"] == duration]
    if num_examples is not None:
        rows = _interleave_rows_by_duration(rows, num_examples)
    examples: list[Example] = []
    for row in rows:
        try:
            examples.append(_build_example(root, row, use_subtitles))
        except (ValueError, KeyError) as e:
            warnings.warn(f"skipping Video-MME row {row.get('question_id')}: {e}")
    return examples


# ---------- sample / score ----------


def video_url_for(local_path: str, template: str | None) -> str:
    """Resolve how the server reaches the video: substitute the template, or
    inline the file as a base64 data URL (read now, per request)."""
    path = Path(local_path)
    if template:
        return template.format(videoID=path.stem, filename=path.name)
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:video/mp4;base64,{data}"


def make_sample_fn(
    sampler: ChatCompletionSampler, gen: GenConfig, video_url_template: str | None
) -> SampleFn:
    def sample_fn(ex: Example, _rep_idx: int) -> Sample:
        media = [
            MediaItem(kind="video", url=video_url_for(m.url, video_url_template), mime=m.mime)
            for m in ex.media
        ]
        # Video first, then the prompt (build_user_content appends video blocks
        # after the text, so assemble explicitly).
        text_content = build_user_content(ex.inputs["problem"], [])
        content = [{"type": "video_url", "video_url": {"url": m.url}} for m in media]
        content.append({"type": "text", "text": text_content})
        return sampler([{"role": "user", "content": content}], gen)

    return sample_fn


def make_score_one_fn() -> ScoreOneFn:
    def score_one(ex: Example, sample: Sample) -> tuple[float, str | None]:
        letter = extract_characters_regex(sample.text or "")
        correct = letter != "" and letter == str(ex.target)
        return (1.0 if correct else 0.0), (letter or None)

    return score_one


# ---------- aggregate ----------


def _acc(pairs: list[tuple[float, str | None]], answered_only: bool) -> float:
    if answered_only:
        pairs = [p for p in pairs if p[1] is not None]
    return sum(s for s, _ in pairs) / len(pairs) if pairs else 0.0


def aggregate_video_mme(results: list[ExampleResult], n_repeats: int) -> dict[str, float]:
    """``score`` = official accuracy over answered samples; ``score_all`` counts
    unparsable responses as wrong; ``task.<duration>`` per bucket (official
    rule); ``no_answer`` = share of unparsable responses. With repeats, the
    vendored MathMetrics adds pass@1[avg-of-k] / pass@k / majority@k (unparsable
    counted wrong)."""
    if not results:
        return {"score": 0.0}
    pairs = [(s, e) for r in results for s, e in zip(r.scores, r.extracted)]
    flat: dict[str, float] = {}
    if n_repeats > 1:
        from sgl_eval.evals._multichoice import aggregate_with_math_metrics

        flat.update(aggregate_with_math_metrics(results, n_repeats))
    flat["score"] = _acc(pairs, answered_only=True)
    flat["score_all"] = _acc(pairs, answered_only=False)
    by_dur: dict[str, list[tuple[float, str | None]]] = {}
    for r in results:
        by_dur.setdefault(r.example.meta.get("duration", "other"), []).extend(
            zip(r.scores, r.extracted)
        )
    for dur in sorted(by_dur, key=lambda d: DURATIONS.index(d) if d in DURATIONS else 99):
        flat[f"task.{dur}"] = _acc(by_dur[dur], answered_only=True)
    flat["no_answer"] = sum(1 for _, e in pairs if e is None) / len(pairs) if pairs else 0.0
    return flat


# ---------- CLI surface ----------


def add_arguments(group: Any) -> None:
    group.add_argument(
        "--video-mme-video-url",
        metavar="TEMPLATE",
        help=(
            "how the SERVER reaches each video, with {videoID} / {filename} placeholders, "
            "e.g. file:///mnt/video-mme/data/{videoID}.mp4 or https://host/vmme/{filename}; "
            "default: inline each file as a base64 data URL (only practical for small setups)"
        ),
    )
    group.add_argument(
        "--video-mme-duration",
        choices=list(DURATIONS),
        help="evaluate one duration bucket only (900 questions), like the official --video_duration_type",
    )
    group.add_argument(
        "--video-mme-subtitles",
        action="store_true",
        default=None,  # None keeps it out of bench_args when not passed
        help="use the official with-subtitles prompt (videos without a subtitle file fall back to the plain prompt)",
    )


# ---------- registration ----------


def run_video_mme_benchmark(
    *,
    name: str,
    sampler: ChatCompletionSampler,
    gen: GenConfig,
    n_repeats: int,
    num_examples: int | None,
    num_threads: int,
    video_url_template: str | None = None,
    use_subtitles: bool = False,
    duration: str | None = None,
    predictions_writer: PredictionsWriter | None = None,
) -> RunResult:
    examples = load_video_mme_examples(num_examples, use_subtitles=use_subtitles, duration=duration)
    return run_examples(
        name=name,
        examples=examples,
        sample_fn=make_sample_fn(sampler, gen, video_url_template),
        score_one_fn=make_score_one_fn(),
        num_threads=num_threads,
        n_repeats=n_repeats,
        aggregate_fn=lambda results: aggregate_video_mme(results, n_repeats),
        on_sample_scored=predictions_writer,
    )


def _run(
    *,
    sampler: ChatCompletionSampler,
    gen: GenConfig,
    n_repeats: int,
    num_examples: int | None,
    num_threads: int,
    predictions_writer: PredictionsWriter | None = None,
    load_examples: Callable[[int | None], list[Example]] | None = None,
    bench_args: dict[str, Any] | None = None,
    **_unused: Any,
) -> RunResult:
    if load_examples is not None:
        sys.exit("error: --from-dataset is not supported for video_mme (video inputs required)")
    args = dict(bench_args or {})
    return run_video_mme_benchmark(
        name="video_mme",
        sampler=sampler,
        gen=gen,
        n_repeats=n_repeats,
        num_examples=num_examples,
        num_threads=num_threads,
        video_url_template=args.get("video_url"),
        use_subtitles=bool(args.get("subtitles")),
        duration=args.get("duration"),
        predictions_writer=predictions_writer,
    )


register(
    EvalSpec(
        name="video_mme",
        category="multichoice",
        description="Video-MME (2700 video MCQs, short/medium/long; official scorer, server-side frame sampling).",
        default_gen=GenConfig(),
        default_n_repeats=1,
        run=_run,
        # Each request makes the server decode and sample a video of up to an
        # hour; 64 concurrent decodes is a lot of CPU and RAM on one box.
        default_num_threads=16,
        add_arguments=add_arguments,
    )
)
