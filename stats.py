"""Per-stage stats logging. Each pipeline stage writes one JSON file
under cache/<vid>/<stage>_stats.json with elapsed time + stage-specific
counters. Read by visualize.py at end of run (and standalone) to build
the pipeline summary image."""
import os
import time

from .cache import save_json, load_json


_STAGES = (
    "download", "transcribe", "loudness", "llm", "post", "encode", "visualize",
)


def stats_path(out_dir: str, stage: str) -> str:
    return os.path.join(out_dir, f"{stage}_stats.json")


def save(out_dir: str, stage: str, elapsed_s: float | None = None, **fields) -> None:
    """Write stats_<stage>.json with elapsed_s plus arbitrary fields.
    Merges with any existing data so multiple writes (e.g. mid-stage
    updates) accumulate instead of overwriting."""
    if stage not in _STAGES:
        # Allow but warn — keeps the door open for future stages.
        print(f"[stats] note: unknown stage '{stage}'")
    path = stats_path(out_dir, stage)
    existing = {}
    if os.path.exists(path):
        try:
            existing = load_json(path)
        except Exception:
            existing = {}
    existing.update(fields)
    if elapsed_s is not None:
        # Cache-hit preservation: if existing has a meaningful elapsed and
        # the new value is tiny (< 0.5s, likely a cache load), keep the old
        # value. The stage didn't actually run this round, the stats represent
        # earlier real work.
        prev = existing.get("elapsed_s")
        if prev is not None and elapsed_s < 0.5 and prev >= 0.5:
            existing["_cached_this_run"] = True
        else:
            existing["elapsed_s"] = round(float(elapsed_s), 3)
            existing["_cached_this_run"] = False
    existing["_stage"] = stage
    existing["_saved_at"] = time.time()
    save_json(path, existing)


def load(out_dir: str, stage: str) -> dict | None:
    """Load stats_<stage>.json or None if missing/unreadable."""
    path = stats_path(out_dir, stage)
    if not os.path.exists(path):
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def load_all(out_dir: str) -> dict:
    """Load every stage's stats; missing ones come back as None values.
    Useful for visualize.py to render whatever exists."""
    return {stage: load(out_dir, stage) for stage in _STAGES}
