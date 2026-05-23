"""CLI entry point: python -m dead_cut <youtube_url> [--iter N] [--dry-run]"""
import argparse
import sys

# Force utf-8 stdout/stderr so non-ASCII characters in log prints (≈ ≥ → ✓ etc)
# don't crash on Windows when stdout is redirected (default cp1252 there).
for _stream in ("stdout", "stderr"):
    _s = getattr(sys, _stream, None)
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import glob
import os
import re

from .pipeline import run
from .cache import cache_dir, video_id_from_url


def _latest_cached_iter(url: str) -> int:
    """Return the highest llm_iter{N}.json index already in the cache for
    this video, or 0 if none found. Used as the default --iter so re-runs
    pick up the latest cached LLM result instead of always re-running iter 0."""
    try:
        vid = video_id_from_url(url)
    except Exception:
        return 0
    out_dir = cache_dir(vid)
    pattern = os.path.join(out_dir, "llm_iter*.json")
    files = glob.glob(pattern)
    if not files:
        return 0
    iters = []
    for f in files:
        m = re.search(r"llm_iter(\d+)\.json$", f)
        if m:
            iters.append(int(m.group(1)))
    return max(iters) if iters else 0


def main():
    ap = argparse.ArgumentParser(description="dead_cut: auto-edit a YouTube vod")
    ap.add_argument("url", help="YouTube video URL or 11-char video ID")
    ap.add_argument("--iter", type=int, default=None,
                    help="LLM cache key. Default: latest already-cached iter. "
                         "Bump to one above the latest to force a fresh LLM call.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run through LLM but skip ffmpeg cut")
    ap.add_argument("--model", default=None,
                    help="Force a specific OpenRouter model id (e.g. qwen/qwen3.6-plus:free)")
    ap.add_argument("--no-snap", action="store_true",
                    help="Disable snap-to-silence post-process (for A/B comparison)")
    ap.add_argument("--no-trim", action="store_true",
                    help="Disable within-keep silence trim (keeps original pacing)")
    args = ap.parse_args()
    if args.iter is None:
        args.iter = _latest_cached_iter(args.url)
        print(f"[main] --iter not specified; using latest cached iter={args.iter}")
    try:
        run(args.url, iteration=args.iter, dry_run=args.dry_run,
            force_model=args.model, snap=not args.no_snap, trim=not args.no_trim)
    except KeyboardInterrupt:
        print("\n[main] interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
