"""CLI entry point: python -m dead_cut <youtube_url> [--iter N] [--dry-run]"""
import argparse
import sys

from .pipeline import run


def main():
    ap = argparse.ArgumentParser(description="dead_cut: auto-edit a YouTube vod")
    ap.add_argument("url", help="YouTube video URL or 11-char video ID")
    ap.add_argument("--iter", type=int, default=0,
                    help="Iteration number — bumps LLM cache key so the prompt re-runs")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run through LLM but skip ffmpeg cut")
    ap.add_argument("--model", default=None,
                    help="Force a specific OpenRouter model id (e.g. qwen/qwen3.6-plus:free)")
    ap.add_argument("--no-snap", action="store_true",
                    help="Disable snap-to-silence post-process (for A/B comparison)")
    ap.add_argument("--no-trim", action="store_true",
                    help="Disable within-keep silence trim (keeps original pacing)")
    args = ap.parse_args()
    try:
        run(args.url, iteration=args.iter, dry_run=args.dry_run,
            force_model=args.model, snap=not args.no_snap, trim=not args.no_trim)
    except KeyboardInterrupt:
        print("\n[main] interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
