"""Synthesize a "kept transcript" by intersecting the original transcript with
the keep ranges from a pipeline iteration. Useful for judging cut quality without
needing to re-transcribe the final mp4."""
import os
import sys

from .cache import cache_dir, load_json
from .parser import parse_cuts, cuts_to_keeps


def _hms(t):
    t = max(0, int(round(t)))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _overlaps(seg_start, seg_end, keep_start, keep_end):
    return seg_start < keep_end and seg_end > keep_start


def synthesize(video_id: str, iteration: int = 1) -> str:
    cdir = cache_dir(video_id)
    transcript = load_json(os.path.join(cdir, "transcript.json"))
    summary = load_json(os.path.join(cdir, f"summary_iter{iteration}.json"))
    keep_ranges = [tuple(r) for r in summary["keep_ranges"]]
    duration = transcript["duration"]
    segments = transcript["segments"]

    out_lines = []
    out_lines.append(f"=== Original duration: {_hms(duration)}  Kept: {_hms(summary['keep_total_s'])} ({100-summary['pct_cut']:.1f}%)")
    out_lines.append(f"=== {summary['n_cuts']} cut ranges, {len(keep_ranges)} keep ranges, model={summary['model']}")
    out_lines.append("")

    for i, (ks, ke) in enumerate(keep_ranges):
        out_lines.append(f"--- KEEP #{i+1}: {_hms(ks)} - {_hms(ke)}  ({ke-ks:.0f}s) ---")
        for seg in segments:
            if _overlaps(seg["start"], seg["end"], ks, ke):
                out_lines.append(f"  [{_hms(seg['start'])}-{_hms(seg['end'])}] {seg['text']}")
        out_lines.append("")

    return "\n".join(out_lines)


if __name__ == "__main__":
    vid = sys.argv[1]
    it = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    print(synthesize(vid, it))
