"""End-to-end orchestrator."""
import os
import time

from .cache import cache_dir, video_id_from_url, save_json
from .download import download
from .transcribe import transcribe
from .loudness import analyze as analyze_loudness
from .llm import detect_cuts
from .parser import parse_cuts, cuts_to_keeps
from .cutter import cut_video


def run(url: str, iteration: int = 0, dry_run: bool = False, force_model: str | None = None) -> dict:
    """Run the full pipeline. Returns a summary dict with stats + paths.

    iteration: if you tweak the prompt and want to re-call the LLM, bump this
    to bypass the LLM response cache. Download/transcript/loudness stays cached.
    """
    t0 = time.time()
    vid = video_id_from_url(url)
    print(f"\n=== dead_cut: {url}  (id={vid}, iter={iteration}) ===\n")
    out_dir = cache_dir(vid)

    # 1. Download
    video_path = download(url, vid)

    # 2. Transcribe
    transcript = transcribe(video_path, vid)
    segments = transcript["segments"]
    duration = transcript["duration"]

    # 3. Loudness
    loud = analyze_loudness(video_path, vid, segments)
    loud_per_seg = loud["per_segment"]

    # 4. LLM cut detection
    model, raw = detect_cuts(vid, duration, segments, loud_per_seg,
                             iteration=iteration, force_model=force_model)

    # 5. Parse + invert (pass duration so malformed ranges get dropped/clamped)
    cuts = parse_cuts(raw, max_duration=duration)
    keeps = cuts_to_keeps(cuts, duration)
    cut_secs = sum(e - s for s, e in cuts)
    keep_secs = sum(e - s for s, e in keeps)
    pct_cut = 100.0 * cut_secs / max(duration, 1e-6)
    print(f"\n[pipeline] {len(cuts)} cuts totalling {cut_secs:.1f}s ({pct_cut:.1f}% of {duration:.1f}s)")
    print(f"[pipeline] keeping {keep_secs:.1f}s in {len(keeps)} segments")

    # 6. Cut. Per-iter file for debugging; also copy to final.mp4 as the
    # canonical "latest" output.
    final_path = os.path.join(out_dir, f"final_iter{iteration}.mp4")
    if dry_run:
        print("[pipeline] DRY RUN — skipping ffmpeg cut")
    else:
        cut_video(video_path, keeps, final_path, work_dir=out_dir)
        import shutil
        shutil.copyfile(final_path, os.path.join(out_dir, "final.mp4"))

    summary = {
        "video_id": vid,
        "url": url,
        "iteration": iteration,
        "model": model,
        "original_duration_s": duration,
        "n_segments": len(segments),
        "n_cuts": len(cuts),
        "cuts_total_s": cut_secs,
        "pct_cut": pct_cut,
        "keep_total_s": keep_secs,
        "keep_ranges": keeps,
        "cut_ranges": cuts,
        "final_path": final_path if not dry_run else None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    save_json(os.path.join(out_dir, f"summary_iter{iteration}.json"), summary)
    print(f"\n[pipeline] DONE in {summary['elapsed_s']}s -> {final_path}")
    return summary
