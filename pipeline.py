"""End-to-end orchestrator."""
import os
import subprocess
import time

from .cache import cache_dir, video_id_from_url, save_json


def _ffprobe_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())
from .download import download
from .transcribe import transcribe
from .loudness import analyze as analyze_loudness
from .llm import detect_cuts
from .parser import parse_cuts, cuts_to_keeps, snap_cuts_to_silence, trim_silences_within_keeps
from .cutter import cut_video


# Snap tolerance: how far a cut boundary may be moved to land on a detected
# silence. 2.0s catches typical transcript-segment misalignment without
# distorting the model's intent.
SNAP_TOLERANCE_S = 2.0

# Within-keep silence trim. The LLM does macro cuts well but can't see fine
# dead air between sentences — these compress long silences to a small gap.
TRIM_MAX_SILENCE_S = 0.6
TRIM_PADDING_S = 0.2


def run(url: str, iteration: int = 0, dry_run: bool = False, force_model: str | None = None,
        snap: bool = True, trim: bool = True) -> dict:
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

    # Source duration is known after download; print a rough ETA banner.
    # Ratios measured on this machine (GTX 1660 Ti, batched int8_float16):
    #   transcribe ~0.07x of source duration  (small model batched)
    #   loudness   ~0.02x of source duration
    #   llm        ~30s (fixed, free model)
    #   reencode   ~0.6x of EXPECTED OUTPUT duration (which is ~0.3-0.5x source)
    # Total ≈ ~0.3x of source duration.
    try:
        src_dur = _ffprobe_duration(video_path)
        est_total_min = src_dur * 0.30 / 60.0
        print(f"[pipeline] source ~{src_dur/60.0:.1f} min — rough ETA ~{est_total_min:.0f} min on this machine\n")
    except Exception:
        pass

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

    # 5. Parse + snap-to-silence + invert.
    # Snap fixes the mid-word-cut / kept-silence problem: the LLM picks cuts
    # against transcript-segment boundaries (which often fall mid-sentence),
    # then we round each boundary to the nearest detected silence within
    # SNAP_TOLERANCE_S. Disable with snap=False to compare A/B.
    cuts_raw = parse_cuts(raw, max_duration=duration)
    if snap and loud.get("silences"):
        cuts = snap_cuts_to_silence(cuts_raw, loud["silences"], tolerance_s=SNAP_TOLERANCE_S)
        snapped_n = sum(1 for a, b in zip(cuts_raw, cuts) if a != b)
        print(f"[pipeline] snapped {snapped_n}/{len(cuts_raw)} cut boundaries to silence "
              f"(tolerance {SNAP_TOLERANCE_S}s, {len(loud['silences'])} silences detected)")
    else:
        cuts = cuts_raw
        if snap:
            print("[pipeline] snap enabled but no silences in loudness cache — skipped")
    keeps_pre_trim = cuts_to_keeps(cuts, duration)
    cut_secs = sum(e - s for s, e in cuts)
    keep_secs_pre = sum(e - s for s, e in keeps_pre_trim)
    pct_cut = 100.0 * cut_secs / max(duration, 1e-6)
    print(f"\n[pipeline] {len(cuts)} cuts totalling {cut_secs:.1f}s ({pct_cut:.1f}% of {duration:.1f}s)")
    print(f"[pipeline] keeping {keep_secs_pre:.1f}s in {len(keeps_pre_trim)} segments (pre-trim)")

    # 5b. Within-keep silence trim. The LLM sees segment-level loudness summary
    # so it can't catch the 15s walking-around silences between two spoken
    # lines inside a keep. This stage compresses them.
    if trim and loud.get("silences"):
        keeps = trim_silences_within_keeps(
            keeps_pre_trim, loud["silences"],
            max_silence_s=TRIM_MAX_SILENCE_S, padding_s=TRIM_PADDING_S,
        )
        trimmed_secs = keep_secs_pre - sum(e - s for s, e in keeps)
        print(f"[pipeline] trimmed {trimmed_secs:.1f}s of inner silence "
              f"(max_silence={TRIM_MAX_SILENCE_S}s, padding={TRIM_PADDING_S}s) "
              f"-> {len(keeps)} sub-keeps")
    else:
        keeps = keeps_pre_trim
        if trim:
            print("[pipeline] trim enabled but no silences in loudness cache — skipped")
    keep_secs = sum(e - s for s, e in keeps)

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
        "keep_total_s_pre_trim": keep_secs_pre,
        "keep_total_s": keep_secs,
        "n_sub_keeps": len(keeps),
        "keep_ranges": keeps,
        "keep_ranges_pre_trim": keeps_pre_trim if trim else None,
        "cut_ranges": cuts,
        "cut_ranges_pre_snap": cuts_raw if snap else None,
        "snap_enabled": snap,
        "trim_enabled": trim,
        "final_path": final_path if not dry_run else None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    save_json(os.path.join(out_dir, f"summary_iter{iteration}.json"), summary)
    print(f"\n[pipeline] DONE in {summary['elapsed_s']}s -> {final_path}")
    return summary
