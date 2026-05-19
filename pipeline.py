"""End-to-end orchestrator."""
import os
import subprocess
import time

from .cache import cache_dir, video_id_from_url, save_json
from .download import download
from .transcribe import transcribe
from .loudness import analyze as analyze_loudness
from .llm import detect_cuts
from .parser import parse_cuts, cuts_to_keeps, snap_cuts_to_silence, trim_silences_within_keeps, enforce_budget
from .cutter import cut_video
from . import progress


# Snap tolerance: how far a cut boundary may be moved to land on a detected
# silence. 2.0s catches typical transcript-segment misalignment without
# distorting the model's intent.
SNAP_TOLERANCE_S = 2.0

# Within-keep silence trim. The LLM does macro cuts well but can't see fine
# dead air between sentences — these compress long silences to a small gap.
TRIM_MAX_SILENCE_S = 0.6
TRIM_PADDING_S = 0.2


def _ffprobe_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def _hms(t: float) -> str:
    t = max(0, int(round(t)))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _print_cut_stats(n_cuts: int, duration_s: float) -> None:
    """Print AI cut stats as soon as the LLM returns: count, target, ratio,
    'lazy' / 'on target' / 'aggressive' verdict. Lets the user see immediately
    whether the model engaged or phoned it in."""
    duration_min = duration_s / 60.0
    target_low = max(8, int(round(duration_min * 0.6)))
    target_high = max(15, int(round(duration_min * 1.2)))
    lazy_floor = max(5, int(round(duration_min * 0.4)))
    actual_ratio = n_cuts / duration_min
    target_ratio_low = target_low / duration_min
    target_ratio_high = target_high / duration_min

    if n_cuts < lazy_floor:
        verdict = "BELOW LAZY FLOOR (model phoned it in)"
    elif n_cuts < target_low:
        verdict = "under target (slightly lazy)"
    elif n_cuts <= target_high:
        verdict = "ON TARGET"
    else:
        verdict = "above target (aggressive)"

    print()
    print(f"  [ai cut stats]")
    print(f"    produced:       {n_cuts} cut ranges")
    print(f"    target:         {target_low}-{target_high} cuts")
    print(f"    actual ratio:   {actual_ratio:.2f} cuts/min")
    print(f"    target ratio:   {target_ratio_low:.2f}-{target_ratio_high:.2f} cuts/min")
    print(f"    verdict:        {verdict}")
    print()


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
    progress.begin_stage("download")
    video_path = download(url, vid)
    progress.end_stage("download")

    # Source duration is known after download — initialise the pipeline-wide
    # progress tracker so all subsequent stages drive both their own bar and
    # the overall bar.
    try:
        src_dur = _ffprobe_duration(video_path)
        progress.init(src_dur)
        print(f"[pipeline] source ~{src_dur/60.0:.1f} min\n")
    except Exception:
        pass

    # 2. Transcribe
    progress.begin_stage("transcribe")
    transcript = transcribe(video_path, vid)
    progress.end_stage("transcribe")
    segments = transcript["segments"]
    duration = transcript["duration"]

    # 3. Loudness
    progress.begin_stage("loudness")
    loud = analyze_loudness(video_path, vid, segments)
    progress.end_stage("loudness")
    loud_per_seg = loud["per_segment"]

    # 4. LLM cut detection
    progress.begin_stage("llm")
    model, raw = detect_cuts(vid, duration, segments, loud_per_seg,
                             iteration=iteration, force_model=force_model)
    progress.end_stage("llm")

    # 5. Parse + snap-to-silence + invert.
    # Snap fixes the mid-word-cut / kept-silence problem: the LLM picks cuts
    # against transcript-segment boundaries (which often fall mid-sentence),
    # then we round each boundary to the nearest detected silence within
    # SNAP_TOLERANCE_S. Disable with snap=False to compare A/B.
    progress.begin_stage("post")
    cuts_raw = parse_cuts(raw, max_duration=duration)

    # Show the AI's cut decision IMMEDIATELY — user can see whether the model
    # engaged or was lazy before waiting on snap/trim/encode.
    _print_cut_stats(len(cuts_raw), duration)

    # Final safety net: if the model + revision both failed to stay under the
    # 65% budget, drop the longest cuts programmatically. This runs even when
    # detect_cuts already triggered a revision — that's intentional: it only
    # trims if STILL over budget after revision.
    cuts_raw, was_trimmed = enforce_budget(cuts_raw, duration, ceiling_frac=0.65)
    if was_trimmed:
        new_total = sum(e - s for s, e in cuts_raw)
        new_pct = 100.0 * new_total / max(duration, 1e-6)
        print(
            f"\n[WARNING] programmatic drop applied — LLM still over budget after "
            f"revision. Trimmed to {len(cuts_raw)} cuts ({new_pct:.1f}% of source). "
            f"This is the safety net firing because the model couldn't self-correct.\n"
        )

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
    print(f"[pipeline] {len(cuts)} cuts totalling {cut_secs:.1f}s ({pct_cut:.1f}% of {duration:.1f}s)")
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
    progress.end_stage("post")

    # 6. Cut. Per-iter file for debugging; also copy to final.mp4 as the
    # canonical "latest" output.
    final_path = os.path.join(out_dir, f"final_iter{iteration}.mp4")
    if dry_run:
        print("[pipeline] DRY RUN — skipping ffmpeg cut")
    else:
        progress.begin_stage("encode")
        cut_video(video_path, keeps, final_path, work_dir=out_dir)
        progress.end_stage("encode")
        import shutil
        shutil.copyfile(final_path, os.path.join(out_dir, "final.mp4"))

    progress.close()

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

    # End-of-run summary block: before/after timings now that we know the
    # actual trimmed output size. Print after `[pipeline] DONE`.
    pct_kept = 100.0 * keep_secs / max(duration, 1e-6)
    print()
    print(f"  [final summary]")
    print(f"    source:         {_hms(duration)}")
    print(f"    after LLM cuts: {_hms(keep_secs_pre)}  ({100*keep_secs_pre/duration:.1f}% of source)")
    print(f"    after trim:     {_hms(keep_secs)}  ({pct_kept:.1f}% of source)")
    print(f"    total removed:  {_hms(duration-keep_secs)}  ({100-pct_kept:.1f}%)")
    print(f"    output:         {final_path}")
    print(f"    elapsed:        {summary['elapsed_s']}s")
    print()
    return summary
