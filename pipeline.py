"""End-to-end orchestrator."""
import os
import subprocess
import time

from .cache import cache_dir, video_id_from_url, save_json, load_json
from .download import download, fetch_duration
from .transcribe import transcribe
from .loudness import analyze as analyze_loudness
from .llm import detect_cuts
from .target import compute_ai_cut_target, estimate_silence_trim_ratio
from . import stats as _stats
from .parser import (
    parse_cuts, cuts_to_keeps, snap_cuts_to_silence, trim_silences_within_keeps,
    enforce_budget, extract_highlights_from_response, protect_highlights,
    merge_close_cuts, merge_close_keeps,
)
from .cutter import cut_video
from . import progress


# Snap tolerance: how far a cut boundary may be moved to land on a detected
# silence. 2.0s catches typical transcript-segment misalignment without
# distorting the model's intent.
SNAP_TOLERANCE_S = 2.0

# Within-keep silence trim. The LLM does macro cuts well but can't see fine
# dead air between sentences — these compress long silences to a small gap.
TRIM_MAX_SILENCE_S = 0.6
TRIM_PADDING_S = 0.15

# Post-trim merge: after silence trim creates many short skips between
# sub-keeps, absorb any skip ≤ this duration that contains no transcript
# speech. Re-flows micro-cuts back into longer takes so playback is less
# choppy without losing actual dialogue.
KEEP_MERGE_MAX_GAP_S = 1.5


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


def _elapsed_fmt(t: float) -> str:
    if t < 60:
        return f"{t:.1f}s"
    m, s = divmod(int(round(t)), 60)
    return f"{m}m {s}s"


def _announce_cached_stages(out_dir: str, iteration: int) -> None:
    """Peek at the cache before any stage runs and shrink the baseline for
    each stage whose output is already on disk. Without this, the overall ETA
    starts assuming a full download+transcribe+loudness even when re-running
    `--iter N` on a fully-cached video (mostly LLM work). 0.3s is realistic
    for a json/path cache hit."""
    CACHED = 0.3
    checks = {
        "download":   "vid_src.mp4",
        "transcribe": "transcribe.json",
        "loudness":   "loudness.json",
        "llm":        f"llm_iter{iteration}.json",
    }
    for stage, filename in checks.items():
        if os.path.exists(os.path.join(out_dir, filename)):
            progress.set_stage_baseline(stage, CACHED)


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
        verdict = "above target (long cuts ok if specific)"

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

    # Init progress tracker BEFORE download so the overall bar covers it.
    # Fetch source duration via cheap yt-dlp metadata call FIRST so the
    # tracker's baselines (transcribe, encode etc. scale with source) are
    # sized correctly from t=0. Without this, a 3hr vod gets sized as a
    # 30min placeholder and the overall bar shows 60%+ during download.
    cached_source = os.path.join(out_dir, "source.mp4")
    if os.path.exists(cached_source) and os.path.getsize(cached_source) > 0:
        try:
            src_dur_pre = _ffprobe_duration(cached_source)
        except Exception:
            src_dur_pre = None
    else:
        print(f"[pipeline] fetching source metadata to size progress baselines...")
        src_dur_pre = fetch_duration(url)
    if src_dur_pre:
        print(f"[pipeline] source ~{src_dur_pre/60.0:.1f} min\n")
    progress.init(src_dur_pre)
    _announce_cached_stages(out_dir, iteration)

    # Pre-warm faster-whisper in a background thread while the (often slow)
    # download is happening. The transcribe call will block on completion
    # if the load isn't done yet, but normally it'll be ready and we skip
    # the ~5s cold-load delay at the top of transcribe.
    try:
        from .transcribe import prewarm as _whisper_prewarm
        _whisper_prewarm()
    except Exception:
        pass

    # 1. Download
    _t = time.time()
    progress.begin_stage("download")
    video_path = download(url, vid)
    progress.end_stage("download")
    # elapsed_s passed only when the stage actually ran (> 0.5s). On a
    # cache hit the stage returns almost instantly and we don't want to
    # overwrite the real elapsed from when it last ran fresh.
    _elapsed = time.time() - _t
    _stats.save(out_dir, "download",
                elapsed_s=_elapsed if _elapsed >= 0.5 else None,
                url=url, video_id=vid, video_path=video_path,
                file_size_bytes=os.path.getsize(video_path) if os.path.exists(video_path) else None,
                source_duration_estimate_s=src_dur_pre)

    # Refine again with actual ffprobed duration (metadata duration can be
    # slightly off vs the actual decoded duration).
    src_dur = None
    try:
        src_dur = _ffprobe_duration(video_path)
        progress.set_source_duration(src_dur)
    except Exception:
        pass

    # 2. Transcribe
    _t = time.time()
    progress.begin_stage("transcribe")
    transcript = transcribe(video_path, vid)
    progress.end_stage("transcribe")
    segments = transcript["segments"]
    duration = transcript["duration"]
    n_words = sum(len(s.get("words") or []) for s in segments)
    _elapsed = time.time() - _t
    _stats.save(out_dir, "transcribe",
                elapsed_s=_elapsed if _elapsed >= 0.5 else None,
                duration_s=duration, language=transcript.get("language"),
                n_segments=len(segments), n_words=n_words,
                words_per_sec=round(n_words / max(duration, 1e-6), 3))

    # 3. Loudness
    _t = time.time()
    progress.begin_stage("loudness")
    loud = analyze_loudness(video_path, vid, segments)
    progress.end_stage("loudness")
    loud_per_seg = loud["per_segment"]
    _elapsed = time.time() - _t
    _stats.save(out_dir, "loudness",
                elapsed_s=_elapsed if _elapsed >= 0.5 else None,
                speech_level_db=loud.get("speech_level_db"),
                silence_threshold_db=loud.get("silence_threshold_db"),
                silence_threshold_min_db=loud.get("silence_threshold_min_db"),
                silence_threshold_max_db=loud.get("silence_threshold_max_db"),
                n_silences_audio_only=loud.get("n_silences_audio_only"),
                n_speech_gaps=loud.get("n_speech_gaps"),
                n_sentence_break_gaps=loud.get("n_sentence_break_gaps"),
                n_silences_dropped_sentence_break=loud.get("n_silences_dropped_sentence_break"),
                n_silences_dropped_loud_peak=loud.get("n_silences_dropped_loud_peak"),
                n_final_silences=len(loud.get("silences", [])))

    # 4. LLM cut detection. Dynamic cut target: longer videos go to a smaller
    # final length on a log curve, back-calculated to an AI cut % using the
    # estimated silence-trim ratio so the model only has to cut what's left
    # over the trim's contribution. Curve + math in target.py.
    silence_trim_ratio = estimate_silence_trim_ratio(
        loud.get("silences", []), duration,
        max_silence_s=TRIM_MAX_SILENCE_S, padding_s=TRIM_PADDING_S,
    )
    target = compute_ai_cut_target(duration, silence_trim_ratio)
    print(f"[pipeline] dynamic target: cut {target['ai_cut_pct']}% "
          f"(floor {target['floor_pct']}%, ceiling {target['ceiling_pct']}%), "
          f"final keep target {target['target_final_keep_pct']}%, "
          f"silence-trim assist ≈ {silence_trim_ratio*100:.1f}%")
    _t = time.time()
    progress.begin_stage("llm")
    model, raw, primary_raw = detect_cuts(vid, duration, segments, loud_per_seg,
                                          target=target, iteration=iteration,
                                          force_model=force_model)
    progress.end_stage("llm")
    # llm_iter{N}.json captures the full per-revision response chain;
    # llm_stats.json captures the high-level summary for the visualizer.
    try:
        from .parser import parse_cuts as _parse_cuts
        _llm_cache = load_json(os.path.join(out_dir, f"llm_iter{iteration}.json"))
    except Exception:
        _llm_cache = {}
    _elapsed = time.time() - _t
    _stats.save(out_dir, "llm",
                elapsed_s=_elapsed if _elapsed >= 0.5 else None,
                model=model, iteration=iteration,
                target_pct=target["ai_cut_pct"],
                floor_pct=target["floor_pct"],
                ceiling_pct=target["ceiling_pct"],
                target_final_keep_pct=target["target_final_keep_pct"],
                silence_trim_ratio_estimate=round(silence_trim_ratio, 4),
                cut_pct_first=_llm_cache.get("cut_pct_first"),
                cut_pct_revised=_llm_cache.get("cut_pct_revised"),
                structure_response_present=bool(_llm_cache.get("structure_response")),
                revised_response_present=bool(_llm_cache.get("revised_response")),
                coverage_response_present=bool(_llm_cache.get("coverage_response")),
                uncovered_response_present=bool(_llm_cache.get("uncovered_response")),
                final_response_present=bool(_llm_cache.get("final_response")))

    # 5. Parse + snap-to-silence + invert.
    # Snap fixes the mid-word-cut / kept-silence problem: the LLM picks cuts
    # against transcript-segment boundaries (which often fall mid-sentence),
    # then we round each boundary to the nearest detected silence within
    # SNAP_TOLERANCE_S. Disable with snap=False to compare A/B.
    _t_post = time.time()
    progress.begin_stage("post")
    try:
        cuts_raw = parse_cuts(raw, max_duration=duration)
    except ValueError as e:
        progress.end_stage("post")
        progress.close()
        bar = "!" * 80
        print()
        print(bar)
        print(f"!! [FATAL] LLM did not produce a usable CUTS block")
        print("!!")
        print(f"!! Error: {e}")
        print("!!")
        print(f"!! detect_cuts already retried once with a corrective message")
        print(f"!! but the model still didn't emit CUTS_BEGIN..CUTS_END. This")
        print(f"!! is most common on very long vods (8h+) where the model")
        print(f"!! exhausts its planning before reaching the cut list, or on")
        print(f"!! degenerate samples that veer into other text formats.")
        print("!!")
        print(f"!! Try:")
        print(f"!!   - rerun with --iter N to get a different sample")
        print(f"!!   - rerun with --model qwen/qwen3.6-plus:free to try a")
        print(f"!!     different model")
        print(f"!!   - for very long vods, consider trimming the source first")
        print(bar)
        print()
        raise SystemExit(1)

    # Protect highlights programmatically: the model commits to a HIGHLIGHTS
    # list (entertaining moments to keep) but empirically can contradict it
    # in CUTS (especially after an under-floor revision asks for more cuts).
    # 3s padding (was 10s) is enough to preserve a one-line joke without
    # forcing long uncut sections around each highlight.
    highlights = extract_highlights_from_response(primary_raw)
    if highlights:
        before_n = len(cuts_raw)
        cuts_raw = protect_highlights(cuts_raw, highlights, padding_s=3.0)
        if len(cuts_raw) != before_n:
            print(f"[pipeline] protect_highlights split {before_n} cuts into "
                  f"{len(cuts_raw)} to preserve {len(highlights)} highlights "
                  f"(3s window each)")

    # Merge cuts that have a small gap that's TRULY silent. The model often
    # emits adjacent cuts with 1-5s gaps — some are intentional (a quick
    # quip / reaction), some are accidental (dead air between two boring
    # stretches the model labeled as separate). We only merge when the gap
    # falls inside a detected silence (no speech, no highlight) — otherwise
    # we'd swallow real content.
    before_n = len(cuts_raw)
    cuts_raw = merge_close_cuts(
        cuts_raw, max_gap_s=5.0,
        silences=loud.get("silences"),
        highlights=highlights,
    )
    if len(cuts_raw) != before_n:
        print(f"[pipeline] merged {before_n - len(cuts_raw)} adjacent cuts "
              f"(gap <5s, fully silent, no highlight)")

    # Show the AI's cut decision IMMEDIATELY — user can see whether the model
    # engaged or was lazy before waiting on snap/trim/encode.
    _print_cut_stats(len(cuts_raw), duration)

    # Final safety net: if the model + revision both failed to stay under the
    # 75% budget, drop the longest cuts programmatically. This runs even when
    # detect_cuts already triggered a revision — that's intentional: it only
    # trims if STILL over budget after revision.
    cuts_raw, was_trimmed = enforce_budget(cuts_raw, duration, ceiling_frac=target["ceiling_pct"] / 100.0)
    if was_trimmed:
        new_total = sum(e - s for s, e in cuts_raw)
        new_pct = 100.0 * new_total / max(duration, 1e-6)
        print(
            f"\n[WARNING] programmatic drop applied — LLM still over budget after "
            f"revision. Trimmed to {len(cuts_raw)} cuts ({new_pct:.1f}% cut). "
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
    print(f"[pipeline] {len(cuts)} cuts totalling {cut_secs:.1f}s ({pct_cut:.1f}% cut of {duration:.1f}s)")
    print(f"[pipeline] expected length after AI cuts: {_hms(keep_secs_pre)} "
          f"(was {_hms(duration)}, {100*(1-keep_secs_pre/duration):.1f}% cut)")

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

    # 5c. Absorb micro-skips back into bigger keeps when the gap is short
    # and contains no transcript speech. Silence trim is per-silence so
    # successive long silences create stacked tiny sub-keeps; this re-flows
    # those into single takes so playback isn't choppy.
    keeps_pre_merge = keeps
    keeps, n_merged_keeps, absorbed_s = merge_close_keeps(
        keeps, segments, max_gap_s=KEEP_MERGE_MAX_GAP_S,
    )
    if n_merged_keeps:
        print(f"[pipeline] merged {n_merged_keeps} adjacent sub-keep(s) "
              f"(gap ≤{KEEP_MERGE_MAX_GAP_S}s, no speech in gap) "
              f"-> {len(keeps)} sub-keeps, absorbed {absorbed_s:.1f}s of dead air "
              f"back into takes")
    keep_secs = sum(e - s for s, e in keeps)
    if trim and keep_secs != keep_secs_pre:
        print(f"[pipeline] expected length after silence trim: {_hms(keep_secs)} "
              f"(was {_hms(duration)}, {100*(1-keep_secs/duration):.1f}% cut)")
    progress.end_stage("post")
    _elapsed = time.time() - _t_post
    _stats.save(out_dir, "post",
                elapsed_s=_elapsed if _elapsed >= 0.5 else None,
                n_ai_cuts=len(cuts),
                ai_cut_seconds=cut_secs,
                ai_cut_pct=round(pct_cut, 2),
                n_highlights=len(highlights or []),
                n_keeps_pre_trim=len(keeps_pre_trim),
                n_sub_keeps_final=len(keeps),
                silence_trim_seconds=round(keep_secs_pre - keep_secs, 2) if trim else 0,
                merge_close_keeps_absorbed_s=round(absorbed_s, 2),
                merge_close_keeps_n=n_merged_keeps,
                budget_drop_fired=was_trimmed)

    # 6. Cut. One canonical output file, overwritten on each --iter rerun.
    # Per-iter debugging context lives in summary_iter{N}.json beside it.
    final_path = os.path.join(out_dir, "final.mp4")
    if dry_run:
        print("[pipeline] DRY RUN — skipping ffmpeg cut")
    else:
        # Refine the encode baseline now that we know exact output duration.
        # ~0.020x of output for smartcut+nvenc (the default H.264 path).
        # Full-reencode fallback is ~0.10x; observed-rate reports from
        # cutter push the estimate up live if that path runs.
        progress.set_stage_baseline("encode", keep_secs * 0.025)
        _t_enc = time.time()
        progress.begin_stage("encode")
        cut_video(video_path, keeps, final_path, work_dir=out_dir)
        progress.end_stage("encode")
        _elapsed = time.time() - _t_enc
        _stats.save(out_dir, "encode",
                    elapsed_s=_elapsed if _elapsed >= 0.5 else None,
                    n_keep_ranges=len(keeps),
                    output_seconds=round(keep_secs, 2),
                    output_path=final_path,
                    output_size_bytes=os.path.getsize(final_path) if os.path.exists(final_path) else None,
                    source_size_bytes=os.path.getsize(video_path) if os.path.exists(video_path) else None)

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

    # JSON cut/keep export for external NLEs (DaVinci, Premiere) or post-
    # processing scripts. Schema: source duration, cut list, keep list,
    # highlight markers — all in source seconds.
    try:
        from .export import write_cuts_json
        # extract_highlights_from_response is module-level imported at the top
        # of this file; do NOT re-import it locally — Python's scope analysis
        # would then treat it as a function-local in run() and break the
        # earlier reference at the post stage (UnboundLocalError).
        _hl = []
        try:
            _hl = extract_highlights_from_response(primary_raw) or []
        except Exception:
            pass
        _cuts_path = write_cuts_json(
            out_dir, duration_s=duration, cuts=cuts, keeps=keeps,
            highlights=_hl,
            metadata={"video_id": vid, "url": url, "model": model,
                      "iteration": iteration},
        )
        print(f"[pipeline] cuts JSON: {_cuts_path}")
    except Exception as e:
        print(f"[pipeline] cuts JSON export failed: {e}")

    # Visualize: always (re)generate pipeline_visual.png. Reads ALL cached
    # stats + intermediate data files + the freshly-written summary, so it
    # works even when stages were cache-hit and didn't contribute new stats
    # this run. Must come AFTER summary_iter is written (viz reads keep_ranges
    # + keep_total_s from there).
    _t_viz = time.time()
    try:
        from .visualize import render as _render_viz
        _viz_path = _render_viz(vid)
        if _viz_path:
            print(f"[pipeline] visualization: {_viz_path}")
        _stats.save(out_dir, "visualize",
                    elapsed_s=time.time() - _t_viz,
                    output_path=_viz_path)
    except Exception as e:
        print(f"[pipeline] visualize failed: {e}")

    # End-of-run summary block: before/after timings + cut counts now that
    # we know the actual trimmed output size. Print after `[pipeline] DONE`.
    pct_cut_total = 100.0 * (1 - keep_secs / max(duration, 1e-6))
    pct_cut_after_ai = 100.0 * (1 - keep_secs_pre / max(duration, 1e-6))
    ai_removed = max(0.0, duration - keep_secs_pre)
    silence_removed = max(0.0, keep_secs_pre - keep_secs)
    # Each silence removed inside a keep splits 1 keep into 2 sub-keeps,
    # so the silence cut count equals the increase in keep-range count.
    silence_cut_count = max(0, len(keeps) - len(keeps_pre_trim))
    print()
    print(f"  [final summary]")
    print(f"    source duration:       {_hms(duration)}")
    print(f"    AI macro cuts:         {len(cuts)}  (removed {_hms(ai_removed)})")
    print(f"    after AI cuts:         {_hms(keep_secs_pre)}  ({pct_cut_after_ai:.1f}% cut)")
    print(f"    silence micro-cuts:    {silence_cut_count}  (removed {_hms(silence_removed)})")
    print(f"    after silence trim:    {_hms(keep_secs)}  ({pct_cut_total:.1f}% cut)")
    print(f"    total cut:             {_hms(duration-keep_secs)}  ({pct_cut_total:.1f}%)")
    print(f"    output:                {final_path}")
    print(f"    elapsed:               {_elapsed_fmt(summary['elapsed_s'])}")
    print()
    return summary
