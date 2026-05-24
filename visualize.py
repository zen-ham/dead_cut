"""Build a single-PNG summary of a dead_cut pipeline run.

Reads every available stats_*.json + loudness.json + llm_iter*.json + the
summary file under cache/<vid>/, draws a stacked layout of timelines and
a time-breakdown bar, saves to cache/<vid>/pipeline_visual.png.

Always regenerated when called. Falls back gracefully when a stage's data
is missing (e.g. a partial run, or rendering an older cache pre-stats).

CLI: python -m dead_cut.visualize <video_id>
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no Tk required
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .cache import cache_dir, load_json
from . import stats as _stats


# Dark-mode palette. All concepts bright enough to stand out against
# the dark panel background.
COLORS = {
    # Foreground content (filled = "this is detected / present")
    "audio_silence":     "#fb8500",  # orange — audio-detected silences
    "speech_gap":        "#facc15",  # yellow — transcription-derived silences
    "highlight":         "#22d3ee",  # cyan — preserve / protected zones
    # AI keeps (filled = "this would play in the final"). Progress from
    # cool purple early stages to warm green at the final stage.
    "ai_primary":        "#a371f7",
    "ai_structure":      "#c08af7",
    "ai_revised":        "#b178f0",
    "ai_uncovered":      "#d670d6",
    "ai_final":          "#3fb950",  # bright green for final keeps
    # Stage colors for the time-breakdown bar
    "stage_download":    "#f85149",  # red
    "stage_transcribe":  "#fb8500",  # orange
    "stage_loudness":    "#ffc233",  # yellow
    "stage_llm":         "#3fb950",  # green
    "stage_post":        "#58a6ff",  # blue
    "stage_encode":      "#bc8cff",  # purple
    "stage_visualize":   "#6e7681",  # grey
    # Theme
    "bg":                "#0d1117",  # page background (github dark)
    "panel":             "#161b22",  # panel / track background
    "track":             "#1c2128",  # individual timeline track bg
    "text":              "#e6edf3",  # primary text
    "text_dim":          "#8b949e",  # secondary text
    "border":            "#30363d",  # subtle separators
}


def _fmt_secs(s: float | None) -> str:
    if s is None:
        return "(cached)"
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m}m {sec}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {sec}s"


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n = n / 1024
    return f"{n:.1f}TB"


def _hms(t: float) -> str:
    t = max(0, int(round(t)))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _draw_intervals(ax, intervals, duration, color, y=0.5, height=0.8, alpha=1.0):
    """Draw a list of (start, end) intervals as filled rectangles on ax.
    ax x-axis is in source seconds [0, duration]."""
    for s, e in intervals or []:
        if e <= s:
            continue
        ax.add_patch(Rectangle((s, y - height / 2), e - s, height,
                                facecolor=color, edgecolor="none", alpha=alpha))


def _draw_envelope(ax, envelope_db, duration, color, alpha=0.45):
    """Draw the audio dB envelope as a faint line behind a timeline row.
    Normalizes dB to [0, 1] (where 1.0 = -10dB, 0 = -60dB) so loud peaks
    span the full row height. Drawn at high z-order under the intervals."""
    if not envelope_db or duration <= 0:
        return
    import numpy as np
    arr = np.array(envelope_db, dtype=np.float32)
    # Normalize dB range -60..-10 → 0..1
    norm = np.clip((arr - (-60.0)) / 50.0, 0.0, 1.0)
    xs = np.linspace(0, duration, len(arr))
    ax.fill_between(xs, 0.05, 0.05 + norm * 0.9, color=color,
                    alpha=alpha, linewidth=0, zorder=0)


def _strip_axis(ax, duration, label):
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    n_ticks = 8
    ticks = [i * duration / n_ticks for i in range(n_ticks + 1)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_hms(t) for t in ticks], fontsize=7, color=COLORS["text_dim"])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9,
                  labelpad=8, color=COLORS["text"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["border"])
    ax.tick_params(left=False, colors=COLORS["text_dim"])
    ax.set_facecolor(COLORS["track"])


def _compute_quality(cuts, duration, target):
    """Score the AI cut output on 4 pass/fail criteria:
       in_band:      total cut % within target [floor, ceiling]
       not_chunked:  no single cut > 15min
       covered:      largest uncut region < 25% of source
       balanced:     cuts spread evenly across video quartiles
                     (stddev / mean of per-quartile count < 0.6)
    Returns (label, color_hex, checks_dict).
    """
    if not cuts or duration <= 0:
        return "no-data", COLORS["text_dim"], {}
    cut_total = sum(e - s for s, e in cuts)
    cut_pct = 100.0 * cut_total / duration
    in_band = True
    if target:
        in_band = target.get("floor_pct", 0) <= cut_pct <= target.get("ceiling_pct", 100)
    max_cut = max(e - s for s, e in cuts)
    not_chunked = max_cut <= 900.0  # 15 min absolute
    # Coverage: longest uncut gap
    sorted_cuts = sorted(cuts)
    gaps = [sorted_cuts[0][0]]
    for i in range(1, len(sorted_cuts)):
        gaps.append(sorted_cuts[i][0] - sorted_cuts[i - 1][1])
    gaps.append(duration - sorted_cuts[-1][1])
    longest_gap = max(gaps)
    covered = (longest_gap / duration) < 0.25
    # Balance: count cuts per quartile, compute coefficient of variation
    n_quartiles = 4
    q_dur = duration / n_quartiles
    counts = [0] * n_quartiles
    for s, _ in cuts:
        q = min(int(s / q_dur), n_quartiles - 1)
        counts[q] += 1
    mean = sum(counts) / n_quartiles
    if mean > 0:
        var = sum((c - mean) ** 2 for c in counts) / n_quartiles
        cov = (var ** 0.5) / mean
        balanced = cov < 0.6
    else:
        balanced = False
    checks = {
        "in_band": in_band,
        "not_chunked": not_chunked,
        "covered": covered,
        "balanced": balanced,
    }
    passes = sum(checks.values())
    if passes == 4:
        return "BALANCED ✓", COLORS["ai_final"], checks
    if passes == 3:
        return "OK", "#facc15", checks
    if passes == 2:
        return "UNEVEN ⚠", COLORS["audio_silence"], checks
    return "POOR ✗", "#f85149", checks


def _invert_intervals(intervals, duration):
    """Return the complement of `intervals` in [0, duration]."""
    if duration <= 0:
        return []
    sorted_iv = sorted(intervals or [])
    out = []
    cursor = 0.0
    for s, e in sorted_iv:
        if s > cursor:
            out.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration:
        out.append((cursor, duration))
    return out


def _clip_to_keeps(intervals, keeps, min_dur_s=0.6):
    """For each interval, keep only the portion(s) that overlap any keep
    region and exceed min_dur_s. Used to compute which detected silences
    would actually get trimmed (only those inside AI keep regions, big
    enough to trigger the trim)."""
    out = []
    for s, e in intervals or []:
        for ks, ke in keeps or []:
            if ks >= e:
                break
            if ke <= s:
                continue
            ov_s, ov_e = max(s, ks), min(e, ke)
            if ov_e - ov_s >= min_dur_s:
                out.append((ov_s, ov_e))
    return out


def _parse_cuts_safe(text, max_duration):
    if not text:
        return []
    try:
        from .parser import parse_cuts
        return parse_cuts(text, max_duration=max_duration)
    except Exception:
        return []


def _cut_block_from_response(resp, max_duration):
    """Extract the LAST CUTS_BEGIN..CUTS_END block from a raw LLM response
    and parse it. Returns [] if missing or malformed."""
    return _parse_cuts_safe(resp, max_duration)


def render(video_id: str, out_path: str | None = None) -> str | None:
    """Build the PNG. Returns the output path, or None if nothing to render."""
    out_dir = cache_dir(video_id)
    all_stats = _stats.load_all(out_dir)

    # Determine source duration from the most reliable available stat.
    transcribe_stats = all_stats.get("transcribe")
    loudness = _load_if(os.path.join(out_dir, "loudness.json"))
    duration = None
    if transcribe_stats and transcribe_stats.get("duration_s"):
        duration = float(transcribe_stats["duration_s"])
    elif loudness and loudness.get("per_segment"):
        try:
            tr = _load_if(os.path.join(out_dir, "transcribe.json"))
            if tr:
                duration = float(tr.get("duration") or 0)
        except Exception:
            pass
    if not duration:
        # Last-ditch from download stats.
        d_stats = all_stats.get("download") or {}
        duration = float(d_stats.get("source_duration_estimate_s") or 0)
    if not duration:
        print(f"[visualize] no duration info available for {video_id}, skipping")
        return None

    # Load AI cut data from llm_iter*.json (pick the latest iter present).
    llm_cache = _load_latest_llm_iter(out_dir)
    primary_cuts = _parse_cuts_safe(llm_cache.get("response"), duration) if llm_cache else []
    structure_cuts = _parse_cuts_safe(llm_cache.get("structure_response"), duration) if llm_cache else []
    revised_cuts = _parse_cuts_safe(llm_cache.get("revised_response"), duration) if llm_cache else []
    coverage_cuts = _parse_cuts_safe(llm_cache.get("coverage_response"), duration) if llm_cache else []
    uncovered_cuts = _parse_cuts_safe(llm_cache.get("uncovered_response"), duration) if llm_cache else []
    final_cuts = _parse_cuts_safe(
        (llm_cache.get("final_response") or llm_cache.get("uncovered_response")
         or llm_cache.get("coverage_response") or llm_cache.get("revised_response")
         or llm_cache.get("structure_response") or llm_cache.get("response")) if llm_cache else None,
        duration,
    )

    # Highlights (gold protected zones)
    highlights = []
    if llm_cache and llm_cache.get("response"):
        try:
            from .parser import extract_highlights_from_response
            highlights = extract_highlights_from_response(llm_cache["response"]) or []
        except Exception:
            pass

    # Per-type silence lists. New loudness.json (post-stats-rework) stores
    # them separately; older caches just have the union.
    final_silences = loudness.get("silences", []) if loudness else []
    audio_silences = (loudness.get("silences_audio_only_list")
                      if loudness else None) or []
    speech_gap_silences = (loudness.get("silences_speech_gaps_list")
                           if loudness else None) or []
    # If new fields are missing (old cache), fall back to splitting the union
    # — we can't truly distinguish but treat all as audio for the orange row.
    if loudness and not audio_silences and final_silences:
        audio_silences = [list(s) for s in final_silences]

    # Summary file for keep ranges (final, post-trim, post-merge_close_keeps)
    summary = _find_summary(out_dir)
    keep_ranges = summary.get("keep_ranges") if summary else None

    # Compute "what was actually cut by what mechanism" for the combined row.
    # Three non-overlapping interval lists:
    #   1. AI cuts (purple): from final_cuts
    #   2. Audio silences inside AI keeps, trimmable size: orange
    #   3. Speech-gap silences inside AI keeps, trimmable size: yellow
    ai_keeps_pre_trim = _invert_intervals(final_cuts, duration) if final_cuts else []
    audio_trimmed = _clip_to_keeps(audio_silences, ai_keeps_pre_trim, min_dur_s=0.6)
    speech_trimmed = _clip_to_keeps(speech_gap_silences, ai_keeps_pre_trim, min_dur_s=0.6)

    # Layout: list of (label, intervals, color, alpha)
    timeline_rows = []
    if audio_silences:
        timeline_rows.append(("audio silences\n(detected)", audio_silences,
                              COLORS["audio_silence"], 0.85))
    if speech_gap_silences:
        timeline_rows.append(("transcript-based\nsilences (detected)",
                              speech_gap_silences, COLORS["speech_gap"], 0.85))
    if highlights:
        hl_pad = min(30.0, max(3.0, duration * 0.0025))
        timeline_rows.append(("highlights\n(protected)",
                              [(h - hl_pad, h + hl_pad) for h in highlights],
                              COLORS["highlight"], 1.0))
    if primary_cuts:
        timeline_rows.append(("AI primary\nCUTS", primary_cuts,
                              COLORS["ai_primary"], 0.85))
    if structure_cuts:
        timeline_rows.append(("AI structure rev\nCUTS", structure_cuts,
                              COLORS["ai_structure"], 0.85))
    if revised_cuts:
        timeline_rows.append(("AI budget rev\nCUTS", revised_cuts,
                              COLORS["ai_revised"], 0.85))
    if coverage_cuts:
        timeline_rows.append(("AI coverage rev\nCUTS", coverage_cuts,
                              COLORS["ai_uncovered"], 0.85))
    if uncovered_cuts:
        timeline_rows.append(("AI uncovered rev\nCUTS", uncovered_cuts,
                              COLORS["ai_uncovered"], 0.85))
    if final_cuts:
        timeline_rows.append(("AI FINAL\nCUTS", final_cuts,
                              COLORS["ai_final"], 1.0))
    if keep_ranges:
        timeline_rows.append(("FINAL keeps\n(plays)", keep_ranges,
                              COLORS["ai_final"], 1.0))

    # Add the combined "what was removed by what mechanism" row last.
    # Three non-overlapping interval sets layered on one axis.
    has_combined = bool(final_cuts or audio_trimmed or speech_trimmed)

    n_timelines = len(timeline_rows) + (1 if has_combined else 0)
    fig_width = 16.0
    header_h = 1.05         # was 0.55 — text was vertically overlapping
    breakdown_h = 0.9
    details_h = 1.7
    timeline_each = 0.45
    combined_extra = 0.35   # combined row gets a bit more height for its legend
    timelines_h = (n_timelines - (1 if has_combined else 0)) * timeline_each \
                  + (timeline_each + combined_extra if has_combined else 0)
    fig_height = header_h + breakdown_h + details_h + timelines_h + 0.6

    # Build per-row heights including the taller combined row
    row_heights = [timeline_each] * len(timeline_rows)
    if has_combined:
        row_heights.append(timeline_each + combined_extra)

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=COLORS["bg"])
    gs = fig.add_gridspec(
        nrows=3 + n_timelines, ncols=1,
        height_ratios=[header_h, breakdown_h, details_h] + row_heights,
        hspace=0.6, left=0.10, right=0.985, top=0.97, bottom=0.04,
    )

    # ----- Header -----
    ax_h = fig.add_subplot(gs[0])
    ax_h.axis("off")
    ax_h.set_facecolor(COLORS["bg"])
    download_stats = all_stats.get("download") or {}
    url = download_stats.get("url", "?")
    keep_secs = (summary or {}).get("keep_total_s") or 0
    model = (all_stats.get('llm') or {}).get('model', '?')
    # Spread the three lines more — was 0.95 / 0.55 / 0.20 (overlapped on a
    # 0.55-inch axis). Now 0.85 / 0.45 / 0.05 on a 1.05-inch axis.
    ax_h.text(0.0, 0.85, f"dead_cut — pipeline visual for {video_id}",
              fontsize=16, weight="bold", va="top", color=COLORS["text"])
    ax_h.text(0.0, 0.45, f"source: {url}",
              fontsize=9, va="top", color=COLORS["text_dim"])
    ax_h.text(0.0, 0.10,
              f"source duration: {_hms(duration)}   |   final cut: {_hms(keep_secs)}   "
              f"({100*keep_secs/duration:.1f}% kept)   |   model: {model}",
              fontsize=9, va="top", color=COLORS["text"])

    # Quality badge — top right. Pass/fail across 4 metrics on the AI cut
    # output: in-band, not-chunked, well-covered, balanced distribution.
    ll_stats = all_stats.get("llm") or {}
    target_for_quality = {
        "floor_pct": ll_stats.get("floor_pct", 0),
        "ceiling_pct": ll_stats.get("ceiling_pct", 100),
    }
    quality_label, quality_color, quality_checks = _compute_quality(
        final_cuts, duration, target_for_quality,
    )
    # Badge background
    badge_w, badge_h = 0.18, 0.50
    badge_x, badge_y = 1.0 - badge_w, 0.45
    ax_h.add_patch(Rectangle(
        (badge_x, badge_y), badge_w, badge_h,
        facecolor=quality_color, edgecolor="none", alpha=0.9,
    ))
    ax_h.text(badge_x + badge_w / 2, badge_y + badge_h - 0.10,
              quality_label, fontsize=11, weight="bold",
              ha="center", va="top", color=COLORS["bg"])
    # Check icons under the label. Single-space separator so the 4 checks
    # fit inside the badge horizontally (the 3-space version overflowed both
    # ends at fontsize=7 with the label "balanced" being the widest).
    if quality_checks:
        check_line = " ".join(
            f"{'✓' if v else '✗'}{k}" for k, v in quality_checks.items()
        )
        ax_h.text(badge_x + badge_w / 2, badge_y + 0.08,
                  check_line, fontsize=7, ha="center", va="bottom",
                  color=COLORS["bg"])

    # ----- Time breakdown bar -----
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_facecolor(COLORS["bg"])
    _draw_time_breakdown(ax_b, all_stats, duration=duration)

    # ----- Per-stage details list -----
    ax_d = fig.add_subplot(gs[2])
    ax_d.set_facecolor(COLORS["bg"])
    _draw_details(ax_d, all_stats, loudness, llm_cache, summary, duration)

    # ----- Timeline rows -----
    envelope = (loudness or {}).get("envelope_db") if loudness else None
    for i, (label, intervals, color, alpha) in enumerate(timeline_rows):
        ax = fig.add_subplot(gs[3 + i])
        # Draw envelope as a faint background ONLY on the audio silences row,
        # so the user can visually confirm silences land on quiet dB regions.
        if envelope and label.startswith("audio silences"):
            _draw_envelope(ax, envelope, duration, COLORS["text_dim"], alpha=0.25)
        _draw_intervals(ax, intervals, duration, color, alpha=alpha)
        # Append "X% of src" to the row label so each timeline shows what
        # fraction of the source the colored bars cover. Reader can interpret
        # as % cut / % protected / % kept depending on the row's semantics.
        total_s = sum(e - s for s, e in (intervals or []))
        pct = 100.0 * total_s / max(duration, 1e-6)
        _strip_axis(ax, duration, f"{label}\n{pct:.1f}% of src")
        ax.text(0.998, 0.5, f"n={len(intervals)}", ha="right", va="center",
                transform=ax.transAxes, fontsize=7, color=COLORS["text_dim"])

    # ----- Combined "removed by what" row -----
    if has_combined:
        ax_c = fig.add_subplot(gs[3 + len(timeline_rows)])
        # Draw intervals in BOTTOM portion only (y=0.15 to 0.65) so the top
        # of the axis (0.7-0.95) is free for the inline legend.
        for src_intervals, color in [
            (final_cuts,      COLORS["ai_primary"]),
            (audio_trimmed,   COLORS["audio_silence"]),
            (speech_trimmed,  COLORS["speech_gap"]),
        ]:
            _draw_intervals(ax_c, src_intervals, duration, color, y=0.4, height=0.5, alpha=0.9)
        combined_s = (sum(e - s for s, e in final_cuts)
                      + sum(e - s for s, e in audio_trimmed)
                      + sum(e - s for s, e in speech_trimmed))
        combined_pct = 100.0 * combined_s / max(duration, 1e-6)
        _strip_axis(ax_c, duration,
                    f"removed by\n(combined)\n{combined_pct:.1f}% of src")
        leg_items = [
            ("AI cuts", COLORS["ai_primary"], len(final_cuts)),
            ("audio sil trim", COLORS["audio_silence"], len(audio_trimmed)),
            ("transcript sil trim", COLORS["speech_gap"], len(speech_trimmed)),
        ]
        leg_x = 0.005
        leg_y = 0.86  # well above interval top (0.65)
        for label, c, n in leg_items:
            ax_c.add_patch(Rectangle((leg_x, leg_y - 0.04), 0.008, 0.10,
                                      facecolor=c, edgecolor="none",
                                      transform=ax_c.transAxes))
            ax_c.text(leg_x + 0.013, leg_y, f"{label} ({n})",
                      fontsize=8, va="center",
                      color=COLORS["text"], transform=ax_c.transAxes)
            leg_x += 0.14

    if out_path is None:
        out_path = os.path.join(out_dir, "pipeline_visual.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    plt.close(fig)
    return out_path


def _fmt_realtime(elapsed_s, duration_s) -> str:
    """Format a 'Nx realtime' multiplier — how many source seconds were
    processed per wall second. Empty string when either value is missing or
    elapsed is sub-second (no meaningful rate)."""
    if not elapsed_s or not duration_s or elapsed_s < 0.05:
        return ""
    rt = duration_s / elapsed_s
    if rt >= 10:
        return f"{rt:.0f}x realtime"
    return f"{rt:.1f}x realtime"


def _draw_time_breakdown(ax, all_stats, duration=None):
    """GitHub-style stacked horizontal bar showing % of pipeline time per stage.
    Skips stages that were cached this run (elapsed < 0.5s) so the breakdown
    represents actual work done in the current pipeline run.
    `duration` is the source video duration; when provided, the header also
    shows the overall pipeline rate vs source length (Nx realtime)."""
    # Skip "post" entirely — it's a programmatic post-processing step that
    # always runs in <0.5s and would only clutter the breakdown bar with a
    # "sub-second: post" annotation that carries no useful information. Its
    # actual stats (final cut count etc.) are visible in other rows.
    stages_order = ["download", "transcribe", "loudness", "llm", "encode"]
    elapsed_by = {}
    cached_by = {}   # genuinely a cache-hit this run (had a prior real value)
    fast_by = {}     # ran fresh but in <0.5s (no bar segment, but not cached)
    for s in stages_order:
        st = all_stats.get(s) or {}
        if not st:
            continue
        e = st.get("elapsed_s")
        if st.get("_cached_this_run"):
            cached_by[s] = e
            continue
        if e is None or e < 0.5:
            fast_by[s] = e
            continue
        elapsed_by[s] = float(e)
    if not elapsed_by and not cached_by and not fast_by:
        ax.axis("off")
        ax.text(0.5, 0.5, "no timing data", ha="center", va="center",
                fontsize=10, color=COLORS["text_dim"])
        return
    total = sum(elapsed_by.values()) if elapsed_by else 0
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    notes = []
    if cached_by:
        notes.append(f"cached this run: {', '.join(cached_by.keys())}")
    if fast_by:
        notes.append(f"sub-second: {', '.join(fast_by.keys())}")
    notes_str = "   |   " + "   |   ".join(notes) if notes else ""
    overall_rt = _fmt_realtime(total, duration)
    rt_note = f"   |   {overall_rt}" if overall_rt else ""
    ax.text(0.0, 1.05,
            f"pipeline time breakdown — total {_fmt_secs(total)}{rt_note}{notes_str}",
            fontsize=10, weight="bold", va="bottom", color=COLORS["text"])
    cursor = 0.0
    bar_y = 0.55
    bar_h = 0.30
    for s in stages_order:
        if s not in elapsed_by:
            continue
        frac = elapsed_by[s] / total
        ax.add_patch(Rectangle((cursor, bar_y), frac, bar_h,
                                facecolor=COLORS[f"stage_{s}"],
                                edgecolor=COLORS["bg"], linewidth=2))
        cursor += frac
    # Legend row below
    leg_y = 0.05
    leg_x = 0.0
    for s in stages_order:
        if s not in elapsed_by:
            continue
        frac = elapsed_by[s] / total
        ax.add_patch(Rectangle((leg_x, leg_y), 0.012, 0.22,
                                facecolor=COLORS[f"stage_{s}"], edgecolor="none",
                                transform=ax.transAxes))
        ax.text(leg_x + 0.018, leg_y + 0.11,
                f"{s}: {_fmt_secs(elapsed_by[s])} ({100*frac:.0f}%)",
                fontsize=8, va="center", color=COLORS["text"],
                transform=ax.transAxes)
        leg_x += 0.16


def _draw_details(ax, all_stats, loudness, llm_cache, summary, duration):
    """Render a multi-column stage detail block. Each stage = one block."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    blocks = []  # (title, [lines])

    # Each stage's "Nx realtime" is the source-duration / wall-elapsed
    # ratio for that stage. Skipped (empty string) on cached/<0.5s stages.
    def _rt(stats):
        return _fmt_realtime(stats.get("elapsed_s"), duration)

    d_stats = all_stats.get("download") or {}
    if d_stats:
        blocks.append(("download", [
            f"{_fmt_secs(d_stats.get('elapsed_s'))}",
            _rt(d_stats),
            f"file: {_fmt_bytes(d_stats.get('file_size_bytes'))}",
            f"dur (est): {_hms(d_stats.get('source_duration_estimate_s') or 0)}",
        ]))

    t_stats = all_stats.get("transcribe") or {}
    if t_stats:
        blocks.append(("transcribe", [
            f"{_fmt_secs(t_stats.get('elapsed_s'))}",
            _rt(t_stats),
            f"lang: {t_stats.get('language') or '?'}",
            f"segs: {t_stats.get('n_segments')}, words: {t_stats.get('n_words')}",
            f"{t_stats.get('words_per_sec', 0):.2f} words/s",
        ]))

    l_stats = all_stats.get("loudness") or {}
    if l_stats:
        thr_range = ""
        if l_stats.get("silence_threshold_min_db") is not None:
            thr_range = f"{l_stats.get('silence_threshold_min_db'):.0f}..{l_stats.get('silence_threshold_max_db'):.0f}dB"
        blocks.append(("loudness", [
            f"{_fmt_secs(l_stats.get('elapsed_s'))}",
            _rt(l_stats),
            f"speech: {l_stats.get('speech_level_db', '?')}dB",
            f"silence thr: {thr_range}",
            f"audio sil: {l_stats.get('n_silences_audio_only', '?')}",
            f"speech gaps: {l_stats.get('n_speech_gaps', '?')}",
            f"dropped: {l_stats.get('n_silences_dropped_sentence_break', 0)} sent-brk, "
            f"{l_stats.get('n_silences_dropped_loud_peak', 0)} loud-pk",
            f"final sil: {l_stats.get('n_final_silences', '?')}",
        ]))

    ll_stats = all_stats.get("llm") or {}
    if ll_stats:
        flow = []
        if ll_stats.get("structure_response_present"): flow.append("structure")
        if ll_stats.get("revised_response_present"): flow.append("budget")
        if ll_stats.get("coverage_response_present"): flow.append("coverage")
        if ll_stats.get("uncovered_response_present"): flow.append("uncovered")
        blocks.append(("llm", [
            f"{_fmt_secs(ll_stats.get('elapsed_s'))}",
            _rt(ll_stats),
            f"model: {(ll_stats.get('model') or '?').split('/')[-1].split(':')[0]}",
            f"target: cut {ll_stats.get('target_pct')}%",
            f"primary: {(ll_stats.get('cut_pct_first') or 0):.1f}% cut",
            f"revisions: {' → '.join(flow) if flow else 'none fired'}",
        ]))

    # POST block intentionally omitted — it's always sub-second and its
    # interesting outputs (final cut count, silence trim seconds, sub-keep
    # count) are already conveyed by the timeline rows + the FINAL keeps row.

    e_stats = all_stats.get("encode") or {}
    if e_stats:
        blocks.append(("encode", [
            f"{_fmt_secs(e_stats.get('elapsed_s'))}",
            _rt(e_stats),
            f"output: {_hms(e_stats.get('output_seconds') or 0)}",
            f"size: {_fmt_bytes(e_stats.get('output_size_bytes'))}",
            f"keeps: {e_stats.get('n_keep_ranges')}",
        ]))

    # Layout: blocks side-by-side. Filter blank lines (e.g. realtime is empty
    # on cached / sub-second stages) so subsequent stats shift up.
    if not blocks:
        return
    n = len(blocks)
    block_w = 1.0 / n
    for i, (title, lines) in enumerate(blocks):
        x = i * block_w
        ax.text(x + 0.005, 0.95, title.upper(), fontsize=9, weight="bold",
                color=COLORS.get(f"stage_{title}", COLORS["text"]), va="top")
        lines = [ln for ln in lines if ln]
        for j, ln in enumerate(lines):
            # Dim "Nx realtime" lines slightly so they read as a sub-stat of
            # the elapsed line above them, not a peer detail.
            color = COLORS["text_dim"] if ln.endswith("realtime") else COLORS["text"]
            ax.text(x + 0.005, 0.82 - j * 0.10, ln, fontsize=8,
                    va="top", color=color)


def _load_if(path):
    if not os.path.exists(path):
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def _load_latest_llm_iter(out_dir):
    import glob
    candidates = sorted(glob.glob(os.path.join(out_dir, "llm_iter*.json")))
    if not candidates:
        return None
    return _load_if(candidates[-1])


def _find_summary(out_dir):
    import glob
    candidates = sorted(glob.glob(os.path.join(out_dir, "summary_iter*.json")))
    if not candidates:
        return None
    return _load_if(candidates[-1])


# ---------------------------------------------------------------------------
# Iter-vs-iter diff visualization
# ---------------------------------------------------------------------------

# Diff palette — distinct from the main viz palette so the two PNGs don't
# look confusingly similar.
DIFF_COLORS = {
    "common":   "#3fb950",  # green — kept by both
    "only_a":   "#58a6ff",  # blue — kept only by iter A
    "only_b":   "#fb8500",  # orange — kept only by iter B
}


def _interval_diff(a_intervals, b_intervals, duration):
    """Classify the source timeline [0, duration] into three non-overlapping
    interval lists: common (kept by both), only_a (kept by A but cut by B),
    only_b (kept by B but cut by A). Implementation: boundary sweep — collect
    all endpoints, walk consecutive pairs, classify each by midpoint
    membership."""
    if duration <= 0:
        return [], [], []
    boundaries = {0.0, float(duration)}
    for s, e in (a_intervals or []):
        boundaries.add(float(s))
        boundaries.add(float(e))
    for s, e in (b_intervals or []):
        boundaries.add(float(s))
        boundaries.add(float(e))
    pts = sorted(b for b in boundaries if 0.0 <= b <= duration)

    def _in(intervals, t):
        for s, e in intervals or []:
            if s <= t < e:
                return True
        return False

    common, only_a, only_b = [], [], []
    for i in range(len(pts) - 1):
        s, e = pts[i], pts[i + 1]
        if e - s < 1e-6:
            continue
        mid = (s + e) / 2.0
        in_a = _in(a_intervals, mid)
        in_b = _in(b_intervals, mid)
        if in_a and in_b:
            common.append((s, e))
        elif in_a:
            only_a.append((s, e))
        elif in_b:
            only_b.append((s, e))

    def _merge(lst):
        if not lst:
            return []
        lst = sorted(lst)
        out = [list(lst[0])]
        for s, e in lst[1:]:
            if s <= out[-1][1] + 1e-6:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return [(s, e) for s, e in out]

    return _merge(common), _merge(only_a), _merge(only_b)


def _load_iter_summary(out_dir, iteration):
    """Load summary_iter{N}.json. Returns None if missing."""
    return _load_if(os.path.join(out_dir, f"summary_iter{iteration}.json"))


def _load_iter_llm(out_dir, iteration):
    """Load llm_iter{N}.json. Returns None if missing."""
    return _load_if(os.path.join(out_dir, f"llm_iter{iteration}.json"))


def render_diff(
    video_id: str, iter_a: int, iter_b: int, out_path: str | None = None,
) -> str | None:
    """Render a PNG comparing two iterations of the LLM cut for the same
    video. Shows each iter's keep timeline, a combined diff timeline, and
    a stats block comparing model / cut count / quality side-by-side.

    Output filename: pipeline_diff_iter{A}_vs_iter{B}.png in the cache dir.
    Returns the path, or None if either iter's summary is missing.
    """
    out_dir = cache_dir(video_id)
    sum_a = _load_iter_summary(out_dir, iter_a)
    sum_b = _load_iter_summary(out_dir, iter_b)
    if not sum_a or not sum_b:
        missing = [f"iter{n}" for n, s in [(iter_a, sum_a), (iter_b, sum_b)] if not s]
        print(f"[visualize] cannot diff — missing summary for: {', '.join(missing)}")
        return None

    duration = float(sum_a.get("original_duration_s")
                     or sum_b.get("original_duration_s") or 0)
    if duration <= 0:
        print(f"[visualize] no duration info; cannot diff")
        return None

    keeps_a = [tuple(k) for k in (sum_a.get("keep_ranges") or [])]
    keeps_b = [tuple(k) for k in (sum_b.get("keep_ranges") or [])]
    common, only_a, only_b = _interval_diff(keeps_a, keeps_b, duration)

    # Per-iter quality on AI cuts (use cut_ranges, mirroring the main viz).
    llm_a = _load_iter_llm(out_dir, iter_a) or {}
    llm_b = _load_iter_llm(out_dir, iter_b) or {}
    target_a = {"floor_pct": llm_a.get("target_floor_pct", 0),
                "ceiling_pct": llm_a.get("target_ceiling_pct", 100)}
    target_b = {"floor_pct": llm_b.get("target_floor_pct", 0),
                "ceiling_pct": llm_b.get("target_ceiling_pct", 100)}
    cuts_a = [tuple(c) for c in (sum_a.get("cut_ranges") or [])]
    cuts_b = [tuple(c) for c in (sum_b.get("cut_ranges") or [])]
    q_a_label, q_a_color, _ = _compute_quality(cuts_a, duration, target_a)
    q_b_label, q_b_color, _ = _compute_quality(cuts_b, duration, target_b)

    # Layout: header + stats block + 3 timelines + legend.
    fig_width = 16.0
    header_h = 0.85
    stats_h = 1.5
    timeline_each = 0.55
    legend_h = 0.45
    n_timelines = 3
    fig_height = header_h + stats_h + n_timelines * timeline_each + legend_h + 0.5

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=COLORS["bg"])
    gs = fig.add_gridspec(
        nrows=3 + n_timelines, ncols=1,
        height_ratios=[header_h, stats_h] + [timeline_each] * n_timelines + [legend_h],
        hspace=0.55, left=0.10, right=0.985, top=0.96, bottom=0.05,
    )

    # ----- Header -----
    ax_h = fig.add_subplot(gs[0])
    ax_h.axis("off")
    ax_h.set_facecolor(COLORS["bg"])
    ax_h.text(0.0, 0.85,
              f"dead_cut diff — iter {iter_a} vs iter {iter_b}  ({video_id})",
              fontsize=15, weight="bold", va="top", color=COLORS["text"])
    common_s = sum(e - s for s, e in common)
    only_a_s = sum(e - s for s, e in only_a)
    only_b_s = sum(e - s for s, e in only_b)
    ax_h.text(0.0, 0.30,
              f"source: {_hms(duration)}   |   "
              f"shared keeps: {_hms(common_s)} ({100*common_s/duration:.1f}%)   "
              f"only iter {iter_a}: {_hms(only_a_s)}   "
              f"only iter {iter_b}: {_hms(only_b_s)}",
              fontsize=9, va="top", color=COLORS["text_dim"])

    # ----- Stats comparison block (two side-by-side columns) -----
    ax_s = fig.add_subplot(gs[1])
    ax_s.axis("off")
    ax_s.set_facecolor(COLORS["bg"])
    ax_s.set_xlim(0, 1)
    ax_s.set_ylim(0, 1)
    col_w = 0.48
    for i, (label, summary, llm, q_label, q_color) in enumerate([
        (f"iter {iter_a}", sum_a, llm_a, q_a_label, q_a_color),
        (f"iter {iter_b}", sum_b, llm_b, q_b_label, q_b_color),
    ]):
        x = 0.0 if i == 0 else 0.50
        ax_s.text(x, 0.95, label.upper(), fontsize=11, weight="bold",
                  color=COLORS["text"], va="top")
        # Quality badge inline.
        bw, bh = 0.13, 0.18
        ax_s.add_patch(Rectangle((x + 0.10, 0.78), bw, bh,
                                  facecolor=q_color, edgecolor="none", alpha=0.9))
        ax_s.text(x + 0.10 + bw / 2, 0.78 + bh / 2, q_label,
                  fontsize=9, weight="bold", ha="center", va="center",
                  color=COLORS["bg"])
        model = (llm.get("model") or "?").split("/")[-1].split(":")[0]
        keep_s = summary.get("keep_total_s") or 0
        pct_kept = 100.0 * keep_s / max(duration, 1e-6)
        lines = [
            f"model:       {model}",
            f"AI cuts:     {summary.get('n_cuts', 0)}  "
            f"({summary.get('pct_cut', 0):.1f}% of source)",
            f"sub-keeps:   {summary.get('n_sub_keeps', 0)}",
            f"final kept:  {_hms(keep_s)}  ({pct_kept:.1f}%)",
            f"elapsed:     {_fmt_secs(summary.get('elapsed_s'))}",
        ]
        for j, ln in enumerate(lines):
            ax_s.text(x, 0.70 - j * 0.13, ln, fontsize=9,
                      color=COLORS["text"], va="top", family="monospace")

    # ----- Timeline rows -----
    # Row 1: iter A keeps (full-row blue bars)
    ax1 = fig.add_subplot(gs[2])
    _draw_intervals(ax1, keeps_a, duration, DIFF_COLORS["only_a"], alpha=0.85)
    _strip_axis(ax1, duration, f"iter {iter_a}\nkeeps")
    ax1.text(0.998, 0.5, f"n={len(keeps_a)}", ha="right", va="center",
             transform=ax1.transAxes, fontsize=7, color=COLORS["text_dim"])

    # Row 2: iter B keeps (full-row orange bars)
    ax2 = fig.add_subplot(gs[3])
    _draw_intervals(ax2, keeps_b, duration, DIFF_COLORS["only_b"], alpha=0.85)
    _strip_axis(ax2, duration, f"iter {iter_b}\nkeeps")
    ax2.text(0.998, 0.5, f"n={len(keeps_b)}", ha="right", va="center",
             transform=ax2.transAxes, fontsize=7, color=COLORS["text_dim"])

    # Row 3: combined diff — color-coded common/only_a/only_b on one axis.
    ax3 = fig.add_subplot(gs[4])
    _draw_intervals(ax3, common, duration, DIFF_COLORS["common"], alpha=0.95)
    _draw_intervals(ax3, only_a, duration, DIFF_COLORS["only_a"], alpha=0.95)
    _draw_intervals(ax3, only_b, duration, DIFF_COLORS["only_b"], alpha=0.95)
    _strip_axis(ax3, duration, "diff\n(combined)")

    # ----- Legend -----
    ax_l = fig.add_subplot(gs[5])
    ax_l.axis("off")
    ax_l.set_facecolor(COLORS["bg"])
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(0, 1)
    leg_items = [
        (f"kept by both ({_hms(common_s)})",     DIFF_COLORS["common"]),
        (f"only iter {iter_a} ({_hms(only_a_s)})", DIFF_COLORS["only_a"]),
        (f"only iter {iter_b} ({_hms(only_b_s)})", DIFF_COLORS["only_b"]),
    ]
    lx = 0.0
    for label, c in leg_items:
        ax_l.add_patch(Rectangle((lx, 0.4), 0.015, 0.32,
                                  facecolor=c, edgecolor="none"))
        ax_l.text(lx + 0.022, 0.56, label, fontsize=9, va="center",
                  color=COLORS["text"])
        lx += 0.27

    if out_path is None:
        out_path = os.path.join(
            out_dir, f"pipeline_diff_iter{iter_a}_vs_iter{iter_b}.png"
        )
    fig.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--diff", nargs=2, metavar=("ITER_A", "ITER_B"), type=int,
                    help="Render a diff PNG comparing two iterations of the LLM"
                         " cut for this video, instead of the standard pipeline"
                         " visual.")
    args = ap.parse_args()
    if args.diff:
        path = render_diff(args.video_id, args.diff[0], args.diff[1],
                           out_path=args.output)
    else:
        path = render(args.video_id, out_path=args.output)
    if path:
        print(f"[visualize] wrote {path}")
    else:
        print(f"[visualize] no output produced (missing data)")
        sys.exit(1)


if __name__ == "__main__":
    main()
