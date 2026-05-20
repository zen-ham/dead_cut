"""LLM prompts for cut detection.

Design notes:
- The transcript is rendered as one line per segment, prefixed by [start-end]
  in HH:MM:SS so the model never has to do arithmetic.
- Loudness is summarised compactly per segment: P=peak_db M=mean_db L=loud_frac.
- The model emits FOUR blocks: HIGHLIGHTS (engagement proof), CANDIDATES
  (first-pass candidates), AUDIT (forced arithmetic — sum the draft, compute
  cut percentage, decide if over budget and which drafts to drop), then the
  FINAL CUTS block which is the only one the parser reads. This scratchpad
  structure forces non-thinking models to do reasoning IN their output —
  by the time they emit final CUTS, they've already written the audit math
  in their own context and committed to dropping over-budget candidates.
  Caught the failure mode where a 3hr vod got 94% cut because the model
  enumerated boring stretches without summing them.
- The parser only reads CUTS_BEGIN..CUTS_END, so CANDIDATES / AUDIT timestamps
  are ignored.
"""

SYSTEM_PROMPT = """You are an expert video editor working on a streamer/vod \
re-cut. Your job: identify the BORING parts of a long lightly-edited video to \
cut so the remaining edit is as entertaining per minute as possible.

Most of these videos are gaming streams, podcasts, or commentary — the \
entertainment is the streamer's reactions, jokes, distinctive personality \
moments, hype reactions, and storytelling. The boring parts are usually game \
tutorial text being read out loud, slow navigation/walking, dead air, repeated \
"alrights" while figuring out what to do next, technical setup, and intro/outro \
filler.

YOUR OUTPUT MUST HAVE FOUR BLOCKS IN THIS EXACT ORDER:

  HIGHLIGHTS_BEGIN
  HH:MM:SS — brief note on the funny/interesting/memorable moment here
  HH:MM:SS — another highlight
  (list at least 6 specific highlights; be concrete about WHAT is funny)
  HIGHLIGHTS_END

  CANDIDATES_BEGIN
  HH:MM:SS-HH:MM:SS — short reason (quote a boring transcript line, or note "dead air L=0.0", "tutorial readout", "repeated 'alrights'", etc)
  HH:MM:SS-HH:MM:SS — short reason
  (list each candidate with a SPECIFIC reason on the same line — separated by " — ")
  CANDIDATES_END

  CRITICAL: CANDIDATES is a list of SPECIFIC boring ranges you identified \
in the transcript. It is NOT a partition of the runtime. If you find yourself \
emitting contiguous ranges that cover the whole video (range1 end == range2 \
start, etc.), you are CHUNKING, not analyzing — stop and re-read the \
transcript for actual boring sections. Each candidate must have a concrete \
reason you can point to. Aim for 10-50 distinct boring sections totaling \
30-60% of the runtime, not 60+ adjacent blocks totaling 100%.

  AUDIT_BEGIN
  Now sum your draft cuts and check the budget. Show the math:

  draft_total: <sum of all your CANDIDATES durations, in HH:MM:SS>
  duration:    <total video duration from the user prompt>
  cut_percent: <draft_total / duration as percent, e.g. 47%>
  verdict:     <WITHIN BUDGET if cut_percent < 65%, else OVER BUDGET>

  If OVER BUDGET, you MUST drop draft cuts to get under 65%. List which ones \
you're dropping with a short reason for each (e.g. "00:14:00-00:16:00 — only \
guessed it was boring, no strong signal"). Pick the cuts you're LEAST \
confident were genuinely boring. Then re-sum and re-verify until under 65%.

  IMPORTANT: Do NOT write the literal strings "CUTS_BEGIN" or "CUTS_END" \
anywhere in this AUDIT block — those are reserved for the final block below.
  AUDIT_END

  CUTS_BEGIN
  <your CANDIDATES list, minus anything the audit told you to drop>
  HH:MM:SS-HH:MM:SS
  HH:MM:SS-HH:MM:SS
  CUTS_END

The parser only applies the CUTS_BEGIN..CUTS_END block. The other three blocks \
are your scratchpad — but they ARE mandatory. A response that skips CANDIDATES \
or AUDIT, or where the AUDIT math doesn't add up, indicates a lazy / sloppy \
edit and will be rejected.

HARD RULES FOR CUTS:
1. Each cut range is HH:MM:SS-HH:MM:SS (or MM:SS-MM:SS), one per line. Use ONE \
format consistently across ALL ranges. NEVER mix formats within a single range \
(e.g. `00:00:45-02:30:00` is wrong because the left is HH:MM:SS=45s but the \
right would parse as 2.5 hours). Always use leading zeros (`00:00:45`, not `45`).
2. Cuts must be SPECIFIC — target boring sections you actually identified. Do \
NOT chunk the runtime into equal blocks. Do NOT cut a single giant range. \
Cut ranges should be varying lengths (10 seconds to a few minutes). The user \
prompt below specifies how many cuts are appropriate for THIS video's length \
— hitting a too-low count for a long video means you missed boring sections.
2b. Cuts do NOT have to be in chronological order. If you finish a first pass \
and realize you missed some boring stretches, just append more ranges to \
CANDIDATES. Order doesn't matter — the parser sorts them.
3. NEVER list a range you want to keep. Only put ranges to REMOVE inside the \
CUTS_BEGIN/CUTS_END block.
4. NEVER cut a HIGHLIGHT you listed above — that's a contradiction.
5. Don't over-fragment. Each individual cut should be at least ~15 seconds \
long; sub-10s micro-cuts make the result feel twitchy without saving \
meaningful time.
6. The CUTS_END line must be the last thing in your response.

LOUDNESS HINT: each transcript line includes P (peak dB), M (mean dB), L (loud \
fraction). Loudness is relative to this video's own mean. High L values \
(>0.5) often indicate hype reactions, laughter, or shouting — usually worth \
keeping. Very low M values across a long stretch indicate dead air — usually \
worth cutting.
"""


USER_PROMPT_HEADER = """Video duration: {duration_str} ({duration_sec:.0f} seconds)
Target cut: roughly 30-60% of runtime (i.e. keep {min_keep_str}-{max_keep_str}). \
If the video is mostly gold, cut less. If most is filler, cut more — but you \
MUST keep at least {min_keep_str} of runtime ({min_keep_sec:.0f} seconds). A \
response that cuts more than 65% will be rejected — that's why the AUDIT step \
exists, to catch yourself before you submit an over-aggressive list.

EXPECTED CUT COUNT for a video this length: aim for {cut_target_low}-{cut_target_high} \
distinct cut ranges. Fewer than {cut_target_low_strict} means you're being lazy — \
a long video has many more boring stretches than a short one, and emitting only a \
handful of giant cuts wastes the granularity.

Transcript follows. Each line: [start-end] P=<peak_dB> M=<mean_dB> L=<loud_frac> | text

TRANSCRIPT:
"""


def format_segment_line(seg: dict, loud: dict) -> str:
    """One transcript line for the prompt."""
    return (
        f"[{_hms(seg['start'])}-{_hms(seg['end'])}] "
        f"P={loud['peak_db']:>5.1f} M={loud['mean_db']:>5.1f} L={loud['loud_frac']:.2f} "
        f"| {seg['text']}"
    )


def _hms(t: float) -> str:
    t = max(0, int(round(t)))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_user_prompt(duration: float, segments: list, loudness_per_seg: list) -> str:
    # Tell the model the minimum keep window in concrete seconds. Without this
    # the v0 prompt got a 90%-cut response.
    min_keep_sec = duration * 0.35
    max_keep_sec = duration * 0.70
    # Cut count target: scale with duration so the model doesn't emit ~20 cuts
    # on a 3hr video the same way it did on a 45min one. Baseline observed:
    # ~0.5 cuts/min on early test runs. We push for higher to combat laziness.
    duration_min = duration / 60.0
    cut_target_low = max(8, int(round(duration_min * 0.6)))
    cut_target_high = max(15, int(round(duration_min * 1.2)))
    cut_target_low_strict = max(5, int(round(duration_min * 0.4)))
    header = USER_PROMPT_HEADER.format(
        duration_str=_hms(duration),
        duration_sec=duration,
        min_keep_str=_hms(min_keep_sec),
        max_keep_str=_hms(max_keep_sec),
        min_keep_sec=min_keep_sec,
        cut_target_low=cut_target_low,
        cut_target_high=cut_target_high,
        cut_target_low_strict=cut_target_low_strict,
    )
    lines = [header]
    for seg, loud in zip(segments, loudness_per_seg):
        lines.append(format_segment_line(seg, loud))
    return "\n".join(lines)
