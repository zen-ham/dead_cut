"""LLM prompts for cut detection.

Design notes:
- The transcript is rendered as one line per segment, prefixed by [start-end]
  in HH:MM:SS so the model never has to do arithmetic.
- Loudness is summarised compactly per segment: P=peak_db M=mean_db L=loud_frac.
- The model is required to emit a HIGHLIGHTS block FIRST, then a brief REASONING
  block, then the CUTS block. The HIGHLIGHTS block forces actual engagement with
  content — without it, lazy models just block-chunk the entire runtime.
- The parser only reads cut ranges from inside CUTS_BEGIN / CUTS_END, so any
  timestamps mentioned in the reasoning or highlights blocks are ignored. The
  model can freely say "keep 9:13-10:40" in prose without it being cut.
- A duration-aware minimum-keep is injected into the prompt to prevent the model
  from cutting 90% of the runtime in one lazy block.
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

YOUR OUTPUT MUST HAVE THREE BLOCKS IN THIS ORDER:

  HIGHLIGHTS_BEGIN
  HH:MM:SS — brief note on the funny/interesting/memorable moment here
  HH:MM:SS — another highlight
  (list at least 6 specific highlights; be concrete about WHAT is funny)
  HIGHLIGHTS_END

  REASONING_BEGIN
  A short paragraph (3-6 sentences) explaining the cut strategy: what kinds of \
sections are boring in THIS specific video, which big chunks you're removing, \
and why you're keeping the parts you're keeping.
  REASONING_END

  CUTS_BEGIN
  HH:MM:SS-HH:MM:SS
  HH:MM:SS-HH:MM:SS
  CUTS_END

The HIGHLIGHTS block is mandatory — it proves you actually read the transcript. \
A response that skips straight to CUTS_BEGIN will be rejected. The HIGHLIGHTS \
should be specific (quote the line or describe the action), not generic.

HARD RULES FOR CUTS:
1. Each cut range is HH:MM:SS-HH:MM:SS (or MM:SS-MM:SS), one per line. Use ONE \
format consistently across ALL ranges. NEVER mix formats within a single range \
(e.g. `00:00:45-02:30:00` is wrong because the left is HH:MM:SS=45s but the \
right would parse as 2.5 hours). Always use leading zeros (`00:00:45`, not `45`).
2. Cuts must be SPECIFIC — target boring sections you actually identified. Do \
NOT chunk the runtime into equal blocks. Do NOT cut a single giant range. \
Typical good output has 5-30 cut ranges of varying lengths (10 seconds to a \
few minutes).
3. NEVER list a range you want to keep. Only put ranges to REMOVE inside \
CUTS_BEGIN/CUTS_END.
4. NEVER cut a HIGHLIGHT you listed above — that's a contradiction.
5. Don't over-fragment. Keep the cut count reasonable. Each individual cut \
should be at least ~15 seconds long; sub-10s micro-cuts make the result feel \
twitchy without saving meaningful time.
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
response that cuts more than 65% will be rejected.

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
    header = USER_PROMPT_HEADER.format(
        duration_str=_hms(duration),
        duration_sec=duration,
        min_keep_str=_hms(min_keep_sec),
        max_keep_str=_hms(max_keep_sec),
        min_keep_sec=min_keep_sec,
    )
    lines = [header]
    for seg, loud in zip(segments, loudness_per_seg):
        lines.append(format_segment_line(seg, loud))
    return "\n".join(lines)
