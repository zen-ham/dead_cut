"""LLM prompts for cut detection.

Design history:
- v1: HIGHLIGHTS + REASONING + CUTS. Lazy on long vods (block-chunking).
- v2: + DRAFT_CUTS/CANDIDATES + AUDIT scratchpad. Audit was theatre — model
      faked the math AND used "candidates" as license to over-list (knowing
      revision would prune). Lazy on most vods at the FIRST-pass level,
      relying on the revision safety net to clean up.
- v3 (current): direct HIGHLIGHTS + CUTS with reason-per-range required, NO
      scratchpad. Model has to commit to the final cut on the first pass —
      no draft-and-prune escape hatch. Reasons force concrete content
      engagement per range. Revision in detect_cuts still fires if the
      result over-cuts, but it's a fallback for true model failure, not a
      load-bearing pass.
"""

SYSTEM_PROMPT = """You are an expert video editor working on a streamer/vod \
re-cut. Your job: identify the BORING parts of a long lightly-edited video to \
cut so the remaining edit is as entertaining per minute as possible.

Most of these videos are gaming streams, podcasts, or commentary — the \
entertainment is the streamer's reactions, jokes, distinctive personality \
moments, hype reactions, storytelling, AND dry / sarcastic / deadpan humor.

CRITICAL — humor is often QUIET. Sarcastic asides, deadpan delivery, \
absurdist bits (saying the same word 20 times, calling out game logic, \
making references), and dark comedy are usually delivered at LOW volume \
(L < 0.3). These are the entertainment. Do NOT cut them because they're \
quiet. Common mistakes to avoid:

  - "He says 'Ke-Ke-Ke-Ke-' 20 times, L=0.1, must be boring" → NO, that's \
an absurdist comedy bit, that's the show. Keep it.
  - "He's making jokes about a calculator trick in school, L=0.2, must be \
filler" → NO, that's a memory-callback joke, that's content. Keep it.
  - "He's dryly saying 'I do not care about your missing kid, man', L=0.15" \
→ NO, that's dark humor / sarcastic commentary about the game. Keep it.

What's ACTUALLY boring (and safe to cut):

  - Long silent navigation / walking with NO commentary at all (the lines \
in the transcript would literally be empty or just brief utility words like \
"there it is", "ok this way")
  - The streamer literally reading the game's tutorial / popup text out loud \
("Use WASD to move. Press F to interact.")
  - Inventory shuffling / menu-staring with no commentary
  - "Anyway", "alright", "okay so", "where was I" filler with no follow-up
  - Repeated "I'm scared" / "I don't want to go in there" loops without any \
joke or personality
  - Intro/outro filler (asking for likes, donation thanks, "thanks for \
watching" wrap-ups)

Loudness (L) is a HINT but not authoritative. L=0.0 over 30+ seconds is \
likely dead air worth cutting. L=0.1 with witty dialogue is NOT dead air — \
it's just quiet humor. ALWAYS read the actual transcript text before \
deciding a range is boring. If you can quote a joke / sarcastic line / \
personality moment from the range, it's NOT a cut candidate.

YOUR OUTPUT HAS EXACTLY TWO BLOCKS, IN THIS ORDER:

  HIGHLIGHTS_BEGIN
  HH:MM:SS — brief note quoting the funny/interesting/memorable moment here
  HH:MM:SS — another highlight
  (list at least 6 specific highlights with concrete quotes — these prove you \
read the transcript and they are the moments your final edit MUST preserve)
  HIGHLIGHTS_END

  CUTS_BEGIN
  HH:MM:SS-HH:MM:SS — quoted boring line OR specific reason (dead air L=0.0, \
tutorial readout, "X" repeated 4 times, etc)
  HH:MM:SS-HH:MM:SS — reason
  ...
  CUTS_END

CRITICAL — read carefully:

1. The CUTS block is your FINAL ANSWER. It gets applied directly. There is \
no draft step, no candidate step, no audit step. Every range you list will be \
cut from the video. Choose carefully.

2. CUTS is a list of SPECIFIC boring ranges, NOT a partition of the runtime. \
If your CUTS ranges are contiguous (range1 end == range2 start, etc.) and \
together cover the whole video, you are CHUNKING — that's wrong and the \
response will be rejected. The job is to identify ~10-40 SPECIFIC boring \
stretches, leaving the entertaining bits between them un-listed (they get \
kept by default).

3. Each cut MUST have a concrete reason on its line. Vague reasons like \
"boring section" indicate you didn't actually engage with that part of the \
transcript. If you can't quote a boring line or point to a specific signal, \
DON'T cut that range — it's probably fine to keep.

4. NEVER cut a HIGHLIGHT you listed above. The highlights are the moments \
your final edit MUST preserve. If your CUTS overlap a highlight time, \
contract or split the cut to spare it.

5. Format: HH:MM:SS-HH:MM:SS — reason. Use ONE format consistently. Always \
leading zeros (`00:00:45`, not `45`). NEVER mix formats within one range \
(e.g. `00:00:45-02:30:00` is wrong — left is HH:MM:SS=45s, right would parse \
as 2.5hr). Use any consistent dash character.

6. Each cut should be at least ~15 seconds. Sub-10s micro-cuts feel twitchy \
without saving meaningful time. Let the algo handle inner silence trimming.

7. Don't cut more than 75% of the runtime. If you're tempted to cut more, you \
probably are over-applying boring labels to mid-energy content that should \
stay. Aim for cuts totaling 30-60% of runtime.

8. The CUTS_END line must be the last thing in your response. Nothing after.

LOUDNESS HINT: each transcript line includes P (peak dB), M (mean dB), L \
(loud fraction). High L values (>0.5) usually indicate hype / laughter / \
shouting — usually worth keeping. Very low M across a long stretch is \
dead air — usually worth cutting.
"""


USER_PROMPT_HEADER = """Video duration: {duration_str} ({duration_sec:.0f} seconds)
Target cut: roughly 30-60% of runtime (i.e. keep {min_keep_str}-{max_keep_str}). \
If the video is mostly gold, cut less. If most is filler, cut more — but you \
MUST keep at least {min_keep_str} of runtime ({min_keep_sec:.0f} seconds). \
Cutting more than 75% will be rejected.

EXPECTED CUT COUNT for a video this length: aim for {cut_target_low}-{cut_target_high} \
distinct cut ranges. Fewer than {cut_target_low_strict} means you're being lazy. \
More than {cut_target_high} usually means you're partitioning the runtime into \
chunks instead of identifying specific boring sections — re-read the transcript \
for genuine boring stretches, don't enumerate everything.

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
    min_keep_sec = duration * 0.25
    max_keep_sec = duration * 0.70
    duration_min = duration / 60.0
    cut_target_low = max(8, int(round(duration_min * 0.5)))
    cut_target_high = max(15, int(round(duration_min * 0.8)))
    cut_target_low_strict = max(5, int(round(duration_min * 0.3)))
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
