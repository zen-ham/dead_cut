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

YOU MUST CUT AT LEAST 50% OF THE VIDEO. The user is editing a long vod \
down to a tight watch; under-cutting (keeping >50%) is failure. If your \
cuts add up to under 50% of the runtime, you have not done the job — go \
through the transcript again and find more boring stretches. Most lightly-\
edited vods have plenty to remove; if you can't find 50% to cut, you're \
not looking hard enough.

WHAT TO CUT (most of the video):

  - Silent navigation / walking with NO commentary
  - Long inventory shuffling, menu-staring, looking at the map
  - The streamer literally reading the game's tutorial / popup text aloud
  - "Anyway", "alright", "okay so", "where was I" filler with no follow-up
  - Drawn-out problem-solving where the streamer is just thinking out loud \
without jokes ("hmm so I need to... wait... no... maybe I should...")
  - Repeated complaints / "I'm scared" loops with no joke payoff
  - Intro/outro fluff (subscribe asks, donation reads, "thanks for watching")
  - Mid-energy commentary that's narrative but not funny — like the \
streamer describing what they're doing in the game without any joke
  - Any segment where you can summarise the whole thing in one sentence \
without losing comedy

WHAT TO KEEP (the entertainment, often quiet):

  - Sarcastic asides and deadpan delivery (often L < 0.3 — DO NOT cut)
  - Absurdist bits (saying the same word 20 times, weird tangents)
  - Dark humor and meta commentary about the game
  - Memory-callbacks, school-reference jokes, weird analogies
  - Hype moments, loud reactions, jump scares, big laughs
  - Any segment where there's a real joke, even if it's quiet

Loudness L is a HINT, NOT authoritative. L=0.0 for a sustained stretch = \
dead air, cut. L=0.1 with a deadpan joke = entertainment, keep. Always \
read the actual transcript text before deciding.

EACH CUT SHOULD BE LONG — at least 30 seconds, ideally 1-5 minutes. Don't \
emit 50 micro-cuts of 2-5 seconds each. There's a separate silence-trim \
step that handles micro pauses. Your job is to identify the BIG boring \
stretches.

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
Cut at LEAST 50% (target 50-75%). Output below 50% cut WILL be rejected — the \
user wants tight edits. Stay under 75% to keep enough content for it to flow. \
For this video specifically, aim to cut between {min_cut_str} and {max_cut_str}.

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
    # 50% MIN cut, 75% MAX cut.
    min_cut_sec = duration * 0.50
    max_cut_sec = duration * 0.75
    duration_min = duration / 60.0
    cut_target_low = max(5, int(round(duration_min * 0.25)))   # longer cuts now
    cut_target_high = max(10, int(round(duration_min * 0.5)))
    cut_target_low_strict = max(3, int(round(duration_min * 0.15)))
    header = USER_PROMPT_HEADER.format(
        duration_str=_hms(duration),
        duration_sec=duration,
        min_cut_str=_hms(min_cut_sec),
        max_cut_str=_hms(max_cut_sec),
        cut_target_low=cut_target_low,
        cut_target_high=cut_target_high,
        cut_target_low_strict=cut_target_low_strict,
    )
    lines = [header]
    for seg, loud in zip(segments, loudness_per_seg):
        lines.append(format_segment_line(seg, loud))
    return "\n".join(lines)
