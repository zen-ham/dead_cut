"""Parse cut ranges from LLM response.

Strict rule: only ranges inside the FINAL CUTS_BEGIN..CUTS_END block count.
Anything outside (reasoning prose, "keep these" mentions, etc.) is ignored.
"""
import bisect
import re
from typing import List, Tuple


# Accept any dash-like Unicode separator. LLMs love non-breaking hyphens (U+2011)
# and similar variants; restricting to ASCII/en/em silently drops their output.
_DASH = "\\-‐‑‒–—―−－"
_RANGE_RE = re.compile(
    rf"(\d{{1,2}}):(\d{{2}})(?::(\d{{2}}))?\s*[{_DASH}]\s*(\d{{1,2}}):(\d{{2}})(?::(\d{{2}}))?"
)


def _ts_to_sec(h: str, m: str, s: str | None) -> float:
    if s is None:
        # mm:ss form: first group is minutes, second is seconds.
        return int(h) * 60 + int(m)
    return int(h) * 3600 + int(m) * 60 + int(s)


def _extract_cut_block(text: str) -> str | None:
    """Return the content of the LAST CUTS_BEGIN..CUTS_END block, or None.

    Leading `\\b` matters — without it, a prefix like `DRAFT_CUTS_BEGIN` would
    match because `_` is a regex word char (no boundary between `T` and `_C`).
    With the leading `\\b`, we only match when "CUTS_BEGIN" is preceded by a
    non-word character (newline, whitespace, start of string)."""
    pat = re.compile(
        r"\bCUTS?_BEGIN\b(.*?)\bCUTS?_END\b",
        re.IGNORECASE | re.DOTALL,
    )
    matches = pat.findall(text)
    if not matches:
        return None
    return matches[-1]


def parse_cuts(llm_response: str, max_duration: float | None = None) -> List[Tuple[float, float]]:
    """Return sorted, non-overlapping list of (start_sec, end_sec) cut ranges.

    max_duration: if given, drop ranges where start exceeds duration and clamp
    end to duration. Catches models that emit malformed mixed mm:ss/HH:MM:SS
    ranges that parse to 2hr ranges on a 27min video (caught in test-video iter 3)."""
    block = _extract_cut_block(llm_response)
    if block is None:
        raise ValueError("No CUTS_BEGIN..CUTS_END block found in LLM response")

    ranges: List[Tuple[float, float]] = []
    dropped: List[Tuple[float, float, str]] = []
    for line in block.splitlines():
        line = line.strip().strip("-*•").strip()
        if not line:
            continue
        m = _RANGE_RE.search(line)
        if not m:
            continue
        h1, m1, s1, h2, m2, s2 = m.groups()
        start = _ts_to_sec(h1, m1, s1)
        end = _ts_to_sec(h2, m2, s2)
        if end <= start:
            dropped.append((float(start), float(end), "end <= start"))
            continue
        if max_duration is not None:
            if start >= max_duration:
                dropped.append((float(start), float(end), f"start beyond duration {max_duration:.0f}"))
                continue
            if end > max_duration:
                end = max_duration  # clamp instead of drop
        ranges.append((float(start), float(end)))

    if dropped:
        print(f"[parser] dropped {len(dropped)} malformed range(s):")
        for s, e, why in dropped:
            print(f"  ({s:.0f}, {e:.0f}) -- {why}")

    return _merge(ranges)


def _merge(ranges: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def snap_cuts_to_silence(
    cuts: List[Tuple[float, float]],
    silences: List[List[float]],
    tolerance_s: float = 2.0,
) -> List[Tuple[float, float]]:
    """Snap each cut's start to the nearest silence_start and end to the nearest
    silence_end within tolerance. This makes the surrounding KEEP regions:
      - end on speech (cut starts when silence begins, not mid-word)
      - begin on speech (cut ends when silence ends, not before next word)

    Without snap the LLM's cut times land on transcript-segment boundaries,
    which themselves often fall mid-sentence — producing mid-word cuts and
    keeping pre-speech silence inside the next clip.

    Args:
      cuts: original cut ranges in seconds.
      silences: list of [start, end] silence intervals from loudness analysis.
      tolerance_s: max distance to snap; boundaries outside this are left alone.
    """
    if not silences or not cuts:
        return cuts

    silence_starts = sorted(s[0] for s in silences)
    silence_ends = sorted(s[1] for s in silences)

    def _nearest(target: float, sorted_list: List[float], tol: float) -> float:
        i = bisect.bisect_left(sorted_list, target)
        cands = []
        if i > 0:
            cands.append(sorted_list[i - 1])
        if i < len(sorted_list):
            cands.append(sorted_list[i])
        if not cands:
            return target
        best = min(cands, key=lambda x: abs(x - target))
        return best if abs(best - target) <= tol else target

    snapped = []
    for cs, ce in cuts:
        new_cs = _nearest(cs, silence_starts, tolerance_s)
        new_ce = _nearest(ce, silence_ends, tolerance_s)
        if new_ce > new_cs:
            snapped.append((new_cs, new_ce))
    return _merge(snapped)


def trim_silences_within_keeps(
    keeps: List[Tuple[float, float]],
    silences: List[List[float]],
    max_silence_s: float = 0.6,
    padding_s: float = 0.2,
) -> List[Tuple[float, float]]:
    """Compress long silences that fall INSIDE keep ranges. The LLM does macro
    cuts well but can't see fine-grained dead air between sentences/words within
    a keep — e.g. a 15s walking-around silence between two spoken lines. We
    detect each silence (start, end) that overlaps a keep, and if it exceeds
    max_silence_s, we convert (start+padding_s, end-padding_s) into a skip,
    leaving padding_s of breathing room on each side of the surrounding speech.

    Args:
      keeps: original keep ranges (post-snap, pre-trim).
      silences: list of [start, end] silence intervals from loudness analysis.
      max_silence_s: silences shorter than this stay intact (preserves natural
        rhythm — typical inter-sentence pauses are 0.2-0.8s).
      padding_s: how much silence to leave at each end of the surrounding speech.
        Effective minimum trimmed-silence gap = 2 * padding_s.

    Returns:
      new list of keep ranges; same as input if no qualifying silences.
    """
    if not silences or not keeps:
        return keeps
    silence_list = sorted([(s[0], s[1]) for s in silences])
    new_keeps: List[Tuple[float, float]] = []
    for ks, ke in keeps:
        # Build the list of "skip" ranges inside this keep.
        skips: List[Tuple[float, float]] = []
        for ss, se in silence_list:
            if se <= ks or ss >= ke:
                continue  # no overlap
            ss_clip = max(ss, ks)
            se_clip = min(se, ke)
            if se_clip - ss_clip < max_silence_s:
                continue  # too short to compress
            skip_start = ss_clip + padding_s
            skip_end = se_clip - padding_s
            if skip_end > skip_start:
                skips.append((skip_start, skip_end))
        if not skips:
            new_keeps.append((ks, ke))
            continue
        # Invert skips within [ks, ke] -> sub-keeps.
        skips.sort()
        cursor = ks
        for cs, ce in skips:
            if cs > cursor:
                new_keeps.append((cursor, cs))
            cursor = max(cursor, ce)
        if cursor < ke:
            new_keeps.append((cursor, ke))
    return new_keeps


def merge_close_keeps(
    keeps: List[Tuple[float, float]],
    segments: List[dict],
    max_gap_s: float = 1.5,
) -> Tuple[List[Tuple[float, float]], int, float]:
    """Merge adjacent sub-keeps separated by a short, speechless gap.

    After silence trim creates many short skips between sub-keeps, some of
    those skips are tiny dead-air slivers (0.2-1.5s) with no speech in
    them. Cutting on those produces choppy flashes. If the gap is short
    AND no transcript segment overlaps it, we can safely absorb the skip
    back into a single contiguous keep.

    A segment overlapping the gap means the transcript thinks someone is
    talking there — never merge across speech, that would drop dialogue.

    Returns:
        (new_keeps, n_merged, total_gap_absorbed_s)
    """
    if len(keeps) < 2:
        return keeps, 0, 0.0
    # Segments sorted by start for fast overlap check.
    seg_sorted = sorted(((s["start"], s["end"]) for s in segments), key=lambda x: x[0])
    seg_starts = [s[0] for s in seg_sorted]

    def speech_in(gap_start: float, gap_end: float) -> bool:
        if not seg_sorted:
            return False
        # First segment whose start >= gap_start, then walk back one to catch
        # a segment that starts before but extends into the gap.
        i = bisect.bisect_left(seg_starts, gap_start)
        for j in (i - 1, i):
            if 0 <= j < len(seg_sorted):
                ss, se = seg_sorted[j]
                if se > gap_start and ss < gap_end:
                    return True
        return False

    merged: List[Tuple[float, float]] = [keeps[0]]
    n_merged = 0
    absorbed_s = 0.0
    for ks, ke in keeps[1:]:
        prev_s, prev_e = merged[-1]
        gap = ks - prev_e
        if 0 < gap <= max_gap_s and not speech_in(prev_e, ks):
            merged[-1] = (prev_s, ke)
            n_merged += 1
            absorbed_s += gap
        else:
            merged.append((ks, ke))
    return merged, n_merged, absorbed_s


def merge_close_cuts(
    cuts: List[Tuple[float, float]],
    max_gap_s: float = 5.0,
    silences: List | None = None,
    highlights: List[float] | None = None,
) -> List[Tuple[float, float]]:
    """If two adjacent cuts have a small gap AND the gap is entirely silent
    in the audio, merge them. Tiny silent gaps between cuts produce
    micro-keep-slivers (1-5s flashes of dead air between long cuts) which
    are useless and choppy.

    Crucially: only merge gaps that are TRULY silent. The model sometimes
    leaves small gaps because there's a quick joke / reaction / quip in
    them that's worth keeping. We rely on the loudness analysis's silence
    intervals — only merge a gap if it falls entirely inside a detected
    silence.

    Without `silences` provided, this is a no-op (safe default — don't
    risk merging over content)."""
    if not cuts or not silences:
        return cuts
    highlights = highlights or []
    sorted_cuts = sorted(cuts)
    sil = sorted([(s[0], s[1]) for s in silences])

    def _gap_fully_silent(prev_e: float, cs: float) -> bool:
        """True if [prev_e, cs] is fully covered by silence intervals
        (allowing multiple, possibly with sub-100ms breaks)."""
        gap_len = cs - prev_e
        if gap_len <= 0:
            return True
        covered = 0.0
        for ss, se in sil:
            if se <= prev_e:
                continue
            if ss >= cs:
                break
            covered += min(se, cs) - max(ss, prev_e)
        return covered >= gap_len - 0.1  # 100ms tolerance for window rounding

    out = [sorted_cuts[0]]
    for cs, ce in sorted_cuts[1:]:
        prev_s, prev_e = out[-1]
        gap = cs - prev_e
        if 0 < gap < max_gap_s:
            has_highlight = any(prev_e < h < cs for h in highlights)
            if _gap_fully_silent(prev_e, cs) and not has_highlight:
                out[-1] = (prev_s, ce)
                continue
        out.append((cs, ce))
    return out


def extract_highlights_from_response(response: str) -> List[float]:
    """Parse the HIGHLIGHTS_BEGIN..HIGHLIGHTS_END block and return the list
    of highlight timestamps in seconds. Returns [] if no highlights block."""
    m = re.search(r"HIGHLIGHTS_BEGIN\b(.*?)\bHIGHLIGHTS_END", response,
                  re.DOTALL | re.IGNORECASE)
    if not m:
        return []
    times: List[float] = []
    ts_re = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        # Use the FIRST timestamp on the line (highlights are typically
        # "HH:MM:SS — description")
        tm = ts_re.search(line)
        if not tm:
            continue
        a, b, c = tm.groups()
        if c is not None:
            times.append(int(a) * 3600 + int(b) * 60 + int(c))
        else:
            times.append(int(a) * 60 + int(b))
    return times


def protect_highlights(
    cuts: List[Tuple[float, float]],
    highlights: List[float],
    padding_s: float = 10.0,
) -> List[Tuple[float, float]]:
    """Split or contract any cut that contains a highlight timestamp.
    Preserves a 2*padding_s window around each highlight so the comedy
    survives even when the model contradicts its own HIGHLIGHTS list.

    Model is bad at following negative constraints ('don't cut highlight X')
    — empirically even when told explicitly in the revision prompt, it
    still cut 4/6 highlights on the test vod. This is the safety net:
    parse highlight timestamps independently and physically prevent any
    cut from swallowing them.
    """
    if not highlights or not cuts:
        return cuts
    out = []
    for cs, ce in cuts:
        contained = sorted(h for h in highlights if cs <= h <= ce)
        if not contained:
            out.append((cs, ce))
            continue
        current = cs
        for h in contained:
            cut_end = max(current, h - padding_s)
            if cut_end > current + 0.5:
                out.append((current, cut_end))
            current = h + padding_s
        if current < ce - 0.5:
            out.append((current, ce))
    return out


def enforce_budget(
    cuts: List[Tuple[float, float]],
    duration: float,
    ceiling_frac: float = 0.65,
) -> tuple[List[Tuple[float, float]], bool]:
    """Final safety net: if total cut time exceeds ceiling_frac of duration,
    drop the LONGEST cuts in descending order until under budget.

    This runs after the LLM revision call. If the model both over-cut on the
    first pass AND failed to fix it in the revision, this trims the largest
    cuts (which are the most likely to be over-aggressive sweeps) until the
    keep fraction is at least 1 - ceiling_frac.

    Returns (cuts_under_budget, was_trimmed). was_trimmed=True means the
    pipeline should print a warning so the user knows the model couldn't
    self-correct.
    """
    max_cut_s = duration * ceiling_frac
    total = sum(e - s for s, e in cuts)
    if total <= max_cut_s:
        return cuts, False

    # Drop longest first. Sort indices by cut length descending; pop until under.
    indexed = list(enumerate(cuts))
    indexed.sort(key=lambda ic: ic[1][1] - ic[1][0], reverse=True)
    keep_mask = [True] * len(cuts)
    dropped_total = 0.0
    for orig_i, (s, e) in indexed:
        if total - dropped_total <= max_cut_s:
            break
        keep_mask[orig_i] = False
        dropped_total += e - s
    out = [c for c, k in zip(cuts, keep_mask) if k]
    return out, True


def cuts_to_keeps(cuts: List[Tuple[float, float]], duration: float) -> List[Tuple[float, float]]:
    """Invert cut ranges into keep ranges over [0, duration]."""
    keeps: List[Tuple[float, float]] = []
    cursor = 0.0
    for s, e in cuts:
        s = max(0.0, min(s, duration))
        e = max(0.0, min(e, duration))
        if s > cursor:
            keeps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration:
        keeps.append((cursor, duration))
    return keeps
