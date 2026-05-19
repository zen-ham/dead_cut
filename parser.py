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
    """Return the content of the LAST CUTS_BEGIN..CUTS_END block, or None."""
    # Tolerate variations: cuts_begin, CUT_BEGIN, **CUTS_BEGIN**, etc.
    pat = re.compile(
        r"CUTS?_BEGIN\b(.*?)\bCUTS?_END\b",
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
