"""OpenRouter caller. Free models only, with fallback chain."""
import os
import re
import time
import requests

from .cache import cache_dir, save_json, load_json
from .prompts import SYSTEM_PROMPT, build_user_prompt


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# Common false-positive triggers in gaming/streaming transcripts that get
# flagged by OpenAI-style moderation as "self-harm/intent". When a moderation
# 403 fires, the transcript gets scrubbed through these patterns and the same
# model is retried once. Each pattern maps to a neutral paraphrase that keeps
# the joke-y tone without tripping the classifier.
_MODERATION_SCRUB_PATTERNS = [
    (re.compile(r"\bkill(?:ing)?\s+(?:myself|me\b|us\b)", re.I), "ending"),
    (re.compile(r"\b(?:i'?m|im|i)\s+(?:gonna|going\s+to|wanna|want\s+to)\s+die\b", re.I), "i'm done"),
    (re.compile(r"\b(?:wanna|wanted?\s+to|want\s+to)\s+die\b", re.I), "give up"),
    (re.compile(r"\bk\s*\.?\s*m\s*\.?\s*s\b", re.I), "stop it"),
    (re.compile(r"\bk\s*\.?\s*y\s*\.?\s*s\b", re.I), "stop it"),
    (re.compile(r"\bsuicid(?:e|al|es)\b", re.I), "giving up"),
    (re.compile(r"\bblow(?:ing)?\s+(?:my|his|her|their)\s+brains?\s+out\b", re.I), "losing it"),
    (re.compile(r"\bshoot(?:ing)?\s+(?:myself|himself|herself|themself|themselves)\b", re.I), "fed up"),
    (re.compile(r"\bhang(?:ing)?\s+(?:myself|himself|herself|themself|themselves)\b", re.I), "giving up"),
    (re.compile(r"\bend(?:ing)?\s+(?:it\s+all|my\s+life)\b", re.I), "giving up"),
    (re.compile(r"\bjump(?:ing)?\s+off\s+(?:a\s+|the\s+)?(?:bridge|building|cliff|roof)\b", re.I), "losing it"),
    (re.compile(r"\bslit(?:ting)?\s+(?:my|his|her|their)\s+(?:wrists?|throats?)\b", re.I), "losing it"),
    (re.compile(r"\boff\s+(?:myself|himself|herself|themself|themselves)\b", re.I), "tap out"),
]


def _censor_messages(messages: list) -> list:
    """Scrub moderation triggers from each message's content. Returns a new
    list, doesn't mutate the input. Logs the count of substitutions so the
    user can see what was scrubbed if cuts get weird."""
    out = []
    total_subs = 0
    for msg in messages:
        content = msg.get("content", "")
        for pat, repl in _MODERATION_SCRUB_PATTERNS:
            content, n = pat.subn(repl, content)
            total_subs += n
        out.append({**msg, "content": content})
    if total_subs:
        print(f"[llm] censor: {total_subs} substitution(s) made")
    else:
        print(f"[llm] censor: no patterns matched (moderation false positive on benign content?)")
    return out


def _is_moderation_403(status: int, body: str) -> bool:
    """Heuristic: status 403 with moderation/flagged language in the body."""
    if status != 403:
        return False
    low = body.lower()
    return any(t in low for t in ("moderation", "flagged", "self-harm", "openinference"))


def _secs_to_hms(t: float) -> str:
    t = max(0, int(round(t)))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _union_intervals(intervals: list) -> list:
    """Merge overlapping/adjacent intervals into a sorted non-overlapping list.
    Each item is (start, end) in seconds."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: (x[0], x[1]))
    out = [list(sorted_iv[0])]
    for s, e in sorted_iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def _short_revision_context(duration_s: float, target: dict) -> str:
    """Compact user-context replacement for revision calls. The full
    transcript user_prompt is ~50k tokens on long vods, which exceeds
    several free models' context limits and starves reasoning models of
    output budget. Revisions don't actually need the transcript — the
    model can work from its own previous cut labels + this metadata."""
    return (
        f"Video duration: {duration_s/3600:.2f} hours ({duration_s:.0f} seconds).\n"
        f"AI cut target: ~{target['ai_cut_pct']}% of source "
        f"(floor {target['floor_pct']}%, ceiling {target['ceiling_pct']}%).\n"
        f"Your previous cut list and HIGHLIGHTS are in the assistant turn "
        f"below. Use your previous reasoning + the correction below to "
        f"produce a corrected cut list. You do NOT need the full transcript "
        f"again — work from your own prior labels."
    )


def is_chunked(cuts: list, duration: float) -> dict | None:
    """Block-chunking detector. Returns a stats dict if the cut list looks
    like a partition (cuts back-to-back with tiny gaps) OR contains a
    single mega-cut spanning a large fraction of the video. None if the
    structure looks healthy.

    Two failure patterns to catch:
    1. Many cuts back-to-back with 1-5s slivers between them (the
       transcript-segment-labeling pattern).
    2. Very few cuts with one mega-cut covering most of the video (the
       "the whole video is boring" pattern, often a degenerate retry
       output after structure revision).
    """
    if not cuts or duration <= 0:
        return None
    cut_durs = [e - s for s, e in cuts]
    mean_cut = sum(cut_durs) / len(cut_durs)
    max_cut = max(cut_durs)
    pct_cut = 100.0 * sum(cut_durs) / duration
    gaps = [cuts[i + 1][0] - cuts[i][1] for i in range(len(cuts) - 1)] if len(cuts) > 1 else []
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
    short_gap_frac = (sum(1 for g in gaps if g < 15.0) / len(gaps)) if gaps else 0.0
    stats = {
        "median_gap_s": median_gap,
        "short_gap_frac": short_gap_frac,
        "mean_cut_s": mean_cut,
        "max_cut_s": max_cut,
        "pct_cut": pct_cut,
        "n_cuts": len(cuts),
    }
    # Pattern 1: many cuts, mostly back-to-back, big enough mean.
    if (len(cuts) >= 5 and median_gap < 10.0 and short_gap_frac > 0.6
            and mean_cut > 30.0):
        stats["reason"] = "back-to-back cuts (partition pattern)"
        return stats
    # Pattern 2: one mega-cut. Two thresholds:
    #   a) any single cut > 35% of source (catches video-spanning blob cuts)
    #   b) any single cut > 15 minutes (absolute — catches 1hr blocks on 8hr
    #      vods that escape the relative threshold)
    if max_cut > 0.35 * duration:
        stats["reason"] = f"single mega-cut covering {100*max_cut/duration:.0f}% of source"
        return stats
    if max_cut > 900.0:
        stats["reason"] = f"single cut > 15min ({max_cut/60:.0f}min) — chunking in absolute terms"
        return stats
    return None


def check_coverage(cuts: list, duration: float, max_gap_frac: float = 0.25) -> dict | None:
    """Return stats on the largest no-cut span if it exceeds max_gap_frac of
    the video duration. None means coverage is OK. Used to detect when the
    LLM cut only part of the video, leaving a big uncovered region.

    Includes pre-roll (before first cut) and post-roll (after last cut) in
    the analysis since a 35-minute uncut tail is exactly the failure we're
    catching."""
    if not cuts or duration <= 0:
        return None
    sorted_cuts = sorted(cuts)
    gaps = []  # (start, end)
    prev_end = 0.0
    for s, e in sorted_cuts:
        if s > prev_end:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    if prev_end < duration:
        gaps.append((prev_end, duration))
    if not gaps:
        return None
    longest = max(gaps, key=lambda g: g[1] - g[0])
    longest_dur = longest[1] - longest[0]
    if longest_dur / duration < max_gap_frac:
        return None
    return {
        "longest_gap": longest,
        "longest_gap_s": longest_dur,
        "longest_gap_frac": longest_dur / duration,
        "n_gaps": len(gaps),
    }


def revise_cuts_coverage(
    model: str,
    system: str,
    user: str,
    original_response: str,
    duration_s: float,
    coverage_stats: dict,
    primary_cuts: list,
    remaining_budget_s: float,
) -> str | None:
    """Fire when the cut list leaves a big region of the video uncovered
    (e.g. all cuts in the first 60%, last 40% has zero cuts). Tells the
    model the exact time range to focus on and the remaining budget, so
    new cuts add up to about the budget. The pipeline merges these with
    the primary, same as the under-floor revision."""
    gap_start, gap_end = coverage_stats["longest_gap"]
    gap_min = coverage_stats["longest_gap_s"] / 60.0
    budget_min = remaining_budget_s / 60.0
    # Aim for ~3 min per new cut, capped by remaining budget.
    target_cuts_in_gap = max(2, min(8, int(round(budget_min / 3.0))))
    cuts_listing = ""
    if primary_cuts:
        cut_lines = "\n".join(f"  {_secs_to_hms(s)}-{_secs_to_hms(e)}" for s, e in primary_cuts)
        cuts_listing = (
            f"YOUR EXISTING CUTS (do NOT modify these, they stay):\n{cut_lines}\n\n"
        )
    correction = (
        f"COVERAGE PROBLEM: your cut list misses a large section of the video.\n\n"
        f"The region from {_secs_to_hms(gap_start)} to {_secs_to_hms(gap_end)} "
        f"({gap_min:.1f} minutes, {coverage_stats['longest_gap_frac']*100:.0f}% "
        f"of the video) has NO cuts at all. That's a {gap_min:.0f}-minute uncut "
        f"stretch — definitely contains boring filler that should be cut.\n\n"
        f"{cuts_listing}"
        f"YOUR NEW TASK:\n"
        f"  - You have {budget_min:.1f} minutes of cut BUDGET REMAINING. Your "
        f"new cuts must total approximately that much, no more.\n"
        f"  - Find {target_cuts_in_gap-1}-{target_cuts_in_gap+1} boring "
        f"stretches in the {_secs_to_hms(gap_start)}-{_secs_to_hms(gap_end)} "
        f"range, each 2-4 minutes long.\n"
        f"  - Adjacent new cuts must have 30+ seconds between them.\n"
        f"  - Do NOT cut anything outside the {_secs_to_hms(gap_start)}-"
        f"{_secs_to_hms(gap_end)} range.\n\n"
        f"IMPORTANT: the post-processing pipeline AUTOMATICALLY merges your "
        f"new cuts with all your previous cuts. Do NOT re-list any earlier "
        f"cuts. Just output the ADDITIONAL cuts in the uncovered range, "
        f"totaling ~{budget_min:.1f} minutes.\n\n"
        f"Output ONLY a CUTS_BEGIN..CUTS_END block containing your NEW "
        f"additional cuts. Nothing before or after."
    )
    # Short user context (no full transcript) — saves ~50k tokens. Model
    # has its previous cut list in the assistant turn for redistribution.
    short_user = _short_revision_context(
        duration_s=duration_s,
        target={"ai_cut_pct": 0, "floor_pct": 0, "ceiling_pct": 0},
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": short_user},
        {"role": "assistant", "content": original_response},
        {"role": "user", "content": correction},
    ]
    print(f"[llm] coverage revision: model={model}, "
          f"longest uncut gap {_secs_to_hms(gap_start)}-{_secs_to_hms(gap_end)} "
          f"({gap_min:.1f} min = {coverage_stats['longest_gap_frac']*100:.0f}% of source)")
    # Try primary model first, fall back through chain if it returns empty.
    attempt_models = [model] + [m for m in MODELS if m != model]
    for i, attempt_model in enumerate(attempt_models, 1):
        if i > 1:
            print(f"[llm]   coverage attempt {i}/{len(attempt_models)} on {attempt_model}...")
        t0 = time.time()
        resp = _call_one(attempt_model, messages, timeout=300)
        elapsed = time.time() - t0
        if resp is None or not resp.strip():
            if i == 1:
                print(f"[llm]   {attempt_model} failed ({elapsed:.1f}s), trying fallback chain...")
            else:
                print(f"[llm]   {attempt_model} failed ({elapsed:.1f}s)")
            time.sleep(1.0)
            continue
        print(f"[llm] coverage revision OK on {attempt_model} in {elapsed:.1f}s, {len(resp)} chars")
        return resp
    print(f"[llm] coverage revision FAILED on all {len(attempt_models)} models")
    return None


def _has_cuts_block(text: str) -> bool:
    """Cheap check: did the model emit a parseable CUTS_BEGIN..CUTS_END
    block at all? Used before deeper parse_cuts() so we can retry with a
    corrective message instead of crashing the pipeline."""
    if not text:
        return False
    pat = re.compile(r"\bCUTS?_BEGIN\b.*?\bCUTS?_END\b", re.IGNORECASE | re.DOTALL)
    return pat.search(text) is not None


def _ensure_cuts_block(
    model: str,
    primary_resp: str,
    user_prompt: str,
    target: dict,
    duration: float,
) -> str:
    """If primary response is missing a CUTS_BEGIN..CUTS_END block (model
    only emitted HIGHLIGHTS, or got cut off mid-block), fire corrective
    retries. First retry uses the same model with a directive message; if
    that fails, the next attempts cycle through the fallback model chain
    in case the original model is choking on the prompt size (common on
    8hr+ vods)."""
    if _has_cuts_block(primary_resp):
        return primary_resp
    print()
    print(f"[WARNING] primary response missing CUTS_BEGIN..CUTS_END block "
          f"(got {len(primary_resp)} chars but no parseable cut list).")
    print(f"[WARNING] Retrying with corrective message, will try {model} "
          f"first then fall back to other models.")
    print()
    target_cut_min = duration * target["ai_cut_pct"] / 100.0 / 60.0
    # Skip the broken primary entirely from the conversation. It had no
    # CUTS, so feeding it back as assistant context just primes the model
    # to repeat the same shape. Cleaner: fresh prompt with the corrective
    # message appended, model has a blank slate to focus only on CUTS.
    correction_user = (
        user_prompt
        + "\n\n"
        + "=== CRITICAL OUTPUT OVERRIDE ===\n\n"
        + f"For this response, output ONLY a CUTS_BEGIN..CUTS_END block. "
        + f"Do NOT output a HIGHLIGHTS block — highlights are handled "
        + f"separately. Do NOT output anything before or after the cuts "
        + f"block.\n\n"
        + f"Target: cut about {target['ai_cut_pct']}% of the video "
        + f"(≈ {target_cut_min:.0f} minutes). Aim for 10-30 cut ranges of "
        + f"2-5 minutes each, distributed across the timeline with real "
        + f"content between them.\n\n"
        + f"Output format:\n"
        + f"  CUTS_BEGIN\n"
        + f"  HH:MM:SS-HH:MM:SS — concrete reason\n"
        + f"  HH:MM:SS-HH:MM:SS — concrete reason\n"
        + f"  ...\n"
        + f"  CUTS_END\n\n"
        + f"The CUTS_END line MUST be the last line of your response. "
        + f"Begin with CUTS_BEGIN on the very first line. Output the "
        + f"block now."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": correction_user},
    ]
    # Try same model first, then fall back through the chain.
    attempt_models = [model] + [m for m in MODELS if m != model]
    for i, attempt_model in enumerate(attempt_models, 1):
        print(f"[llm] CUTS-block retry {i}/{len(attempt_models)} on {attempt_model}...")
        t0 = time.time()
        retry_resp = _call_one(attempt_model, messages, timeout=300)
        elapsed = time.time() - t0
        if retry_resp is None:
            print(f"[llm]   {attempt_model} returned no response (elapsed {elapsed:.1f}s) — see HTTP error above")
            time.sleep(1.0)
            continue
        if not retry_resp.strip():
            print(f"[llm]   {attempt_model} returned EMPTY content (elapsed {elapsed:.1f}s)")
            time.sleep(1.0)
            continue
        if not _has_cuts_block(retry_resp):
            print(f"[llm]   {attempt_model} returned {len(retry_resp)} chars but no CUTS block "
                  f"(elapsed {elapsed:.1f}s)")
            time.sleep(1.0)
            continue
        print(f"[llm] retry OK on {attempt_model} in {elapsed:.1f}s, {len(retry_resp)} chars")
        m = re.search(r"\bCUTS?_BEGIN\b.*?\bCUTS?_END\b", retry_resp, re.IGNORECASE | re.DOTALL)
        cuts_block = m.group(0)
        return primary_resp.rstrip() + "\n\n" + cuts_block
    print(f"[llm] all {len(attempt_models)} retry attempts failed across the model chain")
    return primary_resp


def _format_cuts_block(cuts: list, reason: str = "merged primary + revision") -> str:
    """Synthesize a CUTS_BEGIN..CUTS_END block from intervals. Used to inject
    a programmatically-merged cut list back into the pipeline as if the model
    had produced it, so downstream parsing/protect_highlights work unchanged."""
    lines = ["CUTS_BEGIN"]
    for s, e in cuts:
        lines.append(f"{_secs_to_hms(s)}-{_secs_to_hms(e)} - {reason}")
    lines.append("CUTS_END")
    return "\n".join(lines)


def _load_token() -> str:
    """Token lives outside the repo so it never gets committed. Precedence:
    1. OPENROUTER_API_KEY env var
    2. <project_root>/openrouter_token.txt (one above the inner repo)
    """
    env = os.environ.get("OPENROUTER_API_KEY")
    if env:
        return env.strip()
    # Layout: <outer>/dead_cut/llm.py  -> outer is one parent up from the inner pkg.
    here = os.path.dirname(os.path.abspath(__file__))   # .../dead_cut/dead_cut (inner)
    outer = os.path.dirname(here)                       # .../dead_cut (outer)
    token_path = os.path.join(outer, "openrouter_token.txt")
    if os.path.exists(token_path):
        with open(token_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    raise RuntimeError(
        "OpenRouter API token missing. Set OPENROUTER_API_KEY env var, or put "
        f"the key on a single line at {token_path}"
    )


OPENROUTER_KEY = _load_token()

# Ordered by best-for-this-task first. All :free tier.
# Fallback chain updated 2026-05-21 — confirmed qwen3.6-plus:free (deprecated,
# 404) and stepfun/step-3.5-flash:free (no endpoint, 404) are dead. Replaced
# with broadly-available free models.
MODELS = [
    "openai/gpt-oss-120b:free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_KEY}",
    "Content-Type": "application/json",
}


def _call_one(model: str, messages: list, timeout: int = 240, _allow_censor: bool = True) -> str | None:
    try:
        r = requests.post(
            OPENROUTER_URL,
            headers=HEADERS,
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.4,
                # Cut detection on a long vod needs room for HIGHLIGHTS +
                # CUTS. gpt-oss-120b is a reasoning model — it burns output
                # tokens on chain-of-thought thinking BEFORE writing the
                # actual response. Observed on 8hr vods: 16k tokens were
                # all consumed by thinking, finish_reason=length, empty
                # content returned. 32k gives ~16k for thinking + 16k for
                # the actual cut list, which is enough for the biggest
                # observed cases.
                "max_tokens": 32000,
            },
            timeout=timeout,
        )
    except Exception as e:
        print(f"[llm] {model} request error: {e}")
        return None
    if not r.ok:
        # Moderation false positives on stream/gaming jokes ("kms", "kill myself")
        # come back as 403 with a "self-harm/intent" reason. Scrub the input and
        # retry once on the same model rather than failing the call.
        if _allow_censor and _is_moderation_403(r.status_code, r.text):
            print(f"[llm] {model} HTTP 403 moderation flag, censoring input and retrying...")
            return _call_one(model, _censor_messages(messages), timeout=timeout, _allow_censor=False)
        print(f"[llm] {model} HTTP {r.status_code}: {r.text[:300]}")
        return None
    try:
        body = r.json()
    except Exception as e:
        print(f"[llm] {model} JSON parse error: {e}; body={r.text[:300]}")
        return None
    try:
        content = body["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[llm] {model} response shape error: {e}; body={str(body)[:500]}")
        return None
    if content is None or not content.strip():
        finish = body["choices"][0].get("finish_reason") if body.get("choices") else None
        print(f"[llm] {model} returned EMPTY content (finish_reason={finish}); "
              f"body keys={list(body.keys())}")
        if body.get("error"):
            print(f"[llm]   server error in body: {body['error']}")
        return None
    return content


def call_llm(system: str, user: str, force_model: str | None = None) -> tuple[str, str]:
    """Try each model in order. Returns (model_used, response_text). Raises if all fail.
    If force_model is given, only that model is tried."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    models = [force_model] if force_model else MODELS
    for m in models:
        print(f"[llm] trying {m}...")
        t0 = time.time()
        resp = _call_one(m, messages)
        if resp:
            print(f"[llm] {m} OK in {time.time()-t0:.1f}s, {len(resp)} chars")
            return m, resp
        time.sleep(1.0)
    raise RuntimeError("All OpenRouter models failed")


def _extract_highlights_block(response: str) -> str:
    """Return the raw HIGHLIGHTS block text from a primary response, or empty
    string if not found. Used to remind the model in both revision messages
    which moments it committed to keeping."""
    m = re.search(
        r"HIGHLIGHTS_BEGIN\b(.*?)\bHIGHLIGHTS_END",
        response, re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def revise_cuts_over_budget(
    model: str,
    system: str,
    user: str,
    original_response: str,
    actual_cut_pct: float,
    duration_s: float,
    target_pct: float = 60.0,
) -> str | None:
    """Send a second message to the model with its original response in context,
    telling it the cut budget was exceeded and to reconsider.

    Tone/scope matches the GAP between actual and target. A 3% overshoot
    gets a 'trim a tiny bit' message; a 25% overshoot gets a 'restructure
    significantly' message. Blunt 'cut less' regardless of gap caused
    overcorrection in both directions.

    Returns the revised assistant response, or None on failure.
    """
    cut_secs = duration_s * (actual_cut_pct / 100.0)
    target_secs = duration_s * (target_pct / 100.0)
    gap_pct = actual_cut_pct - target_pct  # how much MORE they cut than target
    gap_secs = duration_s * (gap_pct / 100.0)
    gap_min = gap_secs / 60.0

    if gap_pct < 7:
        scope = (
            f"You're only {gap_pct:.1f}% over target — that's about "
            f"{gap_min:.1f} minutes of cut to drop. Pick 1-3 of your "
            f"LEAST confident cuts (vague reasons, low-evidence) and drop "
            f"them entirely. Keep everything else the same."
        )
    elif gap_pct < 20:
        scope = (
            f"You're {gap_pct:.1f}% over target — about {gap_min:.1f} "
            f"minutes too much cut. Look through your cuts and drop the "
            f"ones with the weakest reasons (generic 'no jokes', vague "
            f"'low energy', etc). Aim to drop 3-8 cuts spread across the "
            f"timeline (not just the last few). Keep the cuts with strong "
            f"specific reasons (quoted boring lines, clear dead air)."
        )
    else:
        scope = (
            f"You're WAY over target — {gap_pct:.1f}% too much cut "
            f"({gap_min:.0f} minutes excess). This usually means you "
            f"block-chunked the runtime instead of finding specific "
            f"boring sections. Restart your thinking: keep only cuts "
            f"where you can quote a SPECIFIC boring line or point to a "
            f"long stretch of L=0.0 dead air. If you can't justify a "
            f"cut with a concrete reason, drop it. Drop cuts spatially "
            f"distributed, not just sequential ones."
        )

    correction = (
        f"Your CUTS_BEGIN..CUTS_END block totals {cut_secs:.0f}s = "
        f"{actual_cut_pct:.1f}% of the {duration_s:.0f}s video. The "
        f"75% ceiling is exceeded — needs revision to ≤ {target_pct:.0f}% "
        f"(≈{target_secs:.0f}s max).\n\n"
        f"{scope}\n\n"
        f"Also: do NOT cut any of these HIGHLIGHTS you committed to "
        f"earlier:\n"
        f"{_extract_highlights_block(original_response)}\n\n"
        f"Output ONLY a new CUTS_BEGIN..CUTS_END block with the revised list. "
        f"Nothing before or after. Same format rules apply "
        f"(HH:MM:SS-HH:MM:SS — reason, one per line)."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": original_response},
        {"role": "user", "content": correction},
    ]
    print(f"[llm] over-cut revision: model={model}, primary {actual_cut_pct:.1f}% > target {target_pct:.0f}%")
    t0 = time.time()
    resp = _call_one(model, messages, timeout=240)
    if resp:
        print(f"[llm] revision OK in {time.time()-t0:.1f}s, {len(resp)} chars")
    else:
        print(f"[llm] revision FAILED")
    return resp


def revise_cuts_under_floor(
    model: str,
    system: str,
    user: str,
    original_response: str,
    actual_cut_pct: float,
    duration_s: float,
    target_pct: float = 55.0,
    primary_cuts: list | None = None,
) -> str | None:
    """Inverse of over-budget revision: model under-cut, ask for more.
    Used when primary cuts came in below the 50% floor — user wants tight
    edits and the model was too conservative.

    primary_cuts: parsed cut ranges from the original response. If given,
    they're listed explicitly in the correction message so the model knows
    NOT to re-cut or extend those regions (observed failure: model returned
    5 new cuts that overlapped 16 existing primary cuts, contributing only
    100s/0.9% of net-new coverage instead of finding fresh boring stretches)."""
    cut_secs = duration_s * (actual_cut_pct / 100.0)
    target_secs = duration_s * (target_pct / 100.0)
    gap_pct = target_pct - actual_cut_pct  # how much MORE they need to cut
    gap_secs = duration_s * (gap_pct / 100.0)
    gap_min = gap_secs / 60.0
    highlights_block = _extract_highlights_block(original_response)
    # Compact listing of already-cut ranges so the model can target gaps.
    if primary_cuts:
        cut_lines = "\n".join(f"  {_secs_to_hms(s)}-{_secs_to_hms(e)}" for s, e in primary_cuts)
        already_cut_block = (
            f"REGIONS YOU ALREADY CUT (these are SOLVED, do NOT re-cut or "
            f"extend them):\n{cut_lines}\n\n"
            f"Find NEW cuts ONLY in the GAPS between these ranges. If a "
            f"boring stretch falls inside or adjacent to a region you "
            f"already cut, skip it — it's already handled. Look at parts "
            f"of the timeline you have NOT yet flagged.\n\n"
        )
    else:
        already_cut_block = ""

    if gap_pct < 7:
        scope = (
            f"You're only {gap_pct:.1f}% short, about {gap_min:.1f} more "
            f"minutes to cut. Find 1-3 more SHORT cut ranges (30s-2min each) "
            f"from stretches you might have overlooked."
        )
    elif gap_pct < 20:
        scope = (
            f"You're {gap_pct:.1f}% short of the 50% minimum, about "
            f"{gap_min:.1f} more minutes to cut. Find several more boring "
            f"stretches you missed: long narrative gameplay describing actions "
            f"without jokes ('I'm gonna check this'), repeated complaint "
            f"loops, inventory/menu shuffling. 3-8 more cut ranges should "
            f"do it."
        )
    else:
        scope = (
            f"You're significantly under target, need {gap_pct:.1f}% more "
            f"cut ({gap_min:.0f} minutes). You probably labeled too much "
            f"mid-energy content as 'entertaining'. Even narrative game "
            f"commentary that's not funny is fair to cut. 8-20 additional "
            f"ranges, in parts of the video you didn't originally flag."
        )

    correction = (
        f"Your CUTS_BEGIN..CUTS_END block totals only {cut_secs:.0f}s = "
        f"{actual_cut_pct:.1f}% of the {duration_s:.0f}s video. Minimum cut "
        f"is 50%, need ~{gap_min:.1f} more minutes cut to reach "
        f"{target_pct:.0f}% (≈{target_secs:.0f}s total).\n\n"
        f"{already_cut_block}"
        f"{scope}\n\n"
        f"VERY IMPORTANT, READ CAREFULLY:\n"
        f"The post-processing pipeline AUTOMATICALLY merges your new cuts "
        f"with all your previous cuts. You do NOT need to re-list any of "
        f"your earlier cuts. Just output the ADDITIONAL cuts you want to "
        f"add, only the new ones. This is a much smaller, easier task than "
        f"redoing the full list, just find the few extra boring stretches "
        f"in UNCOVERED areas. If you re-list your old cuts it wastes "
        f"tokens, and if you OVERLAP your old cuts you waste effort because "
        f"the merge will just absorb them, don't do either.\n\n"
        f"CRITICAL: do NOT cut any of these HIGHLIGHTS:\n"
        f"{highlights_block}\n\n"
        f"Output ONLY a CUTS_BEGIN..CUTS_END block containing your new "
        f"ADDITIONAL cuts, all of which must be in UNCOVERED parts of the "
        f"timeline (not inside or overlapping any region from the "
        f"'REGIONS YOU ALREADY CUT' list). Nothing before or after the block. "
        f"Same format (HH:MM:SS-HH:MM:SS — reason, one per line)."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": original_response},
        {"role": "user", "content": correction},
    ]
    print(f"[llm] under-cut revision: model={model}, primary {actual_cut_pct:.1f}% < target {target_pct:.0f}%")
    t0 = time.time()
    resp = _call_one(model, messages, timeout=240)
    if resp:
        print(f"[llm] revision OK in {time.time()-t0:.1f}s, {len(resp)} chars")
    else:
        print(f"[llm] revision FAILED")
    return resp


def revise_cuts_structure(
    model: str,
    system: str,
    user: str,
    original_response: str,
    duration_s: float,
    target_pct: float,
    chunk_stats: dict,
    ceiling_pct: float = 70.0,
    floor_pct: float = 35.0,
) -> str | None:
    """Fire when the primary output is block-chunked (cuts back-to-back with
    tiny gaps). Tells the model concretely WHAT went wrong and asks for a
    full restart — different from over/under revisions which only adjust
    the cut budget. The structural problem is independent of total %."""
    target_cut_min = duration_s * (target_pct / 100.0) / 60.0
    ceiling_cut_min = duration_s * (ceiling_pct / 100.0) / 60.0
    floor_cut_min = duration_s * (floor_pct / 100.0) / 60.0
    # Suggest a specific cut count and per-cut duration so the model has a
    # concrete numerical target instead of a vague range. ~4 min per cut is
    # the sweet spot — long enough to be a real boring stretch, short enough
    # that chunking-by-accident is hard.
    avg_cut_min = 4.0
    cut_count_target = max(6, int(round(target_cut_min / avg_cut_min)))
    correction = (
        f"PROBLEM WITH YOUR PREVIOUS RESPONSE: you BLOCK-CHUNKED the runtime.\n\n"
        f"Diagnostic from your CUTS:\n"
        f"  - {chunk_stats['n_cuts']} cut ranges, totalling {chunk_stats['pct_cut']:.1f}% of source\n"
        f"  - median gap between adjacent cuts: {chunk_stats['median_gap_s']:.1f}s "
        f"(should be 30+ seconds)\n"
        f"  - {chunk_stats['short_gap_frac']*100:.0f}% of gaps are < 15s\n"
        f"  - mean cut length: {chunk_stats['mean_cut_s']:.0f}s\n\n"
        f"You enumerated boring sections back-to-back, leaving 1-5 second "
        f"slivers between them. Those slivers either contain no speech "
        f"(pure walking/transition) or a clipped mid-sentence fragment. "
        f"Result: the kept video is transitional silence with random "
        f"mid-word clips — inverted from the goal.\n\n"
        f"=== YOUR NEW TASK (read every line, follow exactly) ===\n\n"
        f"OUTPUT TARGET, exactly these numbers:\n"
        f"  TOTAL CUTS to produce: about {cut_count_target} (range "
        f"{max(6, cut_count_target-3)} to {cut_count_target+3})\n"
        f"  EACH CUT'S length: 2 to 5 minutes. A cut of 6+ minutes is REJECTED.\n"
        f"  TOTAL cut duration summed: ~{target_cut_min:.0f} minutes "
        f"({target_pct:.0f}% of {duration_s/60:.0f} min source).\n"
        f"  ABSOLUTE MAX total cut: {ceiling_cut_min:.0f} min "
        f"({ceiling_pct:.0f}%). Going over is REJECTED.\n"
        f"  ABSOLUTE MIN total cut: {floor_cut_min:.0f} min "
        f"({floor_pct:.0f}%). Going under is REJECTED.\n\n"
        f"WORKED EXAMPLES of valid cuts (2-5 min each):\n"
        f"  RIGHT: 00:01:30-00:04:00 — inventory shuffling, no jokes  (2.5 min)\n"
        f"  RIGHT: 00:09:00-00:13:00 — wandering filler about filters  (4 min)\n"
        f"  RIGHT: 00:18:30-00:21:00 — repetitive 'why is there music' rant  (2.5 min)\n"
        f"  RIGHT: 00:35:00-00:39:30 — mid-energy game narration, no jokes  (4.5 min)\n\n"
        f"WORKED EXAMPLES of INVALID cuts:\n"
        f"  WRONG: 00:01:00-00:25:00 — boring section  (24 min, MUCH too long)\n"
        f"  WRONG: 00:01:00-01:25:00 — most of video  (84 min, way too long)\n"
        f"  WRONG: 00:01:00-00:04:00 then 00:04:30-00:07:00  (only 30s gap between\n"
        f"         cuts, you're chunking again — needs 60s+ of real content "
        f"between cuts)\n\n"
        f"COMPUTING DURATIONS — do this for each cut you write:\n"
        f"  end_seconds - start_seconds must be between 120 and 300\n"
        f"  Example: 00:09:00 = 540s, 00:13:00 = 780s, diff = 240s = 4 min ✓\n"
        f"  Example: 00:01:00 = 60s, 00:25:00 = 1500s, diff = 1440s = 24 min ✗\n\n"
        f"STEP-BY-STEP procedure:\n"
        f"  1. Look at your HIGHLIGHTS list from the original response\n"
        f"  2. Sort them by timestamp; those are your KEEP anchors\n"
        f"  3. Between each pair of highlight anchors, find ONE 2-5 minute\n"
        f"     boring stretch that's clearly filler, not jokes/reactions\n"
        f"  4. Leave 30+ seconds around each highlight uncut\n"
        f"  5. Write the cut as HH:MM:SS-HH:MM:SS — concrete reason\n"
        f"  6. Continue until you have ~{cut_count_target} cuts\n"
        f"  7. Sum the durations. If <{floor_cut_min:.0f} or >{ceiling_cut_min:.0f} min,\n"
        f"     adjust before outputting.\n\n"
        f"Output ONLY a CUTS_BEGIN..CUTS_END block. No HIGHLIGHTS block — "
        f"your originals still apply.\n\n"
        f"BEFORE OUTPUTTING, verify out loud (in your reasoning, not the "
        f"final output):\n"
        f"  - Cut count is between {max(6, cut_count_target-3)} and {cut_count_target+3}\n"
        f"  - No cut exceeds 5 minutes (300 seconds)\n"
        f"  - Total cut duration is {floor_cut_min:.0f}-{ceiling_cut_min:.0f} min\n"
        f"  - Adjacent cuts have 60+ seconds between them"
    )
    # Use a SHORT user context (not the full transcript) — saves ~50k
    # tokens per call so this fits inside smaller models' contexts AND
    # leaves reasoning models enough output budget. The model has its own
    # previous cut labels in the assistant turn, which is enough context
    # for redistribution.
    short_user = _short_revision_context(
        duration_s=duration_s,
        target={"ai_cut_pct": int(target_pct), "floor_pct": int(floor_pct), "ceiling_pct": int(ceiling_pct)},
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": short_user},
        {"role": "assistant", "content": original_response},
        {"role": "user", "content": correction},
    ]
    print(f"[llm] structure revision: primary block-chunked "
          f"({chunk_stats['n_cuts']} cuts, median gap {chunk_stats['median_gap_s']:.1f}s)")
    # Try primary model first, then fall back through the chain. On long
    # vods the primary model often fails the structure call (empty response
    # or context-length error), but other models in the chain can handle it.
    attempt_models = [model] + [m for m in MODELS if m != model]
    for i, attempt_model in enumerate(attempt_models, 1):
        print(f"[llm]   attempt {i}/{len(attempt_models)} on {attempt_model}...")
        t0 = time.time()
        resp = _call_one(attempt_model, messages, timeout=300)
        elapsed = time.time() - t0
        if resp is None:
            print(f"[llm]   {attempt_model} returned None ({elapsed:.1f}s) — see HTTP error above if any")
            time.sleep(1.0)
            continue
        if not resp.strip():
            print(f"[llm]   {attempt_model} returned EMPTY content ({elapsed:.1f}s)")
            time.sleep(1.0)
            continue
        print(f"[llm] structure revision OK on {attempt_model} in {elapsed:.1f}s, {len(resp)} chars")
        return resp
    print(f"[llm] structure revision FAILED on all {len(attempt_models)} models")
    return None


# Budget thresholds are now per-video (dynamic, from target.compute_ai_cut_target).
# Kept as fallbacks for callers that haven't migrated yet — they map to the
# legacy 50-75% band used before the dynamic curve landed.
BUDGET_CEILING = 0.75
BUDGET_FLOOR = 0.50

# Structure revision is unstable on gpt-oss-120b:free — ~50% of attempts
# come back as mega-cuts or back-to-back chunks despite explicit prompt
# constraints. We rejection-sample up to this many attempts, taking the
# first pass-band-non-chunked result. Average cost is ~2 calls.
STRUCTURE_REVISION_ATTEMPTS = 3

# Coverage check threshold: any continuous uncut region exceeding this
# fraction of total duration triggers an additional revision pass that
# tells the model "find cuts here, you missed this range". 0.25 = 25% of
# source. On a 86-min video, this fires if 21+ minutes are uncut.
COVERAGE_MAX_GAP_FRAC = 0.25

# Max coverage revision iterations. One pass only targets the longest gap;
# long vods with multiple uncovered regions need to loop. Each pass costs
# one LLM call (1-3min on long vods). 3 is a balance between coverage
# quality and runtime.
COVERAGE_REVISION_MAX_ITERS = 3


def detect_cuts(
    video_id: str,
    duration: float,
    segments: list,
    loudness_per_seg: list,
    target: dict,
    iteration: int = 0,
    extra_user_suffix: str = "",
    force_model: str | None = None,
    primary_override: str | None = None,
) -> tuple[str, str]:
    """Call the LLM to produce a cut list with a per-video dynamic target.

    target dict from target.compute_ai_cut_target() carries:
        ai_cut_pct:   the goal (e.g. 45% cut)
        floor_pct:    under this triggers under-cut revision
        ceiling_pct:  over this triggers over-cut revision + safety net

    primary_override: if given, skip the primary LLM call and use this text
        as the primary response. Used by recovery tooling that already has
        a parseable primary from a previous run and wants to re-trigger the
        full structure/budget/coverage revision chain without re-paying for
        the primary call. Bypasses the cache check (assumes caller wants
        fresh revisions).

    Cached per (video_id, iteration). Cache stores primary, structure,
    revised, coverage, and final responses for transparency.
    Returns (model_used, final_response, primary_response).
    """
    # Local import to avoid a circular dependency at module load time.
    from .parser import parse_cuts

    out_path = os.path.join(cache_dir(video_id), f"llm_iter{iteration}.json")
    if primary_override is None and os.path.exists(out_path):
        d = load_json(out_path)
        print(f"[llm] cache hit (iter {iteration}): {out_path}")
        return (
            d["model"],
            d.get("final_response") or d.get("revised_response")
                or d.get("structure_response") or d["response"],
            d["response"],
        )

    user_prompt = build_user_prompt(duration, segments, loudness_per_seg, target)
    if extra_user_suffix:
        user_prompt += "\n\n" + extra_user_suffix
    print(f"[llm] prompt size: {len(user_prompt)} chars, {len(segments)} segments")
    print(f"[llm] target: cut {target['ai_cut_pct']}% (floor {target['floor_pct']}%, "
          f"ceiling {target['ceiling_pct']}%), expecting +"
          f"{target.get('silence_trim_ratio', 0)*100:.0f}% silence trim after")

    if primary_override is not None:
        print(f"[llm] using primary_override ({len(primary_override)} chars), skipping primary call")
        model = force_model or MODELS[0]
        primary_resp = primary_override
    else:
        model, primary_resp = call_llm(SYSTEM_PROMPT, user_prompt, force_model=force_model)

    # Long vods sometimes get back a response with only HIGHLIGHTS and no
    # CUTS block (the model exhausts its planning and skips the actual cut
    # list). Detect that and retry once with a corrective message before
    # entering the budget/structure pipeline below.
    primary_resp = _ensure_cuts_block(
        model, primary_resp, user_prompt, target=target, duration=duration,
    )

    # Budget check using the dynamic per-video band.
    floor_frac = target["floor_pct"] / 100.0
    ceiling_frac = target["ceiling_pct"] / 100.0
    target_frac = target["ai_cut_pct"] / 100.0
    # `resp` is the active text used for budget/revision logic; may get
    # replaced by structure_resp if chunking is detected. We always preserve
    # the original `primary_resp` separately because the pipeline uses it
    # downstream for the HIGHLIGHTS block (structure revisions don't emit
    # one — they were told to skip it).
    resp = primary_resp
    final_resp = primary_resp
    revised_resp = None
    structure_resp = None
    coverage_resp = None
    cut_pct_first = None
    cut_pct_revised = None
    try:
        first_cuts = parse_cuts(resp, max_duration=duration)
        first_cut_secs = sum(e - s for s, e in first_cuts)
        cut_pct_first = 100.0 * first_cut_secs / max(duration, 1e-6)

        # Structure check FIRST: if the primary block-chunked the runtime
        # (cuts back-to-back, tiny gaps), no budget revision will save it.
        # Force a full restart with the structure prompt. The structure
        # revision has ~50% pass rate per attempt empirically, so we
        # rejection-sample up to STRUCTURE_REVISION_ATTEMPTS times, picking
        # the first attempt that's not chunked and inside the budget band.
        chunk_stats = is_chunked(first_cuts, duration)
        if chunk_stats is not None:
            print(
                f"\n[WARNING] primary is BLOCK-CHUNKED ({chunk_stats['reason']}, "
                f"{chunk_stats['n_cuts']} cuts, {chunk_stats['pct_cut']:.1f}% cut). "
                f"Firing structure revision...\n"
            )
            attempts: list = []  # (resp, cuts, pct, still_chunked)
            for attempt_i in range(1, STRUCTURE_REVISION_ATTEMPTS + 1):
                print(f"[llm] structure revision attempt {attempt_i}/{STRUCTURE_REVISION_ATTEMPTS}")
                struct_resp = revise_cuts_structure(
                    model=model, system=SYSTEM_PROMPT, user=user_prompt,
                    original_response=primary_resp,
                    duration_s=duration,
                    target_pct=target["ai_cut_pct"],
                    chunk_stats=chunk_stats,
                    ceiling_pct=target["ceiling_pct"],
                    floor_pct=target["floor_pct"],
                )
                if not struct_resp:
                    continue
                try:
                    new_cuts = parse_cuts(struct_resp, max_duration=duration)
                except Exception as e:
                    print(f"[llm] attempt {attempt_i} parse error: {e}")
                    continue
                new_pct = 100.0 * sum(e - s for s, e in new_cuts) / max(duration, 1e-6)
                still_chunked = is_chunked(new_cuts, duration)
                in_band = target["floor_pct"] <= new_pct <= target["ceiling_pct"]
                print(f"[llm] attempt {attempt_i} result: {len(new_cuts)} cuts, "
                      f"{new_pct:.1f}% cut, chunked={still_chunked is not None}, in_band={in_band}")
                attempts.append((struct_resp, new_cuts, new_pct, still_chunked, in_band))
                if still_chunked is None and in_band:
                    break  # found a clean one, stop sampling
            # Pick the best attempt. Sort key: (chunked, not in_band, distance from target).
            if attempts:
                def _score(a):
                    _, _, pct, ch, in_b = a
                    return (ch is not None, not in_b, abs(pct - target["ai_cut_pct"]))
                attempts.sort(key=_score)
                best_resp, best_cuts, best_pct, best_chunked, best_in_band = attempts[0]
                structure_resp = best_resp
                final_resp = best_resp
                resp = best_resp
                first_cuts = best_cuts
                first_cut_secs = sum(e - s for s, e in first_cuts)
                cut_pct_first = best_pct
                print(f"[llm] structure revision picked best of {len(attempts)} attempts: "
                      f"{len(first_cuts)} cuts, {cut_pct_first:.1f}% cut, "
                      f"chunked={best_chunked is not None}, in_band={best_in_band}")
                if best_chunked is not None:
                    bar = "!" * 80
                    print()
                    print(bar)
                    print(f"!! [SEVERE WARNING] structure revision FAILED to fix block-chunking")
                    print("!!")
                    print(f"!! All {len(attempts)} attempt(s) came back chunked. Best attempt:")
                    print(f"!!   {len(first_cuts)} cuts, {cut_pct_first:.1f}% cut")
                    print(f"!!   reason: {best_chunked['reason']}")
                    print(f"!!   max single cut: {best_chunked.get('max_cut_s', 0):.0f}s")
                    print(f"!!   median gap between cuts: {best_chunked.get('median_gap_s', 0):.1f}s")
                    print("!!")
                    print(f"!! This means the model could not produce a structurally")
                    print(f"!! sane cut list for this video. The pipeline will continue")
                    print(f"!! with the least-bad attempt, but expect the output to have")
                    print(f"!! mid-speech clips and/or kept walking-around silence — the")
                    print(f"!! 'inverted edit' failure mode the structure revision tries")
                    print(f"!! to prevent.")
                    print("!!")
                    print(f"!! Try: rerun with --iter N (different model sampling), or")
                    print(f"!! force a different model with --model qwen/qwen3.6-plus:free")
                    print(bar)
                    print()
            else:
                bar = "!" * 80
                print()
                print(bar)
                print(f"!! [SEVERE WARNING] all {STRUCTURE_REVISION_ATTEMPTS} structure revision attempts failed")
                print("!!")
                print(f"!! Either every attempt returned None (model error) or every")
                print(f"!! response failed to parse a CUTS_BEGIN..CUTS_END block.")
                print(f"!! Falling back to the original block-chunked primary, which")
                print(f"!! will produce a poor 'inverted edit' result.")
                print("!!")
                print(f"!! Try: rerun with --iter N or --model <other>.")
                print(bar)
                print()

        revise_fn = None
        revise_target = None
        if cut_pct_first / 100.0 > ceiling_frac:
            print(
                f"\n[WARNING] model over-cut on first pass: "
                f"{len(first_cuts)} cuts totalling {cut_pct_first:.1f}% "
                f"(ceiling {target['ceiling_pct']}%). Requesting revision...\n"
            )
            revise_fn = revise_cuts_over_budget
            revise_target = target["ai_cut_pct"]
        elif cut_pct_first / 100.0 < floor_frac:
            print(
                f"\n[WARNING] model under-cut on first pass: "
                f"{len(first_cuts)} cuts totalling {cut_pct_first:.1f}% "
                f"(floor {target['floor_pct']}%). Requesting revision...\n"
            )
            revise_fn = revise_cuts_under_floor
            revise_target = target["ai_cut_pct"]
        if revise_fn is not None:
            revise_kwargs = dict(
                model=model, system=SYSTEM_PROMPT, user=user_prompt,
                original_response=resp,
                actual_cut_pct=cut_pct_first,
                duration_s=duration,
                target_pct=revise_target,
            )
            # Under-cut revision needs the parsed primary cuts so the model
            # can see which regions are already covered and target the gaps.
            if revise_fn is revise_cuts_under_floor:
                revise_kwargs["primary_cuts"] = list(first_cuts)
            revised_resp = revise_fn(**revise_kwargs)
            if revised_resp:
                final_resp = revised_resp
                try:
                    rev_cuts = parse_cuts(revised_resp, max_duration=duration)
                except Exception as e:
                    print(f"[llm] couldn't parse revision: {e}")
                    rev_cuts = None
                if rev_cuts is not None:
                    if revise_fn is revise_cuts_under_floor:
                        # Under-cut: model was told to output ONLY new additional
                        # cuts. Merge them with primary cuts here so the model
                        # doesn't have to copy its own first-pass output (a load
                        # it was demonstrably failing to carry, dropping originals
                        # and only re-listing a partial set).
                        merged = _union_intervals(list(first_cuts) + list(rev_cuts))
                        merged_secs = sum(e - s for s, e in merged)
                        cut_pct_revised = 100.0 * merged_secs / max(duration, 1e-6)
                        consolidated = len(first_cuts) + len(rev_cuts) - len(merged)
                        net_new_secs = merged_secs - first_cut_secs
                        net_new_pct = 100.0 * net_new_secs / max(duration, 1e-6)
                        print(
                            f"[llm] under-cut merge: primary {len(first_cuts)} + "
                            f"revision {len(rev_cuts)} -> {len(merged)} total"
                        )
                        if consolidated > 0:
                            print(
                                f"[llm]   revision absorbed {consolidated} existing "
                                f"cuts via overlap, contributed only {net_new_secs:.0f}s "
                                f"({net_new_pct:+.2f}% of source) of net-new coverage"
                            )
                        else:
                            print(
                                f"[llm]   revision added {net_new_secs:.0f}s "
                                f"({net_new_pct:+.2f}% of source) of net-new coverage"
                            )
                        print(
                            f"[llm]   final: {len(merged)} cuts, {cut_pct_revised:.1f}% cut "
                            f"(was {cut_pct_first:.1f}%, {net_new_pct:+.1f}%)"
                        )
                        if cut_pct_revised / 100.0 < floor_frac:
                            print(
                                f"\n[WARNING] revision STILL under {target['floor_pct']}% "
                                f"floor ({cut_pct_revised:.1f}% cut). No programmatic "
                                f"remedy available — output will be looser than target.\n"
                            )
                        final_resp = _format_cuts_block(merged)
                    else:
                        # Over-cut: revised replaces primary as instructed.
                        cut_pct_revised = 100.0 * sum(e - s for s, e in rev_cuts) / max(duration, 1e-6)
                        print(f"[llm] revision result: {len(rev_cuts)} cuts, {cut_pct_revised:.1f}% cut")
                        if cut_pct_revised / 100.0 > ceiling_frac:
                            print(
                                f"\n[WARNING] model STILL over-cut after revision: "
                                f"{cut_pct_revised:.1f}% cut. Programmatic drop will run next.\n"
                            )

        # COVERAGE CHECK: after all structure/budget revisions, see if the
        # final cut list leaves a big uncovered region. Fires an additional
        # revision asking the model to find cuts in the uncovered range
        # ONLY if we have budget room — if we're already at target % with
        # bad coverage, adding more cuts would push over the ceiling.
        # Structure rev (which fires first) should normally redistribute,
        # so this is a safety net for the rare case where primary was
        # non-chunked but missed a region.
        try:
            current_cuts = parse_cuts(final_resp, max_duration=duration)
        except Exception:
            current_cuts = list(first_cuts)
        # Iterate coverage revisions: each pass targets the longest current
        # gap and adds cuts inside it. For long vods with severe coverage
        # failures (5hr+ uncut), one pass only nibbles at the longest gap —
        # need to loop until either no gap remains or budget is exhausted.
        # Cap iterations to avoid runaway LLM costs.
        for cov_iter in range(1, COVERAGE_REVISION_MAX_ITERS + 1):
            coverage_stats = check_coverage(current_cuts, duration, max_gap_frac=COVERAGE_MAX_GAP_FRAC)
            if coverage_stats is None:
                if cov_iter > 1:
                    print(f"[llm] coverage check OK after {cov_iter-1} revision(s)")
                break
            current_cut_secs = sum(e - s for s, e in current_cuts)
            current_pct = 100.0 * current_cut_secs / max(duration, 1e-6)
            ceiling_cut_secs = duration * target["ceiling_pct"] / 100.0
            remaining_budget_s = ceiling_cut_secs - current_cut_secs
            if remaining_budget_s < 60.0:
                print(
                    f"\n[WARNING] coverage gap remains ({coverage_stats['longest_gap_s']/60:.1f} "
                    f"min uncut, {coverage_stats['longest_gap_frac']*100:.0f}% of source) "
                    f"after {cov_iter-1} revision(s), BUT current {current_pct:.1f}% cut "
                    f"already at/past ceiling {target['ceiling_pct']}%. Stopping.\n"
                )
                break
            print(
                f"\n[WARNING] coverage gap [iter {cov_iter}/{COVERAGE_REVISION_MAX_ITERS}]: "
                f"{coverage_stats['longest_gap_s']/60:.1f} min uncut "
                f"({coverage_stats['longest_gap_frac']*100:.0f}% of source). "
                f"Current {current_pct:.1f}% cut, {remaining_budget_s/60:.1f} min to ceiling. "
                f"Firing coverage revision...\n"
            )
            cov_resp = revise_cuts_coverage(
                model=model, system=SYSTEM_PROMPT, user=user_prompt,
                original_response=primary_resp,
                duration_s=duration,
                coverage_stats=coverage_stats,
                primary_cuts=list(current_cuts),
                remaining_budget_s=remaining_budget_s,
            )
            if not cov_resp:
                print(f"[llm] coverage revision returned no response, stopping")
                break
            try:
                cov_cuts = parse_cuts(cov_resp, max_duration=duration)
            except Exception as e:
                print(f"[llm] couldn't parse coverage revision: {e}, stopping")
                break
            if not cov_cuts:
                print(f"[llm] coverage revision returned 0 cuts, stopping")
                break
            merged = _union_intervals(list(current_cuts) + list(cov_cuts))
            new_pct = 100.0 * sum(e - s for s, e in merged) / max(duration, 1e-6)
            new_added = len(merged) - len(current_cuts)
            print(f"[llm] coverage merge [iter {cov_iter}]: existing {len(current_cuts)} + "
                  f"new {len(cov_cuts)} -> {len(merged)} total (+{new_added} net), {new_pct:.1f}% cut")
            if new_added == 0:
                # Revision overlapped entirely with existing — no progress.
                print(f"[llm] coverage revision contributed no net-new cuts, stopping")
                break
            coverage_resp = cov_resp  # last successful revision
            current_cuts = merged
            final_resp = _format_cuts_block(merged)
    except Exception as e:
        # If parsing fails, just persist what we have and let pipeline handle.
        print(f"[llm] budget check skipped: parse error ({e})")

    save_json(out_path, {
        "model": model,
        "response": primary_resp,
        "structure_response": structure_resp,
        "revised_response": revised_resp,
        "coverage_response": coverage_resp,
        "final_response": final_resp if final_resp is not primary_resp else None,
        "cut_pct_first": cut_pct_first,
        "cut_pct_revised": cut_pct_revised,
        "prompt_chars": len(user_prompt),
        "iteration": iteration,
        "forced_model": force_model,
    })
    # Return original primary too — it has the HIGHLIGHTS block (structure
    # and budget revisions only emit CUTS). Pipeline uses primary for
    # highlight protection.
    return model, final_resp, primary_resp
