"""OpenRouter caller. Free models only, with fallback chain."""
import os
import re
import time
import requests

from .cache import cache_dir, save_json, load_json
from .prompts import SYSTEM_PROMPT, build_user_prompt


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


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
MODELS = [
    "openai/gpt-oss-120b:free",
    "qwen/qwen3.6-plus:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "stepfun/step-3.5-flash:free",
]

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_KEY}",
    "Content-Type": "application/json",
}


def _call_one(model: str, messages: list, timeout: int = 240) -> str | None:
    try:
        r = requests.post(
            OPENROUTER_URL,
            headers=HEADERS,
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.4,
                # Cut detection on a long vod needs room for HIGHLIGHTS +
                # CANDIDATES + AUDIT + CUTS. On a 3hr vod with ~150 cuts the
                # response can be 6-10K tokens.
                "max_tokens": 12000,
            },
            timeout=timeout,
        )
    except Exception as e:
        print(f"[llm] {model} request error: {e}")
        return None
    if not r.ok:
        print(f"[llm] {model} HTTP {r.status_code}: {r.text[:300]}")
        return None
    try:
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[llm] {model} parse error: {e}; body={r.text[:300]}")
        return None


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
    print(f"[llm] over-cut revision: model={model}, primary {actual_cut_pct:.1f}% > {BUDGET_CEILING*100:.0f}%")
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
) -> str | None:
    """Inverse of over-budget revision: model under-cut, ask for more.
    Used when primary cuts came in below the 50% floor — user wants tight
    edits and the model was too conservative."""
    cut_secs = duration_s * (actual_cut_pct / 100.0)
    target_secs = duration_s * (target_pct / 100.0)
    gap_pct = target_pct - actual_cut_pct  # how much MORE they need to cut
    gap_secs = duration_s * (gap_pct / 100.0)
    gap_min = gap_secs / 60.0
    highlights_block = _extract_highlights_block(original_response)

    if gap_pct < 7:
        scope = (
            f"You're only {gap_pct:.1f}% short — about {gap_min:.1f} "
            f"more minutes to cut. Add 1-3 more SHORT cut ranges (30s-2min "
            f"each) from stretches you might have overlooked. Keep all your "
            f"existing cuts — just add a small handful more."
        )
    elif gap_pct < 20:
        scope = (
            f"You're {gap_pct:.1f}% short of the 50% minimum — about "
            f"{gap_min:.1f} more minutes to cut. Look for several more "
            f"boring stretches you missed: long narrative gameplay describing "
            f"actions without jokes ('I'm gonna check this'), repeated "
            f"complaint loops, inventory/menu shuffling. ADD 3-8 more cut "
            f"ranges to your existing list — don't change the cuts you "
            f"already had."
        )
    else:
        scope = (
            f"You're significantly under target — need {gap_pct:.1f}% more "
            f"cut ({gap_min:.0f} minutes). You probably labeled too much "
            f"mid-energy content as 'entertaining'. Even narrative game "
            f"commentary that's not funny is fair to cut. Aim for many more "
            f"cuts — 8-20 additional ranges, in parts of the video you "
            f"didn't originally flag. Keep your existing cuts and add to them."
        )

    correction = (
        f"Your CUTS_BEGIN..CUTS_END block totals only {cut_secs:.0f}s = "
        f"{actual_cut_pct:.1f}% of the {duration_s:.0f}s video. Minimum cut "
        f"is 50%, target ≥ {target_pct:.0f}% (≈{target_secs:.0f}s).\n\n"
        f"{scope}\n\n"
        f"CRITICAL: do NOT cut any of these HIGHLIGHTS:\n"
        f"{highlights_block}\n\n"
        f"Output a NEW CUTS_BEGIN..CUTS_END block with your ORIGINAL cuts "
        f"PLUS new ones you've added. Nothing before or after. Same format "
        f"(HH:MM:SS-HH:MM:SS — reason, one per line)."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": original_response},
        {"role": "user", "content": correction},
    ]
    print(f"[llm] under-cut revision: model={model}, primary {actual_cut_pct:.1f}% < {BUDGET_FLOOR*100:.0f}%")
    t0 = time.time()
    resp = _call_one(model, messages, timeout=240)
    if resp:
        print(f"[llm] revision OK in {time.time()-t0:.1f}s, {len(resp)} chars")
    else:
        print(f"[llm] revision FAILED")
    return resp


# Budget thresholds. Two-sided: catch both over-cutting AND under-cutting.
# Over: if primary > CEILING, revision asks to pull back to UNDER_TARGET.
# Under: if primary < FLOOR, revision asks to be more aggressive, aiming OVER_TARGET.
BUDGET_CEILING = 0.75
REVISION_UNDER_TARGET = 0.55  # when over-cut, pull back to here
BUDGET_FLOOR = 0.50
REVISION_OVER_TARGET = 0.55   # when under-cut, push up to here


def detect_cuts(
    video_id: str,
    duration: float,
    segments: list,
    loudness_per_seg: list,
    iteration: int = 0,
    extra_user_suffix: str = "",
    force_model: str | None = None,
) -> tuple[str, str]:
    """Call the LLM to produce a cut list. Two-tier budget enforcement:

      1. Primary call gets first-pass cuts.
      2. If the parsed cuts exceed BUDGET_CEILING (65%) of duration, send a
         REVISION request — the original response is in context as the
         assistant turn, then a corrective user message asks the model to drop
         its least-confident cuts. This turns a non-thinking model into a
         one-step-thinking model.

    A third tier (programmatic drop of longest cuts) lives in the pipeline
    after this returns, as a final safety net if the revision still over-cuts.

    Cached per (video_id, iteration). Cache stores both primary and revised
    responses for transparency.
    Returns (model_used, final_response).
    """
    # Local import to avoid a circular dependency at module load time.
    from .parser import parse_cuts

    out_path = os.path.join(cache_dir(video_id), f"llm_iter{iteration}.json")
    if os.path.exists(out_path):
        d = load_json(out_path)
        print(f"[llm] cache hit (iter {iteration}): {out_path}")
        # Return (model, final_response, primary_response). The primary is
        # always returned separately because it's the one with HIGHLIGHTS
        # (the revised response, when present, only has the CUTS block).
        return (
            d["model"],
            d.get("revised_response") or d["response"],
            d["response"],
        )

    user_prompt = build_user_prompt(duration, segments, loudness_per_seg)
    if extra_user_suffix:
        user_prompt += "\n\n" + extra_user_suffix
    print(f"[llm] prompt size: {len(user_prompt)} chars, {len(segments)} segments")

    model, resp = call_llm(SYSTEM_PROMPT, user_prompt, force_model=force_model)

    # Budget check: parse cuts and see if we're over the ceiling.
    final_resp = resp
    revised_resp = None
    cut_pct_first = None
    cut_pct_revised = None
    try:
        first_cuts = parse_cuts(resp, max_duration=duration)
        first_cut_secs = sum(e - s for s, e in first_cuts)
        cut_pct_first = 100.0 * first_cut_secs / max(duration, 1e-6)
        revise_fn = None
        revise_target = None
        if cut_pct_first / 100.0 > BUDGET_CEILING:
            print(
                f"\n[WARNING] model over-cut on first pass: "
                f"{len(first_cuts)} cuts totalling {cut_pct_first:.1f}% "
                f"(ceiling {BUDGET_CEILING*100:.0f}%). Requesting revision...\n"
            )
            revise_fn = revise_cuts_over_budget
            revise_target = REVISION_UNDER_TARGET * 100
        elif cut_pct_first / 100.0 < BUDGET_FLOOR:
            print(
                f"\n[WARNING] model under-cut on first pass: "
                f"{len(first_cuts)} cuts totalling {cut_pct_first:.1f}% "
                f"(floor {BUDGET_FLOOR*100:.0f}%). Requesting revision...\n"
            )
            revise_fn = revise_cuts_under_floor
            revise_target = REVISION_OVER_TARGET * 100
        if revise_fn is not None:
            revised_resp = revise_fn(
                model=model, system=SYSTEM_PROMPT, user=user_prompt,
                original_response=resp,
                actual_cut_pct=cut_pct_first,
                duration_s=duration,
                target_pct=revise_target,
            )
            if revised_resp:
                final_resp = revised_resp
                # Diagnostic: report the revision's cut %.
                try:
                    rev_cuts = parse_cuts(revised_resp, max_duration=duration)
                    cut_pct_revised = 100.0 * sum(e - s for s, e in rev_cuts) / max(duration, 1e-6)
                    print(f"[llm] revision result: {len(rev_cuts)} cuts, {cut_pct_revised:.1f}% of source")
                    if cut_pct_revised / 100.0 > BUDGET_CEILING:
                        print(
                            f"\n[WARNING] model STILL over-cut after revision: "
                            f"{cut_pct_revised:.1f}%. Programmatic drop will run next.\n"
                        )
                except Exception as e:
                    print(f"[llm] couldn't re-parse revision: {e}")
    except Exception as e:
        # If parsing fails, just persist what we have and let pipeline handle.
        print(f"[llm] budget check skipped: parse error ({e})")

    save_json(out_path, {
        "model": model,
        "response": resp,
        "revised_response": revised_resp,
        "cut_pct_first": cut_pct_first,
        "cut_pct_revised": cut_pct_revised,
        "prompt_chars": len(user_prompt),
        "iteration": iteration,
        "forced_model": force_model,
    })
    # Return primary response too — it has the HIGHLIGHTS block (revised
    # responses are just a CUTS block). Pipeline uses primary for highlight
    # protection.
    return model, final_resp, resp
