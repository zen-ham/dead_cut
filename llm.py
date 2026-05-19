"""OpenRouter caller. Free models only, with fallback chain."""
import os
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

    This effectively turns a non-thinking model into a one-step-thinking model:
    the original output IS the thought, this second turn is the revision. Most
    of the time the first response is fine and this isn't called. When the
    model massively over-cuts, this lets it correct itself with full awareness
    of what it just emitted, before we resort to programmatic dropping.

    Returns the revised assistant response, or None on failure.
    """
    cut_secs = duration_s * (actual_cut_pct / 100.0)
    target_secs = duration_s * (target_pct / 100.0)
    correction = (
        f"Your CUTS_BEGIN..CUTS_END block totals approximately "
        f"{cut_secs:.0f}s, which is {actual_cut_pct:.1f}% of the {duration_s:.0f}s "
        f"video. That exceeds the 65% maximum — the response will be rejected.\n\n"
        f"Please reconsider. Drop the cut ranges you are LEAST confident were "
        f"genuinely boring (no strong loudness signal, no specific 'boring' "
        f"content you flagged in your reasoning — keep the cuts where you're "
        f"sure). Aim for total cut ≤ {target_pct:.0f}% (≈{target_secs:.0f}s "
        f"max).\n\n"
        f"Output ONLY a new CUTS_BEGIN..CUTS_END block with the revised list. "
        f"Nothing before or after. Same format rules apply (HH:MM:SS-HH:MM:SS, "
        f"one per line)."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": original_response},
        {"role": "user", "content": correction},
    ]
    print(f"[llm] revision request: model={model}, total {actual_cut_pct:.1f}% > 65% budget")
    t0 = time.time()
    resp = _call_one(model, messages, timeout=240)
    if resp:
        print(f"[llm] revision OK in {time.time()-t0:.1f}s, {len(resp)} chars")
    else:
        print(f"[llm] revision FAILED")
    return resp


# Budget threshold for triggering a revision request. If the model's first-pass
# CUTS exceed this fraction of source duration, ask it to reconsider.
BUDGET_CEILING = 0.65
# Target fraction we ask the revision to hit (leaves a buffer under the ceiling).
REVISION_TARGET = 0.60


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

    out_path = os.path.join(cache_dir(video_id), f"llm_response_iter{iteration}.json")
    if os.path.exists(out_path):
        d = load_json(out_path)
        print(f"[llm] cache hit (iter {iteration}): {out_path}")
        # Return revised if present, else primary.
        return d["model"], d.get("revised_response") or d["response"]

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
        if cut_pct_first / 100.0 > BUDGET_CEILING:
            print(
                f"\n[WARNING] model over-cut on first pass: "
                f"{len(first_cuts)} cuts totalling {cut_pct_first:.1f}% "
                f"(ceiling {BUDGET_CEILING*100:.0f}%). Requesting revision...\n"
            )
            revised_resp = revise_cuts_over_budget(
                model=model, system=SYSTEM_PROMPT, user=user_prompt,
                original_response=resp,
                actual_cut_pct=cut_pct_first,
                duration_s=duration,
                target_pct=REVISION_TARGET * 100,
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
    return model, final_resp
