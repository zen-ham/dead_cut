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


def _call_one(model: str, system: str, user: str, timeout: int = 240) -> str | None:
    try:
        r = requests.post(
            OPENROUTER_URL,
            headers=HEADERS,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.4,
                # Cut detection on a long vod needs room for HIGHLIGHTS +
                # REASONING + CUTS. Default ~512 truncated v0 output to 199 chars.
                "max_tokens": 4000,
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
    models = [force_model] if force_model else MODELS
    for m in models:
        print(f"[llm] trying {m}...")
        t0 = time.time()
        resp = _call_one(m, system, user)
        if resp:
            print(f"[llm] {m} OK in {time.time()-t0:.1f}s, {len(resp)} chars")
            return m, resp
        time.sleep(1.0)
    raise RuntimeError("All OpenRouter models failed")


def detect_cuts(
    video_id: str,
    duration: float,
    segments: list,
    loudness_per_seg: list,
    iteration: int = 0,
    extra_user_suffix: str = "",
    force_model: str | None = None,
) -> tuple[str, str]:
    """Call the LLM to produce a cut list. Cached per (video_id, iteration).
    Returns (model_used, raw_response)."""
    out_path = os.path.join(cache_dir(video_id), f"llm_response_iter{iteration}.json")
    if os.path.exists(out_path):
        d = load_json(out_path)
        print(f"[llm] cache hit (iter {iteration}): {out_path}")
        return d["model"], d["response"]

    user_prompt = build_user_prompt(duration, segments, loudness_per_seg)
    if extra_user_suffix:
        user_prompt += "\n\n" + extra_user_suffix
    print(f"[llm] prompt size: {len(user_prompt)} chars, {len(segments)} segments")

    model, resp = call_llm(SYSTEM_PROMPT, user_prompt, force_model=force_model)
    save_json(out_path, {
        "model": model,
        "response": resp,
        "prompt_chars": len(user_prompt),
        "iteration": iteration,
        "forced_model": force_model,
    })
    return model, resp
