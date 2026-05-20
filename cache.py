"""Cache helpers. All intermediates live under <project_root>/cache/<video_id>/."""
import json
import os
import re
from urllib.parse import urlparse, parse_qs


# cache/ lives in the OUTER dir (one above the repo), so it stays local and never
# gets committed. Inner repo .gitignore also ignores it as a belt-and-suspenders.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = os.path.join(_REPO_DIR, "cache")


def video_id_from_url(url: str) -> str:
    """Extract a YouTube video ID from any of the common URL shapes."""
    u = urlparse(url)
    if u.hostname in ("youtu.be",):
        return u.path.lstrip("/")
    if u.hostname and "youtube" in u.hostname:
        q = parse_qs(u.query)
        if "v" in q:
            return q["v"][0]
        # /shorts/<id>, /embed/<id>, /v/<id>
        m = re.match(r"^/(?:shorts|embed|v)/([^/?]+)", u.path)
        if m:
            return m.group(1)
    # Fallback: if it already looks like a bare ID, use it.
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    raise ValueError(f"Could not extract video ID from: {url}")


def cache_dir(video_id: str) -> str:
    d = os.path.join(CACHE_ROOT, video_id)
    os.makedirs(d, exist_ok=True)
    _migrate_legacy_names(d)
    return d


# Rename map: old cache filenames -> new (stage-prefixed) names.
# Run once at cache_dir() so existing caches keep working without re-downloading.
# To redo a pipeline stage now, you can just `rm` the files matching that
# stage's prefix: vid_*, transcribe*, loudness*, llm_*, cutter_*.
_LEGACY_NAMES = {
    "source.mp4":         "vid_src.mp4",
    "audio.wav":          "loudness_audio.wav",
    "transcript.json":    "transcribe.json",
    "audio_full.wav":     "cutter_audio_full.wav",
    "audio_spliced.wav":  "cutter_audio_spliced.wav",
    "filter_complex.txt": "cutter_filter.txt",
    "concat_list.txt":    "cutter_concat_list.txt",
    "segments":           "cutter_segments",
}


def _migrate_legacy_names(cache_dir_path: str) -> None:
    try:
        existing = os.listdir(cache_dir_path)
    except OSError:
        return
    for old in existing:
        new = _LEGACY_NAMES.get(old)
        # Pattern rename for llm_response_iter*.json -> llm_iter*.json
        if new is None and old.startswith("llm_response_iter") and old.endswith(".json"):
            new = old.replace("llm_response_iter", "llm_iter", 1)
        if not new:
            continue
        old_path = os.path.join(cache_dir_path, old)
        new_path = os.path.join(cache_dir_path, new)
        if not os.path.exists(new_path):
            try:
                os.rename(old_path, new_path)
            except OSError:
                pass  # don't crash the pipeline if rename fails


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
