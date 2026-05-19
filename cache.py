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
    return d


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
