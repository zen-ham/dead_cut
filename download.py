"""Download source video with yt-dlp."""
import os
import subprocess

import yt_dlp
from tqdm import tqdm

from .cache import cache_dir
from . import progress


# Codecs that lack hardware decode on common Turing GPUs (GTX 16xx, RTX 20xx).
# Only AV1 is the problem in practice — every other relevant codec has had
# NVDEC support since Maxwell. Ampere (RTX 30xx) added AV1 NVDEC, but the
# warning is still useful because the bottleneck is the SAME on any pre-Ampere
# GPU (and that's still the majority of consumer cards).
_SLOW_DECODE_CODECS = {
    "av1": (
        "no AV1 hardware decoder on Turing GPUs (GTX 16xx, RTX 20xx) — "
        "only Ampere (RTX 30xx) and newer have it. ffmpeg will fall back to "
        "CPU AV1 decode, which is ~3-5x slower than NVDEC on H.264. "
        "Re-download forcing H.264 via vcodec^=avc1 in your yt-dlp selector "
        "if your GPU is older than Ampere."
    ),
}


def _video_codec(path: str) -> str | None:
    """ffprobe the first video stream's codec name. Returns None on failure."""
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        return (r.stdout or "").strip() or None
    except Exception:
        return None


def _warn_if_slow_codec(path: str) -> None:
    """Loud banner-style warning if the downloaded source uses a codec known
    to cripple GPU pipelines on common consumer cards. Better to scream now
    than to silently take 4x longer through the rest of the pipeline."""
    codec = _video_codec(path)
    if codec is None:
        return
    if codec.lower() in _SLOW_DECODE_CODECS:
        why = _SLOW_DECODE_CODECS[codec.lower()]
        bar = "!" * 80
        print()
        print(bar)
        print(f"!! [WARNING] source video codec is {codec.upper()}")
        print("!!")
        for line in why.split(". "):
            line = line.strip(". ")
            if line:
                print(f"!! {line}.")
        print("!!")
        print(f"!! Expect this pipeline to run ~3-5x slower than with H.264.")
        print(f"!! Source file: {path}")
        print(bar)
        print()
    else:
        print(f"[download] source video codec: {codec} (hw-decode friendly)")


class _NullLogger:
    """Silence yt-dlp's print-style cookie-extraction errors during the
    metadata probe. We try browsers in order and tolerate failures — there's
    no point spamming the user with chrome-not-installed errors when firefox
    works fine on the next attempt."""
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def fetch_duration(url: str) -> float | None:
    """Lightweight metadata-only call to yt-dlp. Returns the video duration in
    seconds, or None on failure. Used to size the progress tracker's baselines
    BEFORE the actual download so the overall ETA is realistic from t=0.

    Roughly 2-5s to complete. Tries cookies from the same browsers as the real
    download so it doesn't fail on age-gated/private content."""
    info_opts = {
        "quiet": True, "no_warnings": True, "skip_download": True,
        "logger": _NullLogger(),
    }
    for browser in ("chrome", "firefox", "edge"):
        try:
            opts = dict(info_opts, cookiesfrombrowser=(browser,))
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                d = info.get("duration")
                if d:
                    return float(d)
        except Exception:
            continue
    return None


def download(url: str, video_id: str) -> str:
    """Download best mp4 (video+audio merged) to cache/<id>/vid_src.mp4. Returns path."""
    out_dir = cache_dir(video_id)
    out_path = os.path.join(out_dir, "vid_src.mp4")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[download] cache hit: {out_path}")
        # Still check codec on cache hit — user might have a stale AV1 source
        # from before the H.264 selector was added, and we want them to know
        # the pipeline is about to be much slower than it could be.
        _warn_if_slow_codec(out_path)
        return out_path

    # PREFER H.264 (vcodec=avc1) over AV1/VP9. YouTube serves AV1 as
    # "best video" for most modern uploads, but AV1 has NO hardware decoder
    # on Turing GPUs (GTX 16xx, RTX 20xx) and only partial support on Ampere.
    # CPU-decoding 3 hours of AV1 through a select filter is the difference
    # between a 5-min and a 40-min encode. The selector falls back to AV1
    # only when no H.264 stream is offered.
    # Drive a proper tqdm bar from yt-dlp's progress_hooks so it lives at
    # position=1 alongside the overall bar. Otherwise yt-dlp's raw printf
    # progress lines clobber the overall bar at position=0 (they're not
    # tqdm-aware, just stdout writes).
    dl_bar = [None]
    def _hook(d):
        st = d.get("status")
        if st == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total and dl_bar[0] is None:
                fname = os.path.basename(d.get("filename", "source"))[:40]
                dl_bar[0] = tqdm(
                    total=total, unit="B", unit_scale=True,
                    desc=f"[stage   ] download {fname}",
                    position=1, leave=False,
                    bar_format="{desc} {bar} {percentage:3.0f}% | {n_fmt}/{total_fmt} | eta {remaining}",
                )
            if dl_bar[0] is not None:
                if total and dl_bar[0].total != total:
                    dl_bar[0].total = total
                downloaded = d.get("downloaded_bytes", 0)
                dl_bar[0].n = downloaded
                dl_bar[0].refresh()
                # Report observed download pace for overall ETA.
                if total and downloaded > 0:
                    progress.report_stage_rate("download", downloaded / total)
                progress.tick()
        elif st == "finished":
            if dl_bar[0] is not None:
                dl_bar[0].n = dl_bar[0].total or dl_bar[0].n
                dl_bar[0].refresh()
                dl_bar[0].close()
                dl_bar[0] = None

    ydl_opts = {
        "format": (
            "bestvideo[height<=720][vcodec^=avc1]+bestaudio/"
            "bestvideo[height<=720][vcodec^=h264]+bestaudio/"
            "bestvideo[height<=720]+bestaudio/"
            "best[height<=720]/best"
        ),
        "outtmpl": os.path.join(out_dir, "vid_src.%(ext)s"),
        "merge_output_format": "mp4",
        # Suppress yt-dlp's own stdout output — the tqdm bar from _hook
        # replaces it. Errors still print (quiet=True only kills info-level).
        "quiet": True,
        "noprogress": True,
        "no_warnings": False,
        "progress_hooks": [_hook],
        "retries": 5,
    }
    # YouTube bot-check now blocks anonymous downloads. Pull cookies from a
    # locally-installed browser. Try Chrome first, then Firefox, then Edge.
    browsers_to_try = ("chrome", "firefox", "edge")
    print(f"[download] fetching {url}")
    last_err = None
    for browser in browsers_to_try:
        try:
            opts = dict(ydl_opts, cookiesfrombrowser=(browser,))
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            print(f"[download] succeeded using {browser} cookies")
            last_err = None
            break
        except Exception as e:
            print(f"[download] {browser} cookies failed: {type(e).__name__}: {str(e)[:200]}")
            last_err = e
    if last_err is not None:
        # Final fallback: no cookies (will likely fail on bot-check but try anyway)
        print("[download] all browser cookies failed, trying without cookies")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    # yt-dlp may write vid_src.mp4 or vid_src.mkv/.webm depending on merge;
    # normalise to vid_src.mp4 so downstream stages have a stable filename.
    for cand in ("vid_src.mp4", "vid_src.mkv", "vid_src.webm"):
        p = os.path.join(out_dir, cand)
        if os.path.exists(p):
            if cand != "vid_src.mp4":
                target = os.path.join(out_dir, "vid_src.mp4")
                os.replace(p, target)
                _warn_if_slow_codec(target)
                return target
            _warn_if_slow_codec(p)
            return p
    raise RuntimeError(f"yt-dlp finished but no source file found in {out_dir}")
