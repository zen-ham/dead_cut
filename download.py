"""Download source video with yt-dlp."""
import os
import yt_dlp

from .cache import cache_dir


def download(url: str, video_id: str) -> str:
    """Download best mp4 (video+audio merged) to cache/<id>/source.mp4. Returns path."""
    out_dir = cache_dir(video_id)
    out_path = os.path.join(out_dir, "source.mp4")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[download] cache hit: {out_path}")
        return out_path

    # Selector left format-agnostic for the video stream; merge_output_format
    # re-muxes to mp4 if the chosen video is webm/etc. mp4-only selectors break
    # on streams where YouTube only offers webm at the desired resolution.
    ydl_opts = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "outtmpl": os.path.join(out_dir, "source.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": False,
        "noprogress": False,
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

    # yt-dlp may write source.mp4 or source.mkv depending on merge; normalise.
    for cand in ("source.mp4", "source.mkv", "source.webm"):
        p = os.path.join(out_dir, cand)
        if os.path.exists(p):
            if cand != "source.mp4":
                # rename so downstream stages have a stable filename
                target = os.path.join(out_dir, "source.mp4")
                os.replace(p, target)
                return target
            return p
    raise RuntimeError(f"yt-dlp finished but no source file found in {out_dir}")
