# dead_cut

Automated pipeline that turns a long, lightly-edited YouTube vod into a tight cut of the entertaining parts.

## Pipeline

1. **Download** the source video with `yt-dlp` (mp4, audio + video).
2. **Transcribe** the audio with `faster-whisper`, producing word-level timestamps.
3. **Loudness-analyze** the audio: peak + duration of loud spikes per transcript segment, used as an entertainment proxy.
4. **Cut detection** via OpenRouter (free models, fallback chain). The LLM receives the timestamped transcript + per-segment loudness and returns ranges to **remove**. Prompt is strict: cut-only output, never "keep" ranges.
5. **Trim** with `ffmpeg` using stream-copy concat for speed and zero re-encode loss.

Intermediate artifacts (download, transcript, loudness) are cached per video ID so re-iterating on the prompt is cheap.

## Setup

1. `pip install -r requirements.txt`
2. Install `ffmpeg` (must be on PATH).
3. Provide an OpenRouter API key (free tier is fine). Either:
   - Set `OPENROUTER_API_KEY` in your environment, OR
   - Put the key on a single line in `openrouter_token.txt` in the parent
     directory of this repo (one level above `dead_cut/`).
4. YouTube now requires cookies for most downloads. The downloader auto-loads
   them from a locally-installed Firefox / Chrome / Edge profile via `yt-dlp`'s
   `cookies-from-browser`. If none of those are installed, downloads will fail
   on the bot-check.

## Usage

```bash
python -m dead_cut https://www.youtube.com/watch?v=Ur6X9be0CO8
```

Output written to `cache/<video_id>/final.mp4`. Each pipeline stage caches its
output, so re-running with `--iter N` (any number you haven't used before) will
re-call only the LLM and ffmpeg — download/transcribe/loudness stay cached.

```bash
python -m dead_cut <url> --iter 2                 # try a new cut on cached transcript
python -m dead_cut <url> --iter 2 --model qwen/qwen3.6-plus:free   # try another model
python -m dead_cut <url> --iter 2 --dry-run        # call LLM, skip ffmpeg
python -m dead_cut.judge <video_id> 2              # print synthesized kept transcript
```

## Dependencies

- Python 3.10+
- `ffmpeg` on PATH
- NVIDIA GPU + CUDA optional but ~10x faster transcribe (CPU fallback auto)
- See `requirements.txt`
