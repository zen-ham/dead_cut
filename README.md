Auto-edit long YouTube vods into tight cuts of just the entertaining bits.
=

![lastcommit](https://img.shields.io/github/last-commit/zen-ham/dead_cut) ![python](https://img.shields.io/badge/python-3.10+-blue) ![ffmpeg](https://img.shields.io/badge/ffmpeg-required-green) ![GPU](https://img.shields.io/badge/NVIDIA_GPU-recommended-76b900)

<p align="center">
    <img src="https://github.com/zen-ham/dead_cut/blob/master/repo_assets/logo.svg" width="90%" />
</p>

---
Technical details:
-

Point this at a 27-minute vod, get back a 9-minute cut. Point it at a 45-minute stream, get back 11 minutes of the actually-entertaining bits. One command, fully hands-off, runs end-to-end on a free OpenRouter model with no API costs.

The pipeline downloads with `yt-dlp` (auto-pulls cookies from Firefox/Chrome/Edge to bypass YouTube's bot check), transcribes with `faster-whisper` running batched `int8_float16` on GPU at ~16x realtime (small model + `BatchedInferencePipeline(batch_size=8)` — benchmarked the alternatives, this won), then computes per-segment loudness so every transcript line in the prompt is annotated with peak dB, mean dB, and a loud-fraction so the model has an audio-energy hint for which sections are likely entertaining. The annotated transcript gets fed to a free OpenRouter model (`gpt-oss-120b:free` by default, with fallback chain) under a carefully-tuned prompt that requires the model to enumerate specific highlights *before* emitting cut ranges — without that forcing function, the model lazily chunks the runtime into equal blocks instead of actually engaging with the content (observed on iter 0 of the very first test, cut 90% of the video in one block).

The raw LLM cut ranges then go through two post-processing stages that nobody told the model about. First, **snap-to-silence**: each cut boundary moves to the nearest actual silence in the waveform within ±2s, so cuts land between words instead of mid-sentence — transcript-segment boundaries are NOT word boundaries, and the LLM has no way to know that. The silence threshold is derived from the median loudness of the speech segments themselves (not the overall track mean, which gets pulled around by silence ratio and background music). Second, **in-keep silence trim**: each keep range is scanned for internal silences >0.6s and they get compressed to a 0.4s gap — this kills the 15-second walking-around stretches inside an otherwise-kept clip that the LLM can't see at segment-aggregate resolution. On the test vod this removed an additional 8 minutes of dead air on top of the LLM's macro cuts.

Final output is built with [`smartcut`](https://github.com/skeskinen/smartcut) (PyAV-based partial-reencode lib) with `h264_nvenc` monkey-patched in for the boundary GOPs — every cut splits exactly one or two GOPs around it, those get decoded and re-encoded on the GPU, every other GOP is stream-copied at the bitstream level. Audio is opus passthru'd from source, no re-encode at all. The full-reencode `select`-filter path still lives in `cutter.py` as a fallback (used automatically if smartcut errors, or for non-h264 sources like AV1). Benchmarked at ~2.1x faster than the old full-reencode pipeline on a 45min 720p60 h264 source with 221 cuts (49s vs 102s on a GTX 1660 Ti). Sample-precise output duration regardless of cut count — no keyframe drift across hundreds of micro-cuts (which is what kills the naive stream-copy approach that I tried first; 221 stream-copy cuts on the test vod accumulated ~10 minutes of drift before I caught it).

Speed on a GTX 1660 Ti:
- 27-min vod → ~4 min total
- 45-min vod → ~9 min total
- 3-hour vod → ~24 min total

Usage:
-

```
python -m dead_cut "https://www.youtube.com/watch?v=ID"
```

Output lands at `cache/<video_id>/final.mp4`. Every pipeline stage caches its result keyed by video ID, so re-runs with the same URL skip the slow parts — download, transcribe, loudness, all reused. Bump `--iter N` to re-call the LLM with the same cached upstream data (cheap to iterate on prompt changes or to compare different models). Other flags worth knowing:

```
--iter N             re-call LLM with cached transcribe/loudness, new iter cache key
--model qwen/qwen3.6-plus:free    force a specific OpenRouter model
--no-trim            disable within-keep silence trim (keeps looser pacing)
--no-snap            disable cut-boundary snap-to-silence
--dry-run            run through LLM but skip ffmpeg
```

Setup:
-

`pip install -r requirements.txt` (yt-dlp, faster-whisper, numpy, soundfile, requests, tqdm, smartcut), `ffmpeg` on PATH, and either set `OPENROUTER_API_KEY` in your env or drop a free OpenRouter key on a single line in `openrouter_token.txt` in the parent directory of the repo (token is loaded from there, never committed).

NVIDIA GPU is not required but the speed numbers above assume CUDA. CPU fallback works, just slower — transcribe drops from ~16x realtime to ~1x.
