Auto Edit Long YouTube Vods Into Tight Cuts of Just the Entertaining Bits.
=

![lastcommit](https://img.shields.io/github/last-commit/zen-ham/dead_cut) ![python](https://img.shields.io/badge/python-3.10+-blue) ![ffmpeg](https://img.shields.io/badge/ffmpeg-required-green) ![GPU](https://img.shields.io/badge/NVIDIA_GPU-recommended-76b900)

<p align="center">
    <img src="https://github.com/zen-ham/dead_cut/blob/master/repo_assets/logo.svg" width="90%" />
</p>

---


Give this a 27 minute vod and get back a 9 minute cut. 45 minute stream to 11 minutes of the actually entertaining bits. One command, fully hands off, free openrouter model, no api costs at all forever. The whole pipeline runs at ~50-60x realtime on h.264 sources (a 1h25 vod processes end to end in 2m17s on a gtx 1660 ti).

The way it actually does this is a 2 layer system of editing, first the AI does the macro scale edits, depending on video length it can do 100+ edits on the timeline. Then the second phase is the micro edits, algorithmic silence removal makes the video much more engaging and feels exactly like a well edited vod a youtuber would actually put on main channel.

-
Pipeline Technical Details:
-

**1. Download**. `yt-dlp`, downloads in h264 format for fast processing. Cookies auto load from firefox/chrome/edge to bypass youtube's bot check. It also first grabs the metadata of the video to compute the overall progress bar accuratly.

**2. Transcribe**. `faster-whisper` small model with `BatchedInferencePipeline(batch_size=8)` on `int8_float16`. ~67x realtime on the 1660 ti, a 1h25 vod transcribes in <1 min. I tried other configurations and models but this is 3x faster then the second fastest one, and you don't really need amazing transcript accuracy for this.

**3. Loudness Analysis**. Each transcript line in the prompt gets annotated with `P=<peak> M=<mean> L=<loud-fraction>` so the model has an entertainment proxy alongside the text. Silence detection runs ≥250ms below threshold, and the threshold is derived from the MEDIAN of the per segment loudness (the vad filtered speech regions), not the overall track mean. Without that, vods with background game music get the threshold pulled up by the music and the silence detector misses real silences.

**4. AI Cut Detection**. Annotated transcript goes to a free openrouter model (`gpt-oss-120b:free` default, with fallback chain). The system prompt forces a `HIGHLIGHTS_BEGIN..HIGHLIGHTS_END` block FIRST (concrete quotes of moments to keep) before the `CUTS_BEGIN..CUTS_END` block, without that forcing function the model just block chunks the runtime into equal cuts (observed on iter 0 of the very first test: cut 99.99% of the video, output was empty). Cuts require an inline reason on each line so vague "boring section" doesn't fly. Hard 50% minimum cut to stop the model from wussing out. And the prompt explicitly tells the model that dry/deadpan/sarcastic humor is QUIET (L < 0.3) and must not be cut, caught when the model was butchering jokes labeled "low energy, no humor" that were actually the best deadpan bits in the vod.

If the first pass cuts land outside [50%, 75%], a SECOND call goes back to the same model with the original response in context and a gap aware correction message. Under by 3% gets "you need ~3% more, add 1 to 3 short cuts", under by 30% gets "you're way off, find way more". Similar split for over budget. This turns a non thinking model into a one step thinking model, the original output is the thought, the second turn is the revision, and the gap aware severity stops the model from wildly overcorrecting (which it does on a generic "be more aggressive" prompt).

There's four programmatic layers between the model and the encoder:
- `protect_highlights`: parses the highlights block independently of the cuts block, then splits any cut that contains a highlight timestamp to preserve a 3s window. Model contradicts its own highlights sometimes, this is the guarantee that doesn't.
- `merge_close_cuts`: adjacent cuts with <5s gaps look like contiguous boring stretches the model split for transcript line reasons, but ONLY merged if the gap is fully covered by detected silence. Verified the model often leaves intentional 1 to 3s gaps with real content (a quick quip, a reaction) and we don't want to swallow those.
- `snap_cuts_to_silence`: each cut boundary moves to the nearest detected silence within ±2s, so cuts land between words instead of mid sentence. Transcript segment boundaries are NOT word boundaries, the model has no way to know that.
- `enforce_budget`: final safety net, if cuts are still >75% of runtime after the revision, drop the longest until under. Hasn't fired since the gap aware revision landed but it's there.

**6. Algorithmically trim silence**
Then `trim_silences_within_keeps` runs INSIDE each keep range, compressing any inner silence >0.6s down to a 0.4s gap. Typically removes another 150 to 500s of dead air the model couldn't see at segment aggregate resolution.

**7. Encode**. [`smartcut`](https://github.com/skeskinen/smartcut) (pyav based partial reencode lib) with `h264_nvenc` monkey patched in for boundary GOPs. Every cut splits exactly one or two GOPs around it, those get decoded plus reencoded on the GPU, every other GOP is stream copied at the bitstream level. Audio is opus passthru'd from source, no reencode at all. The patch is required because smartcut's `VideoSettings.codec_override` only applies in `RECODE` mode (full reencode every frame, defeats the point), in `SMARTCUT` mode the encoder is hardcoded to the source codec (h264, libx264 via pyav default). `_patch_smartcut_for_nvenc` wraps `VideoCutter.__init__` to flip `self.codec_name = 'h264_nvenc'` and `init_encoder` to replace the libx264 encoding_options with nvenc appropriate ones. Benchmarked at 2.1x faster than the full reencode fallback on a 45min 720p60 source with 221 cuts (49s vs 102s).

The full reencode `select` filter path still lives in `cutter.py` as a fallback (kicks in for non h264 sources like av1, or if smartcut errors). That path uses snap to frame boundaries plus `gte(t,s)*lt(t,e)` (not `between()`, which is inclusive on both ends and accumulates ~1 extra frame per cut, multi second a/v drift across 600 cuts) to keep audio and video aligned at sample precision.

**Progress Bars**. Two stacked tqdm bars, `[overall]` (eta of entire pipeline) above `[stage]` (eta of current stage). The overall starts with baseline timings (transcribe 0.015x source, encode 0.020x output, etc.) measured on my machine, then refines as each stage finishes. Crucially, each in flight stage reports observed rate via `progress.report_stage_rate(stage, fraction_done)` so the overall total stops trusting the baseline once we have actual pace data. A background ticker thread refreshes the bar every 2s so stages without their own callback (audio extract, etc.) still animate live.

**Caching**. Every stage's output is cached under `cache/<video_id>/` with stage prefixed filenames (`vid_src.mp4`, `transcribe.json`, `loudness.json`, `llm_iter{N}.json`, `cutter_*`, `final.mp4`). Reruns hit the cache for everything they don't need to redo. `--iter N` bumps just the llm cache key so you can iterate on prompt changes without redownloading or retranscribing.

Speed (gtx 1660 ti, cached download):

- 27 min vod  → ~50s total
- 45 min vod  → ~1m 30s total
- 1h25 vod    → ~2m 17s total (real bench)
- 3 hour vod  → ~5m total

Cold cache adds the actual download time (varies with your connection, maybe plus 1 to 3 min for a typical 30min to 3hr 720p h264 video).

Usage:
-

```
python -m dead_cut "https://www.youtube.com/watch?v=ID"
```

Output lands at `cache/<video_id>/final.mp4`. Every pipeline stage caches its result keyed by video ID, so reruns with the same URL skip the slow parts, download, transcribe, loudness, all reused. Bump `--iter N` to recall the LLM with the same cached upstream data (cheap to iterate on prompt changes or to compare different models). Other flags worth knowing:

```
--iter N             recall LLM with cached transcribe/loudness, new iter cache key
--model qwen/qwen3.6-plus:free    force a specific OpenRouter model
--no-trim            disable within keep silence trim (keeps looser pacing)
--no-snap            disable cut boundary snap to silence
--dry-run            run through LLM but skip ffmpeg
```

Setup:
-

`pip install -r requirements.txt` (yt-dlp, faster-whisper, numpy, soundfile, requests, tqdm, smartcut), `ffmpeg` on PATH, and either set `OPENROUTER_API_KEY` in your env or drop a free OpenRouter key on a single line in `openrouter_token.txt` in the parent directory of the repo (token is loaded from there, never committed).

NVIDIA GPU is not required but the speed numbers above assume CUDA. CPU fallback works, just slower, transcribe drops from ~16x realtime to ~1x.
