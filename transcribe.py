"""Transcribe with faster-whisper. Cached per video.

Uses BatchedInferencePipeline w/ batch_size=8 + int8_float16: benchmarked at
~2.3x faster than the basic small/int8 baseline on this GPU, with the same
accuracy. Batched output has fewer, larger segments (VAD chunks instead of
~5s sliding windows) — fine here because silence-snap / silence-trim work on
the raw waveform independent of segment boundaries.
"""
import os
from faster_whisper import WhisperModel, BatchedInferencePipeline
from tqdm import tqdm

from .cache import cache_dir, save_json, load_json


DEFAULT_MODEL = "small"
DEFAULT_COMPUTE = "int8_float16"  # GPU-optimal; falls back gracefully on CPU
DEFAULT_BATCH = 8

_model_cache = {}


def _get_pipeline(name: str, device: str = "auto", compute_type: str = DEFAULT_COMPUTE):
    key = (name, device, compute_type)
    if key not in _model_cache:
        print(f"[transcribe] loading model {name} ({compute_type}, batched)")
        try:
            model = WhisperModel(name, device=device, compute_type=compute_type)
        except ValueError as e:
            # int8_float16 only works on CUDA; fall back to int8 on CPU.
            if "compute type" in str(e).lower():
                print(f"[transcribe] {compute_type} unsupported here, falling back to int8")
                model = WhisperModel(name, device=device, compute_type="int8")
            else:
                raise
        _model_cache[key] = BatchedInferencePipeline(model=model)
    return _model_cache[key]


def transcribe(
    video_path: str,
    video_id: str,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH,
) -> dict:
    """Return {duration, segments: [{start, end, text}], language}. Cached as JSON."""
    out_path = os.path.join(cache_dir(video_id), "transcript.json")
    if os.path.exists(out_path):
        print(f"[transcribe] cache hit: {out_path}")
        return load_json(out_path)

    pipeline = _get_pipeline(model_name)
    print(f"[transcribe] running on {video_path} (batch={batch_size})")
    segments_iter, info = pipeline.transcribe(
        video_path,
        batch_size=batch_size,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    segments = []
    # tqdm: drive progress by audio-time-covered vs total duration, since we
    # know info.duration up front and segments come in chronological order.
    pbar = tqdm(
        total=float(info.duration), unit="s", desc="[transcribe]",
        bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]",
        leave=True,
    )
    prev_end = 0.0
    for s in segments_iter:
        segments.append({
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "text": s.text.strip(),
        })
        delta = max(0.0, s.end - prev_end)
        pbar.update(delta)
        prev_end = s.end
    # Snap to 100% in case the last segment ended slightly short of info.duration.
    pbar.n = pbar.total
    pbar.refresh()
    pbar.close()
    result = {
        "duration": float(info.duration),
        "language": info.language,
        "language_probability": float(info.language_probability),
        "segments": segments,
    }
    save_json(out_path, result)
    print(f"[transcribe] done: {len(segments)} segments, {result['duration']:.1f}s of audio")
    return result
