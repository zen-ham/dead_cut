"""Transcribe with faster-whisper. Cached per video."""
import os
from faster_whisper import WhisperModel

from .cache import cache_dir, save_json, load_json


# `small` is the sweet spot for this kind of monologue/stream content: ~3x faster
# than medium with marginal accuracy loss for clean speech. CPU int8 keeps the
# memory footprint low. Bump to `medium` or `large-v3` if accuracy is the blocker.
DEFAULT_MODEL = "small"

_model_cache = {}


def _get_model(name: str, device: str = "auto", compute_type: str = "int8"):
    key = (name, device, compute_type)
    if key not in _model_cache:
        print(f"[transcribe] loading model {name} ({compute_type})")
        _model_cache[key] = WhisperModel(name, device=device, compute_type=compute_type)
    return _model_cache[key]


def transcribe(video_path: str, video_id: str, model_name: str = DEFAULT_MODEL) -> dict:
    """Return {duration, segments: [{start, end, text}], language}. Cached as JSON."""
    out_path = os.path.join(cache_dir(video_id), "transcript.json")
    if os.path.exists(out_path):
        print(f"[transcribe] cache hit: {out_path}")
        return load_json(out_path)

    model = _get_model(model_name)
    print(f"[transcribe] running on {video_path}")
    segments_iter, info = model.transcribe(
        video_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    segments = []
    for s in segments_iter:
        segments.append({
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "text": s.text.strip(),
        })
        if len(segments) % 50 == 0:
            print(f"[transcribe] {len(segments)} segments, t={segments[-1]['end']:.1f}s")
    result = {
        "duration": float(info.duration),
        "language": info.language,
        "language_probability": float(info.language_probability),
        "segments": segments,
    }
    save_json(out_path, result)
    print(f"[transcribe] done: {len(segments)} segments, {result['duration']:.1f}s of audio")
    return result
