"""Per-segment loudness annotation. Gives the LLM an entertainment proxy."""
import os
import subprocess

import numpy as np
import soundfile as sf

from .cache import cache_dir, save_json, load_json


SAMPLE_RATE = 16000  # mono, matches Whisper internal rate
WINDOW_MS = 100      # 100 ms RMS windows


def _extract_wav(video_path: str, wav_path: str) -> None:
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
        return
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-vn", "-f", "wav", wav_path,
    ]
    print(f"[loudness] extracting audio -> {wav_path}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _to_db(rms: np.ndarray, floor_db: float = -90.0) -> np.ndarray:
    """RMS in [0,1] -> dBFS. Clamps silence to floor."""
    eps = 10 ** (floor_db / 20.0)
    return 20.0 * np.log10(np.maximum(rms, eps))


def analyze(video_path: str, video_id: str, segments: list) -> dict:
    """For each transcript segment, compute peak_db, mean_db, loud_frac (frac of
    segment above (mean+6dB)). Cached as JSON."""
    out_path = os.path.join(cache_dir(video_id), "loudness.json")
    if os.path.exists(out_path):
        print(f"[loudness] cache hit: {out_path}")
        return load_json(out_path)

    wav_path = os.path.join(cache_dir(video_id), "audio.wav")
    _extract_wav(video_path, wav_path)

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    win = int(sr * WINDOW_MS / 1000)
    n_windows = len(audio) // win
    rms = np.sqrt(np.mean(audio[: n_windows * win].reshape(n_windows, win) ** 2, axis=1))
    db = _to_db(rms)

    # Track-wide stats so per-segment loudness is relative, not absolute.
    track_mean = float(np.mean(db))
    track_p90 = float(np.percentile(db, 90))
    loud_threshold = track_mean + 6.0  # 6 dB above mean = "loud" in this track
    # Silence threshold: well below mean but with an absolute floor so
    # quieter-overall tracks don't get the entire track flagged as silence.
    silence_threshold = max(track_mean - 12.0, -45.0)

    win_secs = WINDOW_MS / 1000.0
    per_segment = []
    for seg in segments:
        i0 = max(0, int(seg["start"] / win_secs))
        i1 = min(n_windows, int(seg["end"] / win_secs) + 1)
        if i1 <= i0:
            per_segment.append({"peak_db": -90.0, "mean_db": -90.0, "loud_frac": 0.0})
            continue
        chunk = db[i0:i1]
        per_segment.append({
            "peak_db": round(float(np.max(chunk)), 1),
            "mean_db": round(float(np.mean(chunk)), 1),
            "loud_frac": round(float(np.mean(chunk > loud_threshold)), 3),
        })

    # Detect silence runs for cut-boundary snapping. A run of ≥MIN_SILENCE_MS
    # of windows below silence_threshold counts as a silence interval.
    silences = _detect_silences(db, win_secs, silence_threshold, MIN_SILENCE_MS)

    result = {
        "track_mean_db": round(track_mean, 1),
        "track_p90_db": round(track_p90, 1),
        "loud_threshold_db": round(loud_threshold, 1),
        "silence_threshold_db": round(silence_threshold, 1),
        "window_ms": WINDOW_MS,
        "min_silence_ms": MIN_SILENCE_MS,
        "per_segment": per_segment,
        "silences": silences,
    }
    save_json(out_path, result)
    print(f"[loudness] mean={track_mean:.1f}dB p90={track_p90:.1f}dB loud_thr={loud_threshold:.1f}dB "
          f"silence_thr={silence_threshold:.1f}dB silences={len(silences)}")
    return result


MIN_SILENCE_MS = 250  # natural inter-word pause; shorter is breath/clipping


def _detect_silences(db: np.ndarray, win_secs: float, threshold_db: float, min_ms: int) -> list:
    """Return list of (start_sec, end_sec) intervals where dB < threshold for ≥min_ms.
    Each interval represents a snappable boundary candidate."""
    min_windows = max(1, int(round(min_ms / 1000.0 / win_secs)))
    is_silent = db < threshold_db
    intervals = []
    n = len(is_silent)
    i = 0
    while i < n:
        if is_silent[i]:
            j = i
            while j < n and is_silent[j]:
                j += 1
            if j - i >= min_windows:
                intervals.append([round(i * win_secs, 3), round(j * win_secs, 3)])
            i = j
        else:
            i += 1
    return intervals
