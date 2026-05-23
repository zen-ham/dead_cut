"""Per-segment loudness annotation. Gives the LLM an entertainment proxy."""
import os
import subprocess

import numpy as np
import soundfile as sf
from tqdm import tqdm

from .cache import cache_dir, save_json, load_json
from . import progress


SAMPLE_RATE = 16000  # mono, matches Whisper internal rate
WINDOW_MS = 100      # 100 ms RMS windows


def _extract_wav(video_path: str, wav_path: str, total_secs: float = 0.0, pbar=None) -> None:
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
        if pbar is not None and total_secs > 0:
            pbar.n = total_secs
            pbar.refresh()
        return
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-vn", "-f", "wav",
        "-progress", "pipe:1", "-nostats",
        wav_path,
    ]
    print(f"[loudness] extracting audio -> {wav_path}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        for line in p.stdout:
            line = line.strip()
            if not line.startswith("out_time_us="):
                continue
            try:
                cur = int(line.split("=", 1)[1]) / 1e6
            except (ValueError, IndexError):
                continue
            if pbar is not None and total_secs > 0:
                pbar.n = min(cur, float(total_secs))
                pbar.refresh()
                progress.report_stage_rate("loudness", min(cur / total_secs, 0.95))
                progress.tick()
    finally:
        p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extract failed (returncode {p.returncode})")


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

    wav_path = os.path.join(cache_dir(video_id), "loudness_audio.wav")
    # Source duration approx from last segment end. Drives the stage bar
    # during ffmpeg extract (the dominant slow phase, ~80% of stage time).
    total_secs = float(segments[-1]["end"]) if segments else 0.0
    pbar = tqdm(
        total=float(total_secs) if total_secs > 0 else 1.0,
        unit="s", desc="[stage   ] loudness  ",
        bar_format="{desc} {bar} {percentage:3.0f}% | {n:.0f}/{total:.0f}s | eta {remaining}",
        position=1, leave=False,
    )
    _extract_wav(video_path, wav_path, total_secs=total_secs, pbar=pbar)
    progress.report_stage_rate("loudness", 0.85)
    progress.tick()

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    win = int(sr * WINDOW_MS / 1000)
    n_windows = len(audio) // win
    rms = np.sqrt(np.mean(audio[: n_windows * win].reshape(n_windows, win) ** 2, axis=1))
    db = _to_db(rms)
    win_secs = WINDOW_MS / 1000.0
    duration = n_windows * win_secs  # audio-derived; matches what the cutter sees

    # Track-wide stats so per-segment loudness is relative, not absolute.
    track_mean = float(np.mean(db))
    track_p90 = float(np.percentile(db, 90))
    loud_threshold = track_mean + 6.0  # 6 dB above mean = "loud" in this track
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

    # Speech-level reference: median of per-segment mean_db. Segments are
    # VAD-filtered speech regions, so their median is a clean speech-volume
    # estimate that adapts per-video and handles background music correctly.
    if per_segment:
        speech_means = sorted(s["mean_db"] for s in per_segment if s["mean_db"] > -85)
        speech_level = speech_means[len(speech_means) // 2] if speech_means else track_mean
    else:
        speech_level = track_mean
    global_silence_threshold = max(speech_level - 12.0, -50.0)

    # Local-floor silence detection. Every inter-segment gap (where the
    # transcript VAD says nobody is speaking) is a real silence sample,
    # so the median dB of that gap IS the local noise floor at that time.
    # We build markers from those gaps, then each audio window gets its
    # own threshold = local_floor + headroom, capped below speech_level so
    # actual speech can't false-positive. Adapts per-section: a section
    # with bg music has a higher floor + higher threshold, catching its
    # inter-word silences that the old global threshold missed entirely.
    floor_markers = _compute_floor_markers(
        db=db, win_secs=win_secs, segments=segments,
        duration=duration, min_gap_s=MIN_GAP_SAMPLE_S,
    )
    silence_thresholds = _per_window_thresholds_from_markers(
        n_windows=n_windows, win_secs=win_secs,
        markers=floor_markers,
        headroom_db=SILENCE_HEADROOM_DB,
        cap_db=speech_level - 3.0,
        fallback_db=global_silence_threshold,
    )
    silences_audio = _detect_silences_adaptive(db, silence_thresholds, win_secs, MIN_SILENCE_MS)
    # Speech-gap injection: treat transcript-level no-speech gaps as silences
    # too. The audio dB detector can't catch "walking around with bg music"
    # because the music is louder than the local noise floor, but the
    # transcript knows nobody is talking there. Anything ≥ SPEECH_GAP_MIN_S
    # between two segments becomes a silence we'll let the trim collapse.
    speech_gaps = _compute_speech_gaps(segments, duration, min_gap_s=SPEECH_GAP_MIN_S)

    # CONTENT PROTECTION — applies ONLY to speech-gap silences, NOT audio
    # silences. The audio detector is dB-honest: if it says a region is
    # silent, the audio really IS quiet, so trimming is safe even inside
    # a sentence-break gap. The PW (speech-gap) silences are the ones at
    # risk of catching loud reactions/screams that the transcript missed,
    # so we filter just those.
    #
    # Two heuristics for protecting PW silences:
    #  1. Sentence-break gaps — gaps after a `.`/`!`/`?` are where streamers
    #     typically have reactions/sound effects/screams. Any speech-gap
    #     overlapping such a sentence-break gap gets preserved.
    #  2. Loud-peak — if the audio in the speech-gap range has a window
    #     louder than `speech_level + LOUD_PEAK_OFFSET_DB`, it's content
    #     (scream, hype, sound effect), preserve it.
    sentence_break_gaps = _compute_sentence_break_gaps(
        segments, min_gap_s=SENTENCE_BREAK_MIN_S, max_gap_s=SENTENCE_BREAK_MAX_S,
    )
    loud_peak_threshold = speech_level + LOUD_PEAK_OFFSET_DB
    speech_gaps_filtered, drop_stats = _filter_silences_for_content(
        speech_gaps, db, win_secs,
        sentence_break_gaps=sentence_break_gaps,
        loud_peak_threshold_db=loud_peak_threshold,
    )
    silences = _union_intervals(silences_audio + speech_gaps_filtered)

    # Downsampled dB envelope for visualizer (max-pool per bucket so peaks
    # like screams/laughs survive). N=1000 keeps the file small (~6KB) but
    # gives enough resolution for a smooth plot at typical figure widths.
    env_n = min(1000, n_windows)
    if env_n > 0 and n_windows > 0:
        bucket = max(1, n_windows // env_n)
        usable = (n_windows // bucket) * bucket
        envelope_db = db[:usable].reshape(-1, bucket).max(axis=1)
        envelope = [round(float(v), 1) for v in envelope_db]
    else:
        envelope = []

    result = {
        "track_mean_db": round(track_mean, 1),
        "track_p90_db": round(track_p90, 1),
        "envelope_db": envelope,
        "envelope_window_s": round(duration / max(len(envelope), 1), 4) if envelope else None,
        "speech_level_db": round(speech_level, 1),
        "loud_threshold_db": round(loud_threshold, 1),
        "silence_threshold_db": round(global_silence_threshold, 1),
        "silence_threshold_min_db": round(float(np.min(silence_thresholds)), 1) if len(silence_thresholds) else None,
        "silence_threshold_max_db": round(float(np.max(silence_thresholds)), 1) if len(silence_thresholds) else None,
        "silence_headroom_db": SILENCE_HEADROOM_DB,
        "n_floor_markers": len(floor_markers),
        "floor_markers": [
            {"t": round(t, 2), "floor_db": round(f, 1)} for t, f in floor_markers
        ],
        "speech_gap_min_s": SPEECH_GAP_MIN_S,
        "n_speech_gaps": len(speech_gaps),
        "n_silences_audio_only": len(silences_audio),
        # Per-type silence lists so the visualizer can show audio-detected
        # silences (orange) separately from transcription-derived speech-gap
        # silences (yellow). Both go into the final `silences` union list.
        "silences_audio_only_list": [[round(s, 3), round(e, 3)] for s, e in silences_audio],
        "silences_speech_gaps_list": [list(g) for g in speech_gaps_filtered],
        "sentence_break_min_s": SENTENCE_BREAK_MIN_S,
        "n_sentence_break_gaps": len(sentence_break_gaps),
        "loud_peak_threshold_db": round(loud_peak_threshold, 1),
        "n_silences_dropped_sentence_break": drop_stats["dropped_sentence_break"],
        "n_silences_dropped_loud_peak": drop_stats["dropped_loud_peak"],
        "window_ms": WINDOW_MS,
        "min_silence_ms": MIN_SILENCE_MS,
        "per_segment": per_segment,
        "silences": silences,
    }
    save_json(out_path, result)
    pbar.n = pbar.total
    pbar.refresh()
    pbar.close()
    progress.report_stage_rate("loudness", 1.0)
    progress.tick()
    thr_min = result.get("silence_threshold_min_db")
    thr_max = result.get("silence_threshold_max_db")
    thr_range = f"{thr_min:.1f}..{thr_max:.1f}" if thr_min is not None else f"{global_silence_threshold:.1f}"
    print(f"[loudness] mean={track_mean:.1f}dB p90={track_p90:.1f}dB "
          f"speech_level={speech_level:.1f}dB silence_thr={thr_range}dB "
          f"({len(floor_markers)} gap markers, +{SILENCE_HEADROOM_DB:.0f}dB headroom)")
    print(f"[loudness] silences: {len(silences_audio)} audio (unfiltered) + "
          f"{len(speech_gaps)} speech-gaps "
          f"-> {len(speech_gaps_filtered)} speech-gaps after content-protect "
          f"(dropped {drop_stats['dropped_sentence_break']} sentence-break, "
          f"{drop_stats['dropped_loud_peak']} loud-peak)")
    print(f"[loudness] final silences (union): {len(silences)}")
    return result


MIN_SILENCE_MS = 250        # natural inter-word pause; shorter is breath/clipping
MIN_GAP_SAMPLE_S = 0.5      # only inter-segment gaps ≥ this contribute a marker
SILENCE_HEADROOM_DB = 10.0  # how much louder than the local floor still counts as silence
MIN_SILENCE_THR_DB = -55.0  # absolute lower bound: anything quieter than this is silent
                            # regardless of local floor (otherwise digital-silence pre-roll
                            # drops the local threshold to -80 and we MISS actual silences
                            # at -50 in nearby windows)
SPEECH_GAP_MIN_S = 2.0      # transcript inter-segment gaps ≥ this become silences
                            # regardless of audio dB. Catches "walking around with bg
                            # music" — VAD says no speech, audio is loud (music), so
                            # the dB threshold won't trigger but we still want to trim
                            # the no-content time. 2s is slightly past natural
                            # reaction-pause length (1-1.5s) while being aggressive
                            # enough to catch most no-speech filler.
SENTENCE_BREAK_MIN_S = 0.4  # gaps after a `.`/`!`/`?` lasting ≥ this become content-
                            # protect zones (silences overlapping them are NOT trimmed).
                            # Catches between-sentence reactions like screams that
                            # whisper transcribes as "Oh god." [scream] "Holy shit."
SENTENCE_BREAK_MAX_S = 10.0 # but ignore very long post-period gaps. A 60s+ gap
                            # after a sentence is almost certainly real dead air
                            # (walking around, AFK moment), not a reaction.
                            # Reactions/screams are short (1-5s typically).
LOUD_PEAK_OFFSET_DB = 6.0   # if a silence candidate contains audio louder than
                            # speech_level + this, treat it as content (sound effect,
                            # scream, hype) and don't trim. Catches mid-sentence
                            # reactions where there's no preceding period.


def _compute_speech_gaps(segments: list, duration: float, min_gap_s: float) -> list:
    """Return [(start, end)] intervals where no speech is happening, longer
    than min_gap_s. Includes pre-roll and post-roll.

    Uses word-level timestamps when available (faster-whisper
    word_timestamps=True). Word-level gaps catch inter-word pauses inside
    long segments — a 90s segment of monologue might contain 30+ seconds
    of micro-pauses that segment-level boundaries miss entirely.

    Falls back to segment-level if no per-word data (older cache files,
    or transcribers that don't emit word timing)."""
    gaps: list = []
    if duration <= 0:
        return gaps

    def add(a: float, b: float) -> None:
        if b - a >= min_gap_s and b > a:
            gaps.append((round(a, 3), round(b, 3)))

    if not segments:
        add(0.0, duration)
        return gaps

    # Collect speech spans. Prefer word-level; concat all words from all
    # segments into a single sorted list so gaps between adjacent words
    # (within or across segments) are seen.
    has_words = any(s.get("words") for s in segments)
    if has_words:
        speech_spans = []
        for s in segments:
            for w in s.get("words", []):
                ws, we = float(w["start"]), float(w["end"])
                if we > ws:
                    speech_spans.append((ws, we))
        if not speech_spans:
            # Segments had a `words` field but it was empty everywhere; fall
            # back to segment-level.
            speech_spans = [(float(s["start"]), float(s["end"])) for s in segments]
    else:
        speech_spans = [(float(s["start"]), float(s["end"])) for s in segments]
    speech_spans.sort()
    # Merge overlapping spans into a clean speech-active timeline.
    merged = [list(speech_spans[0])]
    for s, e in speech_spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # Now invert: gaps are the spaces between consecutive speech spans,
    # plus pre-roll and post-roll.
    add(0.0, merged[0][0])
    for i in range(1, len(merged)):
        add(merged[i - 1][1], merged[i][0])
    add(merged[-1][1], duration)
    return gaps


_SENTENCE_END_CHARS = set(".!?")


def _compute_sentence_break_gaps(
    segments: list, min_gap_s: float, max_gap_s: float = 1e9,
) -> list:
    """Find gaps right after a sentence-ending word (`.`/`!`/`?`) longer than
    min_gap_s and shorter than max_gap_s. These spots commonly contain
    non-speech audio that we want to PROTECT from silence trim (screams,
    reactions, sound effects).

    The max_gap_s ceiling rejects long post-period gaps (e.g. 60s+) that
    are actually dead air rather than reactions — reactions are short.

    Requires word-level transcript data. Words come with leading space and
    trailing punctuation attached (e.g. ' god.'). We check the last char
    after stripping whitespace."""
    gaps: list = []

    def end_punct(word: str) -> bool:
        w = word.rstrip()
        return bool(w) and w[-1] in _SENTENCE_END_CHARS

    def add(a: float, b: float) -> None:
        d = b - a
        if min_gap_s <= d <= max_gap_s and b > a:
            gaps.append((round(a, 3), round(b, 3)))

    # Intra-segment: gap after a sentence-ending word to the next word in
    # the same segment.
    for seg in segments:
        words = seg.get("words") or []
        for i in range(len(words) - 1):
            if end_punct(words[i]["word"]):
                add(float(words[i]["end"]), float(words[i + 1]["start"]))
    # Cross-segment: last word of segment N ends a sentence and the next
    # segment has a first word.
    for i in range(len(segments) - 1):
        prev_words = segments[i].get("words") or []
        next_words = segments[i + 1].get("words") or []
        if not prev_words or not next_words:
            continue
        if end_punct(prev_words[-1]["word"]):
            add(float(prev_words[-1]["end"]), float(next_words[0]["start"]))
    gaps.sort()
    return gaps


def _has_loud_audio(
    db: np.ndarray, win_secs: float, start_s: float, end_s: float,
    threshold_db: float,
) -> bool:
    """Return True if any window of db inside [start_s, end_s] exceeds
    threshold_db. Used to detect non-speech sounds (screams, reactions,
    sound effects) inside silence candidates."""
    if end_s <= start_s or len(db) == 0:
        return False
    i0 = max(0, int(start_s / win_secs))
    i1 = min(len(db), int(end_s / win_secs) + 1)
    if i1 <= i0:
        return False
    return bool(np.any(db[i0:i1] > threshold_db))


def _filter_silences_for_content(
    silences: list,
    db: np.ndarray,
    win_secs: float,
    sentence_break_gaps: list,
    loud_peak_threshold_db: float,
) -> tuple[list, dict]:
    """Drop silences that either overlap a sentence-break gap or contain a
    loud audio peak. Returns (kept_silences, stats_dict)."""
    if not silences:
        return [], {"dropped_sentence_break": 0, "dropped_loud_peak": 0}
    # Sentence gaps sorted; use a walking pointer for O(N+M) overlap check.
    sg_idx = 0
    sg_count = len(sentence_break_gaps)
    kept: list = []
    dropped_sg = 0
    dropped_loud = 0
    for s, e in silences:
        # Advance sentence-gap pointer past any gap fully ending before s.
        while sg_idx < sg_count and sentence_break_gaps[sg_idx][1] <= s:
            sg_idx += 1
        # Check overlap with the current and a few following sentence gaps
        # (silences can be longer than a single gap so peek a bit).
        overlap_sg = False
        peek = sg_idx
        while peek < sg_count and sentence_break_gaps[peek][0] < e:
            gs, ge = sentence_break_gaps[peek]
            if gs < e and ge > s:
                overlap_sg = True
                break
            peek += 1
        if overlap_sg:
            dropped_sg += 1
            continue
        if _has_loud_audio(db, win_secs, s, e, loud_peak_threshold_db):
            dropped_loud += 1
            continue
        kept.append([round(s, 3), round(e, 3)])
    return kept, {"dropped_sentence_break": dropped_sg, "dropped_loud_peak": dropped_loud}


def _union_intervals(intervals: list) -> list:
    """Merge overlapping/adjacent intervals into a sorted non-overlapping list."""
    if not intervals:
        return []
    sorted_iv = sorted([(float(s), float(e)) for s, e in intervals])
    out = [list(sorted_iv[0])]
    for s, e in sorted_iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [[round(s, 3), round(e, 3)] for s, e in out]


def _compute_floor_markers(
    db: np.ndarray,
    win_secs: float,
    segments: list,
    duration: float,
    min_gap_s: float,
) -> list:
    """Build (time_midpoint, floor_db) markers from inter-segment gaps.

    Each gap (pre-roll, between two segments, post-roll) of length ≥ min_gap_s
    contributes one marker. floor_db = MEDIAN dB of windows in the gap, which
    is robust to spurious peaks (a click, a brief sound) without dropping the
    floor to the absolute minimum.

    A gap is genuine silence by definition (the VAD said no speech), so the
    median dB there is what 'silent' actually sounds like at that point in
    the video — exactly what we want to threshold against locally."""
    markers: list = []
    n_windows = len(db)
    if n_windows == 0:
        return markers

    def add_marker(gap_start: float, gap_end: float) -> None:
        if gap_end - gap_start < min_gap_s:
            return
        i0 = max(0, int(gap_start / win_secs))
        i1 = min(n_windows, int(gap_end / win_secs) + 1)
        if i1 <= i0:
            return
        floor = float(np.median(db[i0:i1]))
        markers.append(((gap_start + gap_end) / 2.0, floor))

    # Pre-roll.
    first_start = segments[0]["start"] if segments else duration
    add_marker(0.0, first_start)
    # Inter-segment gaps.
    for i in range(1, len(segments)):
        add_marker(segments[i - 1]["end"], segments[i]["start"])
    # Post-roll.
    if segments:
        add_marker(segments[-1]["end"], duration)
    return markers


def _per_window_thresholds_from_markers(
    n_windows: int,
    win_secs: float,
    markers: list,
    headroom_db: float,
    cap_db: float,
    fallback_db: float,
) -> np.ndarray:
    """Stamp each audio window with its silence threshold.
    Threshold for window at time t = nearest_marker_floor + headroom_db,
    capped at cap_db so real speech can't false-positive. If no markers
    exist (extremely sparse transcript), fall back to fallback_db.
    Implementation is vectorised: searchsorted over marker times, then a
    nearest-neighbour pick between idx and idx-1."""
    if n_windows <= 0:
        return np.empty(0, dtype=np.float32)
    if not markers:
        return np.full(n_windows, fallback_db, dtype=np.float32)
    marker_times = np.array([m[0] for m in markers], dtype=np.float32)
    marker_floors = np.array([m[1] for m in markers], dtype=np.float32)
    win_times = (np.arange(n_windows, dtype=np.float32) + 0.5) * win_secs

    idx = np.searchsorted(marker_times, win_times)
    idx_right = np.clip(idx, 0, len(marker_times) - 1)
    idx_left = np.clip(idx - 1, 0, len(marker_times) - 1)
    dist_right = np.abs(win_times - marker_times[idx_right])
    dist_left = np.abs(win_times - marker_times[idx_left])
    use_left = dist_left < dist_right
    nearest = np.where(use_left, idx_left, idx_right)
    floors = marker_floors[nearest]

    thresholds = floors + np.float32(headroom_db)
    np.maximum(thresholds, np.float32(MIN_SILENCE_THR_DB), out=thresholds)
    np.minimum(thresholds, np.float32(cap_db), out=thresholds)
    return thresholds


def _detect_silences_adaptive(
    db: np.ndarray,
    thresholds: np.ndarray,
    win_secs: float,
    min_ms: int,
) -> list:
    """Per-window threshold version: each window compared against its own
    threshold (from _per_window_thresholds_from_markers). Same output shape
    as the legacy global-threshold detector."""
    min_windows = max(1, int(round(min_ms / 1000.0 / win_secs)))
    is_silent = db < thresholds
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
