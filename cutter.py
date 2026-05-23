"""ffmpeg-based trimmer with sample-precise cuts.

Three strategies, auto-selected:
- few large segments  -> stream-copy via concat demuxer (no re-encode, keyframe
  rounded — fine for ~2-10s precision).
- many h264 segments  -> smartcut (PyAV) with h264_nvenc patched in for the
  boundary GOP re-encodes. Stream-copies all non-boundary GOPs, re-encodes only
  the GOPs touching a cut. ~2x faster than the full-reencode path on the same
  GPU and the audio is passed through (no re-encode, no quality loss).
- everything else     -> full filter_complex re-encode via h264_nvenc. Falls
  through here for non-h264 sources or if smartcut+nvenc bails for any reason.

Stream-copy is unsuitable for many small cuts because every cut rounds outward
to the nearest keyframe, accumulating multi-minute duration drift across 100+
tiny segments.
"""
import os
import re
import shutil
import subprocess
import threading
from typing import List, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

from . import progress


# Re-encode triggers: many segments OR short segments mean stream-copy drift
# will be visible. Past these thresholds, switch to a re-encode strategy.
REENCODE_SEG_COUNT = 30
REENCODE_MIN_SEG_S = 3.0


def cut_video(
    source_path: str,
    keep_ranges: List[Tuple[float, float]],
    output_path: str,
    work_dir: str,
) -> None:
    """Write a new mp4 containing only the keep_ranges from source_path.

    Atomicity: writes to <output_path>.partial first, only renames to the
    final name on success. Prevents a corrupt half-mp4 if the process is
    killed mid-encode. Also drops an encode_checkpoint.json before starting
    so an interrupted run is detectable on next launch."""
    if not keep_ranges:
        raise ValueError("No keep_ranges given — would produce empty video")

    import json as _json
    import time as _time
    checkpoint_path = os.path.join(work_dir, "encode_checkpoint.json")
    # Preserve the .mp4 extension on the partial file so ffmpeg can infer the
    # container format. `final.mp4.partial` makes ffmpeg fail with "Unable to
    # choose an output format". `final.partial.mp4` keeps mp4 inference and
    # still sorts/greps obviously as a partial.
    _base, _ext = os.path.splitext(output_path)
    partial_path = f"{_base}.partial{_ext}"

    # If a previous run was interrupted, log it and clear the leftover.
    if os.path.exists(checkpoint_path):
        try:
            ck = _json.load(open(checkpoint_path, encoding="utf-8"))
            print(f"[cutter] previous encode interrupted at "
                  f"{ck.get('started_at_iso', '?')} — clearing partial state")
        except Exception:
            pass
        if os.path.exists(partial_path):
            try:
                os.remove(partial_path)
            except OSError:
                pass
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass

    # Drop checkpoint so we can detect interrupt next time.
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        _json.dump({
            "started_at": _time.time(),
            "started_at_iso": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_keep_ranges": len(keep_ranges),
            "output_path": output_path,
            "partial_path": partial_path,
        }, f, indent=2)

    need_reencode = (
        len(keep_ranges) > REENCODE_SEG_COUNT
        or any(e - s < REENCODE_MIN_SEG_S for s, e in keep_ranges)
    )
    try:
        if not need_reencode:
            _cut_streamcopy(source_path, keep_ranges, partial_path, work_dir)
        else:
            # Prefer smartcut+nvenc when the source codec is h264 and nvenc is
            # available — partial re-encode that benchmarked ~2x faster than the
            # full-reencode path. Fall through to _cut_reencode on any failure or
            # for non-h264 sources.
            src_codec = _get_source_video_codec(source_path)
            done = False
            if src_codec == "h264" and _has_h264_nvenc():
                try:
                    _cut_smartcut_nvenc(source_path, keep_ranges, partial_path, work_dir)
                    done = True
                except Exception as e:
                    print(f"[cutter] smartcut+nvenc failed ({type(e).__name__}: {e}); "
                          f"falling back to full re-encode")
            if not done:
                _cut_reencode(source_path, keep_ranges, partial_path, work_dir)
        # Atomic rename .partial → final, then drop checkpoint.
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(partial_path, output_path)
    finally:
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except OSError:
                pass


def _cut_streamcopy(
    source_path: str,
    keep_ranges: List[Tuple[float, float]],
    output_path: str,
    work_dir: str,
) -> None:
    segments_dir = os.path.join(work_dir, "cutter_segments")
    os.makedirs(segments_dir, exist_ok=True)
    for f in os.listdir(segments_dir):
        if f.startswith("seg_") and f.endswith(".mp4"):
            os.remove(os.path.join(segments_dir, f))

    seg_paths = []
    for i, (s, e) in enumerate(keep_ranges):
        seg_path = os.path.join(segments_dir, f"seg_{i:04d}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{s:.3f}",
            "-to", f"{e:.3f}",
            "-i", source_path,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            seg_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not os.path.exists(seg_path) or os.path.getsize(seg_path) == 0:
            raise RuntimeError(f"ffmpeg produced empty segment for {s:.1f}-{e:.1f}")
        seg_paths.append(seg_path)

    list_path = os.path.join(work_dir, "cutter_concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in seg_paths:
            f.write(f"file '{p.replace(os.sep, '/')}'\n")

    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path,
    ]
    subprocess.run(concat_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"ffmpeg concat failed to produce {output_path}")

    total_keep = sum(e - s for s, e in keep_ranges)
    print(f"[cutter] stream-copy: wrote {output_path} ({len(seg_paths)} segments, {total_keep:.1f}s kept)")


def _get_source_video_codec(source_path: str) -> str | None:
    """Probe the source's video stream codec name (e.g. 'h264', 'av1', 'vp9').
    Returns None on probe failure."""
    try:
        r = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1", source_path,
        ], capture_output=True, text=True, timeout=10)
        out = (r.stdout or "").strip()
        return out or None
    except Exception:
        return None


_nvenc_cache: bool | None = None

def _has_h264_nvenc() -> bool:
    """Cache-check ffmpeg's encoder list for h264_nvenc availability."""
    global _nvenc_cache
    if _nvenc_cache is None:
        try:
            out = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            _nvenc_cache = "h264_nvenc" in out
        except Exception:
            _nvenc_cache = False
    return _nvenc_cache


_smartcut_nvenc_patched = False

def _patch_smartcut_for_nvenc() -> None:
    """Monkey-patch smartcut's VideoCutter so its boundary-GOP re-encodes use
    h264_nvenc instead of the source's default encoder (libx264).

    Smartcut's SMARTCUT mode normally creates the encoder from the source
    codec name (-> libx264 for h264 source). codec_override only takes effect
    in RECODE mode, which re-encodes every frame and defeats the point. The
    wrap is in two places:
      1. __init__: after the original sets self.codec_name from the source,
         flip it to 'h264_nvenc' when source is h264.
      2. init_encoder: after the original sets self.encoding_options (with
         libx264-specific keys like crf and x264-params), replace it with
         nvenc-appropriate options. The encoder context is built lazily from
         self.codec_name + self.encoding_options inside _ensure_enc_codec.
    Idempotent — safe to call multiple times.
    """
    global _smartcut_nvenc_patched
    if _smartcut_nvenc_patched:
        return
    import smartcut.cut_video as scv

    orig_init = scv.VideoCutter.__init__
    orig_init_encoder = scv.VideoCutter.init_encoder

    def patched_init(self, mc, oac, vs, log_level):
        orig_init(self, mc, oac, vs, log_level)
        if vs.mode == scv.VideoExportMode.SMARTCUT and self.codec_name == "h264":
            self.codec_name = "h264_nvenc"

    def patched_init_encoder(self):
        orig_init_encoder(self)
        if self.codec_name == "h264_nvenc":
            # Match the full-reencode path's settings (cutter.py:_pick_video_encoder).
            # p1 = fastest NVENC preset; quality difference vs p4 is invisible on
            # already-lossy YouTube source.
            self.encoding_options = {
                "preset": "p1",
                "rc": "vbr",
                "cq": "23",
                "b:v": "0",
            }

    scv.VideoCutter.__init__ = patched_init
    scv.VideoCutter.init_encoder = patched_init_encoder
    _smartcut_nvenc_patched = True


class _SmartcutProgress:
    """Adapter from smartcut's emit(int) protocol to our tqdm bar + overall
    pipeline progress tracker. First emit = total; subsequent emits = current."""
    def __init__(self, label: str):
        self.label = label
        self.total: int | None = None
        self.pbar: tqdm | None = None

    def emit(self, value: int) -> None:
        if self.total is None:
            self.total = max(1, int(value))
            self.pbar = tqdm(
                total=self.total, unit="seg",
                desc=f"[stage   ] {self.label}",
                bar_format="{desc} {bar} {percentage:3.0f}% | {n}/{total} segs | eta {remaining}",
                position=1, leave=False,
            )
            return
        cur = int(value)
        if self.pbar is not None:
            if cur > self.pbar.n:
                self.pbar.update(cur - self.pbar.n)
            progress.report_stage_rate("encode", cur / max(self.total, 1))
            progress.tick()

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.n = self.pbar.total
            self.pbar.refresh()
            self.pbar.close()


def _cut_smartcut_nvenc(
    source_path: str,
    keep_ranges: List[Tuple[float, float]],
    output_path: str,
    work_dir: str,
) -> None:
    """Partial re-encode via the smartcut library, with h264_nvenc patched in
    for the boundary GOPs. Stream-copies all non-boundary GOPs; only decodes
    + re-encodes the GOPs touching a cut. Audio is passed through unchanged
    (lossless), so the python audio splice path is bypassed.

    Benchmarked ~2x faster than _cut_reencode on a 45min h264 720p60 source
    with 221 cuts (49s vs 102s).
    """
    from fractions import Fraction
    from smartcut.cut_video import (
        smart_cut, VideoSettings, VideoExportMode, VideoExportQuality,
        AudioExportInfo, AudioExportSettings,
    )
    from smartcut.media_container import MediaContainer

    _patch_smartcut_for_nvenc()

    source = MediaContainer(source_path)
    # smartcut uses Fraction internally; limit_denominator avoids
    # float-to-Fraction precision blowup on long timestamps.
    segments = [(Fraction(s).limit_denominator(1_000_000),
                 Fraction(e).limit_denominator(1_000_000))
                for s, e in keep_ranges]

    audio_settings = [AudioExportSettings(codec="passthru")] * len(source.audio_tracks)
    export_info = AudioExportInfo(output_tracks=audio_settings)
    video_settings = VideoSettings(VideoExportMode.SMARTCUT, VideoExportQuality.NORMAL, "copy")

    total_keep = sum(e - s for s, e in keep_ranges)
    print(f"[cutter] smartcut+h264_nvenc: {len(keep_ranges)} segments ({total_keep:.1f}s)...")

    cb = _SmartcutProgress("[cutter] smartcut+nvenc")
    try:
        exc = smart_cut(
            source, segments, output_path,
            audio_export_info=export_info,
            video_settings=video_settings,
            progress=cb, log_level=None,
        )
    finally:
        cb.close()

    if exc is not None:
        raise exc
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"smartcut produced no output at {output_path}")
    print(f"[cutter] wrote {output_path}")


def _get_source_frame_rate(source_path: str) -> float | None:
    """Probe the source's video stream r_frame_rate. Returns fps as a float
    (e.g. 60.0, 29.97), or None on failure."""
    try:
        r = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1", source_path,
        ], capture_output=True, text=True, timeout=10)
        out = (r.stdout or "").strip()
        if "/" in out:
            num, den = out.split("/")
            return int(num) / int(den)
        return float(out)
    except Exception:
        return None


def _snap_ranges_to_frames(
    ranges: List[Tuple[float, float]], fr: float,
) -> List[Tuple[float, float]]:
    """Snap each range's start AND end to the nearest video-frame boundary.
    This is the fix for the audio/video drift bug:

    Audio Python-splice gives EXACTLY (end - start) seconds per range
    (sample-accurate). Video ffmpeg-select with setpts=N/FRAME_RATE/TB gives
    N/FR seconds where N is the number of frames whose source PTS falls in
    [s, e). Per-range duration mismatch is ~1/FR (16.67ms at 60fps). Over
    600+ ranges this accumulates to multi-second drift.

    Snapping both s and e to k/FR boundaries means audio and video share
    the SAME range timings — both produce (L-K)/FR seconds where L=round(e*fr)
    and K=round(s*fr). Zero accumulated drift.
    """
    out = []
    for s, e in ranges:
        s_snap = round(s * fr) / fr
        e_snap = round(e * fr) / fr
        if e_snap > s_snap:
            out.append((s_snap, e_snap))
    return out


def _build_video_filter_script(keep_ranges: List[Tuple[float, float]]) -> str:
    """Video-only filter: single select with per-range gte*lt predicates OR'd
    via +. Audio is pre-spliced in Python (see _splice_audio) and muxed as a
    second input — way faster than 600+ atrim+concat filters which bottleneck
    ffmpeg's filter scheduler.

    Why gte*lt instead of between(): ffmpeg's between(x, min, max) is
    INCLUSIVE on both ends. With ranges snapped to frame boundaries [K/fr, L/fr],
    between() would keep frames at K, K+1, ..., L-1, AND L — that's L-K+1
    frames, one more than expected. Python audio splice uses [start, stop) =
    L-K samples worth of audio. Result: video is 1 frame longer than audio
    per range, accumulating to multi-second drift across hundreds of ranges.

    gte(t,s)*lt(t,e) is inclusive-start, exclusive-end — matches Python's
    [start, stop) and gives the same frame count as audio sample count.
    Use 6 decimal places to avoid float-comparison fuzziness on frame
    boundaries that need to be exact (snap_ranges_to_frames produces these).
    """
    parts = [f"gte(t,{s:.6f})*lt(t,{e:.6f})" for s, e in keep_ranges]
    video_expr = "+".join(parts)
    return f"[0:v]select='{video_expr}',setpts=N/FRAME_RATE/TB[outv]"


def _splice_audio(
    source_path: str,
    keep_ranges: List[Tuple[float, float]],
    work_dir: str,
) -> str:
    """Pre-splice the source audio into a single wav containing only the
    keep_ranges, concatenated in order. Returns the path to the spliced wav.

    Strategy:
      1. Decode source audio to a temp wav once (ffmpeg, one pass).
      2. Read each keep_range as a slice from that wav (soundfile seeks via
         start/stop sample indices — no full-file load).
      3. Write the slices in order to the output wav (incremental write,
         constant memory).

    This avoids ffmpeg's filter graph entirely for audio. On a 600+ segment
    run, the 600+ atrim+concat filter chain becomes the bottleneck (single-
    threaded filter scheduling, frame-by-frame buffer shuffling). This
    approach does the equivalent work in raw numpy seek/copy operations —
    typically 10-50x faster.
    """
    sr = 48000
    channels = 2
    sample_bytes = channels * 2  # int16 stereo = 4 bytes per sample-frame
    out_wav = os.path.join(work_dir, "cutter_audio_spliced.wav")

    # STREAMING APPROACH: pipe ffmpeg's raw PCM output straight into Python.
    # No multi-GB intermediate file on disk. ffmpeg decodes opus → stdout
    # → Python skips/copies bytes per keep range → writes spliced wav.
    #
    # Why this replaces the previous "extract to disk, then seek+read":
    #   1. Eliminates the 2GB+ intermediate (saves disk + I/O).
    #   2. Sidesteps every WAV/libsndfile size-limit gotcha encountered so
    #      far (2GB truncation in ffmpeg's WAV writer, libsndfile's
    #      psf_fseek bug past ~1.4GB on Windows).
    #   3. ffmpeg's stdout pipe is sequential, which matches our ranges:
    #      we sort ranges, read+discard between them, read+keep within them.
    try:
        r = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", source_path,
        ], capture_output=True, text=True, timeout=10)
        src_dur = float((r.stdout or "0").strip())
    except Exception:
        src_dur = 0.0

    sorted_ranges = sorted([(max(0.0, s), max(s, e)) for s, e in keep_ranges])
    total_out_s = sum(e - s for s, e in sorted_ranges)
    print(f"[cutter] streaming + splicing audio: {len(sorted_ranges)} segments, "
          f"{total_out_s:.1f}s output (no intermediate file)...")

    cmd = [
        "ffmpeg", "-y", "-i", source_path,
        "-map", "0:a:0",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-ac", str(channels),
        "-threads", "0",
        "-",  # pipe to stdout
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=8 * 1024 * 1024,  # 8MB pipe buffer
    )
    # Drain stderr in a thread so ffmpeg doesn't block on full stderr buffer.
    stderr_chunks = []
    def _drain_err():
        for line in proc.stderr:
            stderr_chunks.append(line)
            if sum(len(s) for s in stderr_chunks) > 8000:
                stderr_chunks.pop(0)
    drain = threading.Thread(target=_drain_err, daemon=True)
    drain.start()

    pbar = tqdm(
        total=len(sorted_ranges), unit="seg", desc="[stage   ] audio extract+splice",
        bar_format="{desc} {bar} {percentage:3.0f}% | {n}/{total} segs | eta {remaining}",
        position=1, leave=False,
    )

    READ_CHUNK = 4 * 1024 * 1024  # 4MB reads from pipe
    samples_consumed = 0  # how many SAMPLES we've pulled from ffmpeg stdout

    def _read_exact(n: int) -> bytes:
        """Read exactly n bytes from ffmpeg stdout (or until EOF). Bytes-only."""
        buf = bytearray()
        while len(buf) < n:
            chunk = proc.stdout.read(min(n - len(buf), READ_CHUNK))
            if not chunk:
                break
            buf.extend(chunk)
        return bytes(buf)

    try:
        with sf.SoundFile(out_wav, mode="w", samplerate=sr,
                          channels=channels, subtype="PCM_16") as f_out:
            for s, e in sorted_ranges:
                start_sample = int(round(s * sr))
                end_sample = int(round(e * sr))
                if end_sample <= start_sample:
                    pbar.update(1)
                    continue
                # Skip from current position up to start_sample (discard bytes).
                if start_sample > samples_consumed:
                    skip_bytes = (start_sample - samples_consumed) * sample_bytes
                    while skip_bytes > 0:
                        chunk = proc.stdout.read(min(skip_bytes, READ_CHUNK))
                        if not chunk:
                            break
                        skip_bytes -= len(chunk)
                    samples_consumed = start_sample
                # Read the range's worth of audio (keep).
                n_samples = end_sample - start_sample
                buf = _read_exact(n_samples * sample_bytes)
                if buf:
                    arr = np.frombuffer(buf, dtype=np.int16).reshape(-1, channels)
                    f_out.write(arr)
                samples_consumed = end_sample
                pbar.update(1)
                progress.tick()
            # Drain any remaining ffmpeg output so it can exit cleanly.
            while True:
                chunk = proc.stdout.read(READ_CHUNK)
                if not chunk:
                    break
    finally:
        pbar.close()
        proc.stdout.close()
        proc.wait(timeout=10)
        drain.join(timeout=2)

    if proc.returncode != 0:
        tail = b"".join(stderr_chunks).decode("utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"ffmpeg audio decode failed (rc={proc.returncode}):\n{tail}")
    return out_wav


def _pick_video_encoder() -> tuple[list[str], str]:
    """Prefer h264_nvenc preset p1 (fastest NVENC mode), fallback to libx264.
    p1 is meaningfully faster than p4 on this GPU for marginal quality loss —
    the source is already lossy YouTube re-encode so the difference is
    invisible."""
    enc_list = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True,
    ).stdout
    if "h264_nvenc" in enc_list:
        return (
            ["-c:v", "h264_nvenc", "-preset", "p1", "-rc", "vbr", "-cq", "23", "-b:v", "0"],
            "h264_nvenc",
        )
    return (["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"], "libx264")


def _run_ffmpeg_with_progress(cmd: list, total_keep_s: float, label: str) -> tuple[int, str]:
    """Run ffmpeg with -progress pipe:1 and drive a tqdm bar from out_time_us.
    Returns (returncode, last_2KB_of_stderr) for error reporting."""
    full_cmd = list(cmd) + ["-progress", "pipe:1", "-nostats", "-loglevel", "error"]
    proc = subprocess.Popen(
        full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=1, universal_newlines=True,
    )
    stderr_buf = []
    pbar = tqdm(
        total=int(total_keep_s), unit="s", desc=f"[stage   ] {label}",
        bar_format="{desc} {bar} {percentage:3.0f}% | {n}/{total}s | eta {remaining}",
        position=1, leave=False,
    )
    # Stderr drain thread so it doesn't deadlock if ffmpeg's stderr buffer fills.
    def _drain():
        for line in proc.stderr:
            stderr_buf.append(line)
            if sum(len(s) for s in stderr_buf) > 4000:
                stderr_buf.pop(0)
    drain = threading.Thread(target=_drain, daemon=True)
    drain.start()
    out_time_re = re.compile(r"^out_time_us=(\d+)")
    last = 0
    for line in proc.stdout:
        m = out_time_re.match(line.strip())
        if m:
            cur_s = min(int(m.group(1)) // 1_000_000, int(total_keep_s))
            if cur_s > last:
                pbar.update(cur_s - last)
                last = cur_s
                # Report observed encode rate so the overall bar uses our
                # actual pace (not its baseline guess) for the encode stage.
                progress.report_stage_rate("encode", cur_s / max(total_keep_s, 1))
                progress.tick()
    pbar.n = pbar.total
    pbar.refresh()
    pbar.close()
    proc.wait()
    drain.join(timeout=2)
    return proc.returncode, "".join(stderr_buf)


def _cut_reencode(
    source_path: str,
    keep_ranges: List[Tuple[float, float]],
    output_path: str,
    work_dir: str,
) -> None:
    """Single-pass re-encode. VIDEO via select filter (one filter, fast).
    AUDIO is pre-spliced in Python and muxed as a second input — the
    previous approach with 600+ atrim+concat filters made ffmpeg's filter
    scheduler the bottleneck and gave 4hr ETAs on a 3hr vod.

    Stream-copy isn't usable here because keyframe-snap drift compounds
    across hundreds of micro-cuts."""
    # FIX A/V DRIFT: snap each range's boundaries to the source's frame
    # timeline before splicing. Without this, audio is sample-accurate but
    # video rounds to whole frames → per-range mismatch of ~1/fps seconds
    # accumulates across hundreds of cuts into multi-second drift. With
    # snapping, audio and video share the SAME boundaries → zero drift.
    fr = _get_source_frame_rate(source_path)
    if fr and fr > 0:
        keep_ranges = _snap_ranges_to_frames(keep_ranges, fr)
        print(f"[cutter] snapped {len(keep_ranges)} ranges to {fr:.3f}fps frame boundaries (A/V sync)")

    # 1. Pre-splice audio in Python — using the SNAPPED ranges so audio
    # boundaries match video boundaries exactly.
    spliced_audio = _splice_audio(source_path, keep_ranges, work_dir)

    # 2. Build the (now small) video-only filter script — also using snapped.
    filter_script = _build_video_filter_script(keep_ranges)
    script_path = os.path.join(work_dir, "cutter_filter.txt")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(filter_script)

    enc_args, enc_name = _pick_video_encoder()
    # Benchmarked alternatives (cutter_bench.py): SELECT filter without
    # hwaccel is fastest at ~4x realtime. `-hwaccel cuda` actually breaks
    # this filter+dual-input setup (errors out). `-filter_threads 4` lets
    # the select filter parallelize across CPU cores. Per-range ffmpeg
    # invocations and concat-inpoint demuxer are 2-40x slower due to
    # per-range setup overhead.
    cmd = [
        "ffmpeg", "-y",
        "-i", source_path,
        "-i", spliced_audio,
        "-filter_complex_script", script_path,
        "-filter_threads", "4",
        "-map", "[outv]", "-map", "1:a",
        *enc_args,
        "-c:a", "aac", "-b:a", "192k",
        # +faststart moves the moov atom to the start of the file so video
        # players don't need to seek to the end to read metadata. Doesn't
        # rescue mid-encode-killed files (moov still gets written at finish)
        # but helps streaming and tool compatibility.
        "-movflags", "+faststart",
        output_path,
    ]
    total_keep = sum(e - s for s, e in keep_ranges)
    print(f"[cutter] re-encoding {len(keep_ranges)} segments ({total_keep:.1f}s) via {enc_name}...")
    rc, stderr_tail = _run_ffmpeg_with_progress(cmd, total_keep, f"[cutter] {enc_name}")
    if rc != 0:
        # Fallback to libx264 if nvenc choked.
        if enc_name == "h264_nvenc":
            print(f"[cutter] {enc_name} failed, falling back to libx264")
            print(stderr_tail[-500:])
            cmd_fallback = [
                "ffmpeg", "-y",
                "-i", source_path,
                "-i", spliced_audio,
                "-filter_complex_script", script_path,
                "-filter_threads", "4",
                "-map", "[outv]", "-map", "1:a",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k",
                output_path,
            ]
            rc, stderr_tail = _run_ffmpeg_with_progress(cmd_fallback, total_keep, "[cutter] libx264")
        if rc != 0:
            raise RuntimeError(f"ffmpeg re-encode failed:\n{stderr_tail[-1500:]}")
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"ffmpeg re-encode produced no output at {output_path}")
    print(f"[cutter] wrote {output_path}")
