"""ffmpeg-based trimmer with sample-precise cuts.

Two strategies:
- few segments, large each: stream-copy via concat demuxer (fast, no re-encode loss,
  cut points snap to nearest keyframe — fine for ~2-10s precision).
- many small segments (post-trim): SINGLE filter_complex pass with trim+concat,
  re-encoded once. Stream-copy is unsuitable here because every cut rounds
  outward to the nearest keyframe, accumulating multi-minute duration drift
  across 100+ tiny segments.

Auto-selects based on segment count + minimum segment length. Prefers h264_nvenc
when the encoder is available (much faster on NVIDIA GPUs).
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
# will be visible. Past these thresholds, switch to filter_complex re-encode.
REENCODE_SEG_COUNT = 30
REENCODE_MIN_SEG_S = 3.0


def cut_video(
    source_path: str,
    keep_ranges: List[Tuple[float, float]],
    output_path: str,
    work_dir: str,
) -> None:
    """Write a new mp4 containing only the keep_ranges from source_path."""
    if not keep_ranges:
        raise ValueError("No keep_ranges given — would produce empty video")

    need_reencode = (
        len(keep_ranges) > REENCODE_SEG_COUNT
        or any(e - s < REENCODE_MIN_SEG_S for s, e in keep_ranges)
    )
    if need_reencode:
        _cut_reencode(source_path, keep_ranges, output_path, work_dir)
    else:
        _cut_streamcopy(source_path, keep_ranges, output_path, work_dir)


def _cut_streamcopy(
    source_path: str,
    keep_ranges: List[Tuple[float, float]],
    output_path: str,
    work_dir: str,
) -> None:
    segments_dir = os.path.join(work_dir, "segments")
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

    list_path = os.path.join(work_dir, "concat_list.txt")
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


def _build_video_filter_script(keep_ranges: List[Tuple[float, float]]) -> str:
    """Video-only filter: single select with between() expression OR'd via +.
    Audio is pre-spliced in Python (see _splice_audio) and muxed as a second
    input — way faster than 600+ atrim+concat filters on the audio side which
    bottleneck ffmpeg's filter scheduler and freeze the pipeline."""
    video_expr = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in keep_ranges)
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
    full_wav = os.path.join(work_dir, "audio_full.wav")
    out_wav = os.path.join(work_dir, "audio_spliced.wav")

    # 1. Decode source to a clean wav we can random-access.
    if not os.path.exists(full_wav) or os.path.getsize(full_wav) == 0:
        print(f"[cutter] extracting source audio -> wav...")
        cmd = [
            "ffmpeg", "-y", "-i", source_path,
            "-vn",
            "-c:a", "pcm_s16le",
            "-ar", str(sr),
            "-ac", str(channels),
            full_wav,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. Splice in Python via soundfile's seek-based reads.
    info = sf.info(full_wav)
    real_sr = info.samplerate
    real_ch = info.channels
    total_samples = sum(int(round((e - s) * real_sr)) for s, e in keep_ranges)
    print(f"[cutter] splicing {len(keep_ranges)} audio segments "
          f"({total_samples/real_sr:.1f}s output) in Python...")
    with sf.SoundFile(out_wav, mode="w", samplerate=real_sr,
                      channels=real_ch, subtype="PCM_16") as f_out:
        for s, e in keep_ranges:
            start_sample = max(0, int(round(s * real_sr)))
            end_sample = max(start_sample, int(round(e * real_sr)))
            if end_sample <= start_sample:
                continue
            chunk, _ = sf.read(full_wav, start=start_sample, stop=end_sample,
                               dtype="int16", always_2d=True)
            f_out.write(chunk)
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
    # 1. Pre-splice audio in Python.
    spliced_audio = _splice_audio(source_path, keep_ranges, work_dir)

    # 2. Build the (now small) video-only filter script.
    filter_script = _build_video_filter_script(keep_ranges)
    script_path = os.path.join(work_dir, "filter_complex.txt")
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
