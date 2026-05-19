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


def _build_filter_script(keep_ranges: List[Tuple[float, float]]) -> str:
    """Build a filter_complex script: select for video (single filter, fast),
    trim+concat for audio (proven correct, slightly slower but unavoidable).

    Why hybrid: a single `select` filter compresses N video ranges in O(1)
    filters, but the equivalent `aselect` on audio mishandles variable-size
    audio frames — `asetpts=N/SR/TB` underflows because N is FRAME index not
    sample index, leaving audio at original PTS and the container duration
    blown out to the source length. Audio trim+concat is reliable.

    Written to disk and passed via -filter_complex_script (avoids Windows 8KB
    command-line limit on 200+ range expressions)."""
    # Video: single select filter with between() expression OR'd via +.
    video_expr = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in keep_ranges)
    video_chain = f"[0:v]select='{video_expr}',setpts=N/FRAME_RATE/TB[outv]"
    # Audio: trim each range then concat them in order.
    audio_parts = []
    audio_labels = []
    for i, (s, e) in enumerate(keep_ranges):
        audio_parts.append(f"[0:a]atrim={s:.3f}:{e:.3f},asetpts=PTS-STARTPTS[a{i}]")
        audio_labels.append(f"[a{i}]")
    audio_concat = f"{''.join(audio_labels)}concat=n={len(keep_ranges)}:v=0:a=1[outa]"
    return ";\n".join([video_chain] + audio_parts + [audio_concat])


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
    """Single-pass re-encode with filter_complex. Required when there are many
    segments or short segments — stream-copy drifts multi-minutes across many
    keyframe-rounded cuts."""
    filter_script = _build_filter_script(keep_ranges)
    script_path = os.path.join(work_dir, "filter_complex.txt")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(filter_script)

    enc_args, enc_name = _pick_video_encoder()
    cmd = [
        "ffmpeg", "-y",
        "-i", source_path,
        "-filter_complex_script", script_path,
        "-map", "[outv]", "-map", "[outa]",
        *enc_args,
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ]
    total_keep = sum(e - s for s, e in keep_ranges)
    print(f"[cutter] re-encoding {len(keep_ranges)} segments ({total_keep:.1f}s) via {enc_name}...")
    rc, stderr_tail = _run_ffmpeg_with_progress(cmd, total_keep, f"[cutter] {enc_name}")
    if rc != 0:
        # Fallback to libx264 if nvenc choked (driver mismatch etc).
        if enc_name == "h264_nvenc":
            print(f"[cutter] {enc_name} failed, falling back to libx264")
            print(stderr_tail[-500:])
            cmd_fallback = [
                "ffmpeg", "-y",
                "-i", source_path,
                "-filter_complex_script", script_path,
                "-map", "[outv]", "-map", "[outa]",
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
