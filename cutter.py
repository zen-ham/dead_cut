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
import shutil
import subprocess
from typing import List, Tuple


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
    """Build a filter_complex script: trim each range, then concat them all.
    Returned as a single string ready to be written to disk and passed via
    -filter_complex_script (avoids Windows 8KB command-line limit)."""
    parts = []
    labels = []
    for i, (s, e) in enumerate(keep_ranges):
        parts.append(f"[0:v]trim={s:.3f}:{e:.3f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim={s:.3f}:{e:.3f},asetpts=PTS-STARTPTS[a{i}]")
        labels.append(f"[v{i}][a{i}]")
    concat = f"{''.join(labels)}concat=n={len(keep_ranges)}:v=1:a=1[outv][outa]"
    return ";\n".join(parts + [concat])


def _pick_video_encoder() -> tuple[list[str], str]:
    """Prefer h264_nvenc (NVIDIA) for speed, fallback to libx264."""
    enc_list = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True,
    ).stdout
    if "h264_nvenc" in enc_list:
        return (
            ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"],
            "h264_nvenc",
        )
    return (["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"], "libx264")


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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback to libx264 if nvenc choked (driver mismatch etc).
        if enc_name == "h264_nvenc":
            print(f"[cutter] {enc_name} failed, falling back to libx264")
            print(result.stderr[-500:])
            cmd = [
                "ffmpeg", "-y",
                "-i", source_path,
                "-filter_complex_script", script_path,
                "-map", "[outv]", "-map", "[outa]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k",
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg re-encode failed:\n{result.stderr[-1500:]}")
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"ffmpeg re-encode produced no output at {output_path}")
    print(f"[cutter] wrote {output_path}")
