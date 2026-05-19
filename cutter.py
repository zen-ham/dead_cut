"""ffmpeg-based trimmer. Stream-copy concat for speed + zero re-encode loss.

Stream-copy snaps cut points to keyframes (typically 2-10s apart), which is the
right tradeoff for this kind of edit: cut precision is < re-encode quality loss.
"""
import os
import subprocess
from typing import List, Tuple


def cut_video(
    source_path: str,
    keep_ranges: List[Tuple[float, float]],
    output_path: str,
    work_dir: str,
) -> None:
    """Write a new mp4 containing only the keep_ranges from source_path."""
    if not keep_ranges:
        raise ValueError("No keep_ranges given — would produce empty video")

    segments_dir = os.path.join(work_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    # Clear any stale segments from a prior run
    for f in os.listdir(segments_dir):
        if f.startswith("seg_") and f.endswith(".mp4"):
            os.remove(os.path.join(segments_dir, f))

    seg_paths = []
    for i, (s, e) in enumerate(keep_ranges):
        seg_path = os.path.join(segments_dir, f"seg_{i:04d}.mp4")
        cmd = [
            "ffmpeg", "-y",
            # -ss BEFORE -i is fast (seek by index) but rounds to keyframes.
            # -ss AFTER -i is accurate but decodes from start (slow on long vids).
            # We want speed; keyframe snap is acceptable for this use case.
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

    # Concat demuxer
    list_path = os.path.join(work_dir, "concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in seg_paths:
            # ffmpeg concat list wants forward-slashes and escaped single-quotes
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
    print(f"[cutter] wrote {output_path} ({len(seg_paths)} segments, {total_keep:.1f}s kept)")
