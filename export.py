"""Export the final cut + keep lists in formats friendly to external NLEs
(DaVinci, Premiere, etc) or programmatic post-processing.

Writes `cuts.json` to the cache directory next to `final.mp4`. Schema is
intentionally simple — list of cut ranges and list of keep ranges, both
in source-video seconds. NLEs that import a JSON edit list can map these
directly onto the source clip.
"""
import json
import os
from typing import Iterable


def write_cuts_json(
    out_dir: str,
    duration_s: float,
    cuts: Iterable,
    keeps: Iterable,
    highlights: Iterable | None = None,
    metadata: dict | None = None,
    filename: str = "cuts.json",
) -> str:
    """Write cuts/keeps to JSON. Returns the path."""
    payload = {
        "schema_version": 1,
        "source_duration_s": round(float(duration_s), 3),
        "cuts": [
            {"start_s": round(float(s), 3), "end_s": round(float(e), 3),
             "duration_s": round(float(e - s), 3)}
            for s, e in cuts
        ],
        "keeps": [
            {"start_s": round(float(s), 3), "end_s": round(float(e), 3),
             "duration_s": round(float(e - s), 3)}
            for s, e in keeps
        ],
        "highlights_s": [round(float(h), 3) for h in (highlights or [])],
    }
    if metadata:
        payload["metadata"] = metadata
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
