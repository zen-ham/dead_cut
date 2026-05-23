"""Dynamic cut target based on video length.

The user wants longer videos cut more aggressively, on a gentle log curve:
    30 min vod  -> 15 min final  (50% final keep)
   300 min vod  -> 120 min final (40% final keep)

These two anchors fit a log10 curve:
    final_keep_ratio(d_min) = 0.6477 - 0.1 * log10(d_min)

The target is for the FINAL output (after silence trim too), so the AI's
own cut target gets back-calculated from the estimated silence-trim
removal. Silence trim runs after AI cuts but the silences are detected in
stage 3 already, so we have the data needed to estimate.
"""
import math


def final_keep_ratio(duration_s: float) -> float:
    """Logarithmic curve over duration (in minutes). Bounded so very short
    or very long videos don't end up with absurd targets."""
    d_min = max(1.0, duration_s / 60.0)
    raw = 0.6477 - 0.1 * math.log10(d_min)
    # Bounds: never keep less than 30% or more than 65% (safety rails).
    return max(0.30, min(0.65, raw))


def estimate_silence_trim_ratio(
    silences: list,
    duration_s: float,
    max_silence_s: float = 0.6,
    padding_s: float = 0.2,
) -> float:
    """Estimate what fraction of the source the silence-trim stage would
    remove if applied to the whole video. We're estimating the *whole*-video
    bound here; the actual removal scales roughly linearly with the AI's
    keep ratio (silence is approximately uniformly distributed). The model
    target uses this directly because the per-keep silence removal is
    approximately (this_ratio * ai_keep_ratio)."""
    if not silences or duration_s <= 0:
        return 0.0
    removed_s = 0.0
    for s, e in silences:
        dur = e - s
        if dur < max_silence_s:
            continue
        # Each qualifying silence trims (dur - 2*padding) seconds.
        removed_s += max(0.0, dur - 2 * padding_s)
    return removed_s / duration_s


def compute_ai_cut_target(
    duration_s: float,
    silence_trim_ratio: float,
    round_to_pct: int = 5,
) -> dict:
    """Compute the AI cut target % for this video.

    Given duration and the estimated silence-trim ratio, returns a dict with:
        target_final_keep_pct: where we want the final video to land
        silence_trim_ratio:    fraction of source the trim stage will remove
                               (per-keep, scales with ai_keep)
        ai_cut_pct:            % the AI should cut (rounded to nearest 5)
        floor_pct:             min % the AI must cut (target - 7)
        ceiling_pct:           max % the AI may cut (target + 12)

    Math: final_keep = ai_keep * (1 - silence_trim_ratio)
          target_final_keep = final_keep_ratio(d)
          ai_keep = target_final_keep / (1 - silence_trim_ratio)
          ai_cut_pct = 100 * (1 - ai_keep)
    """
    target_keep = final_keep_ratio(duration_s)
    # Cap silence ratio so the divisor doesn't blow up on near-1.0 trim videos.
    eff_silence = min(0.5, max(0.0, silence_trim_ratio))
    ai_keep = target_keep / (1.0 - eff_silence)
    # ai_keep can exceed 1 if target_keep > 1-silence (silly short video with
    # tons of silence). Clamp.
    ai_keep = min(0.95, max(0.05, ai_keep))
    ai_cut_pct = 100.0 * (1.0 - ai_keep)

    # Round to nearest round_to_pct.
    ai_cut_rounded = round(ai_cut_pct / round_to_pct) * round_to_pct
    # Keep within sane bounds AFTER rounding.
    ai_cut_rounded = max(15, min(75, ai_cut_rounded))

    return {
        "duration_min": duration_s / 60.0,
        "target_final_keep_pct": round(100.0 * target_keep, 1),
        "silence_trim_ratio": round(silence_trim_ratio, 3),
        "ai_cut_pct": int(ai_cut_rounded),
        "floor_pct": max(10, int(ai_cut_rounded) - 7),
        "ceiling_pct": min(80, int(ai_cut_rounded) + 12),
    }


if __name__ == "__main__":
    # Quick sanity check across 10 min to 10 hr.
    print("duration | final_keep | (assuming 20% silence trim) ai_cut | (5% trim) ai_cut | (35% trim) ai_cut")
    print("-" * 110)
    for d_min in [10, 20, 30, 45, 60, 90, 120, 180, 240, 300, 400, 500, 600]:
        d_s = d_min * 60.0
        keep = final_keep_ratio(d_s)
        target_20 = compute_ai_cut_target(d_s, 0.20)
        target_05 = compute_ai_cut_target(d_s, 0.05)
        target_35 = compute_ai_cut_target(d_s, 0.35)
        print(f"{d_min:6d} min | keep {keep*100:5.1f}% | "
              f"{target_20['ai_cut_pct']:3d}% ({target_20['floor_pct']}-{target_20['ceiling_pct']}) | "
              f"{target_05['ai_cut_pct']:3d}% ({target_05['floor_pct']}-{target_05['ceiling_pct']}) | "
              f"{target_35['ai_cut_pct']:3d}% ({target_35['floor_pct']}-{target_35['ceiling_pct']})")
