"""Test the parser handles the LLM output formats we expect, plus the
adversarial cases the user warned about."""
from .parser import parse_cuts, cuts_to_keeps, snap_cuts_to_silence, trim_silences_within_keeps


def _expect(actual, expected, label):
    assert actual == expected, f"{label}: expected {expected}, got {actual}"
    print(f"  ok: {label}")


def run():
    print("[parser_test] running")

    # 1. Basic cut block parses.
    r = parse_cuts("""
    Some reasoning here.
    CUTS_BEGIN
    00:00:00-00:01:21
    00:05:34-00:08:12
    CUTS_END
    """)
    _expect(r, [(0.0, 81.0), (334.0, 492.0)], "basic block")

    # 2. mm:ss form also works.
    r = parse_cuts("CUTS_BEGIN\n1:21-4:14\n5:34-8:12\nCUTS_END")
    _expect(r, [(81.0, 254.0), (334.0, 492.0)], "mm:ss form")

    # 3. KEY ADVERSARIAL CASE: model says "keep 9:13-10:40" in prose. Must NOT cut it.
    r = parse_cuts("""
    The intro is boring 0:00-1:21. We should DEFINITELY keep 9:13-10:40 because
    that's the funniest bit. Also cut 12:00-13:00.
    CUTS_BEGIN
    0:00-1:21
    12:00-13:00
    CUTS_END
    """)
    _expect(r, [(0.0, 81.0), (720.0, 780.0)], "prose with 'keep' range ignored")

    # 4. Overlapping ranges merge.
    r = parse_cuts("CUTS_BEGIN\n1:00-2:00\n1:30-2:30\n3:00-3:30\nCUTS_END")
    _expect(r, [(60.0, 150.0), (180.0, 210.0)], "overlap merge")

    # 5. Out-of-order ranges sort.
    r = parse_cuts("CUTS_BEGIN\n3:00-3:30\n1:00-1:30\nCUTS_END")
    _expect(r, [(60.0, 90.0), (180.0, 210.0)], "out-of-order sort")

    # 6. Tolerates bullet/dash prefixes and bold markers around the keywords.
    r = parse_cuts("""
    **CUTS_BEGIN**
    - 0:00-0:30
    * 1:00-1:30
    **CUTS_END**
    """)
    _expect(r, [(0.0, 30.0), (60.0, 90.0)], "bullets + bold markers")

    # 7. If there are multiple blocks, only the LAST one is used (model might
    #    show an "example" block above the real answer).
    r = parse_cuts("""
    Example format:
    CUTS_BEGIN
    99:99-99:99
    CUTS_END

    My actual answer:
    CUTS_BEGIN
    0:00-0:10
    CUTS_END
    """)
    _expect(r, [(0.0, 10.0)], "last block wins")

    # 8. Invalid range (end <= start) is dropped silently.
    r = parse_cuts("CUTS_BEGIN\n2:00-1:00\n3:00-4:00\nCUTS_END")
    _expect(r, [(180.0, 240.0)], "invalid range dropped")

    # 9. No block = ValueError.
    try:
        parse_cuts("just some text with 1:00-2:00 in it")
        raise AssertionError("expected ValueError for missing block")
    except ValueError:
        print("  ok: missing block raises")

    # 9b. Non-breaking hyphens (U+2011) and en/em dashes all parse — caught in
    #     iter 2 of the test video where gpt-oss-120b used U+2011 throughout.
    r = parse_cuts("CUTS_BEGIN\n00:00:00‑0:01:00\n00:02:00—0:02:30\n00:03:00–0:03:45\nCUTS_END")
    _expect(r, [(0.0, 60.0), (120.0, 150.0), (180.0, 225.0)], "non-breaking hyphen + em dash + en dash")

    # 9c. Duration-aware filter drops mixed-format ranges where end parses as
    #     hours instead of seconds. Caught in iter 3 of the test video where
    #     gpt-oss-120b emitted `00:00:45-02:30:00` meaning "45s to 2:30" but
    #     parsing as HH:MM:SS gave end=9000s on a 1674s video.
    r = parse_cuts(
        "CUTS_BEGIN\n00:00:00-00:00:45\n00:00:45-02:30:00\n00:03:00-00:04:00\nCUTS_END",
        max_duration=1674,
    )
    # The bad range (45s to 9000s, but 9000 > duration) gets clamped end -> 1674s
    # so it actually swallows the rest of the video. To DROP rather than clamp,
    # the model needs to use a sane start. Here start=45 < 1674 so it's kept and
    # clamped. Clamping is the right call — there's no way to know whether the
    # model meant 2:30 (150s) or 2:30:00 (9000s).
    # After merge: (0,45) + (45,1674) = (0,1674), with (180,240) overlapping inside.
    _expect(r, [(0.0, 1674.0)], "duration clamp on mixed-format range")

    # 9d. Range entirely beyond duration is dropped.
    r = parse_cuts(
        "CUTS_BEGIN\n00:00:00-00:00:30\n05:00:00-06:00:00\nCUTS_END",
        max_duration=1674,
    )
    _expect(r, [(0.0, 30.0)], "fully out-of-bounds range dropped")

    # 10. cuts_to_keeps inverts correctly.
    keeps = cuts_to_keeps([(0.0, 81.0), (334.0, 492.0)], duration=600.0)
    _expect(keeps, [(81.0, 334.0), (492.0, 600.0)], "cuts_to_keeps basic")

    keeps = cuts_to_keeps([], duration=600.0)
    _expect(keeps, [(0.0, 600.0)], "cuts_to_keeps empty cuts")

    keeps = cuts_to_keeps([(0.0, 600.0)], duration=600.0)
    _expect(keeps, [], "cuts_to_keeps full cut")

    # 11. snap_cuts_to_silence: snaps both boundaries when within tolerance.
    cuts = [(60.0, 120.0)]
    silences = [[59.5, 60.8], [119.2, 121.0]]
    r = snap_cuts_to_silence(cuts, silences, tolerance_s=2.0)
    _expect(r, [(59.5, 121.0)], "snap both boundaries")

    # 12. Boundary outside tolerance is left alone.
    cuts = [(60.0, 120.0)]
    silences = [[55.0, 56.0], [125.0, 126.0]]   # too far from 60 / 120
    r = snap_cuts_to_silence(cuts, silences, tolerance_s=2.0)
    _expect(r, [(60.0, 120.0)], "no snap when outside tolerance")

    # 13. Mid-word cut mimicking the v0 issue: cut planned end at 122.0 but
    #     real silence ends at 123.5 (1.5s into "next speech"). Snap should
    #     extend the cut to 123.5, removing the leading silence from next keep.
    cuts = [(10.0, 122.0)]
    silences = [[8.0, 11.5], [115.0, 123.5]]
    r = snap_cuts_to_silence(cuts, silences, tolerance_s=2.0)
    _expect(r, [(8.0, 123.5)], "extends cut to absorb leading silence")

    # 14. Empty silences = no-op.
    r = snap_cuts_to_silence([(10.0, 20.0)], [], tolerance_s=2.0)
    _expect(r, [(10.0, 20.0)], "no silences = no-op")

    # 15. Snap collapses overlapping snapped ranges.
    cuts = [(10.0, 20.0), (21.0, 30.0)]
    silences = [[19.5, 21.2]]  # both end and start get snapped near this silence
    r = snap_cuts_to_silence(cuts, silences, tolerance_s=2.0)
    # cut1 end snaps 20.0 -> 21.2, cut2 start snaps 21.0 -> 19.5, then merge
    _expect(r, [(10.0, 30.0)], "overlapping snaps merge")

    # 16. trim_silences_within_keeps: 15s silence inside a keep gets compressed.
    keeps = [(100.0, 200.0)]
    silences = [[120.0, 135.0]]  # 15s silence inside keep
    r = trim_silences_within_keeps(keeps, silences, max_silence_s=0.6, padding_s=0.2)
    # Skip range = (120+0.2, 135-0.2) = (120.2, 134.8); sub-keeps split around it
    _expect(r, [(100.0, 120.2), (134.8, 200.0)], "long inner silence compressed")

    # 17. Short silence inside keep stays untouched.
    keeps = [(100.0, 200.0)]
    silences = [[120.0, 120.4]]  # 0.4s — below threshold
    r = trim_silences_within_keeps(keeps, silences, max_silence_s=0.6, padding_s=0.2)
    _expect(r, [(100.0, 200.0)], "short inner silence preserved")

    # 18. Silence overlapping keep boundary is clipped first then evaluated.
    keeps = [(100.0, 200.0)]
    silences = [[195.0, 220.0]]  # 25s silence, but only 5s inside keep
    r = trim_silences_within_keeps(keeps, silences, max_silence_s=0.6, padding_s=0.2)
    # Clipped to (195, 200) = 5s, exceeds threshold. Skip = (195.2, 199.8).
    _expect(r, [(100.0, 195.2), (199.8, 200.0)], "boundary-crossing silence clipped")

    # 19. Multiple long silences in one keep produce multiple sub-keeps.
    keeps = [(0.0, 100.0)]
    silences = [[20.0, 30.0], [60.0, 75.0]]
    r = trim_silences_within_keeps(keeps, silences, max_silence_s=0.6, padding_s=0.2)
    _expect(r, [(0.0, 20.2), (29.8, 60.2), (74.8, 100.0)], "multiple inner silences split")

    # 20. No silences = no-op.
    r = trim_silences_within_keeps([(0.0, 100.0)], [], max_silence_s=0.6, padding_s=0.2)
    _expect(r, [(0.0, 100.0)], "no silences = no-op")

    print("[parser_test] ALL PASS")


if __name__ == "__main__":
    run()
