"""Pipeline-wide progress tracking.

Two stacked tqdm bars:
  [overall]  ████░░░░░░░  35%  → eta 4m12s    (wall-clock vs estimated total)
  [stage  ]  ██████████░  92%  → eta 0m08s    (per-stage progress, from transcribe/ffmpeg)

The overall bar's total is an ESTIMATE built from per-stage baselines (measured
on the dev machine). As each stage completes, its actual elapsed time replaces
its baseline in the total, so the ETA refines.

The overall bar's "current" is wall-clock elapsed time since pipeline start —
no fancy weighted-progress math. This is the most honest representation: "this
is how long it's actually taken, here's how long I think it'll take total."

A background ticker thread refreshes the bar every 2s so even stages without
their own progress callbacks (like the yt-dlp download) animate the bar live.
"""
import threading
import time
from tqdm import tqdm


# Baseline timings on the dev machine (GTX 1660 Ti, batched int8_float16
# transcribe, smartcut+h264_nvenc encode). Calibrated from a 1h25 H.264 run:
# total wall-clock 137.5s on cached download (1h25 source → 34min output).
#   ("constant", X)  → flat X seconds regardless of source duration
#   ("source",   X)  → X * source_duration_seconds
#   ("output",   X)  → X * estimated_output_duration  (output ≈ 0.4 × source)
_BASELINES = {
    "download":   ("constant", 30.0),
    "transcribe": ("source",   0.015),  # ~67x realtime on batched GPU (77s/5155s)
    "loudness":   ("source",   0.002),  # ~500x realtime (11s/5155s incl. ffmpeg extract)
    "llm":        ("constant", 35.0),
    "post":       ("constant", 1.0),    # snap + trim (instant)
    # encode (smartcut+nvenc on H.264): ~0.020x of OUTPUT duration. Before
    # we know the output, estimate as 0.010 × source (assuming ~50% kept).
    # Pipeline calls set_stage_baseline() once snap+trim finishes with the
    # exact output_dur × 0.025. Observed-rate from the cutter further
    # refines mid-encode (pushes up for AV1 fallback path, etc).
    "encode":     ("source",   0.010),
}

_TRACKER = None  # module-level singleton; pipeline initialises, stages call.


class PipelineTracker:
    # If pipeline initialises before source duration is known (e.g. before
    # download), assume a typical-length vod for the initial estimate. Gets
    # refined by set_source_duration() once download finishes.
    DEFAULT_SOURCE_S = 1800.0  # 30min — close enough for the bar to be useful

    def __init__(self, source_duration_s: float | None = None):
        self.source = source_duration_s if source_duration_s else self.DEFAULT_SOURCE_S
        self.start_time = time.time()
        self.actual_elapsed: dict[str, float] = {}
        self._stage_starts: dict[str, float] = {}
        self._baseline_overrides: dict[str, float] = {}
        # Live, observed-rate totals for in-flight stages. Set by per-stage
        # progress callbacks (cutter, transcribe, download) once they have
        # enough samples to extrapolate. These TRUMP the baseline for the
        # in-flight stage — when the cutter says "ETA 10:33 based on actual
        # rate", the overall total should reflect that, not a baseline guess.
        self._stage_observed_total: dict[str, float] = {}
        self.overall = tqdm(
            total=int(self._total_estimate()),
            desc="[overall ]", position=0, unit="s",
            bar_format="{desc} {bar} {percentage:3.0f}% | {n_fmt}/{total_fmt}s | elapsed {elapsed} | eta {remaining}",
            leave=True,
        )
        # Background ticker — keeps the bar live during stages that don't
        # call progress.tick() (e.g. the yt-dlp download).
        self._stop_event = threading.Event()
        self._ticker = threading.Thread(target=self._tick_loop, daemon=True)
        self._ticker.start()

    def _tick_loop(self) -> None:
        while not self._stop_event.wait(2.0):
            try:
                self.tick()
            except Exception:
                # If the bar is closed mid-tick, just stop.
                break

    def set_stage_baseline(self, stage: str, seconds: float) -> None:
        """Override a stage's baseline estimate. Used when we learn the
        actual scale of a stage's work (e.g. encode after snap+trim
        tells us exact output duration). Only applies to not-yet-started
        stages — once a stage is in flight, _total_estimate uses
        max(baseline, current_elapsed) anyway."""
        if stage in _BASELINES and stage not in self.actual_elapsed:
            self._baseline_overrides[stage] = seconds
            self.tick()

    def set_source_duration(self, s: float) -> None:
        """Pipeline calls this once it knows the actual source duration (after
        download). Refreshes the overall ETA — for very long sources the bar
        was probably under-estimated; for short sources, over-estimated."""
        if s and s > 0:
            self.source = s
            new_total = int(self._total_estimate())
            if new_total != self.overall.total:
                self.overall.total = new_total
            self.tick()

    def _estimate(self, stage: str) -> float:
        # Pipeline-supplied override wins (set after we know actual stage
        # work — e.g. encode after snap+trim gives us exact output dur).
        if stage in self._baseline_overrides:
            return self._baseline_overrides[stage]
        mode, val = _BASELINES[stage]
        if mode == "constant":
            return val
        if mode == "source":
            return val * self.source
        if mode == "output":
            return val * self.source * 0.30
        raise ValueError(f"unknown mode {mode}")

    def _total_estimate(self) -> float:
        """Total seconds we expect the pipeline to take. Per stage:
          - completed: actual elapsed
          - in flight + has observed-rate total: use that (the stage knows
            its own ETA from its actual throughput — way more accurate than
            any baseline)
          - in flight + no observed total yet: max(baseline, elapsed)
          - not started: baseline (or pipeline-supplied override)
        """
        now = time.time()
        total = 0.0
        for s in _BASELINES:
            if s in self.actual_elapsed:
                total += self.actual_elapsed[s]
            elif s in self._stage_starts:
                if s in self._stage_observed_total:
                    total += self._stage_observed_total[s]
                else:
                    in_progress = now - self._stage_starts[s]
                    total += max(self._estimate(s), in_progress)
            else:
                total += self._estimate(s)
        return total

    def begin_stage(self, name: str) -> None:
        if name in _BASELINES:
            self._stage_starts[name] = time.time()
            # Clear stale observed-rate carryover from a previous run/stage.
            self._stage_observed_total.pop(name, None)

    def end_stage(self, name: str) -> None:
        if name in self._stage_starts:
            self.actual_elapsed[name] = time.time() - self._stage_starts[name]
            del self._stage_starts[name]
            self._stage_observed_total.pop(name, None)
            # Refresh overall total with the actual time we just observed.
            new_total = int(self._total_estimate())
            if new_total != self.overall.total:
                self.overall.total = new_total
            self.tick()

    def report_stage_rate(self, stage: str, fraction_done: float) -> None:
        """In-flight stages call this with their fraction-of-work-done
        (0.0-1.0). We extrapolate the total time for this stage from
        wall-clock elapsed-so-far × (1 / fraction_done).

        This is how the overall ETA stays honest mid-stage: instead of
        trusting baselines that might be wildly wrong, we use the actual
        observed pace of whatever stage is currently running."""
        if stage not in self._stage_starts:
            return
        if fraction_done <= 0.01:
            return  # too little data, skip
        elapsed = time.time() - self._stage_starts[stage]
        observed_total = elapsed / fraction_done
        self._stage_observed_total[stage] = observed_total

    def tick(self) -> None:
        """Update the overall bar from wall-clock elapsed. Recomputes the
        dynamic total each tick so in-flight stages that run long push the
        total out instead of letting the bar overshoot."""
        elapsed = int(time.time() - self.start_time)
        new_total = int(self._total_estimate())
        if new_total != self.overall.total:
            self.overall.total = new_total
        self.overall.n = min(elapsed, self.overall.total)
        self.overall.refresh()

    def close(self) -> None:
        self._stop_event.set()
        self._ticker.join(timeout=3.0)
        self.overall.n = self.overall.total
        self.overall.refresh()
        self.overall.close()


# Module-level convenience API. Pipeline initialises, stages call.

def init(source_duration_s: float | None = None) -> PipelineTracker:
    global _TRACKER
    _TRACKER = PipelineTracker(source_duration_s)
    return _TRACKER


def set_source_duration(s: float) -> None:
    if _TRACKER is not None:
        _TRACKER.set_source_duration(s)


def set_stage_baseline(stage: str, seconds: float) -> None:
    if _TRACKER is not None:
        _TRACKER.set_stage_baseline(stage, seconds)


def report_stage_rate(stage: str, fraction_done: float) -> None:
    if _TRACKER is not None:
        _TRACKER.report_stage_rate(stage, fraction_done)


def begin_stage(name: str) -> None:
    if _TRACKER is not None:
        _TRACKER.begin_stage(name)


def end_stage(name: str) -> None:
    if _TRACKER is not None:
        _TRACKER.end_stage(name)


def tick() -> None:
    if _TRACKER is not None:
        _TRACKER.tick()


def close() -> None:
    global _TRACKER
    if _TRACKER is not None:
        _TRACKER.close()
        _TRACKER = None
