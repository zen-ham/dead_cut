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
# transcribe, h264_nvenc p1 encode). All in seconds.
#   ("constant", X)  → flat X seconds regardless of source duration
#   ("source",   X)  → X * source_duration_seconds
#   ("output",   X)  → X * estimated_output_duration  (output ≈ 0.3 × source)
_BASELINES = {
    "download":   ("constant", 30.0),
    "transcribe": ("source",   0.012),  # ~80x realtime on batched GPU
    "loudness":   ("source",   0.003),  # plus ~3s ffmpeg extract overhead
    "llm":        ("constant", 35.0),
    "post":       ("constant", 1.0),    # snap + trim (instant)
    "encode":     ("source",   0.18),   # nvenc p1, encoding ~0.3x source output at ~0.6x realtime
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
        mode, val = _BASELINES[stage]
        if mode == "constant":
            return val
        if mode == "source":
            return val * self.source
        if mode == "output":
            return val * self.source * 0.30
        raise ValueError(f"unknown mode {mode}")

    def _total_estimate(self) -> float:
        """Total seconds we expect the pipeline to take. For each stage:
        - completed: use actual elapsed
        - in flight: use max(baseline, current_elapsed_in_stage) — this is
          the key bit. Without this, an in-flight stage that exceeds its
          baseline (e.g. download taking 5min instead of 30s placeholder)
          doesn't push the total, so the overall bar overshoots into
          fake-near-completion territory.
        - not started: use baseline
        """
        now = time.time()
        total = 0.0
        for s in _BASELINES:
            if s in self.actual_elapsed:
                total += self.actual_elapsed[s]
            elif s in self._stage_starts:
                in_progress = now - self._stage_starts[s]
                total += max(self._estimate(s), in_progress)
            else:
                total += self._estimate(s)
        return total

    def begin_stage(self, name: str) -> None:
        if name in _BASELINES:
            self._stage_starts[name] = time.time()

    def end_stage(self, name: str) -> None:
        if name in self._stage_starts:
            self.actual_elapsed[name] = time.time() - self._stage_starts[name]
            del self._stage_starts[name]
            # Refresh overall total with the actual time we just observed.
            new_total = int(self._total_estimate())
            if new_total != self.overall.total:
                self.overall.total = new_total
            self.tick()

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
