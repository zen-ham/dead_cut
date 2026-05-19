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
"""
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
    def __init__(self, source_duration_s: float):
        self.source = source_duration_s
        self.start_time = time.time()
        self.actual_elapsed: dict[str, float] = {}
        self._stage_starts: dict[str, float] = {}
        self.overall = tqdm(
            total=int(self._total_estimate()),
            desc="[overall ]", position=0, unit="s",
            bar_format="{desc} {bar} {percentage:3.0f}% | {n_fmt}/{total_fmt}s | elapsed {elapsed} | eta {remaining}",
            leave=True,
        )

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
        return sum(
            self.actual_elapsed.get(s, self._estimate(s))
            for s in _BASELINES
        )

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
        """Update the overall bar from wall-clock elapsed. Called frequently
        from inside per-stage progress callbacks."""
        elapsed = int(time.time() - self.start_time)
        # Don't show >100%; if a stage runs longer than its baseline we just
        # cap at total. The total gets refreshed at stage end with actuals.
        if elapsed > self.overall.total:
            self.overall.total = elapsed  # let the bar keep moving past the estimate
        self.overall.n = elapsed
        self.overall.refresh()

    def close(self) -> None:
        self.overall.n = self.overall.total
        self.overall.refresh()
        self.overall.close()


# Module-level convenience API. Pipeline initialises, stages call.

def init(source_duration_s: float) -> PipelineTracker:
    global _TRACKER
    _TRACKER = PipelineTracker(source_duration_s)
    return _TRACKER


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
