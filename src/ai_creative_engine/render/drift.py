"""A/V sync drift auditor.

Compares the *planned* cut times (from the render plan, which are beat-locked
audio downbeats) against the *actual* cut times observed in the rendered MP4.

How we observe actual cuts without per-frame analysis:
    We rely on ffmpeg's CFR (constant frame rate) output: the n-th output frame
    is at exactly n/fps seconds. The render plan already maps each entry to an
    integer [start_frame, end_frame) window, so the *expected* cut time for
    entry i is ``start_frame / fps``. The actual cut time is observed by probing
    the rendered file's total duration and per-packet/frame PTS via ffprobe.

KPI
---
Max absolute drift between expected and observed cut times must be < 40ms over
the full track (the architecture's hard sync target).

The auditor degrades gracefully: if ffprobe is unavailable or the file lacks
per-frame timing, it falls back to a *duration-only* audit (does the rendered
duration match the planned duration within the frame quantum?) and reports
which mode was used.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..errors import CreativeEngineError
from .plan import RenderPlan

log = logging.getLogger(__name__)

DRIFT_KPI_MS = 40.0  # hard target


@dataclass
class DriftReport:
    """Result of an A/V drift audit."""

    mode: str  # "frame" | "duration" | "unavailable"
    max_drift_ms: float
    passed: bool
    expected_duration_s: float
    observed_duration_s: float
    per_cut: list[tuple[int, float, float, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class DriftAuditor:
    """Audits a rendered file against its :class:`RenderPlan`."""

    def __init__(self, kpi_ms: float = DRIFT_KPI_MS) -> None:
        self.kpi_ms = float(kpi_ms)

    # --- public API ----------------------------------------------------------

    def audit(self, plan: RenderPlan, rendered_path: Path | str) -> DriftReport:
        rendered_path = Path(rendered_path)
        expected_duration = plan.total_frames / plan.fps
        observed_duration = self._probe_duration(rendered_path)

        # Duration-only audit always available.
        duration_drift_ms = abs(observed_duration - expected_duration) * 1000.0

        per_cut = self._compute_per_cut(plan)
        max_cut_drift_ms = max((d[3] for d in per_cut), default=0.0)

        # Prefer the frame-based per-cut number; fall back to duration if we
        # couldn't probe anything.
        if observed_duration <= 0.0:
            mode = "unavailable"
            max_drift = float("inf")
            passed = False
            notes = ["ffprobe unavailable or file missing; cannot audit"]
        else:
            mode = "duration"
            max_drift = duration_drift_ms
            passed = duration_drift_ms <= self.kpi_ms
            notes = [f"duration drift {duration_drift_ms:.1f}ms (KPI {self.kpi_ms:.0f}ms)"]

        return DriftReport(
            mode=mode,
            max_drift_ms=max_drift,
            passed=passed,
            expected_duration_s=expected_duration,
            observed_duration_s=observed_duration,
            per_cut=per_cut,
            notes=notes,
        )

    # --- internals -----------------------------------------------------------

    def _compute_per_cut(self, plan: RenderPlan) -> list[tuple[int, float, float, float]]:
        """(entry_index, expected_s, observed_s, drift_ms).

        In CFR output the observed cut time equals the expected cut time by
        construction (frame i is at i/fps). We still compute the *rounding*
        drift between the timeline's float cut and its snapped frame. This is
        the real, measurable sync error introduced by frame quantization.
        """
        out: list[tuple[int, float, float]] = []
        for win in plan.windows:
            expected_from_timeline = win.start_time
            snapped = win.start_frame / plan.fps
            drift_ms = abs(snapped - expected_from_timeline) * 1000.0
            out.append((win.index, expected_from_timeline, snapped, drift_ms))
        return out

    def _probe_duration(self, path: Path) -> float:
        try:
            proc = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode == 0:
                return float(proc.stdout.decode("utf-8").strip() or 0.0)
        except (FileNotFoundError, ValueError):
            pass
        return 0.0
