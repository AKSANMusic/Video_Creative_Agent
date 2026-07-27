"""Stage-4 render orchestrator.

Flow
----
1. Load :class:`Timeline` (from JSON or passed in).
2. Build :class:`RenderPlan` (frame-accurate EDL).
3. Build :class:`FilterGraph` (ffmpeg filter_complex).
4. Validate that every referenced image file exists on disk.
5. Run :class:`FFmpegEncoder`.
6. Audit A/V drift with :class:`DriftAuditor`.

Renders are deterministic given identical inputs + encoder params. We do NOT
cache rendered MP4s by default (they are large and the upstream timeline cache
already short-circuits identical timelines); the caller may cache by file hash
if desired.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..errors import CreativeEngineError
from ..narrative.timeline import Timeline
from .drift import DriftAuditor, DriftReport
from .encoder import EncodeResult, FFmpegEncoder
from .filtergraph import FilterGraph, FilterGraphBuilder
from .plan import PlanBuilder, RenderPlan

log = logging.getLogger(__name__)


@dataclass
class RenderResult:
    """Full outcome of a render run."""

    plan: RenderPlan
    graph: FilterGraph
    encode: EncodeResult
    drift: DriftReport


class RenderPipeline:
    """Coordinates plan -> filtergraph -> encode -> audit."""

    def __init__(
        self,
        plan_builder: Optional[PlanBuilder] = None,
        graph_builder: Optional[FilterGraphBuilder] = None,
        encoder: Optional[FFmpegEncoder] = None,
        auditor: Optional[DriftAuditor] = None,
    ) -> None:
        self.plan_builder = plan_builder or PlanBuilder()
        self.graph_builder = graph_builder or FilterGraphBuilder()
        self.encoder = encoder or FFmpegEncoder(fps=self.plan_builder.fps)
        self.auditor = auditor or DriftAuditor()

    # --- public API ----------------------------------------------------------

    def run(
        self,
        timeline: Timeline,
        output_path: Path | str,
    ) -> RenderResult:
        """Render the timeline to ``output_path`` and audit it."""
        output_path = Path(output_path)
        audio_path = Path(timeline.audio_file)
        if not audio_path.is_file():
            raise CreativeEngineError(f"Audio file not found: {audio_path}")

        plan = self.plan_builder.build(timeline)
        self._validate_image_inputs(plan)

        graph = self.graph_builder.build(plan, audio_path=str(audio_path))
        encode = self.encoder.encode(graph, audio_path, output_path)
        drift = self.auditor.audit(plan, output_path)

        if not drift.passed:
            log.warning(
                "A/V drift KPI FAILED: %.1fms > %.1fms (mode=%s)",
                drift.max_drift_ms, self.auditor.kpi_ms, drift.mode,
            )
        else:
            log.info(
                "A/V drift OK: %.1fms <= %.1fms (mode=%s)",
                drift.max_drift_ms, self.auditor.kpi_ms, drift.mode,
            )

        return RenderResult(plan=plan, graph=graph, encode=encode, drift=drift)

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _validate_image_inputs(plan: RenderPlan) -> None:
        missing = [w.image_path for w in plan.windows if not Path(w.image_path).is_file()]
        if missing:
            sample = ", ".join(missing[:3])
            raise CreativeEngineError(
                f"{len(missing)} image(s) missing on disk (e.g. {sample}). "
                "Ensure Stage-1 file_path values point to existing files."
            )
