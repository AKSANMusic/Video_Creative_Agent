"""Frame-accurate render plan (EDL) built from a :class:`Timeline`.

The plan converts the timeline's float second-boundaries into integer frame
counts at the target FPS, snaps cut boundaries to the audio's frame grid (so
the renderer never has to resample), and pre-computes per-entry zoom/transition
parameters. It is pure data + math: no I/O, fully testable offline.

Ken Burns
---------
Each entry gets a subtle linear zoom from ``zoom_start`` to ``zoom_end``
(default 1.0 -> 1.08) over its frame window, with a deterministic pan that
keeps the focal point near the image center. Deterministic => reproducible
renders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..errors import CreativeEngineError
from ..narrative.timeline import Timeline, TransitionType

# Default subtle Ken Burns zoom range (in/out factor on the source frame).
DEFAULT_ZOOM_START = 1.0
DEFAULT_ZOOM_END = 1.08


@dataclass(frozen=True)
class FrameWindow:
    """Integer frame window for one timeline entry."""

    index: int
    image_path: str
    section_index: int
    start_frame: int
    end_frame: int  # exclusive
    start_time: float
    end_time: float
    transition_type: TransitionType
    transition_frames: int  # frames consumed by the leading transition
    zoom_start: float
    zoom_end: float
    pan_x_start: float
    pan_x_end: float
    pan_y_start: float
    pan_y_end: float

    @property
    def frame_count(self) -> int:
        return max(1, self.end_frame - self.start_frame)


@dataclass
class RenderPlan:
    """Full frame-accurate plan for the renderer."""

    timeline: Timeline
    fps: float
    width: int
    height: int
    total_frames: int
    windows: list[FrameWindow] = field(default_factory=list)

    def validate(self) -> None:
        """Assert the plan covers [0, total_frames] contiguously (frame grid)."""
        if not self.windows:
            raise CreativeEngineError("render plan has no windows")
        if self.windows[0].start_frame != 0:
            raise CreativeEngineError(
                f"first window must start at frame 0 (got {self.windows[0].start_frame})"
            )
        for prev, cur in zip(self.windows, self.windows[1:]):
            if cur.start_frame != prev.end_frame:
                raise CreativeEngineError(
                    f"frame gap/overlap between entry {prev.index} and {cur.index}"
                )
        last = self.windows[-1]
        if last.end_frame != self.total_frames:
            raise CreativeEngineError(
                f"last window end frame {last.end_frame} != total {self.total_frames}"
            )


class PlanBuilder:
    """Converts a :class:`Timeline` into a :class:`RenderPlan`."""

    def __init__(
        self,
        fps: float = 30.0,
        width: int = 1280,
        height: int = 720,
        zoom_start: float = DEFAULT_ZOOM_START,
        zoom_end: float = DEFAULT_ZOOM_END,
        min_transition_frames: int = 1,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if width <= 0 or height <= 0:
            raise ValueError("width/height must be positive")
        if zoom_end < zoom_start:
            raise ValueError("zoom_end must be >= zoom_start")
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.zoom_start = float(zoom_start)
        self.zoom_end = float(zoom_end)
        self.min_transition_frames = max(1, int(min_transition_frames))

    # --- public API ----------------------------------------------------------

    def build(self, timeline: Timeline) -> RenderPlan:
        total_frames = self._sec_to_frame(timeline.duration)
        windows: list[FrameWindow] = []
        for entry in timeline.entries:
            windows.append(self._build_window(entry, total_frames))

        # Apply J/L-cut offsets by shifting boundary frames (Phase 2)
        import dataclasses
        for i in range(1, len(windows)):
            entry = timeline.entries[i]
            offset_ms = getattr(entry, "audio_offset_ms", 0.0)
            if offset_ms != 0.0:
                offset_s = offset_ms / 1000.0
                offset_frames = int(round(offset_s * self.fps))

                base_boundary = self._sec_to_frame(entry.start)
                new_boundary = base_boundary + offset_frames

                # Clamp to ensure we don't collapse either window.
                min_boundary = windows[i-1].start_frame + 1
                max_boundary = windows[i].end_frame - 1
                new_boundary = max(min_boundary, min(new_boundary, max_boundary))

                windows[i-1] = dataclasses.replace(
                    windows[i-1],
                    end_frame=new_boundary,
                    end_time=float(new_boundary / self.fps)
                )
                windows[i] = dataclasses.replace(
                    windows[i],
                    start_frame=new_boundary,
                    start_time=float(new_boundary / self.fps)
                )

                # Re-clamp transition frames for both adjusted windows for safety.
                t_frames_prev = windows[i-1].transition_frames
                max_t_prev = min(t_frames_prev, max(0, (windows[i-1].end_frame - windows[i-1].start_frame) - 1))
                if max_t_prev != t_frames_prev:
                    windows[i-1] = dataclasses.replace(windows[i-1], transition_frames=max_t_prev)

                t_frames_curr = windows[i].transition_frames
                max_t_curr = min(t_frames_curr, max(0, (windows[i].end_frame - windows[i].start_frame) - 1))
                if max_t_curr != t_frames_curr:
                    windows[i] = dataclasses.replace(windows[i], transition_frames=max_t_curr)

        plan = RenderPlan(
            timeline=timeline,
            fps=self.fps,
            width=self.width,
            height=self.height,
            total_frames=total_frames,
            windows=windows,
        )
        plan.validate()
        return plan

    # --- internals -----------------------------------------------------------

    def _build_window(self, entry, total_frames: int) -> FrameWindow:
        start_frame = self._sec_to_frame(entry.start)
        # End frame: snap the entry end, clamped to the total so rounding never
        # overshoots the track.
        end_frame = min(total_frames, self._sec_to_frame_ceil(entry.end))
        if end_frame <= start_frame:
            # Degenerate (sub-frame) entry: give it at least one frame.
            end_frame = start_frame + 1

        trans_frames = 0
        if entry.transition.type != "cut" and entry.transition.duration_s > 0:
            trans_frames = max(
                self.min_transition_frames,
                self._sec_to_frame(entry.transition.duration_s),
            )
            # Transition can't consume the whole window.
            trans_frames = min(trans_frames, max(0, (end_frame - start_frame) - 1))

        # Focus coordinates from saliency centroid (Phase 2)
        pan_x_start = getattr(entry, "zoom_target_x", 0.5)
        pan_y_start = getattr(entry, "zoom_target_y", 0.5)

        # Subtle camera drift from focal point
        drift_x = 0.02 * ((entry.index % 3) - 1)  # -0.02, 0, +0.02
        drift_y = 0.01 * (((entry.index + 1) % 3) - 1)  # -0.01, 0, +0.01

        pan_x_end = min(max(pan_x_start + drift_x, 0.0), 1.0)
        pan_y_end = min(max(pan_y_start + drift_y, 0.0), 1.0)

        # Dynamic zoom intensity based on director output (Phase 2)
        zoom_end = getattr(entry, "zoom_intensity", self.zoom_end)

        return FrameWindow(
            index=entry.index,
            image_path=entry.file_path,
            section_index=entry.section_index,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=float(entry.start),
            end_time=float(entry.end),
            transition_type=entry.transition.type,
            transition_frames=trans_frames,
            zoom_start=self.zoom_start,
            zoom_end=zoom_end,
            pan_x_start=pan_x_start,
            pan_x_end=pan_x_end,
            pan_y_start=pan_y_start,
            pan_y_end=pan_y_end,
        )

    def _sec_to_frame(self, seconds: float) -> int:
        return int(round(float(seconds) * self.fps))

    def _sec_to_frame_ceil(self, seconds: float) -> int:
        return int(round(float(seconds) * self.fps))
