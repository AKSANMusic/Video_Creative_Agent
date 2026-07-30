"""Build the FFmpeg ``filter_complex`` for a :class:`RenderPlan`.

Strategy (robust + simple)
--------------------------
For each entry we produce a per-image video sub-chain:

    [N:v] loop=<frames>,scale,setsar=1,format=yuva420p,
          zoompan=<ken burns> -> [vN]

Then we stitch the per-entry streams into one continuous video stream. We use
**xfade** for dissolve/zoom transitions and a plain **concat** for hard cuts.
To keep the filtergraph uniform and easy to reason about, we run every join
through ``xfade`` with ``duration=0`` for cuts (an instant crossfade is a cut)
and ``duration=<transition_frames/fps>`` for dissolves.

This yields a single deterministic filtergraph string that ffmpeg-python (or
the ffmpeg CLI directly) can consume. No runtime decisions inside ffmpeg.

The builder is pure string construction -> fully unit-testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..errors import CreativeEngineError
from .plan import FrameWindow, RenderPlan


@dataclass(frozen=True)
class FilterGraph:
    """A built filter_complex + the mapping of input index -> image path."""

    filter_complex: str
    video_output_label: str  # e.g. "[vout]"
    input_order: list[str]  # image_path per ffmpeg input index
    audio_index: int  # ffmpeg input index of the audio track


class FilterGraphBuilder:
    """Translates a :class:`RenderPlan` into an ffmpeg filter_complex."""

    def __init__(self, sample_fmt: str = "yuva420p", pixel_fmt: str = "yuv420p") -> None:
        self.sample_fmt = sample_fmt
        self.pixel_fmt = pixel_fmt

    # --- public API ----------------------------------------------------------

    def build(
        self,
        plan: RenderPlan,
        audio_path: str,
    ) -> FilterGraph:
        if not plan.windows:
            raise CreativeEngineError("cannot build a filtergraph from an empty plan")

        input_order: list[str] = []
        chains: list[str] = []
        labels: list[str] = []

        for i, win in enumerate(plan.windows):
            input_idx = i
            input_order.append(win.image_path)
            out_label = f"[v{i}]"
            chains.append(self._entry_chain(win, input_idx, out_label, fps=plan.fps))
            labels.append(out_label)

        # Stitch entries pairwise with xfade (duration=0 for cuts).
        stitch_chains, final_label = self._stitch(labels, plan)
        chains.extend(stitch_chains)

        audio_input_index = len(input_order)
        filter_complex = ";".join(chains)
        return FilterGraph(
            filter_complex=filter_complex,
            video_output_label=final_label,
            input_order=input_order,
            audio_index=audio_input_index,
        )

    # --- per-entry chain -----------------------------------------------------

    def _entry_chain(self, win: FrameWindow, input_idx: int, out_label: str, fps: float = 30.0) -> str:
        """loop -> scale -> setsar -> format -> zoompan for one entry."""
        frames = win.frame_count
        fps_val = fps if fps > 0 else 30.0
        # zoompan needs 'd' (duration in output frames), and a time-based zoom
        # expression. We compute zoom over the normalized in-window frame index.
        # zoompan's zoom expr is evaluated per output frame; 'on' is the output
        # frame index (0-based) for THIS zoompan instance.
        zoom = self._zoom_expr(win)
        
        # Panning equations targeting saliency center with drift (Phase 2)
        dx = win.pan_x_end - win.pan_x_start
        dy = win.pan_y_end - win.pan_y_start
        
        frames_denom = max(1, frames)
        step_x = dx / frames_denom
        step_y = dy / frames_denom
        
        zp = (
            f"zoompan=z='{zoom}':"
            f"x='max(0, min(iw - iw/z, iw*({win.pan_x_start:.6f}+on*{step_x:.8f}) - iw/(2*z)))':"
            f"y='max(0, min(ih - ih/z, ih*({win.pan_y_start:.6f}+on*{step_y:.8f}) - ih/(2*z)))':"
            f"d={frames}:s={self._dims_target()}:fps={fps_val:.1f}"
        )
        chain = (
            f"[{input_idx}:v]loop=loop={frames}:size=1:start=0,"
            f"scale={self._dims_target()}:force_original_aspect_ratio=decrease,"
            f"pad={self._dims_crop()}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format={self.sample_fmt},{zp},"
            f"format={self.pixel_fmt},settb=1/{int(fps_val)}"
            f"{out_label}"
        )
        return chain

    def _zoom_expr(self, win: FrameWindow) -> str:
        """Linear zoom from zoom_start to zoom_end across the window frames."""
        n = max(1, win.frame_count)
        z0 = win.zoom_start
        z1 = win.zoom_end
        if z0 == z1:
            return f"{z0:.6f}"
        # on is zoompan's per-output-frame index.
        slope = (z1 - z0) / n
        return f"{z0:.6f}+on*{slope:.6f}"

    def _dims_target(self) -> str:
        """WxH format for scale and zoompan (e.g. '1280x720')."""
        return f"{self._w}x{self._h}" if hasattr(self, "_w") else "1280x720"

    def _dims_crop(self) -> str:
        """W:H format for the crop filter (e.g. '1280:720')."""
        return f"{self._w}:{self._h}" if hasattr(self, "_w") else "1280:720"

    # --- stitching -----------------------------------------------------------

    def _stitch(self, labels: list[str], plan: RenderPlan) -> tuple[list[str], str]:
        """Fold the per-entry streams into one via xfade joins.

        For the i-th join (between entry i-1 and i) we use the *incoming*
        entry's transition. Cut -> duration 0.
        """
        # Lazy-init target dims from the plan.
        self._w = plan.width
        self._h = plan.height

        if len(labels) == 1:
            return [], labels[0]

        chains: list[str] = []
        cur = labels[0]
        stitched_duration = plan.windows[0].frame_count / plan.fps
        for i in range(1, len(labels)):
            win = plan.windows[i]
            trans = win.transition_type
            trans_frames = win.transition_frames
            dur_s = (trans_frames / plan.fps) if trans != "cut" else 0.0
            
            # The transition begins exactly dur_s before the end of the stitched stream
            offset = max(0.0, stitched_duration - dur_s)
            
            # Add the new clip's duration to the total, but subtract the overlapped portion
            incoming_dur = plan.windows[i].frame_count / plan.fps
            stitched_duration = stitched_duration + incoming_dur - dur_s
            
            out_label = f"[vx{i}]"
            xfade = (
                f"{cur}{labels[i]}xfade=transition={self._xfade_mode(trans, plan, i)}:"
                f"duration={dur_s:.6f}:offset={offset:.6f}{out_label}"
            )
            chains.append(xfade)
            cur = out_label

        final = "[vout]"
        # Relabel the last vxN -> vout via a no-op format (keeps it explicit).
        chains.append(f"{cur}format={self.pixel_fmt}{final}")
        return chains, final

    @staticmethod
    def _xfade_mode(trans: str, plan: RenderPlan, i: int) -> str:
        if trans == "dissolve":
            return "dissolve"
        if trans == "zoom":
            return "fade"  # zoom handled by zoompan; xfade just fades.
        return "fade"  # duration=0 fade == hard cut
