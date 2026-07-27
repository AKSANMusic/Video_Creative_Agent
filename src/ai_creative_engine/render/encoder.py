"""FFmpeg encoder: assemble inputs + run the encode.

Uses ``ffmpeg-python`` only as a convenient argument builder; the heavy work
is done by the ``ffmpeg`` binary on the PATH. The actual ``run`` call is
isolated in :meth:`FFmpegEncoder._run` so tests can monkeypatch it and assert
on the assembled argv without invoking a real encode.

Video is muxed with the source audio. We pin:
    - constant frame rate (``-r fps``, ``-vsync cfr``)
    - H.264, CRF (default 18), preset (default medium)
    - AAC audio at 192k
    - ``-shortest`` so output length is bound by the shorter stream
      (timeline video == audio length by construction; -shortest is a safety net)

KPI: A/V drift < 40ms is enforced by the drift auditor (separate module), which
reads the rendered file back via ffprobe.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..errors import CreativeEngineError
from .codec_profile import CodecProfile, resolve_codec_profile
from .codec_profile import quality_args as _quality_args_fn
from .filtergraph import FilterGraph

log = logging.getLogger(__name__)


@dataclass
class EncodeResult:
    """Outcome of an encode run."""

    output_path: Path
    argv: list[str]
    returncode: int
    stderr_tail: str = ""
    duration_s: float = 0.0
    succeeded: bool = False
    notes: list[str] = field(default_factory=list)


class FFmpegEncoder:
    """Assembles and runs the ffmpeg command for a render."""

    def __init__(
        self,
        fps: float = 30.0,
        crf: int = 18,
        preset: str = "medium",
        audio_bitrate: str = "192k",
        pix_fmt: str = "yuv420p",
        loglevel: str = "error",
        video_codec: str = "libx264",
    ) -> None:
        self.fps = float(fps)
        self.crf = int(crf)
        self.preset = str(preset)
        self.audio_bitrate = str(audio_bitrate)
        self.pix_fmt = str(pix_fmt)
        self.loglevel = str(loglevel)
        # Resolve once at construction (fail-fast on a bad codec name).
        self.video_codec = str(video_codec)
        self.codec_profile: CodecProfile = resolve_codec_profile(self.video_codec)

    # --- public API ----------------------------------------------------------

    def encode(
        self,
        graph: FilterGraph,
        audio_path: Path | str,
        output_path: Path | str,
        extra_inputs: Optional[list[Path]] = None,
    ) -> EncodeResult:
        """Run ffmpeg to produce ``output_path``. Returns an :class:`EncodeResult`."""
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        argv = self._build_argv(graph, audio_path, output_path)
        result = EncodeResult(
            output_path=output_path,
            argv=argv,
            returncode=-1,
        )
        try:
            returncode, stderr = self._run(argv)
        except FileNotFoundError as exc:
            raise CreativeEngineError(
                "ffmpeg binary not found on PATH. Install ffmpeg to render."
            ) from exc
        result.returncode = returncode
        result.stderr_tail = stderr[-4000:]
        result.succeeded = returncode == 0 and output_path.exists()
        if not result.succeeded:
            raise CreativeEngineError(
                f"ffmpeg failed (code {returncode}). stderr tail:\n{result.stderr_tail}"
            )
        result.duration_s = _probe_duration(output_path)
        return result

    # --- argv assembly -------------------------------------------------------

    def _build_argv(
        self,
        graph: FilterGraph,
        audio_path: Path,
        output_path: Path,
    ) -> list[str]:
        """Build the ffmpeg command line. Pure + testable."""
        argv: list[str] = ["ffmpeg", "-hide_banner", "-loglevel", self.loglevel, "-y"]

        # Image inputs in filtergraph order (each a single still).
        for img in graph.input_order:
            argv += ["-loop", "1", "-i", str(img)]

        # Audio input last -> its index is graph.audio_index.
        argv += ["-i", str(audio_path)]

        argv += [
            "-filter_complex", graph.filter_complex,
            "-map", graph.video_output_label,
            "-map", f"{graph.audio_index}:a",
            "-r", f"{self.fps}",
            "-vsync", "cfr",
            *self._quality_args(),
            "-pix_fmt", self.pix_fmt,
            "-c:a", "aac",
            "-b:a", self.audio_bitrate,
            "-shortest",
            str(output_path),
        ]
        return argv

    def _quality_args(self) -> list[str]:
        """ffmpeg argv fragment for video codec + quality + preset (codec-aware)."""
        return _quality_args_fn(self.codec_profile, self.crf, self.preset)

    # --- runner (mockable) ---------------------------------------------------

    def _run(self, argv: list[str]) -> tuple[int, str]:
        """Invoke ffmpeg and capture stderr. Monkeypatch in tests."""
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
        return int(proc.returncode), stderr


def _probe_duration(path: Path) -> float:
    """Best-effort duration probe via ffprobe; 0.0 if unavailable."""
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
    def _quality_args(self) -> list[str]:
        """ffmpeg argv fragment for video codec + quality + preset (codec-aware)."""
        return _quality_args_fn(self.codec_profile, self.crf, self.preset)
