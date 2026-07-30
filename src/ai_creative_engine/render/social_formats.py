"""Social media aspect-ratio formatter.

Converts a source video to standard social-media dimensions using FFmpeg.
Supports two adaptation strategies per target format:

    - **Center Crop**:  Fill the target frame completely by cropping the
      source from its center.  No black bars, no stretch.
    - **Letterbox / Pillarbox (Padding)**:  Fit the entire source inside
      the target frame, padding the remaining space with a solid colour
      (default: deep black ``#0a0a0c``).

Target presets
--------------
==========  ========  ===========  ====================================
Name        Ratio     Resolution   Platforms
==========  ========  ===========  ====================================
vertical    9:16      1080×1920    Instagram Reels, TikTok, YT Shorts
portrait    4:5       1080×1350    Instagram Feed
square      1:1       1080×1080    General Feed
landscape   16:9      1920×1080    YouTube, Desktop Web
==========  ========  ===========  ====================================

Usage
-----
>>> from ai_creative_engine.render.social_formats import SocialFormatter
>>> fmt = SocialFormatter()
>>> result = fmt.export("input.mp4", "vertical", strategy="crop")
>>> print(result.output_path)
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FFmpeg / FFprobe binary resolution
# ---------------------------------------------------------------------------
_PROJECT_DIR = Path(__file__).resolve().parents[3]  # …/Video_Creative_Agent

def _find_binary(name: str) -> str:
    """Return an absolute path to *name* (.exe on Windows).

    Search order:
      1. The project root (where we copied ffmpeg.exe earlier).
      2. ``shutil.which`` (system PATH).
      3. Fall back to bare name and let subprocess raise on failure.
    """
    import shutil

    exe = f"{name}.exe" if os.name == "nt" else name
    local = _PROJECT_DIR / exe
    if local.is_file():
        return str(local)
    found = shutil.which(name)
    if found:
        return found
    return name  # best-effort fallback


FFMPEG_BIN  = _find_binary("ffmpeg")
FFPROBE_BIN = _find_binary("ffprobe")


# ---------------------------------------------------------------------------
# Target presets
# ---------------------------------------------------------------------------

class Strategy(str, Enum):
    """How to adapt the source to the target aspect ratio."""
    CROP    = "crop"       # Centre-crop to fill
    LETTERBOX = "letterbox"  # Fit + pad


@dataclass(frozen=True)
class FormatPreset:
    """A named social-media target."""
    name:   str
    width:  int
    height: int
    label:  str   # human-readable description

    @property
    def aspect(self) -> float:
        return self.width / self.height


# Canonical presets (order matches the UI dropdown).
PRESETS: dict[str, FormatPreset] = {
    "vertical":  FormatPreset("vertical",  1080, 1920, "Vertical 9:16 — Reels / TikTok / Shorts"),
    "portrait":  FormatPreset("portrait",  1080, 1350, "Portrait 4:5 — Instagram Feed"),
    "square":    FormatPreset("square",    1080, 1080, "Square 1:1 — General Feed"),
    "landscape": FormatPreset("landscape", 1920, 1080, "Landscape 16:9 — YouTube / Desktop"),
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FormatResult:
    """Outcome of a social-format export."""
    output_path: Path
    preset:      FormatPreset
    strategy:    Strategy
    duration_s:  float  = 0.0
    succeeded:   bool   = False
    stderr_tail: str    = ""


# ---------------------------------------------------------------------------
# Core formatter
# ---------------------------------------------------------------------------

class SocialFormatter:
    """Reformat a video for social-media aspect ratios.

    Parameters
    ----------
    pad_colour:
        Hex colour for the letterbox bars (without leading ``#``).
        Default is ``0a0a0c`` (the app's deep-black background).
    video_codec:
        FFmpeg video codec.  ``libx264`` is safe everywhere.
    crf:
        Constant-Rate Factor.  Lower = higher quality.  18 is visually
        lossless for libx264.
    preset:
        Encoder speed/quality trade-off.
    audio_codec:
        ``copy`` re-muxes without re-encoding audio (fastest).
    """

    def __init__(
        self,
        pad_colour:  str = "0a0a0c",
        video_codec: str = "libx264",
        crf:         int = 18,
        preset:      str = "medium",
        audio_codec: str = "aac",
        audio_bitrate: str = "192k",
    ) -> None:
        self.pad_colour    = pad_colour
        self.video_codec   = video_codec
        self.crf           = crf
        self.preset        = preset
        self.audio_codec   = audio_codec
        self.audio_bitrate = audio_bitrate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(
        self,
        input_path:  str | Path,
        preset_name: str,
        strategy:    str | Strategy = Strategy.CROP,
        output_dir:  Optional[str | Path] = None,
    ) -> FormatResult:
        """Export *input_path* to the named preset with the given strategy.

        Parameters
        ----------
        input_path:
            Path to the source MP4 / video file.
        preset_name:
            One of ``vertical``, ``portrait``, ``square``, ``landscape``.
        strategy:
            ``"crop"`` or ``"letterbox"``.
        output_dir:
            Directory for the output file.  Defaults to the same directory
            as *input_path*.

        Returns
        -------
        FormatResult
        """
        input_path = Path(input_path)
        if not input_path.is_file():
            raise FileNotFoundError(f"Source video not found: {input_path}")

        preset = PRESETS.get(preset_name.lower())
        if preset is None:
            valid = ", ".join(PRESETS)
            raise ValueError(f"Unknown preset {preset_name!r}.  Choose from: {valid}")

        strat = Strategy(strategy) if isinstance(strategy, str) else strategy

        # Build output filename.
        out_dir = Path(output_dir) if output_dir else input_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix  = f"_{preset.name}_{strat.value}"
        out_path = out_dir / f"{input_path.stem}{suffix}.mp4"

        # Probe source dimensions.
        src_w, src_h = self._probe_dimensions(input_path)
        log.info(
            "Social export: %s  src=%dx%d  target=%dx%d  strategy=%s",
            preset.name, src_w, src_h, preset.width, preset.height, strat.value,
        )

        # Build the FFmpeg filter.
        vf = self._build_filter(src_w, src_h, preset, strat)

        # Assemble argv.
        argv = [
            FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(input_path),
            "-vf", vf,
            "-c:v", self.video_codec,
            "-crf", str(self.crf),
            "-preset", self.preset,
            "-c:a", self.audio_codec,
            "-b:a", self.audio_bitrate,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path),
        ]

        result = FormatResult(
            output_path=out_path,
            preset=preset,
            strategy=strat,
        )

        log.info("Running social-format encode: %s", " ".join(argv[:6]) + " ...")
        try:
            proc = subprocess.run(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except FileNotFoundError as exc:
            result.stderr_tail = str(exc)
            log.error("FFmpeg binary not found: %s", exc)
            return result

        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
        result.stderr_tail = stderr[-4000:]
        result.succeeded   = proc.returncode == 0 and out_path.is_file()

        if result.succeeded:
            result.duration_s = self._probe_duration(out_path)
            log.info(
                "✓ Social export complete: %s  (%.1fs, %dx%d)",
                out_path.name, result.duration_s, preset.width, preset.height,
            )
        else:
            log.error(
                "Social export FAILED (code %d): %s",
                proc.returncode, result.stderr_tail[:500],
            )

        return result

    def export_all(
        self,
        input_path: str | Path,
        strategy:   str | Strategy = Strategy.CROP,
        output_dir: Optional[str | Path] = None,
    ) -> list[FormatResult]:
        """Export to **all four** social presets in one call."""
        results = []
        for name in PRESETS:
            results.append(self.export(input_path, name, strategy, output_dir))
        return results

    # ------------------------------------------------------------------
    # Filter construction
    # ------------------------------------------------------------------

    def _build_filter(
        self,
        src_w: int, src_h: int,
        preset: FormatPreset,
        strategy: Strategy,
    ) -> str:
        """Return the ``-vf`` filter string for the given strategy."""
        tw, th = preset.width, preset.height

        if strategy is Strategy.CROP:
            return self._filter_center_crop(src_w, src_h, tw, th)
        else:
            return self._filter_letterbox(src_w, src_h, tw, th)

    @staticmethod
    def _filter_center_crop(src_w: int, src_h: int, tw: int, th: int) -> str:
        """Scale up to cover the target, then centre-crop to exact dims.

        The scale step ensures the *smaller* dimension matches or exceeds
        the target, so the crop never has insufficient pixels.
        """
        target_aspect = tw / th
        src_aspect    = src_w / src_h

        if src_aspect > target_aspect:
            # Source is wider → match height, crop width.
            scale = f"scale=-2:{th}"
        else:
            # Source is taller (or equal) → match width, crop height.
            scale = f"scale={tw}:-2"

        crop = f"crop={tw}:{th}"
        return f"{scale},{crop}"

    def _filter_letterbox(self, src_w: int, src_h: int, tw: int, th: int) -> str:
        """Scale to fit inside the target, then pad with *pad_colour*.

        The scale step ensures the *larger* dimension matches the target
        (maintaining aspect ratio), and ``pad`` centres the result inside
        a ``tw × th`` canvas.
        """
        # scale2ref is overkill; simple math is easier to debug.
        target_aspect = tw / th
        src_aspect    = src_w / src_h

        if src_aspect > target_aspect:
            # Source is wider → width-limited.
            scale = f"scale={tw}:-2"
        else:
            # Source is taller → height-limited.
            scale = f"scale=-2:{th}"

        # pad centres the (possibly smaller) scaled output on a tw×th canvas.
        pad = (
            f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:"
            f"color=#{self.pad_colour}"
        )
        return f"{scale},{pad}"

    # ------------------------------------------------------------------
    # FFprobe helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_dimensions(path: Path) -> tuple[int, int]:
        """Return (width, height) of the first video stream."""
        try:
            proc = subprocess.run(
                [
                    FFPROBE_BIN, "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=p=0:s=x",
                    str(path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode == 0:
                parts = proc.stdout.decode().strip().split("x")
                return int(parts[0]), int(parts[1])
        except Exception as exc:
            log.warning("ffprobe dimensions failed: %s", exc)
        # Safe fallback — the caller's scale filter will handle it.
        return 1280, 720

    @staticmethod
    def _probe_duration(path: Path) -> float:
        """Return duration in seconds (0.0 on failure)."""
        try:
            proc = subprocess.run(
                [
                    FFPROBE_BIN, "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode == 0:
                return float(proc.stdout.decode().strip() or 0.0)
        except (FileNotFoundError, ValueError):
            pass
        return 0.0
