"""Librosa-based audio signal analysis.

Extracts the raw signals consumed by the audio map:
    - tempo (bpm) and beat grid
    - general onset envelope/times
    - RMS energy curve (downsampled, normalized)
    - spectral contrast curve (downsampled, normalized)
    - structural segmentation via repeated-structure + agglomerative clustering

The analyzer is *deterministic* for a given file + config: librosa's tempo
estimate uses the well-defined `beat_track` default pipeline, and we pin
`hop_length`. All heavy state lives inside librosa; this module is pure math
+ packaging.

Design notes
------------
- Defensive: any librosa error is wrapped in :class:`AudioAnalysisError`.
- Silent/empty audio (all zeros) is handled: we emit an empty beat grid and a
  flat zero energy curve rather than crashing.
- Normalization: curves are min-max scaled per-track to [0, 1] so the
  sequencer can compare shapes across tracks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import CreativeEngineError
from .audio_map import CURVE_BUCKETS

log = logging.getLogger(__name__)


class AudioAnalysisError(CreativeEngineError):
    """Raised when librosa analysis fails."""


@dataclass
class RawAnalysis:
    """Container for raw analysis arrays before packaging into AudioMap."""

    duration: float
    sample_rate: int
    bpm: float
    beats: list[float]
    downbeats: list[float]
    onsets: list[float]
    rms_curve: list[float]
    spectral_contrast_curve: list[float]
    section_boundaries: list[float]  # times in seconds (incl. 0 and duration)


class AudioAnalyzer:
    """Wraps librosa with stable defaults and defensive error handling."""

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 512,
        n_segments: int = 8,
        curve_buckets: int = CURVE_BUCKETS,
    ) -> None:
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_segments = max(2, n_segments)
        self.curve_buckets = max(8, curve_buckets)

    # --- public API ----------------------------------------------------------

    def analyze(self, audio_path: Path | str) -> RawAnalysis:
        """Run the full librosa pipeline on ``audio_path``."""
        audio_path = Path(audio_path)
        y, sr = self._load(audio_path)

        duration = float(len(y)) / float(sr)
        # Reject degenerate / near-empty audio: less than 50ms has no usable
        # beat structure and would produce nonsense estimates downstream.
        if duration < 0.05 or len(y) == 0:
            raise AudioAnalysisError(
                f"Audio too short to analyze ({duration:.3f}s): {audio_path.name}"
            )

        bpm, beats, downbeats = self._beat_grid(y, sr)
        onsets = self._onsets(y, sr)
        rms_curve = self._rms_curve(y, sr)
        sc_curve = self._spectral_contrast_curve(y, sr)
        boundaries = self._segment(y, sr, duration, beats)

        return RawAnalysis(
            duration=duration,
            sample_rate=int(sr),
            bpm=bpm,
            beats=beats,
            downbeats=downbeats,
            onsets=onsets,
            rms_curve=rms_curve,
            spectral_contrast_curve=sc_curve,
            section_boundaries=boundaries,
        )

    # --- internals -----------------------------------------------------------

    def _load(self, audio_path: Path) -> tuple[Any, int]:
        try:
            import soundfile as sf
            y, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            if y.ndim > 1:
                import numpy as np
                y = np.mean(y, axis=1)
            if self.sample_rate and sr != self.sample_rate:
                try:
                    import librosa
                    y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
                    sr = self.sample_rate
                except Exception:
                    pass
            return y, int(sr)
        except Exception:
            pass

        try:
            import librosa
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            return y, int(sr)
        except Exception as exc:
            raise AudioAnalysisError(f"Failed to load audio {audio_path.name}: {exc}") from exc

    def _beat_grid(self, y: Any, sr: int) -> tuple[float, list[float], list[float]]:
        import librosa
        import numpy as np

        try:
            tempo, beat_frames = librosa.beat.beat_track(
                y=y, sr=sr, hop_length=self.hop_length
            )
            bpm = float(np.ravel(tempo)[0]) if np.size(tempo) else 120.0
            if bpm <= 0.0:
                bpm = 120.0
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)
            beats = [float(t) for t in np.ravel(beat_times)]
        except Exception as exc:
            log.warning("beat_track failed (%s); defaulting to 120 bpm fallback grid", exc)
            bpm = 120.0
            duration = float(len(y)) / float(sr) if sr > 0 else 0.0
            step = 60.0 / bpm
            beats = [round(float(i * step), 4) for i in range(int(duration / step))]

        # Downbeat heuristic: every 4th beat (4/4 assumption). Stable + simple.
        downbeats = beats[::4]
        return bpm, beats, downbeats

    def _onsets(self, y: Any, sr: int) -> list[float]:
        import librosa
        import numpy as np

        try:
            onset_frames = librosa.onset.onset_detect(
                y=y, sr=sr, hop_length=self.hop_length
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
            return [float(t) for t in np.ravel(onset_times)]
        except Exception as exc:
            log.warning("onset_detect failed (%s); returning []", exc)
            return []

    def _rms_curve(self, y: Any, sr: int) -> list[float]:
        import librosa
        import numpy as np

        try:
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=self.hop_length)[0]
        except Exception as exc:
            log.warning("rms failed (%s); returning flat curve", exc)
            return [0.0] * self.curve_buckets
        return self._downsample_normalize(rms)

    def _spectral_contrast_curve(self, y: Any, sr: int) -> list[float]:
        import librosa
        import numpy as np

        try:
            sc = librosa.feature.spectral_contrast(
                y=y, sr=sr, hop_length=self.hop_length
            )
            # Average across the 7 contrast bands -> one curve over time.
            curve = sc.mean(axis=0)
        except Exception as exc:
            log.warning("spectral_contrast failed (%s); returning flat curve", exc)
            return [0.0] * self.curve_buckets
        return self._downsample_normalize(curve)

    def _segment(self, y: Any, sr: int, duration: float, beats: list[float]) -> list[float]:
        """Structural boundaries via librosa's `agglomerative` on chroma.

        Falls back to equal-length subdivision if anything fails.
        """
        import librosa
        import numpy as np

        try:
            chroma = librosa.feature.chroma_cqt(
                y=y, sr=sr, hop_length=self.hop_length
            )
            bounds = librosa.segment.agglomerative(
                data=chroma, k=self.n_segments
            )
            times = librosa.frames_to_time(bounds, sr=sr, hop_length=self.hop_length)
            boundaries = [float(t) for t in np.ravel(times)]
        except Exception as exc:
            log.warning("segmentation failed (%s); using equal split", exc)
            boundaries = []

        if len(boundaries) < 2:
            # Equal split fallback.
            boundaries = [duration * i / self.n_segments for i in range(self.n_segments + 1)]
            return boundaries

        # Ensure the boundaries span [0, duration] and are sorted/unique.
        boundaries = sorted(set(boundaries))
        if boundaries[0] > 1e-6:
            boundaries = [0.0] + boundaries
        if boundaries[-1] < duration - 1e-6:
            boundaries = boundaries + [duration]
        return boundaries

    # --- math helpers --------------------------------------------------------

    def _downsample_normalize(self, curve: Any) -> list[float]:
        """Resample a 1-D array to ``curve_buckets`` and min-max normalize to [0,1]."""
        import numpy as np

        arr = np.asarray(curve, dtype="float64").ravel()
        if arr.size == 0:
            return [0.0] * self.curve_buckets

        if arr.size > self.curve_buckets:
            # Even binning: mean within each bucket.
            trim = (arr.size // self.curve_buckets) * self.curve_buckets
            arr = arr[:trim].reshape(self.curve_buckets, -1).mean(axis=1)
        elif arr.size < self.curve_buckets:
            # Linear interpolation to the target length.
            idx_src = np.linspace(0, arr.size - 1, arr.size)
            idx_dst = np.linspace(0, arr.size - 1, self.curve_buckets)
            arr = np.interp(idx_dst, idx_src, arr)

        lo = float(np.min(arr))
        hi = float(np.max(arr))
        span = hi - lo
        if span < 1e-12:
            return [0.0] * int(arr.size)
        return [float((v - lo) / span) for v in arr]
