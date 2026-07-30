"""Stage-1 extraction pipeline.

Sequential, cache-each-image orchestrator:

    for image in images_dir:
        image_id = sha1(bytes)
        if cache.exists(image_id): continue   # never re-pay
        metadata = extract(florence + llava + embedding)
        cache.upsert(metadata)                # immediate, crash-safe

One image's failure never aborts the batch: failures are counted and reported.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, UnidentifiedImageError
import numpy as np

from .cache import MetadataCache
from .errors import CreativeEngineError
from .models import ImageMetadata
from .vision.embeddings import Embedder, embedding_to_hex
from .vision.florence import FlorenceClient
from .vision.llava import LLaVAClient

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class PipelineStats:
    """Aggregated counts for a single pipeline run."""

    total: int = 0
    new: int = 0
    cached: int = 0
    failed: int = 0
    failed_files: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.failed_files is None:
            self.failed_files = []


import time

class ExtractionPipeline:
    """Coordinates vision clients + embedder + cache."""

    def __init__(
        self,
        florence: FlorenceClient,
        llava: LLaVAClient,
        embedder: Embedder,
        cache: MetadataCache,
        max_images: Optional[int] = None,
        request_delay_s: float = 11.0,
    ) -> None:
        self.florence = florence
        self.llava = llava
        self.embedder = embedder
        self.cache = cache
        self.max_images = max_images
        self.request_delay_s = float(request_delay_s)

    # --- entrypoints ---------------------------------------------------------

    def run(
        self,
        images_dir: Path | str,
        progress_callback: Optional[Any] = None,
    ) -> PipelineStats:
        """Process all supported images in ``images_dir`` (non-recursive)."""
        images_dir = Path(images_dir)
        if not images_dir.is_dir():
            raise CreativeEngineError(f"images_dir is not a directory: {images_dir}")

        files = sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if self.max_images is not None:
            files = files[: self.max_images]

        stats = self._run_batched(files, progress_callback=progress_callback)
        return stats

    def _run_batched(
        self,
        files: list[Path],
        progress_callback: Optional[Any] = None,
    ) -> PipelineStats:
        """Two-pass run: vision-extract everything, then one batched embed call.

        Preserves fail-soft semantics: any image that fails vision extraction is
        counted as failed and excluded from the embed batch. Cache hits are
        detected *before* any vision call so already-paid images never reach the
        embedder.
        """
        stats = PipelineStats(total=len(files))

        # Pass 1: Filter out cache hits and run vision extraction on the rest
        pending_files = []

        for idx, image_path in enumerate(files, start=1):
            if progress_callback is not None:
                try:
                    progress_callback(idx, len(files), image_path.name)
                except Exception:
                    pass
            try:
                raw = image_path.read_bytes()
                image_id = self._sha1_hex(raw)

                if self.cache.exists(image_id):
                    log.debug("Cache hit for %s (%s)", image_path.name, image_id)
                    stats.cached += 1
                    log.info(
                        "[%d/%d] %s -> new=%d cached=%d failed=%d (cache hit, skipping API)",
                        idx, stats.total, image_path.name, stats.new, stats.cached, stats.failed,
                    )
                    continue

                width, height, exif, color_hist = self._read_image_meta(image_path)
                florence_res = self.florence.extract(image_path)
                llava_res = self.llava.extract(image_path)

                embed_text = " ".join(
                    part for part in (
                        florence_res.caption,
                        llava_res.symbolism,
                        llava_res.mood,
                    ) if part
                )

                pending_files.append((image_path, image_id, width, height, exif, color_hist, florence_res, llava_res, embed_text))

                if self.request_delay_s > 0 and idx < len(files):
                    time.sleep(self.request_delay_s)

            except Exception as exc:  # noqa: BLE001 - fail-soft by design
                stats.failed += 1
                stats.failed_files.append(image_path.name)
                log.error("Failed vision extraction for %s: %s", image_path.name, exc)
                log.info(
                    "[%d/%d] %s -> new=%d cached=%d failed=%d",
                    idx, stats.total, image_path.name, stats.new, stats.cached, stats.failed,
                )

        if not pending_files:
            return stats

        # Pass 2: Batch embed all pending texts in one go
        embed_texts = [item[8] for item in pending_files]
        try:
            embeddings = self.embedder.embed_many(embed_texts)
        except Exception as exc:
            # Embedder failed -> fail all pending files
            for image_path, _, _, _, _, _, _, _, _ in pending_files:
                stats.failed += 1
                stats.failed_files.append(image_path.name)
                log.error("Failed to embed batch due to embedder error: %s (failed file: %s)", exc, image_path.name)
            return stats

        # Save to cache with local visual composition features
        from .vision.features import VisualFeatureExtractor
        extractor = VisualFeatureExtractor()

        for idx, (image_path, image_id, width, height, exif, color_hist, florence_res, llava_res, _) in enumerate(pending_files):
            try:
                emb_bytes = embeddings[idx]
                embedding_hex = embedding_to_hex(emb_bytes)

                features = extractor.extract(image_path, metadata_so_far={
                    "florence_caption": florence_res.caption,
                    "bounding_boxes": florence_res.bounding_boxes,
                    "llava_tension": llava_res.tension,
                    "llava_symbolism": llava_res.symbolism,
                })

                meta = ImageMetadata(
                    image_id=image_id,
                    file_path=str(image_path),
                    width=width,
                    height=height,
                    exif=exif,
                    color_histogram=color_hist,
                    florence_caption=florence_res.caption,
                    objects=florence_res.objects,
                    bounding_boxes=florence_res.bounding_boxes,
                    llava_mood=llava_res.mood,
                    llava_tension=llava_res.tension,
                    llava_symbolism=llava_res.symbolism,
                    embedding_hex=embedding_hex,
                    **features
                )

                self.cache.upsert(meta)
                stats.new += 1
                log.info("Saved to cache: %s (id=%s)", image_path.name, image_id[:8])
            except Exception as exc:
                stats.failed += 1
                stats.failed_files.append(image_path.name)
                log.error("Failed caching/feature extraction for %s: %s", image_path.name, exc)

        return stats

    # --- per-image -----------------------------------------------------------

    def _process_one(self, image_path: Path) -> bool:
        """Extract + cache a single image.

        Returns
        -------
        True if a new record was written, False if the image was already cached.
        """
        raw = image_path.read_bytes()
        image_id = self._sha1_hex(raw)

        if self.cache.exists(image_id):
            log.debug("Cache hit for %s (%s)", image_path.name, image_id)
            return False

        width, height, exif, color_hist = self._read_image_meta(image_path)

        florence_res = self.florence.extract(image_path)
        llava_res = self.llava.extract(image_path)

        embed_text = " ".join(
            part for part in (
                florence_res.caption,
                llava_res.symbolism,
                llava_res.mood,
            ) if part
        )
        emb_bytes = self.embedder.embed(embed_text)
        embedding_hex = embedding_to_hex(emb_bytes)

        # Local Visual Composition & Directing features (Phase 2)
        from .vision.features import VisualFeatureExtractor
        extractor = VisualFeatureExtractor()
        features = extractor.extract(image_path, metadata_so_far={
            "florence_caption": florence_res.caption,
            "bounding_boxes": florence_res.bounding_boxes,
            "llava_tension": llava_res.tension,
            "llava_symbolism": llava_res.symbolism,
        })

        meta = ImageMetadata(
            image_id=image_id,
            file_path=str(image_path),
            width=width,
            height=height,
            exif=exif,
            color_histogram=color_hist,
            florence_caption=florence_res.caption,
            objects=florence_res.objects,
            bounding_boxes=florence_res.bounding_boxes,
            llava_mood=llava_res.mood,
            llava_tension=llava_res.tension,
            llava_symbolism=llava_res.symbolism,
            embedding_hex=embedding_hex,
            **features
        )

        self.cache.upsert(meta)
        return True

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _sha1_hex(data: bytes) -> str:
        return hashlib.sha1(data).hexdigest()

    @staticmethod
    def _read_image_meta(image_path: Path) -> tuple[int, int, Optional[dict], list[float]]:
        """Return (width, height, exif_dict_or_None, color_histogram) using Pillow.

        EXIF missing or unreadable -> returns ``None`` for exif (never raises).
        """
        try:
            with Image.open(image_path) as im:
                width, height = im.size
                exif_dict: Optional[dict] = None
                try:
                    raw_exif = im.getexif()
                    if raw_exif:
                        # Keep it tiny: only a few selected tags.
                        exif_dict = {
                            str(k): str(v)
                            for k, v in raw_exif.items()
                            if v is not None
                        }
                        if not exif_dict:
                            exif_dict = None
                except Exception:  # pragma: no cover - pillow exif edge cases
                    exif_dict = None
                
                # True 16-bin color histogram in HSV space
                color_hist = ExtractionPipeline._extract_color_histogram(im)
                return int(width), int(height), exif_dict, color_hist
        except (UnidentifiedImageError, OSError) as exc:
            raise CreativeEngineError(f"Cannot read image {image_path.name}: {exc}") from exc

    @staticmethod
    def _extract_color_histogram(im: Image.Image, bins: int = 16) -> list[float]:
        """Extract normalized 16-bin color histogram (HSV space) via Pillow."""
        try:
            hsv = im.convert("HSV")
            h_channel = hsv.getchannel("H")
            h_data = list(h_channel.getdata())
            counts = [0] * bins
            scale = 256.0 / bins
            for val in h_data:
                idx = min(bins - 1, int(val / scale))
                counts[idx] += 1
            total = float(len(h_data))
            if total > 0:
                return [round(c / total, 6) for c in counts]
        except Exception:
            pass
        return [0.0] * bins


def iter_supported(images_dir: Path | str) -> Iterable[Path]:
    """Yield supported image files in ``images_dir`` (non-recursive, sorted)."""
    images_dir = Path(images_dir)
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            yield p
