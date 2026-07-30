"""Florence-2 client (via Replicate).

Florence-2 supplies *literal / spatial* understanding: a dense caption.
We call only the caption task ("Detailed Caption") to stay within Replicate's
free-tier rate limit (burst=1, 6 req/min).  Object-detection boxes were
previously a second call per image; LLaVA's symbolism output covers that
semantic gap sufficiently for our pipeline.

The parsing layer is defensive: if the model returns an unexpected shape, we
fall back to safe defaults rather than crash the batch.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..errors import APIError, CreativeEngineError
from .rate_limiter import REPLICATE_LIMITER

log = logging.getLogger(__name__)


@dataclass
class FlorenceResult:
    """Parsed Florence-2 output for a single image."""

    caption: str = ""
    objects: list[str] = field(default_factory=list)
    bounding_boxes: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "caption": self.caption,
            "objects": list(self.objects),
            "bounding_boxes": [list(b) for b in self.bounding_boxes],
        }


def parse_rate_limit_reset(err_msg: str, default_wait: float = 11.0) -> float:
    """Parse reset delay in seconds from 429 error message, e.g. 'resets in ~6s'."""
    match = re.search(r"resets?\s+in\s+~?(\d+)\s*s", err_msg, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0
    match_retry = re.search(r"retry\s+after\s+(\d+)", err_msg, re.IGNORECASE)
    if match_retry:
        return float(match_retry.group(1)) + 1.0
    return default_wait


class FlorenceClient:
    """Thin wrapper over the Replicate Florence-2 model.

    Parameters
    ----------
    model:
        Replicate model identifier, e.g. ``"microsoft/florence-2"``.
    api_token:
        Replicate API token.
    timeout_s:
        Per-call timeout forwarded to the Replicate client.
    """

    def __init__(
        self,
        model: str,
        api_token: str,
        timeout_s: float = 60.0,
        caption_task: str = "Detailed Caption",
        od_task: str = "Object Detection",
    ) -> None:
        self.model = model
        self.api_token = api_token
        self.timeout_s = timeout_s
        self.caption_task = caption_task or "Detailed Caption"
        self.od_task = od_task or "Object Detection"
        self._client: Optional[Any] = None

    # --- lazy client ---------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import replicate  # imported lazily so tests can monkeypatch
            except ImportError as exc:  # pragma: no cover
                raise CreativeEngineError("the 'replicate' package is not installed") from exc
            self._client = replicate.Client(api_token=self.api_token, timeout=self.timeout_s)
        return self._client

    # --- public API ----------------------------------------------------------

    def extract(self, image_path: Path | str) -> FlorenceResult:
        """Run Florence-2 caption-only on ``image_path``.

        Object Detection (OD) has been removed. Each extract() now makes
        exactly ONE Replicate prediction call, halving API usage compared to
        the previous caption+OD design.  LLaVA's symbolism/mood output
        provides sufficient semantic signal for the pipeline.

        Uses REPLICATE_LIMITER to guarantee >= 11 s between any two Replicate
        prediction calls (across both Florence and LLaVA clients).
        """
        image_path = Path(image_path)
        client = self._get_client()

        max_429_retries = 5
        for attempt in range(max_429_retries + 1):
            try:
                REPLICATE_LIMITER.acquire(context=f"Florence caption {image_path.name}")
                with image_path.open("rb") as fh:
                    caption_out = client.run(
                        self.model,
                        input={"image": fh, "task_input": self.caption_task},
                    )
                return self._parse(caption_out, None, image_path.name)
            except Exception as exc:
                err_str = str(exc)
                if any(k in err_str.lower() for k in ("429", "throttled", "too many requests")):
                    if attempt < max_429_retries:
                        wait_s = parse_rate_limit_reset(err_str, default_wait=REPLICATE_LIMITER.interval_s)
                        log.warning(
                            "Florence-2 429 for %s (attempt %d/%d). Sleeping %.1fs...",
                            image_path.name, attempt + 1, max_429_retries, wait_s,
                        )
                        time.sleep(wait_s)
                        continue
                    log.warning("Florence-2 rate limit retries exhausted for %s; using fallback", image_path.name)
                    return FlorenceResult(caption=image_path.stem)
                elif any(k in err_str.lower() for k in ("404", "not found", "credit")):
                    log.warning("Florence-2 unavailable for %s (%s); using fallback", image_path.name, err_str)
                    return FlorenceResult(caption=image_path.stem)
                raise APIError(f"Florence-2 call failed for {image_path.name}: {exc}") from exc

        return FlorenceResult(caption=image_path.stem)

    # --- parsing -------------------------------------------------------------

    @staticmethod
    def _parse(caption_out: Any, od_out: Any, name: str) -> FlorenceResult:
        caption = ""
        if isinstance(caption_out, str):
            caption = caption_out.strip()
        elif caption_out is None:
            caption = ""
        else:
            # Some Florence-2 variants return {"<CAPTION>": "..."} or {"text": "..."}.
            if isinstance(caption_out, dict):
                if "text" in caption_out:
                    caption = str(caption_out["text"]).strip()
                else:
                    caption = str(next(iter(caption_out.values()), "")).strip()
            else:
                log.warning("Florence caption had unexpected type %s for %s", type(caption_out), name)

        objects: list[str] = []
        boxes: list[list[float]] = []

        # OD output commonly looks like {"<OD>": {"bboxes": [[...], ...], "labels": [...]}}
        od_payload: Any = od_out
        if isinstance(od_payload, dict):
            for k in ("<OD>", "<OBJECT_DETECTION>", "Object Detection", "object_detection"):
                if k in od_payload:
                    od_payload = od_payload[k]
                    break

        if isinstance(od_payload, dict):
            labels = od_payload.get("labels") or []
            bboxes = od_payload.get("bboxes") or []
            if not isinstance(labels, list) or not isinstance(bboxes, list):
                raise ParseError(f"Florence OD labels/bboxes not lists for {name}")
            if len(labels) != len(bboxes):
                log.warning(
                    "Florence OD label/box count mismatch (%s vs %s) for %s",
                    len(labels), len(bboxes), name,
                )
            for label, box in zip(labels, bboxes):
                if not isinstance(label, str):
                    continue
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    log.warning("Skipping malformed box for label %r in %s", label, name)
                    continue
                try:
                    x1, y1, x2, y2 = (float(c) for c in box)
                except (TypeError, ValueError):
                    log.warning("Non-numeric box coords for label %r in %s", label, name)
                    continue
                # Normalize to [0, 1]. Florence-2 emits coordinates in [0, 1000];
                # if any coord exceeds 1.5 we assume that scale and divide by 1000.
                coords = (x1, y1, x2, y2)
                if max(coords) > 1.5:
                    log.debug("Box values look unscaled (>1.5) for %s; dividing by 1000", name)
                    coords = tuple(v / 1000.0 for v in coords)
                # Clamp each coord into [0, 1].
                x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in coords)
                objects.append(label.strip())
                boxes.append([x1, y1, x2, y2])

        return FlorenceResult(caption=caption, objects=objects, bounding_boxes=boxes)
