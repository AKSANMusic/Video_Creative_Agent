"""Florence-2 client (via Replicate).

Florence-2 supplies *literal / spatial* understanding: a dense caption plus
object-detection boxes. We invoke two tasks (``<CAPTION>`` and ``<OD>``) and
parse the outputs into a small dict consumed by the pipeline.

The parsing layer is defensive: if the model returns an unexpected shape, we
fall back to safe defaults rather than crash the batch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..errors import APIError, ParseError
from ..retries import api_retry

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

    def __init__(self, model: str, api_token: str, timeout_s: float = 60.0) -> None:
        self.model = model
        self.api_token = api_token
        self.timeout_s = timeout_s
        self._client: Optional[Any] = None

    # --- lazy client ---------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import replicate  # imported lazily so tests can monkeypatch
            except ImportError as exc:  # pragma: no cover
                raise APIError("the 'replicate' package is not installed") from exc
            self._client = replicate.Client(api_token=self.api_token, timeout=self.timeout_s)
        return self._client

    # --- public API ----------------------------------------------------------

    @api_retry
    def extract(self, image_path: Path | str) -> FlorenceResult:
        """Run Florence-2 caption + object detection on ``image_path``."""
        image_path = Path(image_path)
        client = self._get_client()

        try:
            with image_path.open("rb") as fh:
                # Replicate accepts a file handle for local uploads.
                caption_out = client.run(
                    self.model,
                    input={"image": fh, "task": "<CAPTION>"},
                )
            with image_path.open("rb") as fh:
                od_out = client.run(
                    self.model,
                    input={"image": fh, "task": "<OD>"},
                )
        except Exception as exc:
            # Wrap any non-retryable / unexpected error uniformly.
            raise APIError(f"Florence-2 call failed for {image_path.name}: {exc}") from exc

        return self._parse(caption_out, od_out, image_path.name)

    # --- parsing -------------------------------------------------------------

    @staticmethod
    def _parse(caption_out: Any, od_out: Any, name: str) -> FlorenceResult:
        caption = ""
        if isinstance(caption_out, str):
            caption = caption_out.strip()
        elif caption_out is None:
            caption = ""
        else:
            # Some Florence-2 variants return {"<CAPTION>": "..."}.
            if isinstance(caption_out, dict):
                caption = str(next(iter(caption_out.values()), "")).strip()
            else:
                log.warning("Florence caption had unexpected type %s for %s", type(caption_out), name)

        objects: list[str] = []
        boxes: list[list[float]] = []

        # OD output commonly looks like {"<OD>": {"bboxes": [[...], ...], "labels": [...]}}
        od_payload: Any = od_out
        if isinstance(od_payload, dict) and "<OD>" in od_payload:
            od_payload = od_payload["<OD>"]

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
