"""LLaVA-NeXT client (via Replicate).

LLaVA-NeXT supplies *artistic* understanding: mood, narrative tension, and
symbolism. We prompt it to return strict JSON and parse defensively, falling
back to safe defaults (``mood='unknown'``, ``tension=0.5``, ``symbolism=''``)
on any parse failure.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..errors import APIError, ParseError
from ..retries import api_retry

log = logging.getLogger(__name__)

# Prompt engineered for a small, parseable JSON object.
_LLaVA_PROMPT = (
    "Analyze this image as a film director. Return ONLY a JSON object with keys "
    '"mood" (one short word), "tension" (a float between 0.0 and 1.0 where 0.0 is '
    'calm and 1.0 is maximum tension), and "symbolism" (one short sentence). '
    "Do not include any text outside the JSON object."
)

# Defaults used when the model output cannot be parsed.
DEFAULT_MOOD = "unknown"
DEFAULT_TENSION = 0.5
DEFAULT_SYMBOLISM = ""


@dataclass
class LLaVAResult:
    """Parsed LLaVA-NeXT artistic analysis."""

    mood: str = DEFAULT_MOOD
    tension: float = DEFAULT_TENSION
    symbolism: str = DEFAULT_SYMBOLISM

    def to_dict(self) -> dict[str, Any]:
        return {"mood": self.mood, "tension": self.tension, "symbolism": self.symbolism}


class LLaVAClient:
    """Thin wrapper over the Replicate LLaVA-NeXT model."""

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
    def extract(self, image_path: Path | str) -> LLaVAResult:
        """Run LLaVA-NeXT artistic analysis on ``image_path``."""
        image_path = Path(image_path)
        client = self._get_client()

        try:
            with image_path.open("rb") as fh:
                raw_out = client.run(
                    self.model,
                    input={"image": fh, "prompt": _LLaVA_PROMPT},
                )
        except Exception as exc:
            raise APIError(f"LLaVA call failed for {image_path.name}: {exc}") from exc

        text = self._to_text(raw_out)
        return self._parse(text, image_path.name)

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _to_text(raw_out: Any) -> str:
        """Normalize Replicate output (which may be a generator/str) to text."""
        if raw_out is None:
            return ""
        if isinstance(raw_out, str):
            return raw_out.strip()
        # Some LLaVA deployments stream a list of string chunks.
        try:
            return "".join(str(chunk) for chunk in raw_out).strip()
        except TypeError:
            return str(raw_out).strip()

    @staticmethod
    def _parse(text: str, name: str) -> LLaVAResult:
        """Parse model text into :class:`LLaVAResult` with safe defaults."""
        if not text:
            log.warning("Empty LLaVA output for %s; using defaults", name)
            return LLaVAResult()

        # Try direct JSON first.
        data: Optional[dict[str, Any]] = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: extract the first {...} block.
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = None

        if not isinstance(data, dict):
            log.warning("Could not parse LLaVA JSON for %s (got %r); using defaults", name, text[:120])
            return LLaVAResult()

        mood = str(data.get("mood", DEFAULT_MOOD)).strip() or DEFAULT_MOOD

        try:
            tension = float(data.get("tension", DEFAULT_TENSION))
        except (TypeError, ValueError):
            log.warning("Non-numeric tension for %s; defaulting", name)
            tension = DEFAULT_TENSION
        # Clamp to [0, 1].
        tension = min(max(tension, 0.0), 1.0)

        symbolism = str(data.get("symbolism", DEFAULT_SYMBOLISM)).strip()

        return LLaVAResult(mood=mood, tension=tension, symbolism=symbolism)
