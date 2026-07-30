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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..errors import APIError, ParseError
from .rate_limiter import REPLICATE_LIMITER

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


def parse_rate_limit_reset(err_msg: str, default_wait: float = 11.0) -> float:
    """Parse reset delay in seconds from 429 error message, e.g. 'resets in ~6s'."""
    match = re.search(r"resets?\s+in\s+~?(\d+)\s*s", err_msg, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0
    match_retry = re.search(r"retry\s+after\s+(\d+)", err_msg, re.IGNORECASE)
    if match_retry:
        return float(match_retry.group(1)) + 1.0
    return default_wait


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
                raise CreativeEngineError("the 'replicate' package is not installed") from exc
            self._client = replicate.Client(api_token=self.api_token, timeout=self.timeout_s)
        return self._client

    # --- public API ----------------------------------------------------------

    def extract(self, image_path: Path | str) -> LLaVAResult:
        """Run LLaVA-NeXT artistic analysis on ``image_path``.

        Uses REPLICATE_LIMITER to guarantee >= 11 s between Replicate calls
        (shared with FlorenceClient so the global rate is enforced correctly).
        """
        image_path = Path(image_path)
        client = self._get_client()

        max_429_retries = 5
        for attempt in range(max_429_retries + 1):
            try:
                REPLICATE_LIMITER.acquire(context=f"LLaVA {image_path.name}")
                with image_path.open("rb") as fh:
                    raw_out = client.run(
                        self.model,
                        input={"image": fh, "prompt": _LLaVA_PROMPT},
                    )
                text = self._to_text(raw_out)
                return self._parse(text, image_path.name)
            except Exception as exc:
                err_str = str(exc)
                if any(k in err_str.lower() for k in ("429", "throttled", "too many requests")):
                    if attempt < max_429_retries:
                        wait_s = parse_rate_limit_reset(err_str, default_wait=REPLICATE_LIMITER.interval_s)
                        log.warning(
                            "LLaVA 429 for %s (attempt %d/%d). Sleeping %.1fs...",
                            image_path.name, attempt + 1, max_429_retries, wait_s,
                        )
                        time.sleep(wait_s)
                        continue
                    log.warning("LLaVA rate limit retries exhausted for %s; using fallback", image_path.name)
                    return LLaVAResult(mood="neutral", tension=0.5, symbolism="")
                elif any(k in err_str.lower() for k in ("404", "not found", "credit")):
                    log.warning("LLaVA unavailable for %s (%s); using fallback", image_path.name, err_str)
                    return LLaVAResult(mood="neutral", tension=0.5, symbolism="")
                raise APIError(f"LLaVA call failed for {image_path.name}: {exc}") from exc

        return LLaVAResult(mood="neutral", tension=0.5, symbolism="")

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
