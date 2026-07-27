"""Write :class:`AudioMap` to a compact `audio_map.json` file.

Kept as a tiny separate module so the exporter can be reused by the CLI and
by tests without dragging in the analyzer.
"""

from __future__ import annotations

import json
from pathlib import Path

from .audio_map import AudioMap


def export_audio_map(audio_map: AudioMap, out_path: Path | str) -> Path:
    """Serialize ``audio_map`` to ``out_path`` as compact JSON.

    Returns the resolved path. Raises ``OSError`` on write failure (caller
    decides how to surface it).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = audio_map.model_dump_json(indent=2)
    out_path.write_text(payload, encoding="utf-8")
    return out_path


def load_audio_map(path: Path | str) -> AudioMap:
    """Read an :class:`AudioMap` back from a JSON file (round-trip helper)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return AudioMap.model_validate(data)
