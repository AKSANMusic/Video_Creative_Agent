"""Write / read a :class:`Timeline` to/from ``timeline.json``."""

from __future__ import annotations

import json
from pathlib import Path

from .timeline import Timeline


def export_timeline(timeline: Timeline, out_path: Path | str) -> Path:
    """Serialize ``timeline`` to ``out_path`` as indented JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def load_timeline(path: Path | str) -> Timeline:
    """Read a :class:`Timeline` back from a JSON file (round-trip helper)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return Timeline.model_validate(data)
