"""Load cycle landmark JSON (resampled grid indices)."""
from __future__ import annotations

import json
from pathlib import Path


def load_cycle_points_json(path: Path) -> tuple[list[dict], int]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    points = list(data.get("points", []))
    n = data.get("n")
    if n is None:
        raise ValueError("cycle points JSON must contain integer 'n'")
    return points, int(n)
