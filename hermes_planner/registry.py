"""Registry loader and helpers for hermes-planner."""
import json
import os
from pathlib import Path
from typing import Optional

REGISTRY_PATH = Path("/workspace/Projects/.hermes-planner/registry.json")


def load() -> dict:
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def save(data: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_next_project(data: dict) -> Optional[dict]:
    """Return the highest-ranked active non-blocked project."""
    active = [p for p in data["projects"] if p["status"] == "active" and not p["blocker"]]
    if not active:
        return None
    return sorted(active, key=lambda p: p["rank"])[0]


def set_blocker(data: dict, name: str, reason: str) -> None:
    for p in data["projects"]:
        if p["name"] == name:
            p["status"] = "blocked"
            p["blocker"] = reason
            return
    raise KeyError(f"Project {name!r} not found in registry")


def clear_blocker(data: dict, name: str) -> None:
    for p in data["projects"]:
        if p["name"] == name:
            p["status"] = "active"
            p["blocker"] = None
            return
    raise KeyError(f"Project {name!r} not found in registry")
