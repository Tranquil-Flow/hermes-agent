"""Registry loader and helpers for hermes-planner."""
import json
import os
from datetime import date
from pathlib import Path
from typing import Optional


def _find_registry() -> Path:
    """Find registry.json — works on both host (macOS) and container."""
    # Explicit env override
    env = os.environ.get("HERMES_PLANNER_REGISTRY")
    if env:
        return Path(env)
    # Container path
    container = Path("/workspace/Projects/.hermes-planner/registry.json")
    if container.exists():
        return container
    # Host path — ~/Projects/.hermes-planner/registry.json
    home = Path.home() / "Projects" / ".hermes-planner" / "registry.json"
    if home.exists():
        return home
    # Fallback to container path (will error on load if missing)
    return container


REGISTRY_PATH = _find_registry()


def load() -> dict:
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def save(data: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_next_project(data: dict) -> Optional[dict]:
    """Return the highest-ranked active project (status alone is the gate)."""
    active = [p for p in data["projects"] if p["status"] == "active"]
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


def count_tasks(path: str):
    """Count done / total tasks from TASKS.md if it exists.

    Supports two formats:
      - Checkbox:  '- [x]' / '- [X]' = done, '- [ ]' = pending
      - Status:    'Status: Done' = done, 'Status: Open' = pending,
                   'Status: SKIP' = skip (not counted)

    Auto-detects which format is in use. If both are present, prefers
    whichever has more entries. Returns (None, None) if no TASKS.md.
    """
    tasks_path = os.path.join(path, "TASKS.md")
    if not os.path.isfile(tasks_path):
        return None, None

    cb_done = cb_total = 0
    st_done = st_total = 0

    with open(tasks_path) as f:
        for line in f:
            stripped = line.strip()
            # Checkbox format
            if stripped.startswith("- [x]") or stripped.startswith("- [X]"):
                cb_done += 1
                cb_total += 1
            elif stripped.startswith("- [ ]"):
                cb_total += 1
            # Status format
            if "Status: Done" in stripped:
                st_done += 1
                st_total += 1
            elif "Status: Open" in stripped:
                st_total += 1
            # Status: SKIP — skip entirely (don't increment either counter)

    # Pick the format with more entries; tie-break: prefer checkbox
    if st_total > cb_total:
        return st_done, st_total
    return cb_done, cb_total


# Statuses that are manually managed — sync must not touch them
_MANUAL_STATUSES = {"blocked", "deploy-blocked", "archived", "disabled"}


def sync_registry(data: dict) -> list:
    """Auto-update project statuses based on TASKS.md contents.

    Rules:
      - active + all tasks done  -> complete  (blocker set to auto-sync note)
      - complete + tasks pending -> active    (blocker cleared)
      - blocked / deploy-blocked / archived / disabled -> untouched

    Returns a list of change description strings (may be empty).
    """
    changes = []
    today = date.today().isoformat()

    for p in data["projects"]:
        st = p.get("status", "")
        if st in _MANUAL_STATUSES:
            continue

        path = p.get("path", "")
        done, total = count_tasks(path)

        name = p["name"]

        if st == "active" and total is not None and total > 0 and done == total:
            p["status"] = "complete"
            p["blocker"] = f"All tasks complete — auto-synced {today}"
            changes.append(f"  {name}: active -> complete (all {total} tasks done)")

        elif st == "complete" and total is not None and total > 0 and done < total:
            # Only flip complete→active if the blocker was auto-set or is empty.
            # If a human wrote the blocker, respect it — they marked it complete
            # for a reason (e.g. remaining tasks need hardware/keys/user input).
            blocker = p.get("blocker") or ""
            if not blocker or blocker.startswith("All tasks complete — auto-synced"):
                p["status"] = "active"
                p["blocker"] = None
                changes.append(
                    f"  {name}: complete -> active ({total - done}/{total} tasks remain)"
                )

    return changes
