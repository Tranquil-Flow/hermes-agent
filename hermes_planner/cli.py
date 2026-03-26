#!/usr/bin/env python3
"""hermes-planner CLI — view and manage the project registry."""
import argparse
import os
import sys
from . import registry as reg
from .registry import count_tasks


# ── ANSI codes ──────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
CYAN    = "\033[36m"
MAGENTA = "\033[35m"
BLUE    = "\033[34m"
WHITE   = "\033[37m"
BG_RED  = "\033[41m"

STATUS_STYLE = {
    "active":         (GREEN,   "●"),
    "complete":       (BLUE,    "✓"),
    "blocked":        (RED,     "⊘"),
    "deploy-blocked": (YELLOW,  "◧"),
    "archived":       (DIM,     "◌"),
    "disabled":       (DIM,     "∅"),
}

VERSION_STYLE = {
    "v0":      DIM,
    "v1":      CYAN,
    "v2":      YELLOW,
    "v3":      GREEN,
    "contrib": MAGENTA,
}


# ── helpers ─────────────────────────────────────────────────────
def _bar(filled, total, width=16):
    """Tiny progress bar: ████░░░░"""
    if total == 0:
        return DIM + "░" * width + RESET
    n = round(filled / total * width)
    return GREEN + "█" * n + DIM + "░" * (width - n) + RESET


def _trunc(s, n):
    return (s[:n-1] + "…") if len(s) > n else s


# _count_tasks kept as a private shim for any legacy callers inside this module
def _count_tasks(path):
    return count_tasks(path)


# ── commands ────────────────────────────────────────────────────
def cmd_sync(args):
    """Auto-update registry statuses based on TASKS.md contents."""
    data = reg.load()
    changes = reg.sync_registry(data)
    if changes:
        print("Sync changes:")
        for c in changes:
            print(c)
        reg.save(data)
        print("Registry saved.")
    else:
        print("Sync: nothing to update.")


def cmd_status(args):
    """Rich project queue dashboard."""
    # Optional pre-sync
    if getattr(args, "sync", False):
        cmd_sync(args)
        print()

    data = reg.load()
    projects = sorted(data["projects"], key=lambda p: p["rank"])

    # ── header
    active_count   = sum(1 for p in projects if p["status"] == "active")
    complete_count = sum(1 for p in projects if p["status"] == "complete")
    blocked_count  = sum(1 for p in projects if p["status"] in ("blocked", "deploy-blocked"))
    other_count    = len(projects) - active_count - complete_count - blocked_count

    print()
    print(f"  {BOLD}⚕ Hermes Planner — Project Queue{RESET}")
    print(f"  {GREEN}● {active_count} active{RESET}   "
          f"{BLUE}✓ {complete_count} complete{RESET}   "
          f"{RED}⊘ {blocked_count} blocked{RESET}   "
          f"{DIM}◌ {other_count} other{RESET}   "
          f"{DIM}│{RESET} {len(projects)} total")
    print()

    # ── column headers
    print(f"  {DIM}{'#':>3}  {'project':<24} {'ver':<7} {'status':<18} {'tasks':<20} {'blocker'}{RESET}")
    print(f"  {DIM}{'─'*3}  {'─'*24} {'─'*7} {'─'*18} {'─'*20} {'─'*30}{RESET}")

    # ── rows
    next_project = reg.get_next_project(data)

    for p in projects:
        rank = p["rank"]
        name = p["name"]
        ver  = p.get("version", "?")
        st   = p.get("status", "?")
        plans = p.get("plans", False)
        blocker = p.get("blocker", None)

        # style
        st_color, st_icon = STATUS_STYLE.get(st, (DIM, "?"))
        v_color = VERSION_STYLE.get(ver, DIM)

        # is this the next project to work on?
        is_next = next_project and next_project["name"] == name
        marker = f" {BOLD}{CYAN}◀ NEXT{RESET}" if is_next else ""

        # task progress
        done, total = count_tasks(p.get("path", ""))
        tasks_complete = False
        if total is not None and total > 0 and done == total:
            task_str = f"{BLUE}● tasks complete{RESET}  "
            tasks_complete = True
        elif total is not None and total > 0:
            bar = _bar(done, total)
            task_str = f"{bar} {done}/{total}"
        elif total == 0:
            task_str = f"{BLUE}● tasks complete{RESET}  "
            tasks_complete = True
        elif plans:
            task_str = f"{DIM}has plans{RESET}        "
        else:
            task_str = f"{DIM}no tasks{RESET}        "

        # blocker — use remaining terminal width
        try:
            term_width = os.get_terminal_size().columns
        except (ValueError, OSError):
            term_width = 120
        # columns before blocker: rank(5) + name(26) + ver(9) + status(20) + tasks(22) = ~82
        blocker_max = max(term_width - 82, 30)
        if blocker and st in ("blocked", "deploy-blocked"):
            blocker_str = f"{RED}{_trunc(blocker, blocker_max)}{RESET}"
        elif blocker:
            blocker_str = f"{DIM}{_trunc(blocker, blocker_max)}{RESET}"
        else:
            blocker_str = ""

        # print row
        print(f"  {rank:>3}  {name:<24} {v_color}{ver:<7}{RESET} "
              f"{st_color}{st_icon} {st:<16}{RESET} "
              f"{task_str} {blocker_str}{marker}")

    # ── footer
    print()
    if next_project:
        done, total = count_tasks(next_project.get("path", ""))
        task_info = f" ({total - done} tasks remaining)" if total else ""
        print(f"  {BOLD}▶ Next up:{RESET} {next_project['name']} "
              f"{DIM}(rank {next_project['rank']}){RESET}{task_info}")
    else:
        print(f"  {YELLOW}▶ No workable projects — all active projects have blockers{RESET}")
    print()


def cmd_list(args):
    """Simple list view (legacy)."""
    cmd_status(args)


def cmd_block(args):
    data = reg.load()
    reg.set_blocker(data, args.name, args.reason)
    reg.save(data)
    print(f"Marked {args.name!r} as BLOCKED: {args.reason}")


def cmd_unblock(args):
    data = reg.load()
    reg.clear_blocker(data, args.name)
    reg.save(data)
    print(f"Cleared blocker on {args.name!r}")


def cmd_next(args):
    data = reg.load()
    p = reg.get_next_project(data)
    if p:
        print(p["name"])
    else:
        print("none")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="hermes-planner",
        description="Manage the hermes project registry"
    )
    sub = parser.add_subparsers(dest="cmd")

    st_p = sub.add_parser("status", help="Project queue dashboard (default)")
    st_p.add_argument("--sync", action="store_true",
                      help="Run auto-sync before displaying")

    list_p = sub.add_parser("list", help="Alias for status")
    list_p.add_argument("--sync", action="store_true",
                        help="Run auto-sync before displaying")

    blk = sub.add_parser("block", help="Mark a project as blocked")
    blk.add_argument("name", help="Project name")
    blk.add_argument("reason", help="Blocker reason")

    ublk = sub.add_parser("unblock", help="Clear a project blocker")
    ublk.add_argument("name", help="Project name")

    sub.add_parser("next", help="Print name of next project to work on")

    sub.add_parser("sync", help="Auto-sync registry statuses from TASKS.md")

    args = parser.parse_args()

    if args.cmd == "block":
        cmd_block(args)
    elif args.cmd == "unblock":
        cmd_unblock(args)
    elif args.cmd == "next":
        cmd_next(args)
    elif args.cmd == "sync":
        cmd_sync(args)
    else:
        cmd_status(args)


if __name__ == "__main__":
    main()
