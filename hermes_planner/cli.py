#!/usr/bin/env python3
"""hermes-planner CLI — view and manage the project registry."""
import argparse
import sys
from . import registry as reg


STATUS_ICON = {
    "active": "●",
    "blocked": "⊘",
    "complete": "✓",
    "paused": "◌",
}

VERSION_COLOR = {
    "v0": "\033[90m",   # dim
    "v1": "\033[36m",   # cyan
    "v2": "\033[33m",   # yellow
    "v3": "\033[32m",   # green
    "v4": "\033[35m",   # magenta
}
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"


def cmd_list(args):
    data = reg.load()
    projects = sorted(data["projects"], key=lambda p: p["rank"])

    print(f"\n{BOLD}{'#':>3}  {'name':<22} {'ver':<5} {'status':<10} {'description'}{RESET}")
    print("─" * 80)
    for p in projects:
        icon = STATUS_ICON.get(p["status"], "?")
        vc = VERSION_COLOR.get(p["version"], "")
        blocker_note = f"  {RED}BLOCKED: {p['blocker']}{RESET}" if p["blocker"] else ""
        desc = p["description"][:50] + "…" if len(p["description"]) > 50 else p["description"]
        print(f"  {p['rank']:>2}  {p['name']:<22} {vc}{p['version']:<5}{RESET} {icon} {p['status']:<8} {desc}{blocker_note}")
    print()

    next_p = reg.get_next_project(data)
    if next_p:
        print(f"{BOLD}Next to work on:{RESET} {next_p['name']} (rank {next_p['rank']})\n")
    else:
        print(f"{RED}No active non-blocked projects.{RESET}\n")


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
    parser = argparse.ArgumentParser(prog="hermes-planner", description="Manage the hermes project registry")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="Show all projects (default)")
    
    blk = sub.add_parser("block", help="Mark a project as blocked")
    blk.add_argument("name", help="Project name")
    blk.add_argument("reason", help="Blocker reason")

    ublk = sub.add_parser("unblock", help="Clear a project blocker")
    ublk.add_argument("name", help="Project name")

    sub.add_parser("next", help="Print name of next project to work on")

    args = parser.parse_args()

    if args.cmd == "block":
        cmd_block(args)
    elif args.cmd == "unblock":
        cmd_unblock(args)
    elif args.cmd == "next":
        cmd_next(args)
    else:
        cmd_list(args)


if __name__ == "__main__":
    main()
