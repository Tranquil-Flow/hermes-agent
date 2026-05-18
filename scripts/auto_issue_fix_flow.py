#!/usr/bin/env python3
"""Autonomous Hermes-agent upstream bugfix flow.

Pipeline:
- triage upstream issues with duplicate-aware candidate selection
- create isolated worktrees/branches for selected candidates
- dispatch cheap builder agents first
- dispatch expensive reviewer agents only after build/test success
- loop builder follow-up <-> reviewer until approval/rejection/max rounds
- report approved branches immediately in human-approval mode

Public side effects are blocked unless config publication.mode is
`full_auto_after_dual_approval`.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import URLError

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal environments
    yaml = None

BUG_WORDS = {
    "bug", "fix", "crash", "error", "fail", "failed", "failure", "broken",
    "incorrect", "wrong", "missing", "ignore", "misroute", "silent", "leak",
    "regression", "traceback", "exception", "hang", "timeout", "race", "deadlock",
    "kill", "kills", "killed", "overwrite", "overwrites", "escape", "encoding",
}
FEATURE_WORDS = {
    "feature", "request", "add", "support", "implement", "proposal", "enhancement",
}
LOCAL_HARD_WORDS = {"windows", "win32", "ios", "android", "provider key", "api key required"}
FINAL_STATES = {"PUBLISHED", "PUBLISHED_OR_REPORTED", "REJECTED", "STALE_DUPLICATED", "FAILED"}
PR_COVERAGE_WORDS = {
    "fix", "fixes", "fixed", "resolve", "resolves", "resolved", "close", "closes",
    "closed", "address", "addresses", "addressed", "covered", "superseded",
    "duplicate", "implemented", "merged",
}
BOT_LOGINS = {"dependabot[bot]", "github-actions[bot]", "renovate[bot]"}


@dataclasses.dataclass
class Candidate:
    number: int
    title: str
    url: str
    tier: int
    confidence: str
    why_selected: str
    suspected_files: list[str]
    verification_plan: str
    duplicate_check: str
    risk: str


@dataclasses.dataclass
class Skipped:
    number: int
    title: str
    reason: str


class FlowError(RuntimeError):
    pass


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise FlowError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc


def run_shell(command: str, cwd: Path | None = None, timeout: int = 3600, check: bool = False) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise FlowError(f"command failed ({proc.returncode}): {command}\n{proc.stdout}")
    return proc


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)


def utc_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@contextlib.contextmanager
def acquire_lock(repo: Path, timeout_hours: float):
    lock_path = repo / ".automation" / "auto_issue_fix_flow.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    if lock_path.exists():
        age_hours = (now - lock_path.stat().st_mtime) / 3600
        if age_hours < timeout_hours:
            raise FlowError(
                f"previous run still active or lock is fresh: {lock_path} age={age_hours:.2f}h"
            )
    lock_path.write_text(json.dumps({"pid": os.getpid(), "started_at": now}, indent=2))
    try:
        yield lock_path
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()


def discover_github_token() -> str | None:
    """Find an API token in the current env or common Hermes/gh locations.

    The sandbox often runs as root while gh auth from earlier sessions lives under
    /home/hermes. Keep this read-only and never print the token.
    """
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = (os.getenv(name) or "").strip()
        if value and value != "***":
            return value
    for path in (
        Path.home() / ".config" / "gh" / "hosts.yml",
        Path("/home/hermes/.config/gh/hosts.yml"),
        Path("/root/.config/gh/hosts.yml"),
    ):
        if not path.exists():
            continue
        with contextlib.suppress(Exception):
            text = path.read_text(errors="ignore")
            match = re.search(r"oauth_token:\s*(\S+)", text)
            if match and match.group(1).strip() != "***":
                return match.group(1).strip()
    for path in (
        Path.home() / ".hermes" / ".env",
        Path("/home/hermes/.hermes/.env"),
        Path("/root/.hermes/.env"),
        Path.home() / ".git-credentials",
        Path("/home/hermes/.git-credentials"),
        Path("/root/.git-credentials"),
    ):
        if not path.exists():
            continue
        with contextlib.suppress(Exception):
            text = path.read_text(errors="ignore")
            match = re.search(r"^(?:GITHUB_TOKEN|GH_TOKEN)=(\S+)", text, flags=re.M)
            if match and match.group(1).strip() != "***":
                return match.group(1).strip()
            match = re.search(r"https://[^:]+:([^@]+)@github\.com", text)
            if match and match.group(1).strip() != "***":
                return match.group(1).strip()
    return None


def github_api(path: str, token: str | None = None) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "hermes-agent-auto-issue-fix-flow",
    }
    token = token or discover_github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(f"https://api.github.com{path}", headers=headers)
    with urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed GitHub API URL
        return json.loads(resp.read().decode("utf-8"))


def github_api_post(path: str, payload: dict[str, Any], token: str | None = None) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "hermes-agent-auto-issue-fix-flow",
    }
    token = token or discover_github_token()
    if not token:
        raise FlowError("GitHub API token not found; cannot create PR")
    headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload).encode("utf-8")
    req = Request(f"https://api.github.com{path}", data=data, headers=headers, method="POST")
    with urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed GitHub API URL
        return json.loads(resp.read().decode("utf-8"))


def github_api_optional(path: str, token: str | None = None) -> Any | None:
    try:
        return github_api(path, token=token)
    except Exception:
        return None


def gh_available() -> bool:
    return shutil.which("gh") is not None


def fetch_with_gh(kind: str, repo_slug: str, limit: int) -> list[dict[str, Any]]:
    if kind == "issues":
        cmd = [
            "gh", "issue", "list", "-R", repo_slug, "--state", "open", "--limit", str(limit),
            "--json", "number,title,body,labels,url,createdAt,updatedAt",
        ]
    elif kind == "issues_new_scan":
        # Fetch a generous open-issue window; caller filters by issue-number cursor.
        cmd = [
            "gh", "issue", "list", "-R", repo_slug, "--state", "open", "--limit", str(max(limit, 1000)),
            "--json", "number,title,body,labels,url,createdAt,updatedAt",
        ]
    elif kind == "prs_open":
        cmd = [
            "gh", "pr", "list", "-R", repo_slug, "--state", "open", "--limit", str(limit),
            "--json", "number,title,body,url,headRefName,createdAt,updatedAt",
        ]
    elif kind == "prs_closed":
        cmd = [
            "gh", "pr", "list", "-R", repo_slug, "--state", "closed", "--limit", str(limit),
            "--json", "number,title,body,url,headRefName,createdAt,updatedAt,mergedAt",
        ]
    else:
        raise ValueError(kind)
    out = run(cmd, timeout=120).stdout
    return json.loads(out)


def search_issues_api(repo_slug: str, q_bits: list[str], limit: int) -> list[dict[str, Any]]:
    query = "+".join([f"repo:{repo_slug}", *q_bits])
    items: list[dict[str, Any]] = []
    for page in range(1, min(10, (limit + 99) // 100) + 1):
        path = f"/search/issues?q={quote(query, safe=':+')}&sort=updated&order=desc&per_page=100&page={page}"
        data = github_api(path)
        page_items = data.get("items", [])
        items.extend(page_items)
        if len(page_items) < 100 or len(items) >= limit:
            break
        time.sleep(2.1)
    return items[:limit]


def normalize_api_issue(item: dict[str, Any]) -> dict[str, Any]:
    labels = item.get("labels") or []
    if labels and isinstance(labels[0], dict):
        labels = [x.get("name", "") for x in labels]
    return {
        "number": item.get("number"),
        "title": item.get("title") or "",
        "body": item.get("body") or "",
        "labels": labels,
        "url": item.get("html_url") or item.get("url") or "",
        "createdAt": item.get("created_at") or item.get("createdAt"),
        "updatedAt": item.get("updated_at") or item.get("updatedAt"),
    }


def automation_state_path(repo: Path) -> Path:
    return repo / ".automation" / "state.json"


def load_automation_state(repo: Path) -> dict[str, Any]:
    path = automation_state_path(repo)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"warning: failed reading automation state {path}: {exc}", file=sys.stderr)
        return {}


def infer_last_seen_issue_number_from_runs(repo: Path) -> int:
    """Infer a bootstrap issue cursor from the newest cached run artifacts.

    This prevents a newly-added state file from re-scanning the same broad issue
    window after we have already looked at it in earlier runs.
    """
    runs_dir = repo / ".automation" / "runs"
    if not runs_dir.exists():
        return 0
    for run_dir in sorted((p for p in runs_dir.iterdir() if p.is_dir()), reverse=True):
        raw = run_dir / "issues_raw.json"
        if not raw.exists():
            continue
        try:
            issues = json.loads(raw.read_text())
        except Exception:
            continue
        numbers = [int(i.get("number") or 0) for i in issues if isinstance(i, dict)]
        if numbers:
            return max(numbers)
    return 0


def last_seen_issue_number(repo: Path) -> int:
    state = load_automation_state(repo)
    explicit = int(state.get("last_seen_issue_number") or 0)
    return explicit or infer_last_seen_issue_number_from_runs(repo)


def save_last_seen_issue_number(repo: Path, run_id: str, source: str, issues: list[dict[str, Any]]) -> None:
    numbers = [int(i.get("number") or 0) for i in issues if isinstance(i, dict)]
    if not numbers:
        return
    state = load_automation_state(repo)
    previous = int(state.get("last_seen_issue_number") or 0)
    newest = max(numbers)
    if newest < previous:
        newest = previous
    state.update({
        "last_seen_issue_number": newest,
        "last_issue_scan_run_id": run_id,
        "last_issue_scan_source": source,
        "last_issue_scan_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    path = automation_state_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def fetch_issue_reference_context(repo_slug: str, issue_number: int) -> dict[str, Any]:
    """Fetch issue comments/timeline cross-references used to avoid duplicate PR work.

    GitHub often reveals active competing fixes through issue comments, @mentions,
    and timeline cross-reference events before a PR body says "fixes #N". Keep the
    extracted payload compact so it can be cached in run artifacts.
    """
    context: dict[str, Any] = {
        "referenceContextFetched": True,
        "commentsText": "",
        "timelineText": "",
        "referencedPrNumbers": [],
        "referencedPrs": [],
    }
    comments = github_api(f"/repos/{repo_slug}/issues/{issue_number}/comments?per_page=100")
    comment_bits: list[str] = []
    for comment in comments if isinstance(comments, list) else []:
        author = ((comment.get("user") or {}).get("login") or "") if isinstance(comment, dict) else ""
        body = comment.get("body") or "" if isinstance(comment, dict) else ""
        url = comment.get("html_url") or "" if isinstance(comment, dict) else ""
        if body:
            comment_bits.append(f"comment by @{author}: {body}\n{url}")

    timeline = github_api(f"/repos/{repo_slug}/issues/{issue_number}/timeline?per_page=100")
    timeline_bits: list[str] = []
    pr_numbers: set[int] = set()
    referenced_prs: list[dict[str, Any]] = []
    for item in timeline if isinstance(timeline, list) else []:
        if not isinstance(item, dict):
            continue
        event = item.get("event") or ""
        source_issue = (item.get("source") or {}).get("issue") or {}
        if source_issue and source_issue.get("pull_request"):
            pr_number = source_issue.get("number")
            if isinstance(pr_number, int):
                pr_numbers.add(pr_number)
            referenced_prs.append({
                "number": pr_number,
                "title": source_issue.get("title") or "",
                "url": source_issue.get("html_url") or "",
                "state": source_issue.get("state") or "",
                "body": source_issue.get("body") or "",
            })
            timeline_bits.append(
                f"timeline {event}: PR #{pr_number} {source_issue.get('title') or ''} "
                f"{source_issue.get('html_url') or ''} {source_issue.get('body') or ''}"
            )
        else:
            actor = ((item.get("actor") or {}).get("login") or "")
            body = item.get("body") or item.get("commit_id") or item.get("url") or ""
            if event or body:
                timeline_bits.append(f"timeline {event} by @{actor}: {body}")

    # Also mine explicit PR-looking references from comments/timeline text. GitHub issue
    # and PR numbers share one namespace, so these are later intersected with known PRs.
    combined = "\n".join([*comment_bits, *timeline_bits])
    for ref in re.findall(r"(?:pull/(\d+)|PR\s*#(\d+)|pr\s*#(\d+)|#(\d+))", combined, flags=re.I):
        for value in ref:
            if value:
                with contextlib.suppress(ValueError):
                    pr_numbers.add(int(value))

    context["commentsText"] = "\n".join(comment_bits)[:20000]
    context["timelineText"] = "\n".join(timeline_bits)[:20000]
    context["referencedPrNumbers"] = sorted(pr_numbers)
    context["referencedPrs"] = referenced_prs[:50]
    return context


def preliminary_reference_candidate(issue: dict[str, Any]) -> bool:
    title = issue.get("title", "") or ""
    body = issue.get("body", "") or ""
    labels = issue_labels(issue)
    text = f"{title}\n{body}"
    lower = text.lower()
    if label_has(labels, "duplicate") or label_contains(labels, "duplicate"):
        return False
    bug_label = label_has(labels, "bug", "type/bug", "type:bug", "kind/bug", "type/security")
    feature_label = label_has(labels, "feature", "type/feature", "type:feature", "enhancement", "kind/feature")
    title_words = words(title)
    title_bug = bool(title_words & BUG_WORDS) or title.lower().lstrip().startswith("[bug]") or "[bug]:" in title.lower()
    body_strong_bug = any(marker in lower for marker in ["traceback", "stack trace", "regression", "reproduction", "expected behavior", "actual behavior"])
    is_featureish = bool(title_words & FEATURE_WORDS) or feature_shaped_title(title) or feature_label
    return bool((bug_label or title_bug or body_strong_bug) and not (feature_label and not bug_label) and not (is_featureish and not (bug_label or title_bug)))


def enrich_issues_with_reference_context(repo_slug: str, issues: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    fetched = 0
    for issue in issues:
        item = dict(issue)
        number = int(item.get("number") or 0)
        if number and fetched < limit and preliminary_reference_candidate(item):
            try:
                item.update(fetch_issue_reference_context(repo_slug, number))
                fetched += 1
                # Stay gentle with GitHub's secondary rate limits.
                time.sleep(0.25)
            except Exception as exc:
                item["referenceFetchError"] = str(exc)
                print(f"warning: issue #{number} reference-context fetch failed: {exc}", file=sys.stderr)
        else:
            item.setdefault("referenceContextFetched", False)
        enriched.append(item)
    return enriched


def fetch_open_issues_newer_than_api(repo_slug: str, after_number: int, limit: int) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    # REST issue listing is sorted newest-first; stop once we reach the cursor.
    for page in range(1, 11):
        path = f"/repos/{repo_slug}/issues?state=open&sort=created&direction=desc&per_page=100&page={page}"
        data = github_api(path)
        if not isinstance(data, list) or not data:
            break
        stop = False
        for item in data:
            if item.get("pull_request"):
                continue
            number = int(item.get("number") or 0)
            if after_number and number <= after_number:
                stop = True
                continue
            issues.append(normalize_api_issue(item))
            if len(issues) >= limit:
                return issues
        if stop or len(data) < 100:
            break
        time.sleep(1.0)
    return issues


def fetch_repo_state_incremental(repo: Path, repo_slug: str, limit: int, after_number: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    """Fetch only open issues newer than the persisted issue-number cursor.

    This changes --issue-limit into a safety window for newly-created issues, not a
    repeated scan of the same newest N issues every run. If the cursor is 0, we do
    an initial/bootstrap window and then persist the highest issue number observed.
    """
    if gh_available():
        try:
            all_open = fetch_with_gh("issues_new_scan", repo_slug, limit)
            issues = [i for i in all_open if int(i.get("number") or 0) > after_number][:limit]
            prs_open = fetch_with_gh("prs_open", repo_slug, max(limit, 300))
            prs_closed = fetch_with_gh("prs_closed", repo_slug, max(min(limit, 200), 300))
            issues = enrich_issues_with_reference_context(repo_slug, issues, len(issues))
            mode = "gh+issue_refs+incremental" if after_number else "gh+issue_refs+bootstrap"
            return issues, prs_open, prs_closed, f"{mode}:after#{after_number}"
        except Exception as exc:
            print(f"warning: gh incremental fetch failed, falling back to GitHub API: {exc}", file=sys.stderr)
    issues = fetch_open_issues_newer_than_api(repo_slug, after_number, limit)
    prs_open = [normalize_api_issue(x) for x in search_issues_api(repo_slug, ["is:pr", "is:open"], max(limit, 300))]
    prs_closed = [normalize_api_issue(x) for x in search_issues_api(repo_slug, ["is:pr", "is:closed"], max(min(limit, 200), 300))]
    issues = enrich_issues_with_reference_context(repo_slug, issues, len(issues))
    mode = "github_api+issue_refs+incremental" if after_number else "github_api+issue_refs+bootstrap"
    return issues, prs_open, prs_closed, f"{mode}:after#{after_number}"


def fetch_repo_state(repo_slug: str, limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    if gh_available():
        try:
            issues = fetch_with_gh("issues", repo_slug, limit)
            prs_open = fetch_with_gh("prs_open", repo_slug, max(limit, 300))
            prs_closed = fetch_with_gh("prs_closed", repo_slug, max(min(limit, 200), 300))
            issues = enrich_issues_with_reference_context(repo_slug, issues, limit)
            return issues, prs_open, prs_closed, "gh+issue_refs"
        except Exception as exc:
            print(f"warning: gh fetch failed, falling back to GitHub API: {exc}", file=sys.stderr)
    issues = [normalize_api_issue(x) for x in search_issues_api(repo_slug, ["is:issue", "is:open"], limit)]
    prs_open = [normalize_api_issue(x) for x in search_issues_api(repo_slug, ["is:pr", "is:open"], max(limit, 300))]
    prs_closed = [normalize_api_issue(x) for x in search_issues_api(repo_slug, ["is:pr", "is:closed"], max(min(limit, 200), 300))]
    issues = enrich_issues_with_reference_context(repo_slug, issues, limit)
    return issues, prs_open, prs_closed, "github_api+issue_refs"


def load_latest_cached_repo_state(repo: Path, limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    """Load the newest complete GitHub issue/PR snapshot from prior run artifacts."""
    runs_dir = repo / ".automation" / "runs"
    if not runs_dir.exists():
        raise FlowError(f"no cached repo-state directory exists: {runs_dir}")
    for run_dir in sorted((p for p in runs_dir.iterdir() if p.is_dir()), reverse=True):
        required = [run_dir / "issues_raw.json", run_dir / "prs_open_raw.json", run_dir / "prs_closed_raw.json"]
        if not all(p.exists() for p in required):
            continue
        try:
            issues = json.loads(required[0].read_text())[:limit]
            prs_open = json.loads(required[1].read_text())[:limit]
            prs_closed = json.loads(required[2].read_text())[: min(limit, 200)]
        except Exception as exc:
            print(f"warning: failed reading cached repo-state from {run_dir}: {exc}", file=sys.stderr)
            continue
        if issues:
            return issues, prs_open, prs_closed, f"cache:{run_dir.name}"
    raise FlowError(f"no complete cached repo-state found under {runs_dir}")


def fetch_repo_state_with_cache(repo: Path, repo_slug: str, limit: int, allow_cache: bool, force_cache: bool = False, incremental: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    if force_cache:
        return load_latest_cached_repo_state(repo, limit)
    try:
        if incremental:
            return fetch_repo_state_incremental(repo, repo_slug, limit, last_seen_issue_number(repo))
        return fetch_repo_state(repo_slug, limit)
    except Exception as exc:
        if not allow_cache:
            raise
        print(f"warning: live GitHub state fetch failed, using cached repo-state if available: {exc}", file=sys.stderr)
        return load_latest_cached_repo_state(repo, limit)


def words(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zA-Z0-9_./-]{4,}", text)}


def issue_labels(issue: dict[str, Any]) -> set[str]:
    labels = issue.get("labels") or []
    out = set()
    for label in labels:
        if isinstance(label, dict):
            out.add(str(label.get("name", "")).lower())
        else:
            out.add(str(label).lower())
    return out


def linked_prs(issue: dict[str, Any], prs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    number = issue.get("number")
    title_words = words(issue.get("title", ""))
    issue_context = "\n".join([
        issue.get("title", "") or "",
        issue.get("body", "") or "",
        issue.get("commentsText", "") or "",
        issue.get("timelineText", "") or "",
    ]).lower()
    referenced_numbers = {int(n) for n in issue.get("referencedPrNumbers", []) if str(n).isdigit()}
    matches = []
    for pr in prs:
        pr_number = pr.get("number")
        hay = f"{pr.get('title','')}\n{pr.get('body','')}\n{pr.get('headRefName','')}".lower()
        if isinstance(pr_number, int) and pr_number in referenced_numbers:
            matches.append(pr)
            continue
        if pr_number and re.search(rf"(?:pull/{pr_number}|pr\s*#?{pr_number}|#{pr_number})(?!\d)", issue_context):
            matches.append(pr)
            continue
        if number and re.search(rf"(?<!\d)#?{number}(?!\d)", hay):
            matches.append(pr)
            continue
        pr_title_words = words(pr.get("title", ""))
        overlap = title_words & pr_title_words
        if len(overlap) >= 3 and re.search(r"fix|resolve|close|bug|crash|error|fail", hay):
            matches.append(pr)
    return matches


def suspect_files(title_body: str) -> list[str]:
    found = sorted(set(re.findall(r"[a-zA-Z0-9_./-]+\.(?:py|ts|tsx|js|md|yaml|yml|toml)", title_body)))
    lower = title_body.lower()
    hints = []
    if "discord" in lower:
        hints.append("gateway/platforms/discord.py")
    if "gateway" in lower:
        hints.append("gateway/run.py")
    if "approval" in lower:
        hints.extend(["tools/approval.py", "gateway/run.py"])
    if "cron" in lower:
        hints.append("cron/")
    if "model" in lower or "provider" in lower or "openrouter" in lower or "deepseek" in lower:
        hints.extend(["run_agent.py", "agent/transports/chat_completions.py", "providers/"])
    if "config" in lower:
        hints.append("hermes_cli/config.py")
    if "prompt" in lower or "context" in lower or "injection" in lower:
        hints.extend(["agent/prompt_builder.py", "tools/skills_guard.py", "run_agent.py"])
    return list(dict.fromkeys([*found, *hints]))[:8]


def label_has(labels: set[str], *needles: str) -> bool:
    for label in labels:
        compact = re.sub(r"[^a-z0-9]+", "", label.lower())
        for needle in needles:
            if compact == re.sub(r"[^a-z0-9]+", "", needle.lower()):
                return True
    return False


def label_contains(labels: set[str], *needles: str) -> bool:
    for label in labels:
        compact = re.sub(r"[^a-z0-9]+", "", label.lower())
        for needle in needles:
            needle_compact = re.sub(r"[^a-z0-9]+", "", needle.lower())
            if needle_compact and needle_compact in compact:
                return True
    return False


def feature_shaped_title(title: str) -> bool:
    lower = title.lower().strip()
    if lower.startswith(("feat", "feature", "enhancement", "proposal")):
        return True
    # `words()` intentionally ignores 3-letter tokens, so catch common feature
    # verbs explicitly. Without this, titles like "Add pagination..." slip into
    # the bugfix queue even with a type/feature label.
    return bool(re.match(r"^(add|support|implement|proposal|enhancement)\b", lower))


def contributor_guideline_summary(repo: Path) -> dict[str, Any]:
    """Return the repository contribution gates the automation must enforce."""
    contributing = repo / "CONTRIBUTING.md"
    return {
        "source": str(contributing) if contributing.exists() else None,
        "priority": "Bug fixes, cross-platform compatibility, security hardening, robustness before features/docs.",
        "branch_naming": "fix/description for bug fixes.",
        "tests": "Before submitting, run scripts/run_tests.sh when possible; otherwise run targeted pytest with -o 'addopts=' and be honest about scope.",
        "manual_test": "Exercise the changed Hermes code path manually when possible.",
        "cross_platform": "Consider macOS, Linux, and WSL2; run scripts/check-windows-footguns.py for file/process/terminal changes.",
        "scope": "One logical change per PR; no refactor, feature, or cleanup mixed into bugfix branches.",
        "commit": "Conventional Commits: fix(scope): concise description.",
        "pr_description": "Include what/why, how to test, platforms tested, and related issue references.",
        "dependency_policy": "New PyPI deps need upper bounds; GitHub Actions/Git URLs need full SHAs.",
        "memory_provider_policy": "New memory providers must be standalone plugins, not in-tree plugins/memory additions.",
    }


def pr_author_login(pr: dict[str, Any]) -> str:
    user = pr.get("user") or {}
    if isinstance(user, dict):
        return str(user.get("login") or "")
    return ""


def pr_is_bot(pr: dict[str, Any]) -> bool:
    return pr_author_login(pr).lower() in BOT_LOGINS


def pr_is_open_or_merged(pr: dict[str, Any]) -> bool:
    return pr.get("state") == "open" or bool(pr.get("merged_at") or pr.get("mergedAt"))


def pr_says_it_covers_issue(pr: dict[str, Any], issue_number: int, issue_title: str) -> bool:
    text = f"{pr.get('title', '')}\n{pr.get('body', '')}\n{pr.get('headRefName', '')}".lower()
    if re.search(rf"(?<!\d)#?{issue_number}(?!\d)", text):
        return True
    if words(issue_title) & words(text) and (words(text) & PR_COVERAGE_WORDS):
        return True
    return False


def resolve_referenced_pr_numbers(repo_slug: str, issue: dict[str, Any], known_prs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    """Resolve issue comment/timeline references into actual PR records.

    Plain `#123` in issue comments can mean either an issue or PR. This function
    treats every referenced number as suspicious, intersects it with already
    fetched PR windows, then asks `/pulls/{n}` for unknown numbers. If GitHub is
    unavailable we return the unresolved numbers so strict selection can skip the
    issue instead of guessing.
    """
    referenced = {int(n) for n in issue.get("referencedPrNumbers", []) if str(n).isdigit()}
    for pr in issue.get("referencedPrs", []) or []:
        if isinstance(pr, dict) and str(pr.get("number", "")).isdigit():
            referenced.add(int(pr["number"]))
    if not referenced:
        return [], []

    known_by_number = {int(p.get("number") or 0): p for p in known_prs if int(p.get("number") or 0)}
    resolved: dict[int, dict[str, Any]] = {}
    unresolved: list[int] = []
    for number in sorted(referenced)[:20]:
        if number in known_by_number:
            resolved[number] = known_by_number[number]
            continue
        data = github_api_optional(f"/repos/{repo_slug}/pulls/{number}")
        if isinstance(data, dict) and data.get("number"):
            resolved[number] = {
                "number": data.get("number"),
                "title": data.get("title") or "",
                "body": data.get("body") or "",
                "url": data.get("html_url") or "",
                "state": data.get("state") or "",
                "mergedAt": data.get("merged_at"),
                "headRefName": ((data.get("head") or {}).get("ref") or ""),
                "user": data.get("user") or {},
            }
        else:
            unresolved.append(number)
        time.sleep(0.25)
    return list(resolved.values()), unresolved


def referenced_pr_blockers(repo_slug: str, issue: dict[str, Any], known_prs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    issue_number = int(issue.get("number") or 0)
    issue_title = issue.get("title") or ""
    resolved, unresolved = resolve_referenced_pr_numbers(repo_slug, issue, known_prs)
    blockers = []
    for pr in resolved:
        if pr_is_bot(pr):
            continue
        if pr_is_open_or_merged(pr) and pr_says_it_covers_issue(pr, issue_number, issue_title):
            blockers.append(pr)
    return blockers, unresolved


def classify_candidates(repo: Path, cfg: dict[str, Any], issues: list[dict[str, Any]], prs_open: list[dict[str, Any]], prs_closed: list[dict[str, Any]]) -> tuple[list[Candidate], list[Skipped]]:
    candidates: list[Candidate] = []
    skipped: list[Skipped] = []
    seen_candidate_titles: dict[str, int] = {}
    all_prs = prs_open + prs_closed
    repo_slug = cfg["repo"]["upstream_repo"]
    guidelines = contributor_guideline_summary(repo)
    for issue in issues:
        number = int(issue.get("number") or 0)
        title = issue.get("title", "")
        body = issue.get("body", "") or ""
        labels = issue_labels(issue)
        normalized_title = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
        text = f"{title}\n{body}"
        lower = text.lower()
        if issue.get("referenceFetchError"):
            skipped.append(Skipped(number, title, f"strict mode: could not fetch issue comments/timeline: {issue.get('referenceFetchError')}"))
            continue
        if label_has(labels, "duplicate") or label_contains(labels, "duplicate"):
            skipped.append(Skipped(number, title, "issue is labeled duplicate"))
            continue
        guideline_disallowed = []
        if "new memory provider" in lower or "plugins/memory" in lower:
            guideline_disallowed.append(guidelines["memory_provider_policy"])
        if re.search(r"\b(add|new|create)\b.{0,30}\b(tool|skill)\b", lower) and not (label_has(labels, "bug", "type/bug", "type:bug") or "regression" in lower):
            guideline_disallowed.append("CONTRIBUTING says new tools/skills are lower-priority and must follow dedicated standards; this is not a tight bugfix.")
        if guideline_disallowed:
            skipped.append(Skipped(number, title, "contributor-guideline mismatch: " + " ".join(guideline_disallowed)))
            continue
        if normalized_title in seen_candidate_titles:
            skipped.append(Skipped(number, title, f"duplicate candidate title of issue #{seen_candidate_titles[normalized_title]}"))
            continue
        pr_matches = linked_prs(issue, all_prs)
        focused_open = linked_prs(issue, prs_open)
        reference_blockers, unresolved_refs = referenced_pr_blockers(repo_slug, issue, all_prs)
        if reference_blockers:
            skipped.append(Skipped(number, title, "issue comments/timeline reference covering PR: " + ", ".join('#'+str(p.get('number')) for p in reference_blockers[:3])))
            continue
        if unresolved_refs:
            skipped.append(Skipped(number, title, "strict mode: unresolved PR/issue references in comments/timeline need human or later authenticated check: " + ", ".join('#'+str(n) for n in unresolved_refs[:5])))
            continue
        if focused_open:
            skipped.append(Skipped(number, title, f"open PR match: {', '.join('#'+str(p.get('number')) for p in focused_open[:3])}"))
            continue
        if pr_matches and any(p.get("mergedAt") or "merged" in str(p).lower() for p in pr_matches):
            skipped.append(Skipped(number, title, "closed/merged PR appears to cover it"))
            continue
        title_words = words(title)
        bug_label = label_has(labels, "bug", "type/bug", "type:bug", "kind/bug", "type/security")
        feature_label = label_has(labels, "feature", "type/feature", "type:feature", "enhancement", "kind/feature")
        title_bug = bool(title_words & BUG_WORDS) or title.lower().lstrip().startswith("[bug]") or "[bug]:" in title.lower()
        body_strong_bug = any(marker in lower for marker in ["traceback", "stack trace", "regression", "reproduction", "expected behavior", "actual behavior"])
        is_bug = bug_label or title_bug or body_strong_bug
        is_featureish = bool(title_words & FEATURE_WORDS) or feature_shaped_title(title) or feature_label
        if feature_label and not bug_label:
            skipped.append(Skipped(number, title, "type/feature issue rather than bugfix candidate"))
            continue
        if is_featureish and not (bug_label or title_bug):
            skipped.append(Skipped(number, title, "feature/enhancement-shaped rather than clear bug"))
            continue
        if not is_bug:
            skipped.append(Skipped(number, title, "not clearly a bug from title/labels/body preview"))
            continue
        if not issue.get("referenceContextFetched"):
            skipped.append(Skipped(number, title, "strict mode: issue comments/timeline were not fetched for this potential bug"))
            continue
        if any(w in lower for w in LOCAL_HARD_WORDS):
            skipped.append(Skipped(number, title, "may require unavailable platform/provider access"))
            continue
        files = suspect_files(text)
        tier = 1 if (bug_label or "crash" in lower or "traceback" in lower or files) else 2
        confidence = "high" if tier == 1 else "medium"
        why = "Bug-shaped issue with no focused open PR found; appears locally investigable."
        if files:
            why += f" Suspected area: {', '.join(files[:3])}."
        candidates.append(Candidate(
            number=number,
            title=title,
            url=issue.get("url", ""),
            tier=tier,
            confidence=confidence,
            why_selected=why,
            suspected_files=files,
            verification_plan="Reproduce from issue body or write regression around suspected code path; run focused pytest with -o 'addopts='; prove fail-without-fix where possible.",
            duplicate_check="Strict gate passed: fetched issue comments and timeline; resolved referenced PR numbers; checked open/closed PRs by issue number, title overlap, PR body/head refs, comments, and timeline cross-references.",
            risk="low" if tier == 1 and files else "medium",
        ))
        seen_candidate_titles[normalized_title] = number
    candidates.sort(key=lambda c: (c.tier, {"high": 0, "medium": 1, "low": 2}.get(c.confidence, 9), c.number))
    return candidates, skipped


def issue_search_items(repo_slug: str, query_bits: list[str], limit: int = 20) -> list[dict[str, Any]]:
    query = "+".join([f"repo:{repo_slug}", *query_bits])
    path = f"/search/issues?q={quote(query, safe=':+')}&sort=updated&order=desc&per_page={min(limit, 100)}"
    data = github_api(path)
    return data.get("items", [])[:limit]


def focused_competing_prs_for_candidate(cfg: dict[str, Any], c: Candidate) -> list[dict[str, Any]]:
    """Final, per-candidate duplicate guard before spending builder/reviewer tokens."""
    repo_slug = cfg["repo"]["upstream_repo"]
    try:
        fresh_issue = {
            "number": c.number,
            "title": c.title,
            "body": "",
            **fetch_issue_reference_context(repo_slug, c.number),
        }
        ref_blockers, unresolved_refs = referenced_pr_blockers(repo_slug, fresh_issue, [])
        if ref_blockers:
            return [{
                "number": p.get("number"),
                "title": p.get("title"),
                "url": p.get("url"),
                "reason": "fresh issue comment/timeline reference to covering PR",
            } for p in ref_blockers]
        if unresolved_refs:
            return [{
                "number": n,
                "title": "unresolved referenced PR/issue number",
                "url": None,
                "reason": "strict mode could not resolve issue comment/timeline reference before build",
            } for n in unresolved_refs[:5]]
    except Exception as exc:
        return [{
            "number": None,
            "title": "issue reference-context fetch failed",
            "url": None,
            "reason": f"strict mode refused candidate because comments/timeline could not be refreshed: {exc}",
        }]
    candidate_words = words(c.title) - {
        "bug", "fix", "error", "failed", "failure", "with", "from", "that", "this",
        "should", "report", "issue", "missing", "misses", "gets", "does", "when",
    }
    query_sets = [["type:pr", "state:open", str(c.number)]]
    title_query_words = [w for w in sorted(candidate_words, key=lambda w: (-len(w), w))[:6] if not w.endswith(".py")]
    if len(title_query_words) >= 2:
        query_sets.append(["type:pr", "state:open", *title_query_words[:5]])
    items_by_number: dict[int, dict[str, Any]] = {}
    for q_bits in query_sets:
        try:
            for item in issue_search_items(repo_slug, q_bits, limit=25):
                number = int(item.get("number") or 0)
                if number:
                    items_by_number[number] = item
            time.sleep(1.0)
        except Exception:
            continue
    hits: list[dict[str, Any]] = []
    for item in items_by_number.values():
        number = int(item.get("number") or 0)
        title = item.get("title") or ""
        body = item.get("body") or ""
        text = f"{title}\n{body}".lower()
        title_words = words(title)
        overlap_words = candidate_words & title_words
        overlap = len(overlap_words)
        mentions_issue = bool(re.search(rf"(?<!\d)#?{c.number}(?!\d)", text))
        file_overlap = any(f.lower() in text for f in c.suspected_files[:3])
        # Accept either explicit issue references, file references, or multiple
        # distinctive title-word overlaps. The title search catches common PRs
        # that say "fix: widen prompt injection scanner patterns" without using
        # "Fixes #27284" in their body.
        if mentions_issue or file_overlap or overlap >= 2:
            reason_bits = []
            if mentions_issue:
                reason_bits.append(f"mentions issue #{c.number}")
            if file_overlap:
                reason_bits.append("mentions suspected file")
            if overlap >= 2:
                reason_bits.append("title overlap: " + ", ".join(sorted(overlap_words)[:6]))
            hits.append({
                "number": number,
                "title": title,
                "url": item.get("html_url") or item.get("url"),
                "reason": "; ".join(reason_bits) or f"open PR search hit for issue #{c.number}",
            })
    return hits


def pr_display(pr: dict[str, Any]) -> str:
    number = pr.get("number")
    title = (pr.get("title") or "").strip()
    url = pr.get("html_url") or pr.get("url") or ""
    reason = (pr.get("reason") or "").strip()
    label = f"#{number}" if number is not None else "unresolved PR/reference"
    bits = [label]
    if title:
        bits.append(title)
    if url:
        bits.append(url)
    if reason:
        bits.append(f"({reason})")
    return " ".join(bits)


def format_duplicate_prs(dupes: list[dict[str, Any]], limit: int = 3) -> str:
    return "; ".join(pr_display(d) for d in dupes[:limit])


def filter_candidates_with_live_duplicate_prs(cfg: dict[str, Any], candidates: list[Candidate], skipped: list[Skipped], live_check: bool) -> list[Candidate]:
    """Remove candidates that have a good existing PR before they are reported.

    The cached issue/PR snapshot is already checked in classify_candidates(). This
    optional live pass catches PRs created after the snapshot, but is disabled for
    --use-cache/offline runs so network failures do not manufacture '#None' false
    duplicates.
    """
    if not live_check:
        return candidates
    viable: list[Candidate] = []
    for c in candidates:
        dupes = focused_competing_prs_for_candidate(cfg, c)
        if dupes:
            skipped.append(Skipped(
                c.number,
                c.title,
                "live open PR duplicate during candidate selection: " + format_duplicate_prs(dupes),
            ))
            continue
        viable.append(c)
    return viable


def select_candidates_for_wave(candidates: list[Candidate], max_candidates: int | None) -> list[Candidate]:
    """Select the next processing wave from already duplicate-filtered candidates."""
    if max_candidates is None:
        return list(candidates)
    return list(candidates[:max_candidates])


def ensure_repo(repo: Path, upstream: str, base_branch: str) -> None:
    if not (repo / ".git").exists():
        raise FlowError(f"repo path is not a git checkout: {repo}")
    run(["git", "fetch", upstream, "--prune"], cwd=repo, timeout=300)
    run(["git", "rev-parse", "--verify", f"refs/remotes/{upstream}/{base_branch}"], cwd=repo)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def resolve_repo_path(cfg_path: Path, cfg: dict[str, Any]) -> Path:
    """Resolve repo path robustly across host cron and sandbox/container runs.

    The config normally stores the host checkout path. When the same file is tested
    from a container or alternate mount, prefer the configured path if it exists;
    otherwise fall back to the parent of `.automation/` containing this config.
    """
    configured = Path(cfg["repo"]["path"]).expanduser().resolve()
    if (configured / ".git").exists() or configured.exists():
        return configured
    fallback = cfg_path.parent.parent.resolve()
    if (fallback / ".git").exists() or fallback.exists():
        return fallback
    return configured


def mention_for(cfg: dict[str, Any]) -> str:
    configured = str(cfg.get("reporting", {}).get("mention_prefix") or "").strip()
    if configured and "***" not in configured:
        return configured
    return f"<@{cfg.get('publication', {}).get('mention_user', '385694377655271424')}>"


def build_start_notice(run_id: str, cfg: dict[str, Any], repo: Path, run_dir: Path, args: argparse.Namespace) -> str:
    return "\n".join([
        f"{mention_for(cfg)} 🌙 Hermes-agent autonomous bugfix flow started",
        "",
        f"Run: {run_id}",
        f"Phase: {args.phase}",
        f"Agent mode: {args.agent_mode}",
        f"New-issue scan window: {args.issue_limit}",
        f"Issue cursor: after #{last_seen_issue_number(repo) if not getattr(args, 'full_issue_window', False) else 0}",
        f"Max candidates this run: {args.max_candidates if args.max_candidates is not None else 'none'}",
        f"Repository: {repo}",
        f"Artifacts: {run_dir}",
        "",
        "Publication mode: human_approval — no public push/PR/comment without approval.",
    ])


def write_delivery_helper(path: Path, name: str, deliver: str, message: str) -> None:
    prompt = "Output this EXACT text as your final response. Nothing else. No prefix, no commentary:\n\n" + message
    write_json(path, {
        "action": "create",
        "name": name,
        "schedule": "1m",
        "repeat": 1,
        "deliver": deliver,
        "enabled_toolsets": [],
        "prompt": prompt,
    })


def send_discord_notice_via_hermes_cron(run_dir: Path, cfg: dict[str, Any], name: str, message: str) -> None:
    """Best-effort immediate Discord notice from inside the one scheduled flow.

    Hermes cron jobs only deliver their final response. To get a start chime
    without maintaining a separate scheduled job, the script creates a short-lived
    one-shot delivery job itself when the host has the Hermes CLI available. If
    that is unavailable, the helper JSON artifact records the exact cronjob-tool
    payload for manual replay, but the main flow continues.
    """
    deliver = cfg.get("publication", {}).get("report_destination") or f"discord:{cfg['publication']['discord_channel']}"
    helper_name = f"{name}_cron_delivery.json"
    write_delivery_helper(run_dir / helper_name, name, deliver, message)
    hermes = shutil.which("hermes")
    if not hermes:
        (run_dir / f"{name}_delivery_warning.txt").write_text("hermes CLI not found on PATH; wrote helper JSON only.\n")
        return
    prompt = "Output this EXACT text as your final response. Nothing else. No prefix, no commentary:\n\n" + message
    proc = run([
        hermes, "cron", "create", "1m", prompt,
        "--name", name,
        "--deliver", deliver,
        "--repeat", "1",
    ], timeout=120, check=False)
    if proc.returncode != 0:
        (run_dir / f"{name}_delivery_warning.txt").write_text(proc.stdout)


def candidate_to_dict(c: Candidate) -> dict[str, Any]:
    return dataclasses.asdict(c)


def skipped_to_dict(s: Skipped) -> dict[str, Any]:
    return dataclasses.asdict(s)


def slugify(text: str, max_len: int = 58) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return (slug[:max_len].strip("-") or "bugfix")


def branch_for(cfg: dict[str, Any], c: Candidate) -> str:
    prefix = cfg.get("git", {}).get("branch_prefix", "fix/")
    return f"{prefix}{c.number}-{slugify(c.title)}"


def worktree_for(cfg: dict[str, Any], branch: str) -> Path:
    root = Path(cfg["worktrees"]["root"]).expanduser().resolve()
    safe = branch.replace("/", "__")
    return root / safe


def resolved_worktree_for(repo: Path, cfg: dict[str, Any], branch: str) -> Path:
    """Resolve worktree path on the same mount as the resolved repo.

    Cron/gateway sessions may run on the host while sandbox tests run under
    /workspace. The config stores the host root, but if the repo itself was
    resolved via the portable fallback, deriving the worktree root from the repo
    avoids split-brain git metadata such as a /Users worktree path pointing at a
    /workspace control checkout.
    """
    configured = worktree_for(cfg, branch)
    if configured.exists() or str(repo).startswith("/Users/"):
        return configured
    safe = branch.replace("/", "__")
    return repo.parent / ".automation-worktrees" / repo.name / safe


def existing_worktrees_for_branch(repo: Path, branch: str) -> list[Path]:
    proc = run(["git", "worktree", "list", "--porcelain"], cwd=repo, timeout=60, check=False)
    paths: list[Path] = []
    current_path: Path | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("worktree "):
            current_path = Path(line.split(" ", 1)[1])
        elif line == f"branch refs/heads/{branch}" and current_path is not None:
            paths.append(current_path)
    return paths


def prepare_worktree(repo: Path, cfg: dict[str, Any], branch: str, dry_run: bool) -> Path:
    worktree = resolved_worktree_for(repo, cfg, branch)
    worktree.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return worktree
    upstream = cfg["repo"].get("upstream_remote", "upstream")
    base = cfg["repo"].get("base_branch", "main")
    base_ref = f"refs/remotes/{upstream}/{base}"
    base_sha = run(["git", "rev-parse", "--verify", base_ref], cwd=repo, timeout=60).stdout.strip()
    for old_path in existing_worktrees_for_branch(repo, branch):
        run(["git", "worktree", "remove", "--force", str(old_path)], cwd=repo, timeout=180, check=False)
    if worktree.exists():
        run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo, timeout=180, check=False)
        if worktree.exists():
            shutil.rmtree(worktree, ignore_errors=True)
    run(["git", "worktree", "prune"], cwd=repo, timeout=120, check=False)
    run(["git", "branch", "-D", branch], cwd=repo, timeout=60, check=False)
    run(["git", "worktree", "add", "-B", branch, str(worktree), base_sha], cwd=repo, timeout=300)
    actual = run(["git", "rev-parse", "HEAD"], cwd=worktree, timeout=60).stdout.strip()
    if actual != base_sha:
        raise FlowError(f"worktree {worktree} did not start from {base_ref}: expected {base_sha}, got {actual}")
    return worktree


def read_prompt(repo: Path, name: str) -> str:
    return (repo / ".automation" / "prompts" / name).read_text()


def issue_body_for(c: Candidate, issues: list[dict[str, Any]]) -> str:
    for issue in issues:
        if int(issue.get("number") or 0) == c.number:
            return issue.get("body") or ""
    return ""


def render_builder_prompt(repo: Path, cfg: dict[str, Any], c: Candidate, issue_body: str, branch: str, worktree: Path) -> str:
    base = read_prompt(repo, "builder.md")
    return f"""{base}

ASSIGNMENT
Issue: #{c.number} {c.title}
Issue URL: {c.url}
Branch: {branch}
Worktree: {worktree}
Repository control checkout: {repo}
Base: {cfg['repo'].get('upstream_remote', 'upstream')}/{cfg['repo'].get('base_branch', 'main')}
Suspected files: {', '.join(c.suspected_files) if c.suspected_files else '(none guessed)'}
Candidate rationale: {c.why_selected}
Verification plan: {c.verification_plan}

CONTRIBUTOR GUIDELINE GATES
{json.dumps(contributor_guideline_summary(repo), indent=2)}

ISSUE BODY
{textwrap.indent(issue_body[:12000] or '(empty)', '  ')}

EXECUTION RULES
1. cd to the assigned worktree before editing.
2. Confirm you are on the assigned branch.
3. Re-check for existing focused PRs before implementation.
4. Implement minimal production change plus meaningful regression test.
5. Run focused tests. If possible, prove fail-without-fix.
6. Commit locally only. Do not push.
7. Output JSON only as specified above.
"""


def render_followup_prompt(repo: Path, c: Candidate, branch: str, worktree: Path, review: dict[str, Any]) -> str:
    base = read_prompt(repo, "fix_followup.md")
    return f"""{base}

ASSIGNMENT
Issue: #{c.number} {c.title}
Branch: {branch}
Worktree: {worktree}

REVIEWER_JSON
{json.dumps(review, indent=2)}

Address only these findings, amend the existing commit, rerun requested tests, and return JSON only.
"""


def render_reviewer_prompt(repo: Path, cfg: dict[str, Any], c: Candidate, issue_body: str, branch: str, worktree: Path, builder_result: dict[str, Any]) -> str:
    base = read_prompt(repo, "reviewer.md")
    upstream = cfg["repo"].get("upstream_remote", "upstream")
    base_branch = cfg["repo"].get("base_branch", "main")
    diff = run(["git", "diff", f"refs/remotes/{upstream}/{base_branch}...HEAD"], cwd=worktree, timeout=120, check=False).stdout[:50000]
    stat = run(["git", "diff", "--stat", f"refs/remotes/{upstream}/{base_branch}...HEAD"], cwd=worktree, timeout=60, check=False).stdout
    files = run(["git", "diff", "--name-only", f"refs/remotes/{upstream}/{base_branch}...HEAD"], cwd=worktree, timeout=60, check=False).stdout
    log = run(["git", "log", "--oneline", f"refs/remotes/{upstream}/{base_branch}..HEAD"], cwd=worktree, timeout=60, check=False).stdout
    return f"""{base}

ASSIGNMENT
Issue: #{c.number} {c.title}
Issue URL: {c.url}
Branch: {branch}
Worktree: {worktree}
Base: {upstream}/{base_branch}
Candidate rationale: {c.why_selected}
Suspected files: {', '.join(c.suspected_files) if c.suspected_files else '(none guessed)'}

CONTRIBUTOR GUIDELINE GATES
{json.dumps(contributor_guideline_summary(repo), indent=2)}

ISSUE BODY
{textwrap.indent(issue_body[:12000] or '(empty)', '  ')}

BUILDER_RESULT_JSON
{json.dumps(builder_result, indent=2)}

GIT LOG
{log}

DIFF STAT
{stat}

CHANGED FILES
{files}

DIFF
{diff}

Review this branch. Do not edit files. Return JSON only.
"""


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except Exception:
        pass
    # Collect ALL top-level parseable JSON objects from the text, then pick the
    # one that looks like a builder/reviewer result.  Agent output often contains
    # code diffs with embedded JSON literals (e.g. test fixtures writing
    # `{"name": "disk-cleanup", "version": "1.0.0"}`); the FIRST object is
    # frequently noise from a displayed diff, not the actual result.
    #
    # Selection priority:
    #   1. Objects with a top-level "status" key  →  builder results
    #   2. Objects with a top-level "verdict" key  →  reviewer results
    #   3. Last parseable object                   →  fallback (metadata is
    #      usually AFTER the result, not before)
    starts = [m.start() for m in re.finditer(r"\{", text)]
    parsed: list[tuple[int, dict[str, Any]]] = []  # (start_offset, obj)
    for start in starts:
        chunk = text[start:]
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(chunk):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed.append((start, json.loads(chunk[: i + 1])))
                        except Exception:
                            pass
                        break
    if not parsed:
        raise FlowError("agent output did not contain parseable JSON")
    # Priority 1: builder-shaped (has "status")
    for _off, obj in parsed:
        if "status" in obj:
            return obj
    # Priority 2: reviewer-shaped (has "verdict")
    for _off, obj in parsed:
        if "verdict" in obj:
            return obj
    # Priority 3: last object (closest to the end of the output)
    return parsed[-1][1]


def command_from_template(template: str, prompt_file: Path, prompt_text: str, worktree: Path, branch: str, issue_number: int) -> str:
    return template.format(
        prompt_file=shlex.quote(str(prompt_file)),
        prompt_path=shlex.quote(str(prompt_file)),
        prompt_shell=shlex.quote(prompt_text),
        worktree=shlex.quote(str(worktree)),
        branch=shlex.quote(branch),
        issue_number=issue_number,
    )


def dispatch_agent(kind: str, cfg: dict[str, Any], prompt_file: Path, prompt_text: str, worktree: Path, branch: str, issue_number: int, agent_mode: str, timeout: int) -> tuple[dict[str, Any], str]:
    if agent_mode == "dry-run":
        return {
            "status": "DRY_RUN" if kind != "reviewer" else None,
            "verdict": "DRY_RUN" if kind == "reviewer" else None,
            "issue": issue_number,
            "branch": branch,
            "commit": None,
            "files_changed": [],
            "tests_run": [],
            "fail_without_fix": "not_done",
            "notes": f"Dry-run: would dispatch {kind} with prompt {prompt_file}",
        }, ""
    if agent_mode == "mock":
        if kind == "reviewer":
            return {
                "verdict": "APPROVED",
                "confidence": "medium",
                "blockers": [],
                "non_blocking": ["Mock reviewer approval; no expensive GPT-5.5 call was made."],
                "tests_to_run": [],
                "summary": "Mock review approved the orchestration path only.",
            }, "mock reviewer"
        return {
            "status": "BUILT",
            "issue": issue_number,
            "branch": branch,
            "commit": None,
            "files_changed": [],
            "tests_run": [{"command": "mock", "result": "passed", "summary": "No real builder was launched."}],
            "fail_without_fix": "not_done",
            "notes": "Mock builder result; no code was changed.",
        }, "mock builder"
    template = cfg["agents"][kind]["command_template"]
    command = command_from_template(template, prompt_file, prompt_text, worktree, branch, issue_number)
    proc = run_shell(command, cwd=worktree, timeout=timeout, check=False)
    raw = proc.stdout
    if proc.returncode != 0:
        return {
            "status": "FAILED" if kind != "reviewer" else None,
            "verdict": "REJECT_BRANCH" if kind == "reviewer" else None,
            "issue": issue_number,
            "branch": branch,
            "commit": None,
            "files_changed": [],
            "tests_run": [],
            "fail_without_fix": "not_done",
            "notes": f"{kind} command failed with exit {proc.returncode}",
            "raw_output_path": str(prompt_file.with_suffix(f".{kind}.raw.txt")),
        }, raw
    return extract_json_object(raw), raw


def branch_commit(worktree: Path) -> str | None:
    proc = run(["git", "rev-parse", "HEAD"], cwd=worktree, timeout=60, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def branch_files(worktree: Path, cfg: dict[str, Any]) -> list[str]:
    upstream = cfg["repo"].get("upstream_remote", "upstream")
    base = cfg["repo"].get("base_branch", "main")
    proc = run(["git", "diff", "--name-only", f"refs/remotes/{upstream}/{base}...HEAD"], cwd=worktree, timeout=60, check=False)
    return [x for x in proc.stdout.splitlines() if x.strip()]


def report_branch_immediately(run_dir: Path, cfg: dict[str, Any], c: Candidate, branch: str, worktree: Path, review: dict[str, Any], builder: dict[str, Any], agent_mode: str) -> str:
    commit = builder.get("commit") or (branch_commit(worktree) if worktree.exists() and agent_mode == "real" else None)
    files = builder.get("files_changed") or (branch_files(worktree, cfg) if worktree.exists() and agent_mode == "real" else [])
    mention = mention_for(cfg)
    lines = [
        f"{mention} Hermes-agent branch approved for human review",
        "",
        f"Issue: #{c.number} {c.title}",
        f"URL: {c.url}",
        f"Branch: {branch}",
        f"Commit: {commit or '(mock/dry-run/no commit reported)'}",
        f"Worktree: {worktree}",
        f"Files: {', '.join(files) if files else '(none reported)'}",
        "",
        f"Reviewer confidence: {review.get('confidence', 'unknown')}",
        f"Reviewer summary: {review.get('summary', '')}",
        "",
        "Publication mode: human_approval — no push/PR/comment was performed.",
        f"Artifacts: {run_dir}",
    ]
    msg = "\n".join(lines)
    max_chars = int(cfg.get("reporting", {}).get("max_discord_chars", 1900))
    if len(msg) > max_chars:
        msg = msg[: max_chars - 80].rstrip() + f"\n\n[truncated; full artifacts at {run_dir}]"
    approved_dir = run_dir / "approved"
    approved_dir.mkdir(parents=True, exist_ok=True)
    (approved_dir / f"issue-{c.number}.md").write_text(msg + "\n")
    return msg


def process_candidate(repo: Path, cfg: dict[str, Any], run_dir: Path, c: Candidate, issues: list[dict[str, Any]], agent_mode: str, dry_run_git: bool) -> dict[str, Any]:
    branch = branch_for(cfg, c)
    preflight_dupes = [] if agent_mode in {"dry-run", "mock"} else focused_competing_prs_for_candidate(cfg, c)
    item_dir = run_dir / "branches" / f"issue-{c.number}"
    item_dir.mkdir(parents=True, exist_ok=True)
    if preflight_dupes:
        state = {
            "issue": candidate_to_dict(c),
            "state": "STALE_DUPLICATED",
            "branch": branch,
            "worktree": None,
            "rounds": [],
            "approved_report": None,
            "duplicate_prs": preflight_dupes,
        }
        write_json(item_dir / "state.json", state)
        return state
    worktree = prepare_worktree(repo, cfg, branch, dry_run=(dry_run_git or agent_mode in {"dry-run", "mock"}))
    issue_body = issue_body_for(c, issues)
    state: dict[str, Any] = {
        "issue": candidate_to_dict(c),
        "state": "CANDIDATE",
        "branch": branch,
        "worktree": str(worktree),
        "rounds": [],
        "approved_report": None,
    }

    builder_prompt = render_builder_prompt(repo, cfg, c, issue_body, branch, worktree)
    builder_prompt_file = item_dir / "builder_round_0.md"
    builder_prompt_file.write_text(builder_prompt)
    state["state"] = "BUILDING"
    builder_result, raw = dispatch_agent(
        "builder", cfg, builder_prompt_file, builder_prompt, worktree, branch, c.number,
        agent_mode, int(cfg.get("testing", {}).get("full_timeout_seconds", 3600)),
    )
    if raw:
        (item_dir / "builder_round_0.raw.txt").write_text(raw)
    write_json(item_dir / "builder_round_0.json", builder_result)
    state["builder_result"] = builder_result
    status = builder_result.get("status")
    if status == "STALE_DUPLICATED":
        state["state"] = "STALE_DUPLICATED"
        write_json(item_dir / "state.json", state)
        return state
    if status not in {"BUILT", "FIXED"}:
        state["state"] = "FAILED" if agent_mode != "dry-run" else "DRY_RUN_READY"
        write_json(item_dir / "state.json", state)
        return state

    current_builder = builder_result
    if agent_mode == "real":
        for completion_idx in range(1, 3):
            missing: list[str] = []
            if not current_builder.get("commit"):
                missing.append("local commit SHA")
            if not current_builder.get("tests_run"):
                missing.append("tests_run evidence")
            if current_builder.get("fail_without_fix") in {None, "", "not_done"}:
                missing.append("fail-without-fix proof or explicit reason it is impossible")
            if not missing:
                break
            completion_review = {
                "verdict": "REQUEST_CHANGES",
                "confidence": "high",
                "blockers": [{
                    "severity": "blocker",
                    "file": None,
                    "line": None,
                    "issue": "Builder output is incomplete: missing " + ", ".join(missing),
                    "required_fix": "Run the focused tests, prove fail-without-fix where possible, commit locally only, and return JSON with commit/tests_run/fail_without_fix populated.",
                }],
                "tests_to_run": [cfg.get("testing", {}).get("pytest_base_command", "python3 -m pytest -o 'addopts='")],
                "summary": "Complete the builder checklist before expensive review.",
            }
            followup_prompt = render_followup_prompt(repo, c, branch, worktree, completion_review)
            followup_prompt_file = item_dir / f"builder_completion_round_{completion_idx}.md"
            followup_prompt_file.write_text(followup_prompt)
            followup, raw_followup = dispatch_agent(
                "builder", cfg, followup_prompt_file, followup_prompt, worktree, branch, c.number,
                agent_mode, int(cfg.get("testing", {}).get("focused_timeout_seconds", 600)),
            )
            if raw_followup:
                (item_dir / f"builder_completion_round_{completion_idx}.raw.txt").write_text(raw_followup)
            write_json(item_dir / f"builder_completion_round_{completion_idx}.json", followup)
            state.setdefault("builder_completion_rounds", []).append(followup)
            current_builder = followup

        if not current_builder.get("commit") or not current_builder.get("tests_run"):
            state["state"] = "FAILED"
            state["failure"] = "Builder did not provide commit/tests evidence after completion follow-up; skipping expensive reviewer."
            state["builder_result"] = current_builder
            write_json(item_dir / "state.json", state)
            return state

    max_rounds = int(cfg.get("review_loop", {}).get("max_rounds_per_branch", 5))
    for round_idx in range(1, max_rounds + 1):
        review_prompt = render_reviewer_prompt(repo, cfg, c, issue_body, branch, worktree, current_builder) if agent_mode == "real" else read_prompt(repo, "reviewer.md")
        review_prompt_file = item_dir / f"review_round_{round_idx}.md"
        review_prompt_file.write_text(review_prompt)
        state["state"] = "REVIEWING"
        review, raw_review = dispatch_agent(
            "reviewer", cfg, review_prompt_file, review_prompt, worktree, branch, c.number,
            agent_mode, int(cfg.get("testing", {}).get("focused_timeout_seconds", 600)),
        )
        if raw_review:
            (item_dir / f"review_round_{round_idx}.raw.txt").write_text(raw_review)
        write_json(item_dir / f"review_round_{round_idx}.json", review)
        verdict = review.get("verdict")
        state["rounds"].append({"round": round_idx, "review": review})
        if verdict == "APPROVED":
            state["state"] = "APPROVED_READY_TO_PUBLISH"
            msg = report_branch_immediately(run_dir, cfg, c, branch, worktree, review, current_builder, agent_mode)
            state["approved_report"] = msg
            state["state"] = "PUBLISHED_OR_REPORTED"
            if cfg.get("publication", {}).get("mode") == "full_auto_after_dual_approval":
                push_output = push_branch_to_fork(worktree, cfg, branch, dry_run=False)
                pr = create_pull_request(cfg, state, dry_run=False)
                state["publication_result"] = {
                    "pushed": True,
                    "push_output": push_output[-1000:],
                    "pr": pr,
                }
                state["state"] = "PUBLISHED"
            write_json(item_dir / "state.json", state)
            return state
        if verdict == "REJECT_BRANCH":
            state["state"] = "REJECTED"
            write_json(item_dir / "state.json", state)
            return state
        if verdict != "REQUEST_CHANGES":
            state["state"] = "FAILED"
            state["failure"] = f"Unknown reviewer verdict: {verdict}"
            write_json(item_dir / "state.json", state)
            return state

        followup_prompt = render_followup_prompt(repo, c, branch, worktree, review)
        followup_prompt_file = item_dir / f"followup_round_{round_idx}.md"
        followup_prompt_file.write_text(followup_prompt)
        state["state"] = "FIXING_REVIEW_FINDINGS"
        followup, raw_followup = dispatch_agent(
            "builder", cfg, followup_prompt_file, followup_prompt, worktree, branch, c.number,
            agent_mode, int(cfg.get("testing", {}).get("focused_timeout_seconds", 600)),
        )
        if raw_followup:
            (item_dir / f"followup_round_{round_idx}.raw.txt").write_text(raw_followup)
        write_json(item_dir / f"followup_round_{round_idx}.json", followup)
        state["rounds"][-1]["followup"] = followup
        followup_status = followup.get("status")
        if followup_status == "STALE_DUPLICATED":
            state["state"] = "STALE_DUPLICATED"
            write_json(item_dir / "state.json", state)
            return state
        if followup_status not in {"FIXED", "BUILT"}:
            state["state"] = "FAILED"
            write_json(item_dir / "state.json", state)
            return state
        current_builder = followup

    state["state"] = "FAILED"
    state["failure"] = f"Exceeded max review rounds: {max_rounds}"
    write_json(item_dir / "state.json", state)
    return state


def find_run_dir(repo: Path, run_id: str | None) -> Path:
    runs_dir = repo / ".automation" / "runs"
    if run_id:
        run_dir = runs_dir / run_id
        if not run_dir.exists():
            raise FlowError(f"approved run not found: {run_dir}")
        return run_dir
    candidates = sorted((p for p in runs_dir.iterdir() if p.is_dir() and (p / "branch_states.json").exists()), reverse=True) if runs_dir.exists() else []
    if not candidates:
        raise FlowError(f"no run with branch_states.json found under {runs_dir}")
    return candidates[0]


def load_publishable_states(run_dir: Path, approve_issue: int | None) -> list[dict[str, Any]]:
    path = run_dir / "branch_states.json"
    if not path.exists():
        raise FlowError(f"no branch states found for approval: {path}")
    states = json.loads(path.read_text())
    publishable = []
    for state in states:
        issue = state.get("issue") or {}
        number = int(issue.get("number") or 0)
        if approve_issue is not None and number != approve_issue:
            continue
        if state.get("state") not in {"PUBLISHED_OR_REPORTED", "PUBLISHED"}:
            continue
        if not state.get("branch") or not state.get("worktree"):
            continue
        publishable.append(state)
    if not publishable:
        target = f"issue #{approve_issue}" if approve_issue is not None else "latest run"
        raise FlowError(f"no approved human-review branch found for {target} in {run_dir}")
    return publishable


def git_ssh_prefix() -> str:
    key = Path("/home/hermes/.ssh/id_ed25519")
    if key.exists():
        return "GIT_SSH_COMMAND=" + shlex.quote(f"ssh -i {key} -o StrictHostKeyChecking=accept-new") + " "
    return ""


def push_branch_to_fork(worktree: Path, cfg: dict[str, Any], branch: str, dry_run: bool) -> str:
    origin = cfg["repo"].get("origin_remote", "origin")
    command = f"{git_ssh_prefix()}git push {shlex.quote(origin)} {shlex.quote(branch)}:{shlex.quote(branch)}"
    if dry_run:
        command += " --dry-run"
    proc = run_shell(command, cwd=worktree, timeout=300, check=False)
    if proc.returncode != 0:
        raise FlowError(f"git push failed for {branch} (exit {proc.returncode})\n{proc.stdout}")
    return proc.stdout.strip()


def create_pull_request(cfg: dict[str, Any], state: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    issue = state.get("issue") or {}
    issue_number = int(issue.get("number") or 0)
    issue_title = issue.get("title") or ""
    branch = state["branch"]
    fork_repo = cfg["repo"].get("fork_repo")
    base_branch = cfg["repo"].get("base_branch", "main")
    upstream_repo = cfg["repo"]["upstream_repo"]
    if not fork_repo:
        raise FlowError("repo.fork_repo is required to create a PR")
    owner = fork_repo.split("/", 1)[0]
    title = f"fix: {issue_title}"[:240]
    body = textwrap.dedent(f"""
    Fixes #{issue_number}

    Automated dual-agent bugfix flow approved this branch for publication.

    Issue: https://github.com/{upstream_repo}/issues/{issue_number}
    Branch: {branch}

    Verification evidence is in the automation run artifacts and branch state.
    """).strip()
    payload = {
        "title": title,
        "head": f"{owner}:{branch}",
        "base": base_branch,
        "body": body,
        "maintainer_can_modify": True,
    }
    if dry_run:
        return {"dry_run": True, "payload": payload}
    return github_api_post(f"/repos/{upstream_repo}/pulls", payload)


def publish_approved_branches(repo: Path, cfg: dict[str, Any], run_id: str | None, approve_issue: int | None, dry_run: bool) -> str:
    run_dir = find_run_dir(repo, run_id)
    states = load_publishable_states(run_dir, approve_issue)
    results = []
    for state in states:
        branch = state["branch"]
        worktree = Path(state["worktree"])
        if not worktree.exists():
            raise FlowError(f"worktree for {branch} does not exist: {worktree}")
        push_output = push_branch_to_fork(worktree, cfg, branch, dry_run=dry_run)
        pr = create_pull_request(cfg, state, dry_run=dry_run)
        issue = state.get("issue") or {}
        result = {
            "issue": issue.get("number"),
            "title": issue.get("title"),
            "branch": branch,
            "dry_run": dry_run,
            "push_output": push_output[-1000:],
            "pr": pr,
        }
        results.append(result)
    out_path = run_dir / ("publish_dry_run.json" if dry_run else "publish_results.json")
    write_json(out_path, results)
    lines = [
        "Hermes-agent approved branch publication " + ("dry-run" if dry_run else "complete"),
        f"Run: {run_dir.name}",
        f"Artifacts: {out_path}",
        "",
    ]
    for r in results:
        pr = r.get("pr") or {}
        pr_url = pr.get("html_url") or pr.get("url") or "(dry-run; no PR created)"
        lines.append(f"- #{r.get('issue')} {r.get('title')}: {r.get('branch')} -> {pr_url}")
    return "\n".join(lines)


def build_report(run_id: str, cfg: dict[str, Any], source: str, candidates: list[Candidate], skipped: list[Skipped], run_dir: Path, branch_states: list[dict[str, Any]] | None = None, selected_count: int | None = None) -> str:
    mention = mention_for(cfg)
    mode = cfg.get("publication", {}).get("mode", "human_approval")
    branch_states = branch_states or []
    if selected_count is None:
        selected_count = len(branch_states) if branch_states else len(candidates)
    counts = {state: 0 for state in ["PUBLISHED", "PUBLISHED_OR_REPORTED", "REJECTED", "STALE_DUPLICATED", "FAILED", "DRY_RUN_READY"]}
    for s in branch_states:
        counts[s.get("state", "FAILED")] = counts.get(s.get("state", "FAILED"), 0) + 1
    lines = [
        f"{mention} Hermes-agent autonomous fix cycle report",
        "",
        f"Run: {run_id}",
        f"Mode: {mode}",
        f"Cadence: {cfg.get('schedule', {}).get('cadence', 'every 4h')}",
        "Publication: human approval required" if mode != "full_auto_after_dual_approval" else "Publication: full auto after dual approval",
        f"GitHub source: {source}",
        "",
        "Summary:",
        f"- Good candidates found: {len(candidates)}",
        f"- Candidates selected for processing this run: {selected_count}",
        f"- Skipped/rejected during triage: {len(skipped)}",
        f"- Branches processed this run: {len(branch_states)}",
        f"- Branches approved/reported: {counts.get('PUBLISHED_OR_REPORTED', 0) + counts.get('PUBLISHED', 0)}",
        f"- Branches rejected/stale/failed: {counts.get('REJECTED', 0) + counts.get('STALE_DUPLICATED', 0) + counts.get('FAILED', 0)}",
        "- Total issue cap: none; candidates should be processed in waves",
        "- Approved branches policy: report/publish immediately; do not wait for unrelated review loops",
        "",
    ]
    if branch_states:
        lines.append("Branch results:")
        for s in branch_states[:20]:
            issue = s.get("issue", {})
            lines.append(f"- #{issue.get('number')} {s.get('state')}: {s.get('branch')}")
        lines.append("")
    if candidates:
        lines.append("Top candidates:")
        for c in candidates[:10]:
            files = f" files={', '.join(c.suspected_files[:3])}" if c.suspected_files else ""
            lines.append(f"- #{c.number} T{c.tier} {c.confidence}: {c.title}{files}")
        if len(candidates) > 10:
            lines.append(f"- ... plus {len(candidates) - 10} more candidates in candidates.json")
    else:
        lines.append("No suitable candidates found in this pass.")
    lines.extend(["", "Artifacts:", str(run_dir)])
    report = "\n".join(lines)
    max_chars = int(cfg.get("reporting", {}).get("max_discord_chars", 1900))
    if len(report) > max_chars:
        report = report[: max_chars - 120].rstrip() + f"\n\n[truncated for Discord; full report at {run_dir / 'REPORT.md'}]"
    return report


def write_full_report(path: Path, short_report: str, candidates: list[Candidate], skipped: list[Skipped], branch_states: list[dict[str, Any]] | None = None) -> None:
    lines = [short_report, "", "## Branch states", ""]
    for s in branch_states or []:
        issue = s.get("issue", {})
        lines.extend([
            f"### #{issue.get('number')} {issue.get('title')}",
            f"- State: {s.get('state')}",
            f"- Branch: {s.get('branch')}",
            f"- Worktree: {s.get('worktree')}",
            f"- Approved report artifact: {'yes' if s.get('approved_report') else 'no'}",
            "",
        ])
    lines.extend(["## Candidate details", ""])
    for c in candidates:
        lines.extend([
            f"### #{c.number} {c.title}",
            f"- URL: {c.url}",
            f"- Tier: {c.tier}",
            f"- Confidence: {c.confidence}",
            f"- Why: {c.why_selected}",
            f"- Suspected files: {', '.join(c.suspected_files) if c.suspected_files else '(none guessed)'}",
            f"- Verification: {c.verification_plan}",
            "",
        ])
    lines.extend(["## Skipped", ""])
    for s in skipped[:200]:
        lines.append(f"- #{s.number} {s.title}: {s.reason}")
    path.write_text("\n".join(lines) + "\n")


def write_discord_cron_helper(run_dir: Path, cfg: dict[str, Any], short_report: str) -> None:
    deliver = cfg.get("publication", {}).get("report_destination") or f"discord:{cfg['publication']['discord_channel']}"
    write_delivery_helper(run_dir / "discord_cron_delivery.json", "hermes-agent-auto-fix-report", deliver, short_report)


def build_failure_report(run_id: str, cfg: dict[str, Any], run_dir: Path, exc: Exception, repo: Path, args: argparse.Namespace) -> str:
    kind = "network unreachable" if isinstance(exc, URLError) or "urlopen error" in str(exc) else "failed"
    lines = [
        f"{mention_for(cfg)} 🌙 Hermes-agent autonomous bugfix flow — FAILED",
        "",
        f"Run: {run_id}",
        f"Status: {kind}",
        f"Phase: {args.phase}",
        f"Agent mode: {args.agent_mode}",
        f"Repository: {repo}",
        f"Artifacts: {run_dir}",
        "",
        f"Error: {exc}",
    ]
    if kind == "network unreachable":
        lines.extend([
            "",
            "Root cause: the runtime could not reach GitHub/provider endpoints. `--no-fetch` only skips `git fetch`; issue triage still needs GitHub API or gh access, and real builder/reviewer agents need outbound provider access.",
            "Next step: run this cron job in the host/gateway environment with DNS/network access and provider auth, or add a cached/offline repo-state input for triage tests.",
        ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=".automation/config.yaml")
    parser.add_argument("--phase", choices=["triage", "build", "full", "publish"], default="triage")
    parser.add_argument("--issue-limit", type=int, default=300, help="fetch window, not a total fix cap")
    parser.add_argument("--max-candidates", type=int, default=None, help="test/run throttle; omit for no total cap")
    parser.add_argument("--no-fetch", action="store_true", help="skip git fetch for local testing")
    parser.add_argument("--agent-mode", choices=["real", "mock", "dry-run"], default="dry-run")
    parser.add_argument("--dry-run-git", action="store_true", help="do not create/reset worktrees")
    parser.add_argument("--no-start-notice", action="store_true", help="do not send the best-effort Discord start notification")
    parser.add_argument("--no-cache-fallback", action="store_true", help="fail instead of using cached GitHub issue/PR state when live fetch is unavailable")
    parser.add_argument("--use-cache", action="store_true", help="use the latest cached GitHub issue/PR state without attempting network access")
    parser.add_argument("--full-issue-window", action="store_true", help="ignore the saved issue-number cursor and scan the latest --issue-limit open issues")
    parser.add_argument("--approve-run", default=None, help="run id containing approved branch_states.json for --phase publish")
    parser.add_argument("--approve-issue", type=int, default=None, help="only publish the approved branch for this issue number")
    parser.add_argument("--dry-run-publish", action="store_true", help="for --phase publish, validate push/PR payload without creating a PR")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    repo = resolve_repo_path(cfg_path, cfg)
    run_id = utc_run_id()
    run_dir = repo / ".automation" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "publish":
        try:
            result = publish_approved_branches(repo, cfg, args.approve_run, args.approve_issue, args.dry_run_publish)
            print(result)
            return 0
        except Exception as exc:
            print(f"publish failed: {exc}", file=sys.stderr)
            return 1

    if not args.no_start_notice:
        start_notice = build_start_notice(run_id, cfg, repo, run_dir, args)
        (run_dir / "STARTED.md").write_text(start_notice + "\n")
        send_discord_notice_via_hermes_cron(run_dir, cfg, "hermes-agent-auto-fix-started", start_notice)

    try:
        with acquire_lock(repo, float(cfg.get("schedule", {}).get("lock_timeout_hours", 4))):
            if not args.no_fetch:
                ensure_repo(repo, cfg["repo"].get("upstream_remote", "upstream"), cfg["repo"].get("base_branch", "main"))
            issues, prs_open, prs_closed, source = fetch_repo_state_with_cache(
                repo,
                cfg["repo"]["upstream_repo"],
                args.issue_limit,
                allow_cache=not args.no_cache_fallback,
                force_cache=args.use_cache,
                incremental=not args.full_issue_window,
            )
            candidates, skipped = classify_candidates(repo, cfg, issues, prs_open, prs_closed)
            live_duplicate_check = not (args.use_cache or source.startswith("cache:"))
            candidates = filter_candidates_with_live_duplicate_prs(cfg, candidates, skipped, live_duplicate_check)
            selected = select_candidates_for_wave(candidates, args.max_candidates)

            write_json(run_dir / "issues_raw.json", issues)
            write_json(run_dir / "prs_open_raw.json", prs_open)
            write_json(run_dir / "prs_closed_raw.json", prs_closed)
            write_json(run_dir / "contributor_guidelines.json", contributor_guideline_summary(repo))
            write_json(run_dir / "candidates.json", [candidate_to_dict(c) for c in candidates])
            write_json(run_dir / "selected_candidates.json", [candidate_to_dict(c) for c in selected])
            write_json(run_dir / "skipped.json", [skipped_to_dict(s) for s in skipped])
            if not args.use_cache and not source.startswith("cache:"):
                save_last_seen_issue_number(repo, run_id, source, issues)

            branch_states: list[dict[str, Any]] = []
            if args.phase in {"build", "full"}:
                max_parallel = int(cfg.get("agents", {}).get("builder", {}).get("max_parallel", 3))
                # The state machine reports approvals immediately as each candidate finishes.
                # This loop is sequential in-process for safety; external agent commands may
                # still run their own internal tools. Concurrency can be added by launching
                # process_candidate in a small worker pool once the first real run is stable.
                for c in selected:
                    state = process_candidate(repo, cfg, run_dir, c, issues, args.agent_mode, args.dry_run_git)
                    branch_states.append(state)
                    write_json(run_dir / "branch_states.json", branch_states)
                    if max_parallel < 1:
                        raise FlowError("agents.builder.max_parallel must be >= 1")

            short_report = build_report(run_id, cfg, source, candidates, skipped, run_dir, branch_states, selected_count=len(selected))
            write_full_report(run_dir / "REPORT.md", short_report, candidates, skipped, branch_states)
            write_discord_cron_helper(run_dir, cfg, short_report)
            print(short_report)
            return 0
    except Exception as exc:
        err = build_failure_report(run_id, cfg, run_dir, exc, repo, args)
        (run_dir / "ERROR.txt").write_text(err + "\n")
        write_discord_cron_helper(run_dir, cfg, err)
        print(err, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
