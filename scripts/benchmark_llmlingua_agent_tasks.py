#!/usr/bin/env python3
"""Agent-task A/B benchmark for LLMLingua-2 tool compression.

This benchmark answers a different question than the simple compression-ratio
smoke test:

    Can Hermes still complete realistic tasks when old tool-result context is
    compressed with LLMLingua-2 instead of the historical drop-body summary?

It compares three lanes:

* full              — no compression oracle; upper-bound evidence quality
* drop              — historical Hermes context-pruning behavior before this PR
* llmlingua_default — PR behavior using the default LLMLingua tool allowlist

Optionally, ``llmlingua_all`` can be enabled to diagnose why structured tools are
excluded by default. Official PR claims should use ``llmlingua_default``.

The score is deterministic and evidence-based: each task has required and optional
facts, expressed as regexes. A lane is "answerable" only if all required facts
survive in the context the model would see. This is not a replacement for a paid
LLM-as-judge run, but it is reproducible, cheap, and directly probes information
loss at the compression seam.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from agent.tool_result_compressor import (
    CompressionResult,
    DropBodyCompressor,
    LLMLinguaConfig,
    LLMLinguaLocalCompressor,
    ToolResultCompressor,
    count_tokens,
)


@dataclass(frozen=True)
class Fact:
    """A fact that must or should survive compression."""

    id: str
    pattern: str
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class TaskCase:
    """One realistic Hermes task and the tool output that supports it."""

    id: str
    category: str
    tool_name: str
    tool_args: str
    question: str
    content: str
    facts: tuple[Fact, ...]
    notes: str = ""


class FullContextCompressor(ToolResultCompressor):
    """Oracle lane: preserve the full tool result body."""

    def compress(
        self,
        tool_name: str,
        tool_args: str,
        content: str,
        question: str | None = None,
    ) -> CompressionResult:
        return CompressionResult(
            compressed_text=content,
            original_tokens=count_tokens(content),
            compressed_tokens=count_tokens(content),
            latency_ms=0.0,
            cache_hit=False,
            fell_back=False,
        )


def _pad(topic: str, n: int = 18) -> str:
    """Generate realistic distractor prose so size gates are exercised."""
    paras = []
    for i in range(n):
        paras.append(
            f"Background note {i + 1}: the {topic} team also discussed unrelated "
            "rollout details, migration windows, telemetry dashboards, internal "
            "review meetings, stale screenshots, and user feedback summaries. "
            "These paragraphs are intentionally similar enough to act as noise, "
            "but they do not answer the benchmark question."
        )
    return "\n\n".join(paras)


def task_fixtures() -> tuple[TaskCase, ...]:
    """Representative Hermes tasks with close distractors and exact answers."""

    return (
        TaskCase(
            id="web_security_advisory",
            category="web_research",
            tool_name="web_extract",
            tool_args=json.dumps({"urls": ["https://example.org/security/hs-2026-17"]}),
            question="Which product versions are affected, what CVE is fixed, and what mitigation is recommended?",
            content=(
                "HermesShield Security Advisory HS-2026-17\n"
                "Affected versions: HermesShield Gateway 4.8.0 through 4.8.6.\n"
                "Fixed in: HermesShield Gateway 4.8.7.\n"
                "Identifier: CVE-2026-4242.\n"
                "Mitigation: rotate webhook signing secrets and disable legacy callback forwarding.\n"
                "Unaffected distractor: HermesVault 2.1.0 was patched for CVE-2026-1111.\n\n"
                + _pad("security advisory", 20)
            ),
            facts=(
                Fact("affected_versions", r"4\.\s*8\.\s*0\s+through\s+4\.\s*8\.\s*6"),
                Fact("fixed_version", r"4\.\s*8\.\s*7"),
                Fact("cve", r"CVE\s*-\s*2026\s*-\s*4242"),
                Fact("mitigation", r"rotate webhook signing secrets", True),
            ),
        ),
        TaskCase(
            id="api_migration_docs",
            category="docs_research",
            tool_name="web_extract",
            tool_args=json.dumps({"urls": ["https://docs.example.dev/migration/v3"]}),
            question="How do we migrate the export API to v3 without breaking idempotency?",
            content=(
                "Export API v3 Migration Guide\n"
                "Endpoint change: POST /v2/exports becomes POST /v3/jobs/export.\n"
                "Required header: Idempotency-Key must be a UUIDv7 value.\n"
                "Deprecated field: callback_url is replaced by webhook.destination_url.\n"
                "Safe rollout: send X-Export-Dry-Run: true for the first 24 hours.\n"
                "Distractor: Import API v3 still accepts UUIDv4 keys until 2027.\n\n"
                + _pad("API migration", 22)
            ),
            facts=(
                Fact("endpoint", r"POST\s*/\s*v3\s*/\s*jobs\s*/\s*export"),
                Fact("idempotency", r"Idempotency\s*-\s*Key\s+must\s+(?:be\s+)?(?:a\s+)?UUIDv7"),
                Fact("callback_replacement", r"webhook\s*\.\s*destination\s*_\s*url"),
                Fact("dry_run", r"X\s*-\s*Export\s*-\s*Dry\s*-\s*Run\s*:\s*true"),
            ),
        ),
        TaskCase(
            id="pricing_policy_research",
            category="web_research",
            tool_name="web_extract",
            tool_args=json.dumps({"urls": ["https://example.com/pricing/enterprise"]}),
            question="What are the enterprise pricing constraints and overage policy?",
            content=(
                "Enterprise Pricing Policy 2026\n"
                "Base plan: 2,000,000 included tool-call tokens per month.\n"
                "Overage: $0.42 per additional 10,000 tool-call tokens.\n"
                "Hard cap: customers may set a monthly spend ceiling in dollars.\n"
                "Exception: education tenants receive a 30% overage credit.\n"
                "Distractor: the Pro plan includes 200,000 tokens and a $0.60 overage.\n\n"
                + _pad("pricing policy", 19)
            ),
            facts=(
                Fact("included_tokens", r"2,\s*000,\s*000 included tool\s*-\s*call tokens"),
                Fact("overage", r"\$\s*0\.\s*42 per additional 10,\s*000"),
                Fact("hard_cap", r"monthly spend ceiling"),
                Fact("education_credit", r"30\s*% overage credit", required=False),
            ),
        ),
        TaskCase(
            id="browser_checkout_snapshot",
            category="browser_snapshot",
            tool_name="browser_snapshot",
            tool_args="{}",
            question="Which checkout error should the agent report to the user?",
            content=(
                "Accessibility snapshot for checkout page\n"
                "[@e11] textbox 'Email' value='evi@example.com'\n"
                "[@e12] combobox 'Country' value='France'\n"
                "[@e21] alert 'Payment failed: card requires 3-D Secure authentication.'\n"
                "[@e22] button 'Retry payment' enabled=true\n"
                "[@e23] link 'Use another payment method'\n"
                "Distractor alert: Shipping estimate delayed by 1 day.\n\n"
                + _pad("checkout UI", 18)
            ),
            facts=(
                Fact("payment_failed", r"Payment failed"),
                Fact("3ds", r"3\s*-\s*D Secure authentication"),
                Fact("retry_button", r"Retry payment", required=False),
            ),
        ),
        TaskCase(
            id="web_search_release_notes",
            category="web_search",
            tool_name="web_search",
            tool_args=json.dumps({"query": "AstraDB local-first release notes 1.14"}),
            question="Which release added encrypted local-first sync and where is the changelog?",
            content=json.dumps(
                {
                    "data": {
                        "web": [
                            {
                                "title": "AstraDB 1.14 Release Notes",
                                "url": "https://astradb.example/releases/1.14",
                                "description": "Adds encrypted local-first sync using XChaCha20-Poly1305 and background conflict repair.",
                            },
                            {
                                "title": "AstraDB 1.13 Maintenance",
                                "url": "https://astradb.example/releases/1.13",
                                "description": "Improves compaction and query planner hints.",
                            },
                        ]
                    }
                },
                indent=2,
            )
            + "\n\n"
            + _pad("release search", 16),
            facts=(
                Fact("version", r"AstraDB 1\.\s*14"),
                Fact("feature", r"encrypted local\s*-\s*first sync"),
                Fact("cipher", r"XChaCha20\s*-\s*Poly1305"),
                Fact("url", r"https://astradb\.example/releases/1\.14"),
            ),
        ),
        TaskCase(
            id="incident_feed_json",
            category="api_json_via_web",
            tool_name="web_extract",
            tool_args=json.dumps({"urls": ["https://status.example.net/api/incidents"]}),
            question="What is the active incident, which region is affected, and what is the ETA?",
            content=(
                json.dumps(
                    {
                        "incidents": [
                            {
                                "id": "inc_9af31",
                                "status": "active",
                                "title": "Webhook delivery delays",
                                "region": "eu-west-3",
                                "eta_minutes": 45,
                                "workaround": "switch callbacks to polling fallback",
                            },
                            {
                                "id": "inc_old_771",
                                "status": "resolved",
                                "title": "Dashboard CSS regression",
                                "region": "us-east-1",
                                "eta_minutes": 0,
                            },
                        ]
                    },
                    indent=2,
                )
                + "\n\n"
                + _pad("incident status feed", 18)
            ),
            facts=(
                Fact("incident_id", r"inc\s*_\s*9af31"),
                Fact("title", r"Webhook delivery delays"),
                Fact("region", r"eu\s*-\s*west\s*-\s*3"),
                Fact("eta", r"45"),
                Fact("workaround", r"polling fallback", required=False),
            ),
        ),
        TaskCase(
            id="terminal_failure_log_excluded",
            category="structured_excluded",
            tool_name="terminal",
            tool_args=json.dumps({"command": "pytest tests/gateway/test_webhooks.py -q"}),
            question="Which test failed and why?",
            content=(
                "============================= test session starts =============================\n"
                "FAILED tests/gateway/test_webhooks.py::test_signature_rejects_replayed_timestamp\n"
                "E   AssertionError: expected HTTP 401 but got HTTP 204\n"
                "Captured log: replay window was 900 seconds, expected 300 seconds\n"
                "exit_code: 1\n"
                + "\n".join(f"noise line {i}: unrelated passing test output" for i in range(400))
            ),
            facts=(
                Fact("test_name", r"test_signature_rejects_replayed_timestamp"),
                Fact("expected", r"expected HTTP 401"),
                Fact("actual", r"got HTTP 204"),
                Fact("window", r"900 seconds, expected 300 seconds"),
            ),
            notes="Structured terminal output is intentionally outside the default LLMLingua allowlist.",
        ),
        TaskCase(
            id="code_search_result_excluded",
            category="structured_excluded",
            tool_name="search_files",
            tool_args=json.dumps({"pattern": "trust_external_endpoint", "path": "agent"}),
            question="Which file and line enforces the external endpoint trust flag?",
            content=(
                "agent/tool_result_compressor.py:768:        if self.config.trust_external_endpoint:\n"
                "agent/tool_result_compressor.py:771:            'refusing to send tool-result text over the network'\n"
                "website/docs/user-guide/configuration.md:722: trust_external_endpoint must be explicitly enabled.\n"
                + "\n".join(f"docs/noise_{i}.md:{i}: unrelated endpoint mention" for i in range(300))
            ),
            facts=(
                Fact("file", r"agent/tool_result_compressor\.py"),
                Fact("line", r":768:"),
                Fact("flag", r"trust_external_endpoint"),
                Fact("warning", r"refusing to send tool-result text"),
            ),
            notes="Search/code output is intentionally outside the default LLMLingua allowlist.",
        ),
        TaskCase(
            id="browser_dashboard_status",
            category="browser_snapshot",
            tool_name="browser_snapshot",
            tool_args="{}",
            question="What health issue does the dashboard show and which queue is affected?",
            content=(
                "Dashboard snapshot\n"
                "[@e4] heading 'Operations'\n"
                "[@e9] status 'ingest-normal priority queue: degraded'\n"
                "[@e10] text 'Oldest pending job age: 37 minutes'\n"
                "[@e11] text 'Error rate: 6.2% over the last 15 minutes'\n"
                "[@e12] button 'Drain queue'\n"
                "Distractor: batch-low queue healthy, 0.1% errors.\n\n"
                + _pad("ops dashboard", 18)
            ),
            facts=(
                Fact("queue", r"ingest\s*-\s*normal priority queue"),
                Fact("status", r"degraded"),
                Fact("age", r"37 minutes"),
                Fact("error_rate", r"6\.\s*2\s*%"),
            ),
        ),
        TaskCase(
            id="oidc_docs",
            category="docs_research",
            tool_name="web_extract",
            tool_args=json.dumps({"urls": ["https://identity.example.dev/oidc"]}),
            question="What OIDC settings should the agent configure?",
            content=(
                "OIDC Integration Reference\n"
                "Issuer URL: https://identity.example.dev/oauth2/default\n"
                "JWKS endpoint: https://identity.example.dev/oauth2/default/.well-known/jwks.json\n"
                "Audience: hermes-agent-prod\n"
                "Clock skew allowance: 120 seconds.\n"
                "Required scope: agent.gateway.write\n"
                "Distractor: staging audience is hermes-agent-staging.\n\n"
                + _pad("OIDC configuration", 21)
            ),
            facts=(
                Fact("issuer", r"https://identity\.example\.dev/oauth2/default"),
                Fact("jwks", r"jwks\.json"),
                Fact("audience", r"hermes\s*-\s*agent\s*-\s*prod"),
                Fact("skew", r"120 seconds"),
                Fact("scope", r"agent\s*\.\s*gateway\s*\.\s*write", required=False),
            ),
        ),
        TaskCase(
            id="regulatory_policy_article",
            category="web_research",
            tool_name="web_extract",
            tool_args=json.dumps({"urls": ["https://policy.example.eu/ai-records"]}),
            question="What retention policy applies to model audit records?",
            content=(
                "AI Records Policy Summary\n"
                "High-impact automated decision systems must retain model audit records for 7 years.\n"
                "Records must include prompt templates, model version, reviewer identity, and appeal outcome.\n"
                "Deletion exception: user content may be redacted after 30 days if audit hashes remain verifiable.\n"
                "Distractor: low-risk chatbots keep analytics for 13 months.\n\n"
                + _pad("regulatory records", 20)
            ),
            facts=(
                Fact("retention", r"7 years"),
                Fact("records", r"prompt templates\W+model version\W+reviewer identity\W+(?:and\W+)?appeal outcome"),
                Fact("redaction", r"redacted after 30 days"),
                Fact("hashes", r"audit hashes remain verifiable"),
            ),
        ),
        TaskCase(
            id="package_changelog",
            category="docs_research",
            tool_name="web_extract",
            tool_args=json.dumps({"urls": ["https://packages.example.dev/vega-agent/changelog"]}),
            question="Which changelog entry fixes the streaming cancel bug?",
            content=(
                "vega-agent changelog\n"
                "Version 0.9.12: fixes streaming cancellation leak where cancelled SSE tasks kept provider sockets open.\n"
                "Patch note: cleanup now awaits aclose() with a 2 second timeout.\n"
                "Regression test: test_stream_cancel_closes_provider_socket.\n"
                "Distractor: version 0.9.11 fixed markdown table rendering.\n\n"
                + _pad("package changelog", 18)
            ),
            facts=(
                Fact("version", r"Version 0\.\s*9\.\s*12"),
                Fact("bug", r"streaming cancellation leak"),
                Fact("fix", r"awaits aclose\s*\(\s*\)\s*(?:with\s+a\s+)?2 second timeout"),
                Fact("test", r"test\s*_\s*stream\s*_\s*cancel\s*_\s*closes\s*_\s*provider\s*_\s*socket", required=False),
            ),
        ),
    )


def build_compressor(mode: str, *, force_all_tools: Sequence[str] = ()) -> ToolResultCompressor:
    """Construct the compressor lane using the real production classes."""
    if mode == "full":
        return FullContextCompressor()
    if mode == "drop":
        return DropBodyCompressor()
    if mode == "llmlingua_default":
        return LLMLinguaLocalCompressor(LLMLinguaConfig())
    if mode == "llmlingua_all":
        tools = tuple(dict.fromkeys(force_all_tools))
        return LLMLinguaLocalCompressor(LLMLinguaConfig(only_tools=tools))
    raise ValueError(f"unknown mode: {mode}")


def _scoring_variants(text: str) -> tuple[str, ...]:
    """Return text variants for robust deterministic scoring.

    LLMLingua-2 is a token-classifier compressor, not a byte-preserving filter:
    it often inserts spaces around punctuation (`CVE - 2026 - 4242`,
    `webhook. destination _ url`). A human or LLM can still read those facts, so
    the benchmark scorer must not require byte-exact punctuation survival.
    """
    compact_punctuation = re.sub(r"\s*([./_:@$%(),\[\]])\s*", r"\1", text)
    compact_hyphen = re.sub(r"\s*-\s*", "-", compact_punctuation)
    compact_numbers = re.sub(r"(?<=\d),\s+(?=\d)", ",", compact_hyphen)
    collapsed = re.sub(r"\s+", " ", compact_numbers)
    return (text, collapsed)


def evaluate_text(task: TaskCase, text: str) -> dict:
    """Score preserved evidence using fact regexes."""
    variants = _scoring_variants(text)
    found_required: list[str] = []
    missing_required: list[str] = []
    found_optional: list[str] = []
    missing_optional: list[str] = []

    for fact in task.facts:
        found = any(
            re.search(fact.pattern, variant, flags=re.IGNORECASE | re.MULTILINE)
            is not None
            for variant in variants
        )
        if fact.required:
            (found_required if found else missing_required).append(fact.id)
        else:
            (found_optional if found else missing_optional).append(fact.id)

    total = len(task.facts)
    found_count = len(found_required) + len(found_optional)
    required_total = sum(1 for f in task.facts if f.required)
    optional_total = total - required_total
    return {
        "required_found": len(found_required),
        "required_total": required_total,
        "optional_found": len(found_optional),
        "optional_total": optional_total,
        "fact_recall": found_count / total if total else 1.0,
        "required_recall": len(found_required) / required_total if required_total else 1.0,
        "answerable": not missing_required,
        "found_required": found_required,
        "missing_required": missing_required,
        "found_optional": found_optional,
        "missing_optional": missing_optional,
    }


def is_default_llmlingua_candidate(task: TaskCase) -> bool:
    """Whether default LLMLingua config should preserve this tool body.

    Non-candidates are intentionally delegated to DropBodyCompressor because
    structured outputs such as terminal logs and code-search results are too
    easy to damage with natural-language compression.
    """
    cfg = LLMLinguaConfig()
    return task.tool_name in cfg.only_tools and len(task.content) >= cfg.min_output_chars


def run_task(task: TaskCase, compressor: ToolResultCompressor, *, mode: str | None = None) -> dict:
    """Run one task through a compressor lane and score evidence survival."""
    if mode is None:
        if isinstance(compressor, FullContextCompressor):
            mode = "full"
        elif isinstance(compressor, DropBodyCompressor):
            mode = "drop"
        else:
            mode = "llmlingua"

    result = compressor.compress(task.tool_name, task.tool_args, task.content, task.question)
    text = result.compressed_text
    eval_result = evaluate_text(task, text)
    ratio = len(task.content) / max(len(text), 1)
    token_ratio = result.original_tokens / max(result.compressed_tokens, 1)
    return {
        "id": task.id,
        "category": task.category,
        "tool_name": task.tool_name,
        "default_llmlingua_candidate": is_default_llmlingua_candidate(task),
        "mode": mode,
        "input_chars": len(task.content),
        "compressed_chars": len(text),
        "input_tokens_est": result.original_tokens,
        "compressed_tokens_est": result.compressed_tokens,
        "compression_ratio": ratio,
        "token_compression_ratio": token_ratio,
        "latency_ms": result.latency_ms,
        "fell_back": result.fell_back,
        "cache_hit": result.cache_hit,
        "compressed_preview": text[:500].replace("\n", "\\n"),
        **eval_result,
    }


def _aggregate_subset(rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "tasks": 0,
            "answerable": 0,
            "answerability": 0.0,
            "fact_recall_mean": 0.0,
            "required_recall_mean": 0.0,
            "input_tokens_est": 0,
            "compressed_tokens_est": 0,
            "token_compression_ratio": 0.0,
            "chars_in": 0,
            "chars_out": 0,
            "char_compression_ratio": 0.0,
            "latency_ms_mean": 0.0,
            "fell_back_count": 0,
        }
    return {
        "tasks": len(rows),
        "answerable": sum(1 for r in rows if r["answerable"]),
        "answerability": sum(1 for r in rows if r["answerable"]) / len(rows),
        "fact_recall_mean": statistics.fmean(r["fact_recall"] for r in rows),
        "required_recall_mean": statistics.fmean(r["required_recall"] for r in rows),
        "input_tokens_est": sum(r["input_tokens_est"] for r in rows),
        "compressed_tokens_est": sum(r["compressed_tokens_est"] for r in rows),
        "token_compression_ratio": (
            sum(r["input_tokens_est"] for r in rows)
            / max(sum(r["compressed_tokens_est"] for r in rows), 1)
        ),
        "chars_in": sum(r["input_chars"] for r in rows),
        "chars_out": sum(r["compressed_chars"] for r in rows),
        "char_compression_ratio": sum(r["input_chars"] for r in rows) / max(sum(r["compressed_chars"] for r in rows), 1),
        "latency_ms_mean": statistics.fmean(r["latency_ms"] for r in rows),
        "fell_back_count": sum(1 for r in rows if r["fell_back"]),
    }


def aggregate(rows: Sequence[dict]) -> dict:
    """Aggregate per-task rows for one benchmark lane."""
    all_rows = _aggregate_subset(rows)
    candidates = [r for r in rows if r.get("default_llmlingua_candidate")]
    excluded = [r for r in rows if not r.get("default_llmlingua_candidate")]
    all_rows["default_llmlingua_candidates"] = _aggregate_subset(candidates)
    all_rows["default_llmlingua_excluded"] = _aggregate_subset(excluded)
    return all_rows


def run_benchmark(modes: Sequence[str], tasks: Sequence[TaskCase]) -> dict:
    """Run all requested lanes."""
    all_tools = tuple(t.tool_name for t in tasks)
    results: dict[str, list[dict]] = {}
    started = time.perf_counter()
    for mode in modes:
        compressor = build_compressor(mode, force_all_tools=all_tools)
        rows = [run_task(task, compressor, mode=mode) for task in tasks]
        results[mode] = rows
    return {
        "schema_version": 1,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "duration_s": round(time.perf_counter() - started, 3),
        "modes": list(modes),
        "task_count": len(tasks),
        "tasks": [asdict(t) for t in tasks],
        "results": results,
        "summary": {mode: aggregate(rows) for mode, rows in results.items()},
    }


def print_table(payload: dict) -> None:
    """Terminal-friendly summary table."""
    print("\nAgent-task LLMLingua A/B benchmark")
    print("=" * 72)
    print(f"Tasks: {payload['task_count']}   Duration: {payload['duration_s']}s")
    print()
    print(f"{'Mode':20s} {'Answerable':>12s} {'FactRecall':>11s} {'ReqRecall':>10s} {'TokRatio':>9s} {'Fallbacks':>9s} {'Latency':>9s}")
    print("-" * 92)
    for mode in payload["modes"]:
        s = payload["summary"].get(mode, {})
        print(
            f"{mode:20s} "
            f"{s.get('answerable', 0):>4}/{s.get('tasks', 0):<7} "
            f"{s.get('fact_recall_mean', 0):>10.1%} "
            f"{s.get('required_recall_mean', 0):>9.1%} "
            f"{s.get('token_compression_ratio', 0):>8.2f}x "
            f"{s.get('fell_back_count', 0):>9} "
            f"{s.get('latency_ms_mean', 0):>8.1f}ms"
        )

    print("\nDefault-allowlist subset (web/browser outputs LLMLingua is meant to preserve)")
    print("-" * 92)
    for mode in payload["modes"]:
        s = payload["summary"].get(mode, {}).get("default_llmlingua_candidates", {})
        print(
            f"{mode:20s} "
            f"{s.get('answerable', 0):>4}/{s.get('tasks', 0):<7} "
            f"{s.get('fact_recall_mean', 0):>10.1%} "
            f"{s.get('required_recall_mean', 0):>9.1%} "
            f"{s.get('token_compression_ratio', 0):>8.2f}x "
            f"{s.get('fell_back_count', 0):>9} "
            f"{s.get('latency_ms_mean', 0):>8.1f}ms"
        )

    print("\nPer-task deltas (drop → llmlingua_default)")
    print("-" * 72)
    if "drop" in payload["results"] and "llmlingua_default" in payload["results"]:
        drop = {r["id"]: r for r in payload["results"]["drop"]}
        lng = {r["id"]: r for r in payload["results"]["llmlingua_default"]}
        for task_id, after in lng.items():
            before = drop[task_id]
            mark = "✓" if after["answerable"] else "✗"
            print(
                f"{mark} {task_id:34s} "
                f"req {before['required_recall']:.0%}→{after['required_recall']:.0%} "
                f"facts {before['fact_recall']:.0%}→{after['fact_recall']:.0%} "
                f"ratio {after['token_compression_ratio']:.2f}x"
            )


def write_markdown_report(payload: dict, path: Path) -> None:
    """Write a compact report suitable for PR descriptions."""
    lines = [
        "# LLMLingua-2 Agent-Task A/B Benchmark",
        "",
        f"Generated: {payload['timestamp']}",
        f"Tasks: {payload['task_count']}",
        "",
        "## Summary",
        "",
        "| Mode | Answerable | Fact recall | Required recall | Token compression | Fallbacks | Mean latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in payload["modes"]:
        s = payload["summary"][mode]
        lines.append(
            f"| {mode} | {s['answerable']}/{s['tasks']} ({s['answerability']:.1%}) "
            f"| {s['fact_recall_mean']:.1%} | {s['required_recall_mean']:.1%} "
            f"| {s['token_compression_ratio']:.2f}x | {s['fell_back_count']} "
            f"| {s['latency_ms_mean']:.1f} ms |"
        )
    lines.extend([
        "",
        "## Default allowlist subset",
        "",
        "These are the web/browser tool outputs LLMLingua is expected to preserve. "
        "Structured outputs such as terminal and code search are intentionally "
        "delegated to the historical drop-body compressor by default.",
        "",
        "| Mode | Answerable | Fact recall | Required recall | Token compression | Fallbacks | Mean latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for mode in payload["modes"]:
        s = payload["summary"][mode]["default_llmlingua_candidates"]
        lines.append(
            f"| {mode} | {s['answerable']}/{s['tasks']} ({s['answerability']:.1%}) "
            f"| {s['fact_recall_mean']:.1%} | {s['required_recall_mean']:.1%} "
            f"| {s['token_compression_ratio']:.2f}x | {s['fell_back_count']} "
            f"| {s['latency_ms_mean']:.1f} ms |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "`full` is the no-compression oracle. `drop` is the historical Hermes "
        "context-pruning behavior. `llmlingua_default` is the PR path with the "
        "default allowlist. A task is answerable only if every required fact "
        "survives in the context that would be sent to the model.",
        "",
        "## Per-task results",
        "",
        "| Task | Tool | Mode | Answerable | Required recall | Fact recall | Token compression | Fell back | Missing required |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for mode in payload["modes"]:
        for row in payload["results"][mode]:
            missing = ", ".join(row["missing_required"]) or "—"
            lines.append(
                f"| {row['id']} | {row['tool_name']} | {mode} | {row['answerable']} "
                f"| {row['required_recall']:.1%} | {row['fact_recall']:.1%} "
                f"| {row['token_compression_ratio']:.2f}x | {row['fell_back']} | {missing} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        default="full,drop,llmlingua_default",
        help="Comma-separated modes: full,drop,llmlingua_default,llmlingua_all",
    )
    parser.add_argument("--json-out", default="benchmarks/results/llmlingua_agent_tasks.json")
    parser.add_argument("--markdown-out", default="benchmarks/results/llmlingua_agent_tasks.md")
    parser.add_argument("--no-report", action="store_true", help="Do not write JSON/Markdown reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    tasks = task_fixtures()
    payload = run_benchmark(modes, tasks)
    print_table(payload)
    if not args.no_report:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_path = Path(args.markdown_out)
        write_markdown_report(payload, md_path)
        print(f"\nWrote JSON: {json_path}")
        print(f"Wrote report: {md_path}")


if __name__ == "__main__":
    main()
