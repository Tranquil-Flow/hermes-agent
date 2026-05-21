# LLMLingua-2 Agent-Task A/B Benchmark

Generated: 2026-05-21T06:31:28.982978+00:00
Tasks: 12

## Summary

| Mode | Answerable | Fact recall | Required recall | Token compression | Fallbacks | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| full | 12/12 (100.0%) | 100.0% | 100.0% | 1.00x | 0 | 0.0 ms |
| drop | 0/12 (0.0%) | 2.1% | 2.1% | 122.34x | 0 | 0.0 ms |
| llmlingua_default | 10/12 (83.3%) | 85.4% | 85.4% | 2.66x | 0 | 31425.5 ms |

## Default allowlist subset

These are the web/browser tool outputs LLMLingua is expected to preserve. Structured outputs such as terminal and code search are intentionally delegated to the historical drop-body compressor by default.

| Mode | Answerable | Fact recall | Required recall | Token compression | Fallbacks | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| full | 10/10 (100.0%) | 100.0% | 100.0% | 1.00x | 0 | 0.0 ms |
| drop | 0/10 (0.0%) | 0.0% | 0.0% | 102.13x | 0 | 0.0 ms |
| llmlingua_default | 10/10 (100.0%) | 100.0% | 100.0% | 1.75x | 0 | 37689.9 ms |

## Interpretation

`full` is the no-compression oracle. `drop` is the historical Hermes context-pruning behavior. `llmlingua_default` is the PR path with the default allowlist. A task is answerable only if every required fact survives in the context that would be sent to the model.

## Per-task results

| Task | Tool | Mode | Answerable | Required recall | Fact recall | Token compression | Fell back | Missing required |
|---|---|---|---:|---:|---:|---:|---:|---|
| web_security_advisory | web_extract | full | True | 100.0% | 100.0% | 1.00x | False | — |
| api_migration_docs | web_extract | full | True | 100.0% | 100.0% | 1.00x | False | — |
| pricing_policy_research | web_extract | full | True | 100.0% | 100.0% | 1.00x | False | — |
| browser_checkout_snapshot | browser_snapshot | full | True | 100.0% | 100.0% | 1.00x | False | — |
| web_search_release_notes | web_search | full | True | 100.0% | 100.0% | 1.00x | False | — |
| incident_feed_json | web_extract | full | True | 100.0% | 100.0% | 1.00x | False | — |
| terminal_failure_log_excluded | terminal | full | True | 100.0% | 100.0% | 1.00x | False | — |
| code_search_result_excluded | search_files | full | True | 100.0% | 100.0% | 1.00x | False | — |
| browser_dashboard_status | browser_snapshot | full | True | 100.0% | 100.0% | 1.00x | False | — |
| oidc_docs | web_extract | full | True | 100.0% | 100.0% | 1.00x | False | — |
| regulatory_policy_article | web_extract | full | True | 100.0% | 100.0% | 1.00x | False | — |
| package_changelog | web_extract | full | True | 100.0% | 100.0% | 1.00x | False | — |
| web_security_advisory | web_extract | drop | False | 0.0% | 0.0% | 98.47x | False | affected_versions, fixed_version, cve, mitigation |
| api_migration_docs | web_extract | drop | False | 0.0% | 0.0% | 106.94x | False | endpoint, idempotency, callback_replacement, dry_run |
| pricing_policy_research | web_extract | drop | False | 0.0% | 0.0% | 93.12x | False | included_tokens, overage, hard_cap |
| browser_checkout_snapshot | browser_snapshot | drop | False | 0.0% | 0.0% | 186.38x | False | payment_failed, 3ds |
| web_search_release_notes | web_search | drop | False | 0.0% | 0.0% | 68.85x | False | version, feature, cipher, url |
| incident_feed_json | web_extract | drop | False | 0.0% | 0.0% | 91.06x | False | incident_id, title, region, eta |
| terminal_failure_log_excluded | terminal | drop | False | 0.0% | 0.0% | 211.18x | False | test_name, expected, actual, window |
| code_search_result_excluded | search_files | drop | False | 25.0% | 25.0% | 179.33x | False | file, line, warning |
| browser_dashboard_status | browser_snapshot | drop | False | 0.0% | 0.0% | 185.75x | False | queue, status, age, error_rate |
| oidc_docs | web_extract | drop | False | 0.0% | 0.0% | 109.81x | False | issuer, jwks, audience, skew |
| regulatory_policy_article | web_extract | drop | False | 0.0% | 0.0% | 105.50x | False | retention, records, redaction, hashes |
| package_changelog | web_extract | drop | False | 0.0% | 0.0% | 75.55x | False | version, bug, fix |
| web_security_advisory | web_extract | llmlingua_default | True | 100.0% | 100.0% | 1.74x | False | — |
| api_migration_docs | web_extract | llmlingua_default | True | 100.0% | 100.0% | 1.74x | False | — |
| pricing_policy_research | web_extract | llmlingua_default | True | 100.0% | 100.0% | 1.76x | False | — |
| browser_checkout_snapshot | browser_snapshot | llmlingua_default | True | 100.0% | 100.0% | 1.76x | False | — |
| web_search_release_notes | web_search | llmlingua_default | True | 100.0% | 100.0% | 1.72x | False | — |
| incident_feed_json | web_extract | llmlingua_default | True | 100.0% | 100.0% | 1.74x | False | — |
| terminal_failure_log_excluded | terminal | llmlingua_default | False | 0.0% | 0.0% | 211.18x | False | test_name, expected, actual, window |
| code_search_result_excluded | search_files | llmlingua_default | False | 25.0% | 25.0% | 179.33x | False | file, line, warning |
| browser_dashboard_status | browser_snapshot | llmlingua_default | True | 100.0% | 100.0% | 1.76x | False | — |
| oidc_docs | web_extract | llmlingua_default | True | 100.0% | 100.0% | 1.77x | False | — |
| regulatory_policy_article | web_extract | llmlingua_default | True | 100.0% | 100.0% | 1.72x | False | — |
| package_changelog | web_extract | llmlingua_default | True | 100.0% | 100.0% | 1.76x | False | — |
