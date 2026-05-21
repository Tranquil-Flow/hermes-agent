# Tier 1: LLM Generation + Judge Benchmark

Generated: 2026-05-21T06:42:30.738475+00:00
Generation model: MiniMax-M2.7
Tasks: 12
Duration: 348.9s

## Overall Results

| Mode | Judge correct | Accuracy | Score | Facts answerable | Avg gen tokens | Avg wall-clock | Avg comp latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 12/12 (100.0%) | 1.00 | 12/12 (100.0%) | 1910 | 6228 ms | 0 ms |
| drop | 0/12 (0.0%) | 0.11 | 0/12 (0.0%) | 342 | 10032 ms | 0 ms |
| llmlingua_default | 8/12 (66.7%) | 0.81 | 10/12 (83.3%) | 780 | 12794 ms | 1412 ms |

## Default-allowlist subset (web/browser)

| Mode | Judge correct | Accuracy | Score | Facts answerable | Avg gen tokens | Avg wall-clock | Avg comp latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 10/10 (100.0%) | 1.00 | 10/10 (100.0%) | 1412 | 6196 ms | 0 ms |
| drop | 0/10 (0.0%) | 0.10 | 0/10 (0.0%) | 349 | 11011 ms | 0 ms |
| llmlingua_default | 8/10 (80.0%) | 0.94 | 10/10 (100.0%) | 873 | 14216 ms | 1694 ms |

## Per-task detail

| Task | Tool | Mode | Judge ✓ | Score | Facts ✓ | Gen tokens | Wall-clock |
|---|---|---|---:|---:|---:|---:|---:|
| web_security_advisory | web_extract | full | ✓ | 1.00 | ✓ | 1489 | 1937 ms |
| api_migration_docs | web_extract | full | ✓ | 1.00 | ✓ | 1737 | 6762 ms |
| pricing_policy_research | web_extract | full | ✓ | 1.00 | ✓ | 1398 | 3881 ms |
| browser_checkout_snapshot | browser_snapshot | full | ✓ | 1.00 | ✓ | 1307 | 4150 ms |
| web_search_release_notes | web_search | full | ✓ | 1.00 | ✓ | 1221 | 2880 ms |
| incident_feed_json | web_extract | full | ✓ | 1.00 | ✓ | 1367 | 9938 ms |
| terminal_failure_log_excluded | terminal | full | ✓ | 1.00 | ✓ | 4513 | 1802 ms |
| code_search_result_excluded | search_files | full | ✓ | 1.00 | ✓ | 4288 | 10974 ms |
| browser_dashboard_status | browser_snapshot | full | ✓ | 1.00 | ✓ | 1394 | 10264 ms |
| oidc_docs | web_extract | full | ✓ | 1.00 | ✓ | 1518 | 3801 ms |
| regulatory_policy_article | web_extract | full | ✓ | 1.00 | ✓ | 1412 | 14902 ms |
| package_changelog | web_extract | full | ✓ | 1.00 | ✓ | 1273 | 3449 ms |
| web_security_advisory | web_extract | drop | ✗ | 0.00 | ✗ | 309 | 4679 ms |
| api_migration_docs | web_extract | drop | ✗ | 0.00 | ✗ | 276 | 5652 ms |
| pricing_policy_research | web_extract | drop | ✗ | 0.00 | ✗ | 312 | 22951 ms |
| browser_checkout_snapshot | browser_snapshot | drop | ✗ | 0.00 | ✗ | 410 | 14092 ms |
| web_search_release_notes | web_search | drop | ✗ | 0.67 | ✗ | 228 | 3515 ms |
| incident_feed_json | web_extract | drop | ✗ | 0.00 | ✗ | 275 | 3502 ms |
| terminal_failure_log_excluded | terminal | drop | ✗ | 0.00 | ✗ | 269 | 3875 ms |
| code_search_result_excluded | search_files | drop | ✗ | 0.33 | ✗ | 351 | 6394 ms |
| browser_dashboard_status | browser_snapshot | drop | ✗ | 0.00 | ✗ | 477 | 9721 ms |
| oidc_docs | web_extract | drop | ✗ | 0.33 | ✗ | 303 | 1685 ms |
| regulatory_policy_article | web_extract | drop | ✗ | 0.00 | ✗ | 366 | 10658 ms |
| package_changelog | web_extract | drop | ✗ | 0.00 | ✗ | 530 | 33660 ms |
| web_security_advisory | web_extract | llmlingua_default | ✓ | 1.00 | ✓ | 943 | 17701 ms |
| api_migration_docs | web_extract | llmlingua_default | ✗ | 0.75 | ✓ | 1079 | 60552 ms |
| pricing_policy_research | web_extract | llmlingua_default | ✓ | 1.00 | ✓ | 997 | 11463 ms |
| browser_checkout_snapshot | browser_snapshot | llmlingua_default | ✓ | 1.00 | ✓ | 790 | 2479 ms |
| web_search_release_notes | web_search | llmlingua_default | ✓ | 1.00 | ✓ | 790 | 19088 ms |
| incident_feed_json | web_extract | llmlingua_default | ✓ | 1.00 | ✓ | 752 | 3488 ms |
| terminal_failure_log_excluded | terminal | llmlingua_default | ✗ | 0.00 | ✗ | 309 | 5958 ms |
| code_search_result_excluded | search_files | llmlingua_default | ✗ | 0.33 | ✗ | 319 | 5413 ms |
| browser_dashboard_status | browser_snapshot | llmlingua_default | ✓ | 1.00 | ✓ | 880 | 11768 ms |
| oidc_docs | web_extract | llmlingua_default | ✓ | 1.00 | ✓ | 981 | 6231 ms |
| regulatory_policy_article | web_extract | llmlingua_default | ✓ | 1.00 | ✓ | 798 | 7271 ms |
| package_changelog | web_extract | llmlingua_default | ✗ | 0.67 | ✓ | 718 | 2119 ms |

## Delta: full vs llmlingua_default

| Task | Full judge | LLMLingua judge | Full tokens | LLMLingua tokens | Token delta | Full wall | LLMLingua wall | Wall delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| web_security_advisory | ✓ | ✓ | 1489 | 943 | -546 | 1937ms | 17701ms | +15764ms |
| api_migration_docs | ✓ | ✗ | 1737 | 1079 | -658 | 6762ms | 60552ms | +53790ms |
| pricing_policy_research | ✓ | ✓ | 1398 | 997 | -401 | 3881ms | 11463ms | +7582ms |
| browser_checkout_snapshot | ✓ | ✓ | 1307 | 790 | -517 | 4150ms | 2479ms | -1671ms |
| web_search_release_notes | ✓ | ✓ | 1221 | 790 | -431 | 2880ms | 19088ms | +16208ms |
| incident_feed_json | ✓ | ✓ | 1367 | 752 | -615 | 9938ms | 3488ms | -6451ms |
| terminal_failure_log_excluded | ✓ | ✗ | 4513 | 309 | -4204 | 1802ms | 5958ms | +4156ms |
| code_search_result_excluded | ✓ | ✗ | 4288 | 319 | -3969 | 10974ms | 5413ms | -5561ms |
| browser_dashboard_status | ✓ | ✓ | 1394 | 880 | -514 | 10264ms | 11768ms | +1504ms |
| oidc_docs | ✓ | ✓ | 1518 | 981 | -537 | 3801ms | 6231ms | +2430ms |
| regulatory_policy_article | ✓ | ✓ | 1412 | 798 | -614 | 14902ms | 7271ms | -7631ms |
| package_changelog | ✓ | ✗ | 1273 | 718 | -555 | 3449ms | 2119ms | -1331ms |
