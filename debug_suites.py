"""Debug failing scenarios in suites B-E."""
import json
import os

# Load the last results file
with open("benchmarks/results/cognitive.json") as f:
    data = json.load(f)

# Show per-run details for failures
for run in data.get("runs", []):
    for cat_name, cat_data in run.get("categories", {}).items():
        failures = [d for d in cat_data.get("details", []) if not d.get("correct")]
        if failures:
            print(f"\n{'='*60}")
            print(f"FAILURES in {cat_name} ({len(failures)} failures)")
            print(f"{'='*60}")
            for f_item in failures:
                print(f"\n  ID: {f_item.get('id', '?')}")
                for k, v in f_item.items():
                    if k != 'id':
                        val_str = str(v)[:120]
                        print(f"    {k}: {val_str}")
