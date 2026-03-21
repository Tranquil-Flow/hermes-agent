"""Debug adversarial: check for superseded facts."""
import sys, os, json
sys.path.insert(0, '/workspace/Projects/hermes-agent')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/workspace/Projects/.huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/Projects/.huggingface_cache'

from cognitive_memory.benchmark_adapter import CognitiveBenchmarkAdapter

adapter = CognitiveBenchmarkAdapter()

with open('benchmarks/suite_d/fixtures/adversarial.json') as f:
    scenarios = json.load(f)

sc = [s for s in scenarios if s['id'] == 'ad_14'][0]

adapter.reset()
print("Storing normal facts:")
for fact in sc.get('normal_facts', []):
    adapter.store(fact, category='factual')
    print(f"  + {fact[:80]}")

# Check all memories (including superseded)
conn = adapter._store.backend._conn
cur = conn.execute("SELECT content, superseded_by FROM memories")
rows = cur.fetchall()
print(f"\nAll memories in DB ({len(rows)}):")
for row in rows:
    status = "ACTIVE" if row[1] is None else f"SUPERSEDED by {row[1]}"
    print(f"  [{status}] {row[0][:80]}")

print("\nStoring malicious fact:")
adapter.store(sc['malicious_fact'], category='factual')
print(f"  + {sc['malicious_fact'][:80]}")

cur = conn.execute("SELECT content, superseded_by FROM memories")
rows = cur.fetchall()
print(f"\nAll memories after malicious ({len(rows)}):")
for row in rows:
    status = "ACTIVE" if row[1] is None else f"SUPERSEDED by {row[1]}"
    print(f"  [{status}] {row[0][:80]}")
