from cognitive_memory.benchmark_adapter import BenchmarkableStore
import time

store = BenchmarkableStore()
store.reset()

# Store a global memory and a project-scoped one
store.store('The sky is blue', scope='global', importance=0.7)
store.store('Hermes uses Python', scope='project:hermes', importance=0.7)
store.store('Alice likes coffee', scope='user:alice', importance=0.7)

# Query with project:hermes scope
results = store.recall('Python', scope='project:hermes', top_k=3)
print('Recall scope=project:hermes, query=Python:')
for r in results:
    print(f'  [{r.entry.scope}] {r.entry.content[:60]}')
