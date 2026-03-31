# Memory Benchmark Suite

A system-agnostic benchmark for evaluating AI agent memory systems. Tests retrieval quality, temporal reasoning, contradiction handling, scale performance, and advanced features like typed facts and scope management.

## Quick Start

```bash
# Run with the built-in baseline (no dependencies)
python -m benchmarks.runner --backend baseline-flat --suite a,b,c,d,e,f

# Run all suites including advanced features
python -m benchmarks.runner --backend baseline-flat --suite all

# Compare two backends
python -m benchmarks.runner --backend my-backend --suite all --compare results/baseline.json
```

## Adding Your Own Backend

Implement the `BenchmarkableStore` interface and register it:

```python
# my_memory/benchmark_adapter.py
from benchmarks.interface import BenchmarkableStore
from typing import Dict, List, Any, Optional

class MyMemoryBenchmarkAdapter(BenchmarkableStore):
    def __init__(self, **kwargs):
        # Initialize your memory system
        self.store_data = {}
    
    def store(self, content: str, category: str = "factual",
              scope: str = "global", importance: float = 0.5) -> None:
        """Store a memory."""
        ...
    
    def recall(self, query: str, top_k: int = 10,
               scope: Optional[str] = None) -> List[str]:
        """Recall memories matching the query. Return content strings ranked by relevance."""
        ...
    
    def simulate_time(self, days: float) -> None:
        """Advance simulated clock. For decay/rehearsal testing."""
        ...
    
    def simulate_access(self, content_substring: str) -> None:
        """Simulate accessing/rehearsing a memory."""
        ...
    
    def consolidate(self) -> None:
        """Run consolidation/compaction. Noop if not supported."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Return backend statistics."""
        return {"fact_count": len(self.store_data)}
    
    def reset(self) -> None:
        """Clear all stored memories."""
        self.store_data.clear()
    
    # Optional overrides:
    def reward_memory(self, memory_id: str, signal: float) -> None:
        """Apply reward signal (for RL-based backends)."""
        pass
    
    def explore(self, query: str, top_k: int = 20,
                scope: Optional[str] = None) -> List[str]:
        """Multi-hop exploration (for graph-based backends)."""
        return self.recall(query, top_k=top_k, scope=scope)
```

Register it:

```python
# In your code, before running benchmarks:
from benchmarks.runner import register_backend
from my_memory.benchmark_adapter import MyMemoryBenchmarkAdapter

register_backend("my-backend", MyMemoryBenchmarkAdapter)
```

Or use the plugin system (auto-discovery):

```python
# Create benchmarks/backends/my_backend.py
from benchmarks.interface import BenchmarkableStore

class MyBackend(BenchmarkableStore):
    ...

# Export for auto-discovery
BACKEND_NAME = "my-backend"
BACKEND_CLASS = MyBackend
```

## Benchmark Suites

### Suite A — Core Retrieval (200 scenarios)
| Category | Scenarios | Tests |
|----------|-----------|-------|
| semantic_recall | 50 | Store a fact, recall with paraphrased query |
| contradictions | 20 | Newer fact should supersede older contradicting fact |
| temporal_decay | 45 | Recent facts should rank higher than old ones |
| cross_reference | 45 | Recall requiring multiple related facts |
| importance_filtering | 40 | High-importance facts should beat noise |

### Suite B — Consolidation (30 scenarios)
| Category | Scenarios | Tests |
|----------|-----------|-------|
| consolidation | 20 | Facts survive consolidation cycles |
| compression | 10 | Key info preserved after compression/merging |

### Suite C — Scopes (20 scenarios)
| Category | Scenarios | Tests |
|----------|-----------|-------|
| scopes | 20 | Scope isolation + global fact accessibility |

### Suite D — Adversarial (15 scenarios)
| Category | Scenarios | Tests |
|----------|-----------|-------|
| adversarial | 15 | Malicious injection detection + legitimate fact preservation |

### Suite E — Scale (8 scenarios)
| Category | Scenarios | Tests |
|----------|-----------|-------|
| scale | 8 | Retrieval quality with 10-200 facts, time-series |

### Suite F — Integration (11 scenarios)
| Category | Scenarios | Tests |
|----------|-----------|-------|
| integration | 11 | Multi-step store/recall/time/consolidate sequences |

### Suite G — Q-Learning (variable)
| Category | Scenarios | Tests |
|----------|-----------|-------|
| qlearning | variable | Multi-round recall/reward feedback loops |

### Suite H — Advanced Memory (58 scenarios)
| Category | Scenarios | Tests |
|----------|-----------|-------|
| supersession | 15 | Type+target auto-replacement of old facts |
| typed_decay | 10 | Different decay rates for different fact types |
| scope_lifecycle | 10 | Scope creation, closing, isolation |
| notation_parsing | 10 | Structured notation (TYPE[target]: content) |
| deduplication | 8 | Exact + near-duplicate handling |

## Scoring

- **Keyword judge** (default): Token overlap matching, no LLM needed
- **LLM judge**: Uses Claude/GPT for semantic answer matching
  ```bash
  python -m benchmarks.runner --judge-model claude-3-haiku-20240307
  ```

## Output

Results saved to `benchmarks/results/{backend}.json` with:
- Per-category scores and sub-scores
- Retrieval metrics (Recall@K, MRR, NDCG, Token F1)
- Token/cost efficiency metrics
- Per-scenario details for debugging

## Reference Scores

| Backend | Suites A-F | Suite H | Notes |
|---------|-----------|---------|-------|
| baseline-flat | 80.4% | 66.0% | Simple list + TF-IDF |
| cognitive | 96.8% | 90.6% | ACT-R + Hebbian + Q-value |
| unified | 97.2% | 73.6%* | All features combined |

*Suite H with TF-IDF. With sentence-transformers: significantly higher.

## Interface Reference

```python
class BenchmarkableStore(ABC):
    # Required (7 methods):
    store(content, category, scope, importance) -> None
    recall(query, top_k, scope) -> List[str]
    simulate_time(days) -> None
    simulate_access(content_substring) -> None
    consolidate() -> None
    get_stats() -> Dict[str, Any]
    reset() -> None
    
    # Optional (2 methods with default implementations):
    reward_memory(memory_id, signal) -> None  # default: no-op
    explore(query, top_k, scope) -> List[str]  # default: recall()
```
