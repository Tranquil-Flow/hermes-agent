"""
Common interfaces for the benchmark framework.

All memory backends (baseline, builtin cognitive, engram) must implement
BenchmarkableStore so the benchmark runner can swap them transparently.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


class BenchmarkableStore(ABC):
    """Interface that all benchmarkable memory systems must implement."""

    @abstractmethod
    def store(self, content: str, category: str = "factual",
              scope: str = "global", importance: float = 0.5) -> None:
        """Store a memory. Category/scope/importance may be ignored by simple backends."""
        ...

    @abstractmethod
    def recall(self, query: str, top_k: int = 10,
               scope: Optional[str] = None) -> List[str]:
        """
        Recall memories matching the query.
        Returns list of memory content strings, ranked by relevance.
        """
        ...

    @abstractmethod
    def simulate_time(self, days: float) -> None:
        """Advance the simulated clock by N days. For decay/rehearsal testing."""
        ...

    @abstractmethod
    def simulate_access(self, content_substring: str) -> None:
        """Simulate accessing/rehearsing a memory (by content substring match)."""
        ...

    @abstractmethod
    def consolidate(self) -> None:
        """Run consolidation cycle. Noop for backends that don't support it."""
        ...

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return backend statistics."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored memories. Called between benchmark runs."""
        ...


# --- Result dataclasses ---

@dataclass
class JudgeResult:
    """Result from the LLM judge evaluating a single answer."""
    correct: bool
    raw_response: str = ""
    question_type: str = ""
    tokens_used: int = 0


@dataclass
class CategoryResult:
    """Results for a single test category (e.g., A1 semantic recall)."""
    category: str
    total: int
    correct: int
    score: float  # correct / total
    sub_scores: Dict[str, float] = field(default_factory=dict)
    # sub_scores: e.g., {"easy": 0.93, "medium": 0.80, "hard": 0.67}
    details: List[Dict[str, Any]] = field(default_factory=list)
    # per-question details for debugging


@dataclass
class RunResult:
    """Results from a single benchmark run (one seed)."""
    seed: int
    results_by_category: Dict[str, CategoryResult] = field(default_factory=dict)
    overall_score: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    # token_usage: {"total_input": N, "total_output": N, "judge_input": N, ...}
    wall_time_seconds: float = 0.0


@dataclass
class AggregateResult:
    """Aggregated results across multiple runs."""
    num_runs: int = 0
    mean_score: float = 0.0
    std_score: float = 0.0
    ci_95_lower: float = 0.0
    ci_95_upper: float = 0.0
    per_category_mean: Dict[str, float] = field(default_factory=dict)
    per_category_std: Dict[str, float] = field(default_factory=dict)
    total_tokens: int = 0
    total_cost_usd: float = 0.0


@dataclass
class SignificanceResult:
    """Result of statistical significance test between two systems."""
    test_name: str = ""  # "paired_t_test" or "wilcoxon"
    p_value: float = 1.0
    effect_size: float = 0.0
    significant: bool = False  # p < 0.05
    baseline_mean: float = 0.0
    experiment_mean: float = 0.0
    improvement: float = 0.0  # percentage points


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    backend_name: str = "baseline-flat"
    profile: str = "balanced"
    embedding_model: str = "auto"
    parameters: Dict[str, Any] = field(default_factory=dict)
    num_runs: int = 5
    judge_model: str = "claude-haiku-4.5"
    output_path: str = "benchmarks/results/"
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
