"""
Cognitive Memory Configuration — all tunable parameters with profile presets.

Parameters are documented in docs/COGNITIVE_MEMORY_DESIGN.md Section 6.
Defaults are initial guesses; final values determined by benchmark grid search.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class CognitiveMemoryConfig:
    """All tunable parameters for the cognitive memory system."""

    # --- ACT-R Parameters ---
    d: float = 0.3
    """ACT-R decay parameter in tᵢ^(-d). Lower = slower decay. Optimized from
    grid search: d=0.3 scores 85.0% vs d=0.5 at 79.5% on Suite A."""

    w_semantic: float = 0.4
    """Weight of cosine similarity (query vs memory embedding) in activation score."""

    w_importance: float = 0.4
    """Weight of importance score in activation. Optimized from grid search:
    w=0.4 scores 83.0% vs w=0.2 at 79.5% on Suite A. Plateaus above 0.4."""

    hebbian_learning_rate: float = 0.05
    """How fast co-activation strengthens Hebbian links. η in ΔW = η × aᵢ × aⱼ."""

    # --- Thresholds ---
    contradiction_threshold: float = 0.82
    """Cosine similarity above which to check for contradictions."""

    semantic_link_threshold: float = 0.70
    """Minimum cosine similarity to create a semantic link between memories."""

    high_link_threshold: float = 0.90
    """Cosine above which to link even across different categories."""

    links_per_memory: int = 5
    """Maximum outgoing links per memory."""

    prune_threshold: float = 0.01
    """Activation below which to prune archived memories."""

    scope_multiplier: float = 1.5
    """Boost multiplier for memories matching the current query scope."""

    top_k: int = 10
    """Number of memories returned per recall query."""

    # --- Consolidation ---
    core_promotion_count: int = 3
    """Minimum access count to promote a memory from working → core."""

    core_promotion_importance: float = 0.5
    """Minimum importance to promote from working → core."""

    archive_threshold: float = 0.1
    """Activation below which to demote from core → archive."""

    link_decay_rate: float = 0.95
    """Hebbian link weight multiplier per consolidation cycle."""

    link_prune_threshold: float = 0.01
    """Link weight below which to prune the link."""

    # --- System ---
    embedding_model: str = "auto"
    """Embedding provider: 'auto', 'sentence-transformers', 'ollama', 'openai', 'tfidf'."""

    db_path: str = ""
    """Path to SQLite database. Empty = default (~/.hermes/cognitive_memory.db)."""

    max_access_times: int = 100
    """Cap on stored access timestamps per memory. ACT-R sum converges quickly."""

    contradiction_llm_model: Optional[str] = None
    """Model for LLM-assisted contradiction detection. None = use embedding-only."""

    def __post_init__(self):
        if not self.db_path:
            hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
            self.db_path = os.path.join(hermes_home, "cognitive_memory.db")

    @classmethod
    def developer(cls) -> "CognitiveMemoryConfig":
        """Profile for developers/coders. Moderate decay; balanced weights."""
        return cls(
            d=0.4,
            w_semantic=0.45,
            w_importance=0.35,
            hebbian_learning_rate=0.06,
        )

    @classmethod
    def researcher(cls) -> "CognitiveMemoryConfig":
        """Profile for researchers. Slow decay; high importance weight."""
        return cls(
            d=0.3,
            w_semantic=0.50,
            w_importance=0.45,
            hebbian_learning_rate=0.03,
        )

    @classmethod
    def personal(cls) -> "CognitiveMemoryConfig":
        """Profile for personal assistant use. Slow decay; high importance."""
        return cls(
            d=0.3,
            w_semantic=0.35,
            w_importance=0.45,
            hebbian_learning_rate=0.05,
        )

    @classmethod
    def balanced(cls) -> "CognitiveMemoryConfig":
        """Default balanced profile."""
        return cls()  # All defaults

    @classmethod
    def from_profile(cls, name: str) -> "CognitiveMemoryConfig":
        """Create config from profile name."""
        profiles = {
            "developer": cls.developer,
            "researcher": cls.researcher,
            "personal": cls.personal,
            "balanced": cls.balanced,
        }
        factory = profiles.get(name)
        if factory is None:
            raise ValueError(f"Unknown profile '{name}'. Available: {list(profiles.keys())}")
        return factory()

    def to_dict(self) -> dict:
        """Serialize all parameters to dict (for benchmark result storage)."""
        return {
            "d": self.d,
            "w_semantic": self.w_semantic,
            "w_importance": self.w_importance,
            "hebbian_learning_rate": self.hebbian_learning_rate,
            "contradiction_threshold": self.contradiction_threshold,
            "semantic_link_threshold": self.semantic_link_threshold,
            "high_link_threshold": self.high_link_threshold,
            "links_per_memory": self.links_per_memory,
            "prune_threshold": self.prune_threshold,
            "scope_multiplier": self.scope_multiplier,
            "top_k": self.top_k,
            "core_promotion_count": self.core_promotion_count,
            "core_promotion_importance": self.core_promotion_importance,
            "archive_threshold": self.archive_threshold,
            "link_decay_rate": self.link_decay_rate,
            "link_prune_threshold": self.link_prune_threshold,
            "embedding_model": self.embedding_model,
            "max_access_times": self.max_access_times,
        }
