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

    w_semantic: float = 0.55
    """Weight of cosine similarity (query vs memory embedding) in activation score."""

    w_importance: float = 0.4
    """Weight of importance score in activation. Optimized from grid search:
    w=0.4 scores 83.0% vs w=0.2 at 79.5% on Suite A. Plateaus above 0.4."""

    hebbian_learning_rate: float = 0.12
    """How fast co-activation strengthens Hebbian links. η in ΔW = η × aᵢ × aⱼ."""

    # --- Thresholds ---
    contradiction_threshold: float = 0.12
    """Combined entity-overlap + embedding score above which to flag contradictions.
    Uses _contradiction_score() which combines entity overlap (product names,
    acronyms, domain words) with embedding similarity, gated by update-language
    detection. 0.12 detects 90% of contradictions with zero false positives."""

    semantic_link_threshold: float = 0.70
    """Minimum cosine similarity to create a semantic link between memories."""

    high_link_threshold: float = 0.90
    """Cosine above which to link even across different categories."""

    links_per_memory: int = 5
    """Maximum outgoing links per memory."""

    enable_keyword_links: bool = True
    """If True, create weak keyword-overlap links at store() time to densify the graph for PPR walks."""

    keyword_link_threshold: float = 0.15
    """Minimum Jaccard similarity (shared tokens / union tokens) to create a keyword link."""

    keyword_link_min_shared: int = 3
    """Minimum number of shared tokens required to create a keyword link."""

    keyword_link_max_recent: int = 50
    """Maximum number of recent memories to check for keyword overlap (avoids O(n²))."""

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

    # --- RRF Fusion (BM25 keyword signal + score-weighted Reciprocal Rank Fusion) ---
    enable_rrf_fusion: bool = False
    """Enable BM25 keyword scoring + score-weighted RRF fusion in recall().
    Fuses activation score and BM25 term-frequency signal to improve recall
    precision. Disable to revert to raw activation-only scoring."""

    rrf_k: int = 60
    """RRF rank constant. Higher values reduce the impact of rank differences.
    Standard value from the RRF literature (Cormack et al. 2009)."""

    rrf_activation_weight: float = 0.85
    """Weight applied to the activation (semantic/ACT-R) signal in RRF fusion."""

    rrf_keyword_weight: float = 0.15
    """Weight applied to the BM25 keyword signal in RRF fusion.
    Lower values prevent BM25 from overriding temporal/importance signals."""

    # --- Hebbian Upgrade: Ori-Mnemos co-occurrence learning ---
    hebbian_glove_xmax: int = 100
    """GloVe-style saturation count for co-occurrence frequency weighting.
    co_signal = min(count, xmax)^0.75 / xmax^0.75.  Prevents rare
    co-occurrences from having outsized weight.  Default: 100."""

    hebbian_strength_rate: float = 0.2
    """Ebbinghaus strength accumulation rate.  strength = 1 + rate * log1p(count).
    Higher = stronger resistance to decay for frequently reinforced links."""

    hebbian_homeostasis_target: float = 0.5
    """Turrigiano homeostasis target mean weight per node.
    After strengthening, any node whose outgoing updated-link mean exceeds
    this value is scaled DOWN to prevent hub absorption.  Default: 0.5."""

    enable_hebbian_homeostasis: bool = True
    """Enable Turrigiano homeostatic scaling after each Hebbian update round.
    Prevents hub memories from accumulating unbounded link weight."""

    # --- Dampening Pipeline (ported from Ori-Mnemos dampening.ts) ---
    enable_dampening: bool = True
    """Enable post-scoring dampening pipeline (gravity, hub, resolution boost).
    Validated by ablation testing in Ori-Mnemos. Disable to revert to raw scores."""

    gravity_dampening_factor: float = 0.5
    """Score multiplier for 'cosine ghosts' — high-similarity but zero term overlap.
    0.5 = halve their score. Ori-Mnemos default."""

    hub_dampening_max_penalty: float = 0.3
    """Maximum fractional penalty for hub memories (those with unusually many links).
    Applied as: penalty = 1.0 - hub_dampening_max_penalty * ratio. Ori-Mnemos default."""

    resolution_boost_factor: float = 1.25
    """Score multiplier for actionable-knowledge categories (decision, correction,
    procedural, causal). Boosts signal over passive observations. Ori-Mnemos default."""

    # --- Q-Value Reranking (ported from Ori-Mnemos qvalue.ts / rerank.ts) ---
    enable_qvalue_reranking: bool = True
    """Enable Phase B Q-value reranking in recall(). Blends learned Q-values with
    activation scores. Grows influence as the system learns (cold-start protection)."""

    qvalue_lambda_min: float = 0.05
    """Minimum blend weight for Q-values (at cold start, 0 updates)."""

    qvalue_lambda_max: float = 0.35
    """Maximum blend weight for Q-values (reached at ~200 total updates)."""

    qvalue_exploration_c: float = 0.2
    """UCB exploration constant. Higher = more exploration of under-retrieved memories."""

    # --- Explore (Personalized PageRank multi-hop retrieval) ---
    enable_explore: bool = True
    """Enable explore() / explore_recursive() multi-hop retrieval via PPR.
    When disabled, explore() falls back to plain recall()."""

    ppr_alpha: float = 0.45
    """Teleport probability for Personalized PageRank. 0.45 validated by HippoRAG.
    Higher values = more weight on seed nodes vs. graph diffusion."""

    ppr_boost: float = 0.2
    """How much PPR score boosts an existing memory's activation in explore().
    score += ppr_boost * score * ppr_norm. Keep small to avoid over-weighting."""

    explore_max_rounds: int = 3
    """Maximum recursion depth for explore_recursive(). Each round generates
    sub-questions and explores them. Budget: O(rounds * max_sub_questions * top_k)."""

    explore_convergence_threshold: float = 0.1
    """Stop recursion when new_notes / total_notes < this fraction.
    Prevents wasted LLM calls when exploration has saturated the graph."""

    explore_max_notes: int = 50
    """Hard cap on total memories visited across all rounds in explore_recursive().
    Prevents runaway memory usage on large graphs."""

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
