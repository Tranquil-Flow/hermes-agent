"""
Cognitive Memory System for Hermes Agent.

A neuroscience-inspired memory layer with:
- ACT-R activation scoring (frequency × recency × importance)
- Semantic embeddings with fallback chain
- Hebbian co-activation links
- Three-layer consolidation (working → core → archive)
- Contradiction detection
- Scope filtering (global / project / topic)
"""

from cognitive_memory.config import CognitiveMemoryConfig
from cognitive_memory.store import CognitiveMemoryStore, ConsolidationReport
from cognitive_memory.backends.base import MemoryEntry, MemoryLink, ScoredMemory
from cognitive_memory.embeddings import EmbeddingProvider, cosine_similarity
from cognitive_memory.encoding import classify_category, estimate_importance, encode

__all__ = [
    "CognitiveMemoryConfig",
    "CognitiveMemoryStore",
    "ConsolidationReport",
    "MemoryEntry",
    "MemoryLink",
    "ScoredMemory",
    "EmbeddingProvider",
    "cosine_similarity",
    "classify_category",
    "estimate_importance",
    "encode",
]
