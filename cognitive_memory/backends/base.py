"""
Abstract storage backend and core data types for cognitive memory.

All backends (builtin SQLite, Engram, etc.) implement StorageBackend.
All benchmark adapters also implement BenchmarkableStore (see benchmarks/interface.py).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class MemoryEntry:
    """A single memory in the cognitive store."""

    id: str
    content: str
    category: str  # factual | preference | procedural | environment | episodic | semantic | causal
    scope: str = "global"  # global | project:<name> | topic:<name>
    importance: float = 0.5
    embedding: Optional[np.ndarray] = None
    created_at: float = 0.0  # unix timestamp
    last_accessed: float = 0.0
    access_count: int = 0
    access_times: List[float] = field(default_factory=list)  # for ACT-R base level
    layer: str = "working"  # working | core | archive
    superseded_by: Optional[str] = None  # FK to another memory's id
    source: str = "agent"  # agent | user | consolidation | honcho
    pinned: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryLink:
    """A directed link between two memories."""

    source_id: str
    target_id: str
    weight: float = 0.1
    link_type: str = "hebbian"  # hebbian | semantic | contradiction | causal
    created_at: float = 0.0
    last_coactivated: Optional[float] = None


@dataclass
class ScoredMemory:
    """A memory with its activation score and component breakdown."""

    entry: MemoryEntry
    score: float
    components: Dict[str, float] = field(default_factory=dict)
    # components keys: base_level, spreading, importance_boost, scope_boost


class StorageBackend(ABC):
    """Abstract interface for cognitive memory storage backends."""

    @abstractmethod
    def store(self, entry: MemoryEntry) -> str:
        """Persist a memory entry. Returns the memory id."""
        ...

    @abstractmethod
    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a single memory by id. Returns None if not found."""
        ...

    @abstractmethod
    def get_all_active(self) -> List[MemoryEntry]:
        """Get all memories where superseded_by is None (not superseded)."""
        ...

    @abstractmethod
    def get_by_layer(self, layer: str) -> List[MemoryEntry]:
        """Get all active memories in a specific layer (working/core/archive)."""
        ...

    @abstractmethod
    def get_by_scope(self, scope: str) -> List[MemoryEntry]:
        """Get all active memories matching a scope prefix."""
        ...

    @abstractmethod
    def update(self, memory_id: str, **fields) -> None:
        """Update specific fields on a memory. Only provided fields are changed."""
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """Hard delete a memory (used only in tests/reset). Normal flow uses superseded_by."""
        ...

    # --- Link operations ---

    @abstractmethod
    def get_links(self, memory_id: str) -> List[MemoryLink]:
        """Get all links where this memory is the source."""
        ...

    @abstractmethod
    def get_backlinks(self, memory_id: str) -> List[MemoryLink]:
        """Get all links where this memory is the target."""
        ...

    @abstractmethod
    def create_link(self, source_id: str, target_id: str,
                    weight: float = 0.1, link_type: str = "hebbian") -> None:
        """Create a link between two memories. Upsert if exists."""
        ...

    @abstractmethod
    def update_link(self, source_id: str, target_id: str, **fields) -> None:
        """Update link fields (typically weight and last_coactivated)."""
        ...

    @abstractmethod
    def prune_links(self, min_weight: float = 0.01) -> int:
        """Remove all links with weight below threshold. Returns count pruned."""
        ...

    @abstractmethod
    def decay_links(self, decay_rate: float = 0.95) -> None:
        """Multiply all link weights by decay_rate."""
        ...

    # --- Lifecycle ---

    @abstractmethod
    def reset(self) -> None:
        """Delete all data. Used in benchmarks between runs."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Total number of active (non-superseded) memories."""
        ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Return storage statistics: counts by layer, by category, total links, etc."""
        ...
