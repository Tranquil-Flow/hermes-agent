"""Recursive exploration engine for multi-hop memory retrieval.

When a single recall() pass isn't enough, explore() walks the link graph
via Personalized PageRank and optionally decomposes queries into sub-questions.

Two phases:
  Phase 1: PPR graph expansion (no LLM needed)
  Phase 3: Recursive sub-question decomposition (opt-in, needs LLM)

Ported from Ori-Mnemos RMH Constraint 2 (explore.ts).

Reference: Zhang, Krassa & Khattab (2026), Recursive Language Models.
"""
import math
import logging
from typing import List, Dict, Optional, Set, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExploreResult:
    """Result from recursive exploration."""
    query: str
    results: list  # List of ScoredMemory
    rounds: int = 1
    sub_questions: List[str] = field(default_factory=list)
    total_candidates_visited: int = 0
    converged: bool = False


def personalized_pagerank(
    seeds: Dict[str, float],  # node_id -> seed weight
    adjacency: Dict[str, List[tuple]],  # node_id -> [(neighbor_id, weight), ...]
    alpha: float = 0.45,
    iterations: int = 20,
    tolerance: float = 1e-6,
) -> Dict[str, float]:
    """Run Personalized PageRank on the link graph.

    Args:
        seeds: Starting nodes with weights (from initial recall)
        adjacency: Link graph as adjacency list
        alpha: Teleport probability (0.45 from HippoRAG)
        iterations: Max iterations
        tolerance: Convergence threshold

    Returns:
        Dict of node_id -> PPR score
    """
    # Normalize seed weights
    total_seed = sum(seeds.values())
    if total_seed <= 0:
        return {}
    teleport = {k: v / total_seed for k, v in seeds.items()}

    # Initialize scores
    scores = dict(teleport)

    for _ in range(iterations):
        new_scores: Dict[str, float] = {}

        # Teleport contribution
        for node, weight in teleport.items():
            new_scores[node] = new_scores.get(node, 0) + alpha * weight

        # Walk contribution
        for node, score in scores.items():
            neighbors = adjacency.get(node, [])
            if not neighbors:
                continue
            total_weight = sum(w for _, w in neighbors)
            if total_weight <= 0:
                continue
            for neighbor, weight in neighbors:
                contribution = (1 - alpha) * score * (weight / total_weight)
                new_scores[neighbor] = new_scores.get(neighbor, 0) + contribution

        # Check convergence
        diff = sum(abs(new_scores.get(k, 0) - scores.get(k, 0))
                   for k in set(list(new_scores.keys()) + list(scores.keys())))
        scores = new_scores
        if diff < tolerance:
            break

    return scores


def explore(
    store,  # CognitiveMemoryStore
    query: str,
    initial_top_k: int = 10,
    expand_top_k: int = 20,
    ppr_alpha: float = 0.45,
    ppr_boost: float = 0.2,
    scope: Optional[str] = None,
) -> ExploreResult:
    """Phase 1: PPR graph expansion.

    1. Run initial recall() to get seed results
    2. Build adjacency graph from Hebbian links
    3. Run PPR from seed nodes
    4. Boost/discover memories via PPR scores
    5. Re-rank and return expanded result set

    Args:
        store: CognitiveMemoryStore instance
        query: The query to explore
        initial_top_k: Number of seeds from initial recall
        expand_top_k: Total results to return after expansion
        ppr_alpha: PPR teleport probability (0.45 = HippoRAG validated)
        ppr_boost: How much PPR score boosts a memory's activation
        scope: Optional scope filter
    """
    # Step 1: Initial recall for seeds
    initial = store.recall(query, scope=scope, top_k=initial_top_k)
    if not initial:
        return ExploreResult(query=query, results=[], rounds=1)

    # Step 2: Build adjacency from all links
    all_links = store._backend.get_all_links()
    adjacency: Dict[str, List[tuple]] = {}
    for link in all_links:
        if link.source_id not in adjacency:
            adjacency[link.source_id] = []
        adjacency[link.source_id].append((link.target_id, link.weight))
        # Bidirectional
        if link.target_id not in adjacency:
            adjacency[link.target_id] = []
        adjacency[link.target_id].append((link.source_id, link.weight))

    # Step 3: Run PPR from seed nodes
    seeds = {item.entry.id: max(item.score, 0.01) for item in initial}
    ppr_scores = personalized_pagerank(seeds, adjacency, alpha=ppr_alpha)

    if not ppr_scores:
        return ExploreResult(query=query, results=initial, rounds=1,
                             total_candidates_visited=len(initial))

    # Step 4: Get ALL active memories and re-score with PPR boost
    all_candidates = store._backend.get_by_scope(scope) if scope else store._backend.get_all_active()
    query_embedding = store._embedder.encode(query)
    now = store._now()

    seen_ids = {item.entry.id for item in initial}
    expanded = list(initial)  # start with initial results

    # Score PPR-discovered memories
    for mem in all_candidates:
        if mem.id in seen_ids:
            # Boost existing results with PPR
            for item in expanded:
                if item.entry.id == mem.id and mem.id in ppr_scores:
                    ppr_norm = ppr_scores[mem.id] / max(max(ppr_scores.values()), 0.001)
                    item.score += ppr_boost * item.score * ppr_norm
                    item.components['ppr_score'] = ppr_scores.get(mem.id, 0)
                    item.components['ppr_boost'] = ppr_boost * item.score * ppr_norm
            continue

        if mem.id in ppr_scores:
            # New discovery via PPR
            ppr_val = ppr_scores[mem.id]
            ppr_norm = ppr_val / max(max(ppr_scores.values()), 0.001)

            # Compute basic activation for this memory
            from cognitive_memory.embeddings import cosine_similarity
            sim = cosine_similarity(query_embedding, store._embedder.encode(mem.content))

            # Use PPR-weighted score: median of initial scores * PPR rank
            initial_scores = [item.score for item in initial]
            median_score = sorted(initial_scores)[len(initial_scores) // 2] if initial_scores else 0
            score = median_score * ppr_norm * (0.5 + 0.5 * sim)

            if score > 0:
                from cognitive_memory.backends.base import ScoredMemory
                expanded.append(ScoredMemory(
                    entry=mem,
                    score=score,
                    components={
                        'ppr_score': ppr_val,
                        'ppr_discovery': True,
                        'semantic_sim': sim,
                    }
                ))
                seen_ids.add(mem.id)

    # Step 5: Sort and return top results
    expanded.sort(key=lambda s: s.score, reverse=True)
    results = expanded[:expand_top_k]

    return ExploreResult(
        query=query,
        results=results,
        rounds=1,
        total_candidates_visited=len(seen_ids),
    )


def explore_recursive(
    store,
    query: str,
    llm_fn: Optional[Callable[[str, str], List[str]]] = None,
    max_rounds: int = 3,
    max_sub_questions: int = 3,
    max_total_notes: int = 50,
    convergence_threshold: float = 0.1,
    initial_top_k: int = 10,
    expand_top_k: int = 20,
    scope: Optional[str] = None,
) -> ExploreResult:
    """Phase 3: Recursive sub-question decomposition.

    Requires an LLM function for generating sub-questions.
    Falls back to Phase 1 (PPR only) if no LLM is available.

    Args:
        store: CognitiveMemoryStore instance
        query: The original query
        llm_fn: Callable(context, query) -> List[str] of sub-questions.
                If None, falls back to explore() (Phase 1 only).
        max_rounds: Maximum recursion depth
        max_sub_questions: Max sub-questions per round
        max_total_notes: Budget cap on total notes visited
        convergence_threshold: Stop when new_notes/total < this
        initial_top_k: Seeds from initial recall
        expand_top_k: Final result count
        scope: Optional scope filter
    """
    if llm_fn is None:
        # No LLM available — fall back to Phase 1 PPR only
        return explore(store, query, initial_top_k, expand_top_k, scope=scope)

    # Phase 1: Initial exploration
    result = explore(store, query, initial_top_k, expand_top_k, scope=scope)
    all_results = {item.entry.id: item for item in result.results}
    all_sub_questions = []
    round_num = 0

    for round_num in range(max_rounds):
        # Build context from top results
        context_snippets = []
        for item in sorted(all_results.values(), key=lambda x: x.score, reverse=True)[:10]:
            context_snippets.append(item.entry.content[:200])
        context = "\n".join(context_snippets)

        # Ask LLM for sub-questions
        try:
            sub_questions = llm_fn(context, query)
        except Exception as e:
            logger.warning("LLM sub-question generation failed: %s", e)
            break

        if not sub_questions:
            result.converged = True
            break

        sub_questions = sub_questions[:max_sub_questions]
        all_sub_questions.extend(sub_questions)

        # Explore each sub-question
        new_count = 0
        for sq in sub_questions:
            sq_result = explore(store, sq, initial_top_k=5, expand_top_k=10, scope=scope)
            for item in sq_result.results:
                if item.entry.id not in all_results:
                    all_results[item.entry.id] = item
                    new_count += 1

        # Convergence check
        if len(all_results) > 0 and new_count / len(all_results) < convergence_threshold:
            result.converged = True
            break

        # Budget check
        if len(all_results) >= max_total_notes:
            break

    # Re-rank all accumulated results
    final = sorted(all_results.values(), key=lambda x: x.score, reverse=True)

    return ExploreResult(
        query=query,
        results=final[:expand_top_k],
        rounds=round_num + 2,  # +1 for initial, +1 for 0-indexing
        sub_questions=all_sub_questions,
        total_candidates_visited=len(all_results),
        converged=result.converged,
    )
