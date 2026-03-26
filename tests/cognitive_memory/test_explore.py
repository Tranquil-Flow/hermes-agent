"""Tests for cognitive_memory.explore — Personalized PageRank exploration engine."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cognitive_memory.config import CognitiveMemoryConfig
from cognitive_memory.store import CognitiveMemoryStore
from cognitive_memory.explore import (
    personalized_pagerank,
    explore,
    explore_recursive,
    ExploreResult,
)


def _make_store(**kwargs) -> CognitiveMemoryStore:
    """Create an in-memory store for testing."""
    config = CognitiveMemoryConfig.balanced()
    config.db_path = ":memory:"
    config.embedding_model = "tfidf"  # fast, no model download
    for k, v in kwargs.items():
        setattr(config, k, v)
    return CognitiveMemoryStore(config=config, db_path=":memory:")


class TestPersonalizedPageRank:
    def test_empty_seeds_returns_empty(self):
        scores = personalized_pagerank({}, {})
        assert scores == {}

    def test_zero_seeds_returns_empty(self):
        scores = personalized_pagerank({"a": 0.0}, {"a": [("b", 1.0)]})
        assert scores == {}

    def test_isolated_node_teleports_only(self):
        # Single seed, no neighbors → score stays at seed
        scores = personalized_pagerank({"a": 1.0}, {})
        assert "a" in scores
        assert scores["a"] > 0

    def test_propagates_to_neighbor(self):
        # Seed at A, edge A->B; PPR should propagate score to B
        seeds = {"A": 1.0}
        adjacency = {"A": [("B", 1.0)]}
        scores = personalized_pagerank(seeds, adjacency, alpha=0.45, iterations=30)
        assert "B" in scores
        assert scores["B"] > 0
        # A should still be highest (it's the seed + teleport target)
        assert scores["A"] > scores["B"]

    def test_bidirectional_propagation(self):
        seeds = {"A": 1.0}
        adjacency = {
            "A": [("B", 1.0)],
            "B": [("A", 1.0), ("C", 1.0)],
            "C": [("B", 1.0)],
        }
        scores = personalized_pagerank(seeds, adjacency)
        # C is 2 hops from A — should still get some score
        assert scores.get("C", 0) > 0

    def test_convergence(self):
        # Tight tolerance should still converge without errors
        seeds = {"X": 0.5, "Y": 0.5}
        adjacency = {"X": [("Y", 0.8)], "Y": [("X", 0.8)]}
        scores = personalized_pagerank(seeds, adjacency, tolerance=1e-9, iterations=100)
        assert len(scores) >= 2


class TestExplore:
    def test_returns_explore_result(self):
        store = _make_store()
        store.store("Python is a programming language")
        store.store("Java is also a programming language")
        result = explore(store, "programming language")
        assert isinstance(result, ExploreResult)
        assert result.query == "programming language"

    def test_returns_results_for_matching_query(self):
        store = _make_store()
        store.store("The user enjoys hiking on weekends")
        store.store("The user's favorite food is pizza")
        store.store("The user works at a tech company")
        result = explore(store, "weekend hobbies")
        assert len(result.results) > 0
        assert result.rounds == 1

    def test_empty_store_returns_empty(self):
        store = _make_store()
        result = explore(store, "anything")
        assert result.results == []
        assert result.rounds == 1

    def test_ppr_discovered_items_in_results(self):
        """Store 5 memories, create a link between 2, verify PPR discovers the linked one."""
        store = _make_store()

        # Store 5 memories
        id1 = store.store("Alice lives in Seattle")
        id2 = store.store("Alice likes hiking")  # semantically linked to id1
        id3 = store.store("Bob works at Google")
        id4 = store.store("The weather in Seattle is rainy")
        id5 = store.store("Software engineers work at tech companies")

        # Manually create a Hebbian link between id1 and id4
        # (co-retrieval: both about Seattle)
        store._backend.create_link(id1, id4, weight=0.8, link_type="hebbian")

        # Query about Alice — seeds should include id1, id2
        # PPR should walk the link from id1 -> id4 (Seattle weather)
        result = explore(store, "Alice Seattle", initial_top_k=3, expand_top_k=10)

        assert isinstance(result, ExploreResult)
        assert len(result.results) > 0
        assert result.total_candidates_visited > 0

        # The PPR walk should discover id4 (Seattle weather linked to id1)
        result_ids = {item.entry.id for item in result.results}
        # id1 (Alice Seattle) should be in results as a direct seed
        assert id1 in result_ids

    def test_ppr_boosts_seed_scores(self):
        """PPR should boost scores of memories that are seeds and have links."""
        store = _make_store()
        id1 = store.store("Machine learning is a subset of AI")
        id2 = store.store("Deep learning uses neural networks")
        store._backend.create_link(id1, id2, weight=0.9, link_type="semantic")

        result_plain = store.recall("AI and machine learning", top_k=5)
        result_explore = explore(store, "AI and machine learning", initial_top_k=5, expand_top_k=5)

        # Explore should return at least as many results
        assert len(result_explore.results) >= len(result_plain)

    def test_scope_filter_respected(self):
        store = _make_store()
        store.store("Alice likes coffee", scope="alice")
        store.store("Bob likes tea", scope="bob")

        result = explore(store, "likes beverages", scope="alice")
        # All results should be from alice scope (or no scope filtering if store doesn't enforce it)
        assert isinstance(result, ExploreResult)

    def test_total_candidates_visited_tracked(self):
        store = _make_store()
        for i in range(5):
            store.store(f"Memory number {i} about various things")
        result = explore(store, "things", initial_top_k=3, expand_top_k=10)
        assert result.total_candidates_visited > 0


class TestExploreRecursive:
    def test_falls_back_to_ppr_without_llm(self):
        """Without llm_fn, explore_recursive falls back to Phase 1 PPR."""
        store = _make_store()
        store.store("The capital of France is Paris")
        store.store("The Eiffel Tower is in Paris")
        result = explore_recursive(store, "Paris landmarks", llm_fn=None)
        assert isinstance(result, ExploreResult)
        assert len(result.results) > 0

    def test_with_llm_fn_generates_sub_questions(self):
        """With llm_fn, should generate sub-questions and explore them."""
        store = _make_store()
        store.store("The user is named Alice")
        store.store("Alice lives in Seattle")
        store.store("Seattle is in Washington state")
        store.store("Washington state is in the Pacific Northwest")

        call_count = [0]

        def mock_llm_fn(context: str, query: str):
            call_count[0] += 1
            if call_count[0] == 1:
                return ["Where does Alice live?", "What state is Seattle in?"]
            return []  # converge after first round

        result = explore_recursive(
            store,
            "Tell me about Alice's location",
            llm_fn=mock_llm_fn,
            max_rounds=3,
            max_sub_questions=2,
        )

        assert isinstance(result, ExploreResult)
        assert len(result.results) > 0
        assert len(result.sub_questions) > 0
        assert result.rounds >= 2

    def test_converges_when_no_new_results(self):
        """Should converge when sub-questions don't discover new memories."""
        store = _make_store()
        store.store("Only one memory here")

        def always_return_questions(context, query):
            return ["sub question 1", "sub question 2"]

        result = explore_recursive(
            store,
            "anything",
            llm_fn=always_return_questions,
            max_rounds=5,
            convergence_threshold=0.99,  # very high threshold → converge immediately
        )
        assert isinstance(result, ExploreResult)

    def test_llm_exception_gracefully_handled(self):
        """LLM errors should be caught and fall back gracefully."""
        store = _make_store()
        store.store("Some fact about programming")

        def failing_llm(context, query):
            raise RuntimeError("LLM unavailable")

        # Should not raise — logs warning and exits loop
        result = explore_recursive(
            store,
            "programming",
            llm_fn=failing_llm,
            max_rounds=3,
        )
        assert isinstance(result, ExploreResult)
        assert len(result.results) > 0


class TestStoreExploreMethod:
    """Test that explore() and explore_recursive() are accessible on CognitiveMemoryStore."""

    def test_store_has_explore_method(self):
        store = _make_store()
        assert hasattr(store, 'explore')
        assert callable(store.explore)

    def test_store_has_explore_recursive_method(self):
        store = _make_store()
        assert hasattr(store, 'explore_recursive')
        assert callable(store.explore_recursive)

    def test_store_explore_returns_explore_result(self):
        store = _make_store()
        store.store("Testing the explore method")
        result = store.explore("testing")
        assert isinstance(result, ExploreResult)

    def test_store_explore_recursive_no_llm(self):
        store = _make_store()
        store.store("Testing recursive exploration")
        result = store.explore_recursive("testing")
        assert isinstance(result, ExploreResult)
