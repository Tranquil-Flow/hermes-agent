"""Tests for conversation ingestion and semantic deduplication."""

import pytest
from unified_memory.ingestion import (
    extract_facts, _factuality_score, _classify_type,
    _extract_target, find_near_duplicates, compute_memorability,
)
from unified_memory.types import FactType


class TestFactExtraction:
    def test_extract_factual_statement(self):
        facts = extract_facts("The API uses PostgreSQL 15 with pgvector extension.")
        assert len(facts) >= 1
        assert any("PostgreSQL" in f["content"] for f in facts)

    def test_skip_greeting(self):
        facts = extract_facts("Hi! How are you today?")
        # Greetings should be filtered out
        assert len(facts) == 0

    def test_extract_multiple(self):
        text = "The server runs on port 8080. Authentication uses JWT tokens. The database is PostgreSQL."
        facts = extract_facts(text)
        assert len(facts) >= 2

    def test_skip_short(self):
        facts = extract_facts("Yes ok.")
        assert len(facts) == 0


class TestFactualityScore:
    def test_factual_high(self):
        score = _factuality_score("The API endpoint is https://api.example.com/v2")
        assert score > 0.5

    def test_conversational_low(self):
        score = _factuality_score("Hi there, how are you doing?")
        assert score < 0.4

    def test_constraint_high(self):
        score = _factuality_score("Passwords must be at least 12 characters long")
        assert score > 0.5

    def test_technical_high(self):
        score = _factuality_score("The database server runs PostgreSQL 15 on port 5432")
        assert score > 0.6


class TestTypeClassification:
    def test_constraint(self):
        assert _classify_type("Users must enable 2FA") == FactType.CONSTRAINT

    def test_decision(self):
        assert _classify_type("We decided to use React for the frontend") == FactType.DECISION

    def test_unknown(self):
        assert _classify_type("The deployment strategy is still TBD") == FactType.UNKNOWN

    def test_done(self):
        assert _classify_type("Migration to v2 has been completed") == FactType.DONE

    def test_value_default(self):
        assert _classify_type("The server runs on port 8080") == FactType.VALUE


class TestTargetExtraction:
    def test_subject_verb(self):
        target = _extract_target("The API uses JWT tokens")
        assert "api" in target.lower()

    def test_colon_pattern(self):
        target = _extract_target("Database: PostgreSQL 15")
        assert "database" in target.lower()

    def test_fallback(self):
        target = _extract_target("Something happened yesterday")
        assert target == "general"


class TestMemorability:
    def test_constraint_boost(self):
        c = compute_memorability("Must use HTTPS", FactType.CONSTRAINT)
        v = compute_memorability("Must use HTTPS", FactType.VALUE)
        assert c > v

    def test_obsolete_penalty(self):
        o = compute_memorability("Old config", FactType.OBSOLETE)
        v = compute_memorability("Old config", FactType.VALUE)
        assert o < v

    def test_bounds(self):
        score = compute_memorability("test", FactType.VALUE, 0.5)
        assert 0.0 <= score <= 1.0


class TestSemanticDedup:
    def test_find_near_duplicates(self):
        import sqlite3
        import numpy as np
        from unified_memory.schema import get_connection

        conn = get_connection(":memory:")

        # Insert a fact with embedding
        emb = np.random.randn(10).astype(np.float32)
        conn.execute(
            "INSERT INTO um_facts (id, content, embedding, type, target, status, "
            "created_at, updated_at, last_accessed) "
            "VALUES ('f1', 'API uses JWT tokens', ?, 'V', 'api', 'active', 0, 0, 0)",
            (emb.tobytes(),)
        )
        conn.commit()

        # Search with same embedding and overlapping words (should find duplicate)
        dupes = find_near_duplicates(conn, "API uses JWT tokens for auth", emb, threshold=0.8)
        # With identical embedding + word overlap, should match
        assert len(dupes) >= 1

    def test_no_duplicates_different_content(self):
        import sqlite3
        import numpy as np
        from unified_memory.schema import get_connection

        conn = get_connection(":memory:")

        emb1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        conn.execute(
            "INSERT INTO um_facts (id, content, embedding, type, target, status, "
            "created_at, updated_at, last_accessed) "
            "VALUES ('f1', 'API uses JWT tokens', ?, 'V', 'api', 'active', 0, 0, 0)",
            (emb1.tobytes(),)
        )
        conn.commit()

        dupes = find_near_duplicates(conn, "Database uses PostgreSQL", emb2, threshold=0.8)
        assert len(dupes) == 0
