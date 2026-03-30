"""
Intent classification tests for classify_intent().
"""

from __future__ import annotations

import pytest

from unified_memory.intent import classify_intent, QueryIntent
from unified_memory.types import FactType


def test_value_intent():
    """'What is the URL?' classifies as VALUE intent."""
    intent = classify_intent("What is the URL?")
    assert intent.intent == QueryIntent.VALUE
    assert intent.type_boost == FactType.VALUE


def test_constraint_intent():
    """'What are the requirements?' classifies as CONSTRAINT intent."""
    intent = classify_intent("What are the requirements?")
    assert intent.intent == QueryIntent.CONSTRAINT
    assert intent.type_boost == FactType.CONSTRAINT


def test_decision_intent():
    """'Why did we choose React?' classifies as DECISION intent."""
    intent = classify_intent("Why did we choose React?")
    assert intent.intent == QueryIntent.DECISION
    assert intent.type_boost == FactType.DECISION


def test_procedural_intent():
    """'How do I deploy?' classifies as PROCEDURAL intent."""
    intent = classify_intent("How do I deploy?")
    assert intent.intent == QueryIntent.PROCEDURAL
    # PROCEDURAL has no type_boost (None)
    assert intent.type_boost is None


def test_episodic_intent():
    """'When did we release v2?' classifies as EPISODIC intent."""
    intent = classify_intent("When did we release v2?")
    assert intent.intent == QueryIntent.EPISODIC
    assert intent.type_boost is None


def test_general_intent():
    """'Tell me about the project' classifies as GENERAL intent."""
    intent = classify_intent("Tell me about the project")
    assert intent.intent == QueryIntent.GENERAL
    assert intent.type_boost is None
    assert intent.confidence == 0.0
