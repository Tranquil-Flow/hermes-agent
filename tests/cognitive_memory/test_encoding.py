"""Tests for cognitive_memory.encoding — category classification and importance estimation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cognitive_memory.encoding import classify_category, estimate_importance, encode


class TestClassifyCategory:
    def test_preference(self):
        assert classify_category("User prefers dark mode") == "preference"
        assert classify_category("I like Python over Java") == "preference"
        assert classify_category("They hate tabs, always use spaces") == "preference"

    def test_procedural(self):
        assert classify_category("Run pip install numpy to install") == "procedural"
        assert classify_category("To deploy, first build the Docker image") == "procedural"
        assert classify_category("```python\nprint('hello')\n```") == "procedural"

    def test_environment(self):
        assert classify_category("The server is at 192.168.1.100 port 8080") == "environment"
        assert classify_category("Running on macOS Sonoma with Python 3.11") == "environment"
        assert classify_category("Database lives at /var/lib/postgres") == "environment"

    def test_causal(self):
        assert classify_category("The crash was caused by a null pointer") == "causal"
        assert classify_category("Memory leak leads to OOM after 3 hours") == "causal"

    def test_episodic(self):
        assert classify_category("Yesterday we discussed the API redesign") == "episodic"
        assert classify_category("Last session we agreed on the schema") == "episodic"

    def test_factual_default(self):
        assert classify_category("The sky is blue") == "factual"
        assert classify_category("Pi is approximately 3.14159") == "factual"

    def test_semantic(self):
        assert classify_category("ACT-R means Adaptive Control of Thought") == "semantic"
        assert classify_category("Hebbian learning refers to connection strengthening") == "semantic"


class TestEstimateImportance:
    def test_correction_high_importance(self):
        imp = estimate_importance("Actually, the correct port is 8443", "factual")
        assert imp >= 0.85

    def test_preference_moderate_high(self):
        imp = estimate_importance("User prefers vim over emacs", "preference")
        assert imp >= 0.70

    def test_general_lower(self):
        imp = estimate_importance("The weather is nice", "factual")
        assert imp <= 0.60

    def test_clamp_range(self):
        imp = estimate_importance("x", "factual")
        assert 0.1 <= imp <= 1.0

    def test_long_content_bonus(self):
        short = estimate_importance("Short fact", "factual")
        long_text = "This is a detailed explanation " * 5
        long_imp = estimate_importance(long_text, "factual")
        assert long_imp >= short


class TestEncode:
    def test_returns_tuple(self):
        cat, imp = encode("User prefers dark mode")
        assert isinstance(cat, str)
        assert isinstance(imp, float)
        assert cat == "preference"
        assert 0.1 <= imp <= 1.0

    def test_factual_default(self):
        cat, imp = encode("The library version is 2.3.1")
        assert cat in ("factual", "environment")
        assert 0.1 <= imp <= 1.0
