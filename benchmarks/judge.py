"""
LLM-as-Judge for Memory Benchmark

Uses a configurable LLM to evaluate whether an actual memory recall
matches the expected gold answer semantically.
"""

from dataclasses import dataclass
from typing import Optional

from benchmarks.interface import JudgeResult


JUDGE_SYSTEM_PROMPT = """You are a strict but fair judge evaluating memory recall accuracy.

Given a question, the expected gold answer, and the actual answer produced by a memory system, determine if the actual answer is CORRECT.

Rules:
- CORRECT: The actual answer contains the key information from the gold answer. Paraphrasing, reordering, and minor wording differences are acceptable.
- INCORRECT: The actual answer is missing critical information, contains wrong information, or fails to address the question.
- Partial matches: If the answer contains SOME but not all key facts, mark INCORRECT. We need complete recall.
- Extra information: If the answer includes correct extra details beyond the gold answer, still mark CORRECT.
- No answer: If the actual answer is empty or clearly irrelevant, mark INCORRECT.

Respond with exactly one word: CORRECT or INCORRECT
Then on a new line, a brief explanation (one sentence max)."""


JUDGE_USER_TEMPLATE = """Question: {question}
Expected answer: {gold_answer}
Actual answer: {actual_answer}

Verdict:"""


class MemoryJudge:
    """Evaluates memory recall answers using an LLM judge."""

    def __init__(self, model: str = "claude-haiku-4.5",
                 api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self._call_count = 0
        self._total_tokens = 0

    def judge_answer(self, question: str, gold_answer: str,
                     actual_answer: str) -> JudgeResult:
        """
        Judge whether actual_answer matches gold_answer for the given question.

        Returns JudgeResult with correct=True/False.
        """
        if not actual_answer or not actual_answer.strip():
            return JudgeResult(
                correct=False,
                raw_response="Empty answer",
                question_type="",
            )

        prompt = JUDGE_USER_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer,
            actual_answer=actual_answer,
        )

        response = self._call_llm(prompt)
        self._call_count += 1

        correct = self._parse_verdict(response)

        return JudgeResult(
            correct=correct,
            raw_response=response,
        )

    def _call_llm(self, prompt: str) -> str:
        """Call the judge LLM. Override in subclasses for different backends."""
        raise NotImplementedError(
            "LLM judge requires API implementation. "
            "Use HeuristicJudge for testing without LLM calls."
        )

    def _parse_verdict(self, response: str) -> bool:
        """Parse CORRECT/INCORRECT from judge response."""
        first_line = response.strip().split("\n")[0].strip().upper()
        return first_line == "CORRECT"

    @property
    def stats(self):
        return {
            "calls": self._call_count,
            "total_tokens": self._total_tokens,
            "model": self.model,
        }


class HeuristicJudge(MemoryJudge):
    """Substring-matching judge for benchmarking without LLM calls.

    Uses keyword overlap between gold answer and actual answer.
    Threshold: 60% of significant words must match.
    This is intentionally simple — the point is to test the memory system,
    not the judge. For publication-quality results, use the LLM judge.
    """

    def _call_llm(self, prompt: str) -> str:
        """Use keyword matching instead of LLM."""
        # Extract gold and actual from the prompt
        lines = prompt.strip().split("\n")
        gold = ""
        actual = ""
        for line in lines:
            if line.startswith("Expected answer:"):
                gold = line.split(":", 1)[1].strip().lower()
            elif line.startswith("Actual answer:"):
                actual = line.split(":", 1)[1].strip().lower()

        if not actual:
            return "INCORRECT\nEmpty answer"

        # Check if key words from gold appear in actual
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "with",
                      "and", "or", "for", "in", "on", "at", "to", "of", "it",
                      "its", "by", "as", "that", "this", "from", "has", "have",
                      "be", "been", "we", "our", "they", "do", "does"}
        gold_words = set(gold.split()) - stop_words
        if not gold_words:
            return "CORRECT\nNo significant words to check"

        matches = sum(1 for w in gold_words if w in actual)
        ratio = matches / len(gold_words) if gold_words else 0

        if ratio >= 0.6:
            return f"CORRECT\n{ratio:.0%} keyword match"
        else:
            return f"INCORRECT\n{ratio:.0%} keyword match (threshold: 60%)"
