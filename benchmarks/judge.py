"""
LLM-as-Judge for Memory Benchmark

Evaluates whether actual memory recall matches the expected gold answer.
Supports both LLM judges (Haiku) and heuristic fallback.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

from benchmarks.interface import JudgeResult

logger = logging.getLogger(__name__)

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
    """Evaluates memory recall answers using an LLM judge (Claude Haiku)."""

    def __init__(self, model: str = "claude-haiku-4-5-20241022",
                 api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._client = None

    def _get_client(self):
        """Lazy-init the Anthropic client.
        
        Routes through aegis proxy if available (container environment).
        The proxy injects API keys, so we don't need them in the container.
        """
        if self._client is None:
            import anthropic
            import httpx

            # Try aegis proxy first (container environment)
            proxy_url = "http://host.docker.internal:8443"
            ca_cert = "/certs/mitmproxy-ca-cert.pem"

            if os.path.exists(ca_cert):
                # Route through aegis proxy — it injects the API key
                http_client = httpx.Client(
                    proxy=proxy_url,
                    verify=ca_cert,
                )
                self._client = anthropic.Anthropic(
                    api_key=self.api_key or "placeholder-aegis-injects",
                    http_client=http_client,
                )
                logger.info("LLM judge using aegis proxy")
            else:
                # Direct connection (host environment)
                self._client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("LLM judge using direct API")
        return self._client

    def judge_answer(self, question: str, gold_answer: str,
                     actual_answer: str) -> JudgeResult:
        """Judge whether actual_answer matches gold_answer."""
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
        """Call Claude Haiku for judgment."""
        try:
            client = self._get_client()
            msg = client.messages.create(
                model=self.model,
                max_tokens=100,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            # Track token usage
            self._total_input_tokens += msg.usage.input_tokens
            self._total_output_tokens += msg.usage.output_tokens
            return msg.content[0].text
        except Exception as e:
            logger.warning(f"LLM judge call failed: {e}")
            # Fall back to heuristic if LLM fails
            return self._heuristic_fallback(prompt)

    def _heuristic_fallback(self, prompt: str) -> str:
        """Heuristic judge using keyword matching + substring containment.
        
        Uses multiple signals:
        1. Exact substring containment of gold in actual
        2. Keyword overlap (significant words)
        3. Number/identifier matching (port numbers, versions, names)
        """
        import re
        lines = prompt.strip().split("\n")
        gold = ""
        actual = ""
        for line in lines:
            if line.startswith("Expected answer:"):
                gold = line.split(":", 1)[1].strip()
            elif line.startswith("Actual answer:"):
                actual = line.split(":", 1)[1].strip()

        if not actual:
            return "INCORRECT\nEmpty answer"

        gold_lower = gold.lower()
        actual_lower = actual.lower()

        # Signal 1: exact substring match (strongest signal)
        if gold_lower in actual_lower:
            return f"CORRECT\nExact substring match"

        # Signal 2: extract key identifiers (numbers, proper nouns, technical terms)
        # These are the most important things to match
        identifiers = re.findall(r'\b(?:\d+(?:\.\d+)*|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,})\b', gold)
        if identifiers:
            id_matches = sum(1 for ident in identifiers if ident.lower() in actual_lower)
            id_ratio = id_matches / len(identifiers)
            if id_ratio >= 0.8:
                return f"CORRECT\n{id_ratio:.0%} identifier match ({identifiers})"
            if id_ratio == 0:
                return f"INCORRECT\nNo key identifiers found in actual ({identifiers})"

        # Signal 3: keyword overlap
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "with",
                      "and", "or", "for", "in", "on", "at", "to", "of", "it",
                      "its", "by", "as", "that", "this", "from", "has", "have",
                      "be", "been", "we", "our", "they", "do", "does", "not",
                      "but", "so", "if", "than", "then", "about", "up", "out"}
        gold_words = set(gold_lower.split()) - stop_words
        if not gold_words:
            return "CORRECT\nNo significant words to check"

        matches = sum(1 for w in gold_words if w in actual_lower)
        ratio = matches / len(gold_words) if gold_words else 0

        if ratio >= 0.6:
            return f"CORRECT\n{ratio:.0%} keyword match"
        else:
            return f"INCORRECT\n{ratio:.0%} keyword match (threshold: 60%)"

    def _parse_verdict(self, response: str) -> bool:
        """Parse CORRECT/INCORRECT from judge response."""
        first_line = response.strip().split("\n")[0].strip().upper()
        return "CORRECT" in first_line and "INCORRECT" not in first_line

    @property
    def stats(self):
        return {
            "calls": self._call_count,
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "model": self.model,
        }


class HeuristicJudge(MemoryJudge):
    """Substring-matching judge for fast iteration without LLM calls.

    Uses keyword overlap between gold answer and actual answer.
    Threshold: 60% of significant words must match.
    """

    def _call_llm(self, prompt: str) -> str:
        """Use keyword matching instead of LLM."""
        return self._heuristic_fallback(prompt)
