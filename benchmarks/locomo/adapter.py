"""
LoCoMo adapter for Hermes cognitive memory.

Loads questions from the snap-research/locomo HuggingFace dataset (or a local
JSON file), ingests each question's conversation into a CognitiveMemoryStore,
then answers questions via recall.

Each question is isolated: a fresh store is created per question to avoid
cross-question contamination.

Reference:
    Maharana et al. (2024). "Building a Long-term Memory for Conversational
    Agents." arXiv:2402.15929. https://arxiv.org/abs/2402.15929
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.metrics import compute_metric_suite

logger = logging.getLogger(__name__)

# ── Constants ──

DATASET_NAME = "snap-research/locomo"
DATASET_SPLIT = "test"

QUESTION_TYPES = [
    "single_hop",
    "multi_hop",
    "temporal",
    "open_domain",
]

# ── Data structures ──


@dataclass
class LoCoMoQuestion:
    """A single LoCoMo question with its full conversation context."""

    question_id: str
    question_type: str   # single_hop | multi_hop | temporal | open_domain
    question: str
    answer: str
    conversation: list[dict[str, Any]]  # list of {role, content} message dicts

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LoCoMoQuestion":
        return cls(
            question_id=str(d.get("question_id", d.get("id", ""))),
            question_type=d.get("question_type", "single_hop"),
            question=d.get("question", d.get("query", "")),
            answer=d.get("answer", d.get("gold_answer", "")),
            conversation=d.get("conversation", d.get("context", [])),
        )


@dataclass
class LoCoMoResult:
    """Result of evaluating one LoCoMo question."""

    question_id: str
    question_type: str
    question: str
    gold_answer: str
    recalled: str        # top recalled memory content
    context: str         # top-5 recalled memories joined
    correct: bool
    recall_count: int    # total memories retrieved
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class LoCoMoSummary:
    """Aggregated results across all LoCoMo questions."""

    total: int
    correct: int
    score: float
    by_type: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: list[LoCoMoResult] = field(default_factory=list)
    # Aggregate retrieval metrics averaged across all questions
    mean_metrics: dict[str, float] = field(default_factory=dict)


# ── Dataset loading ──


def _normalize_hf_row(row: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a raw HuggingFace LoCoMo row to the canonical internal format.

    The actual LoCoMo schema on HuggingFace may differ from our internal
    representation.  This function maps any known field variants to the
    expected keys so that LoCoMoQuestion.from_dict() always works.
    """
    # Possible conversation field names in the HF dataset
    conversation = (
        row.get("conversation")
        or row.get("dialog")
        or row.get("history")
        or row.get("context")
        or []
    )

    # Normalize message list: each entry should be {role, content}
    normalized_msgs = []
    for msg in conversation:
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("speaker", "user")
            content = msg.get("content") or msg.get("text") or msg.get("utterance", "")
            normalized_msgs.append({"role": role, "content": content})
        elif isinstance(msg, str):
            # Some datasets encode raw strings alternating user/assistant
            idx = len(normalized_msgs)
            role = "user" if idx % 2 == 0 else "assistant"
            normalized_msgs.append({"role": role, "content": msg})

    # Possible question-type field names
    qtype = (
        row.get("question_type")
        or row.get("type")
        or row.get("category")
        or "single_hop"
    )

    return {
        "question_id": str(row.get("question_id") or row.get("id") or row.get("qid") or ""),
        "question_type": qtype,
        "question": row.get("question") or row.get("query") or "",
        "answer": row.get("answer") or row.get("gold_answer") or row.get("label") or "",
        "conversation": normalized_msgs,
    }


def load_locomo_dataset(
    hf_cache: str | None = None,
    sample: int | None = None,
    question_type_filter: str | None = None,
) -> list[LoCoMoQuestion]:
    """
    Load LoCoMo questions from HuggingFace (snap-research/locomo).

    The HuggingFace dataset may not be publicly available or may use a
    different schema than documented.  This function fails gracefully with
    an actionable error message if the dataset cannot be loaded.

    Args:
        hf_cache: Optional path to HuggingFace cache directory.
        sample: If set, only return the first N questions.
        question_type_filter: If set, only return questions of this type.

    Returns:
        List of LoCoMoQuestion objects.

    Raises:
        ImportError: If the `datasets` package is not installed.
        RuntimeError: If the dataset cannot be downloaded or has an unexpected
            schema, with instructions for using a local JSON file instead.
    """
    if hf_cache:
        os.environ["HF_DATASETS_CACHE"] = hf_cache

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required for LoCoMo. "
            "Install with: pip install datasets"
        )

    logger.info(
        "Loading LoCoMo from HuggingFace (%s, split=%s)...",
        DATASET_NAME,
        DATASET_SPLIT,
    )

    try:
        ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load LoCoMo from HuggingFace ({DATASET_NAME!r}): {exc}\n\n"
            "The dataset may not be publicly available yet or the name may have changed.\n"
            "Try loading from a local JSON file instead:\n"
            "  python -m benchmarks.locomo.runner --local /path/to/locomo.json\n\n"
            "You can download the data from:\n"
            "  https://github.com/snap-research/LoCoMo\n"
            "  https://arxiv.org/abs/2402.15929"
        ) from exc

    questions: list[LoCoMoQuestion] = []
    for row in ds:
        try:
            normalized = _normalize_hf_row(row)
        except Exception as exc:
            logger.warning("Skipping malformed row: %s", exc)
            continue

        if question_type_filter and normalized["question_type"] != question_type_filter:
            continue

        questions.append(LoCoMoQuestion.from_dict(normalized))

        if sample and len(questions) >= sample:
            break

    logger.info("Loaded %d questions from LoCoMo (HuggingFace)", len(questions))
    return questions


def load_locomo_local(
    path: str | Path,
    sample: int | None = None,
    question_type_filter: str | None = None,
) -> list[LoCoMoQuestion]:
    """
    Load LoCoMo questions from a local JSON file.

    Accepts two JSON layouts:
    - A list of question dicts (most common)
    - A dict mapping question IDs to question dicts

    Args:
        path: Path to the JSON file.
        sample: If set, only return the first N questions.
        question_type_filter: If set, only return questions of this type.

    Returns:
        List of LoCoMoQuestion objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON structure is unrecognized.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LoCoMo data file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Could be {"questions": [...]} or {"id1": {...}, "id2": {...}}
        if "questions" in data:
            data = data["questions"]
        elif "data" in data:
            data = data["data"]
        else:
            data = list(data.values())

    if not isinstance(data, list):
        raise ValueError(
            f"Unrecognized LoCoMo JSON structure in {path}. "
            "Expected a list of question dicts or a dict with a 'questions' key."
        )

    questions: list[LoCoMoQuestion] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if question_type_filter and item.get("question_type") != question_type_filter:
            continue
        questions.append(LoCoMoQuestion.from_dict(item))
        if sample and len(questions) >= sample:
            break

    logger.info("Loaded %d questions from %s", len(questions), path)
    return questions


# ── Ingestion ──


def ingest_conversation_into_store(
    store: Any,
    conversation: list[dict[str, Any]],
) -> int:
    """
    Ingest a conversation into a CognitiveMemoryStore.

    Strategy:
    - Each message's content is stored as a factual memory.
    - User messages receive importance 0.6 (user-stated facts are often the
      target of retrieval).
    - Assistant messages receive importance 0.4 (assistant knowledge is
      useful context but less often the direct answer).
    - simulate_time(0.01) is called between messages to create temporal
      ordering so the store can leverage recency signals.

    Args:
        store: A CognitiveBenchmarkAdapter (reset() should already be called).
        conversation: List of {role, content} message dicts.

    Returns:
        Total number of memories stored.
    """
    count = 0

    for i, msg in enumerate(conversation):
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()

        if not content:
            continue

        # User turns are more likely to contain the facts being asked about
        importance = 0.6 if role == "user" else 0.4

        store.store(content, category="factual", importance=importance)
        count += 1

        # Advance simulated time between messages to create temporal ordering
        if i < len(conversation) - 1:
            store.simulate_time(0.01)

    return count


# ── Evaluation ──


def evaluate_question(
    store: Any,
    question: LoCoMoQuestion,
    judge: Any,
    top_k: int = 10,
) -> LoCoMoResult:
    """
    Evaluate a single LoCoMo question against the ingested store.

    Recalls top_k memories, builds context from the top 5, judges the
    context against the gold answer, and computes the full metric suite.

    Args:
        store: A CognitiveBenchmarkAdapter already populated via
            ingest_conversation_into_store.
        question: The LoCoMoQuestion to evaluate.
        judge: A MemoryJudge instance for scoring.
        top_k: Number of memories to recall (default 10).

    Returns:
        LoCoMoResult with binary correctness, context, and retrieval metrics.
    """
    results = store.recall(question.question, top_k=top_k)
    recalled = results[0] if results else ""
    context = " | ".join(results[:5]) if results else ""

    jr = judge.judge_answer(question.question, question.answer, context)

    # Use the gold answer as the single "relevant" item for retrieval metrics.
    # This gives a coarse signal: did we surface a memory containing the answer?
    metrics = compute_metric_suite(
        retrieved=results,
        relevant=[question.answer],
        gold_answer=question.answer,
        predicted_answer=context,
    )

    return LoCoMoResult(
        question_id=question.question_id,
        question_type=question.question_type,
        question=question.question,
        gold_answer=question.answer,
        recalled=recalled,
        context=context,
        correct=jr.correct,
        recall_count=len(results),
        metrics=metrics,
    )


def run_locomo(
    questions: list[LoCoMoQuestion],
    judge: Any,
    backend_cls: Any | None = None,
    backend_kwargs: dict | None = None,
    top_k: int = 10,
    verbose: bool = False,
) -> LoCoMoSummary:
    """
    Run LoCoMo evaluation on a list of questions.

    For each question:
    1. Create a fresh CognitiveBenchmarkAdapter store.
    2. Ingest the question's conversation.
    3. Evaluate the question.

    Then aggregate overall accuracy and per-question-type breakdowns.

    Args:
        questions: List of LoCoMoQuestion objects.
        judge: A MemoryJudge instance.
        backend_cls: Backend class to instantiate (default: CognitiveBenchmarkAdapter).
        backend_kwargs: Init kwargs for the backend.
        top_k: Number of memories to recall per question (default 10).
        verbose: If True, print per-question results to stdout.

    Returns:
        LoCoMoSummary with aggregated scores and per-type breakdowns.
    """
    if backend_cls is None:
        from cognitive_memory.benchmark_adapter import CognitiveBenchmarkAdapter
        backend_cls = CognitiveBenchmarkAdapter

    backend_kwargs = backend_kwargs or {}

    results: list[LoCoMoResult] = []
    correct = 0

    # Accumulate metric values for computing means
    metric_accumulator: dict[str, list[float]] = {}

    for i, question in enumerate(questions):
        store = backend_cls(**backend_kwargs)
        store.reset()

        n_stored = ingest_conversation_into_store(store, question.conversation)

        result = evaluate_question(store, question, judge, top_k=top_k)
        results.append(result)

        if result.correct:
            correct += 1

        # Accumulate retrieval metrics
        for k, v in result.metrics.items():
            metric_accumulator.setdefault(k, []).append(v)

        if verbose:
            status = "✓" if result.correct else "✗"
            print(
                f"  [{i+1}/{len(questions)}] {status} {question.question_type} "
                f"q={question.question_id} stored={n_stored} "
                f"recalled={result.recall_count}"
            )

    # ── Aggregate by question type ──
    by_type: dict[str, dict[str, Any]] = {}
    for qtype in QUESTION_TYPES:
        subset = [r for r in results if r.question_type == qtype]
        if not subset:
            continue
        type_correct = sum(1 for r in subset if r.correct)
        # Average retrieval metrics for this type
        type_metrics: dict[str, float] = {}
        for k in (subset[0].metrics if subset else {}).keys():
            vals = [r.metrics.get(k, 0.0) for r in subset]
            type_metrics[k] = sum(vals) / len(vals) if vals else 0.0
        by_type[qtype] = {
            "total": len(subset),
            "correct": type_correct,
            "score": type_correct / len(subset),
            "metrics": type_metrics,
        }

    # ── Also capture any question types not in QUESTION_TYPES ──
    seen_types = {r.question_type for r in results}
    for qtype in seen_types - set(QUESTION_TYPES):
        subset = [r for r in results if r.question_type == qtype]
        type_correct = sum(1 for r in subset if r.correct)
        type_metrics = {}
        for k in (subset[0].metrics if subset else {}).keys():
            vals = [r.metrics.get(k, 0.0) for r in subset]
            type_metrics[k] = sum(vals) / len(vals) if vals else 0.0
        by_type[qtype] = {
            "total": len(subset),
            "correct": type_correct,
            "score": type_correct / len(subset),
            "metrics": type_metrics,
        }

    total = len(questions)
    overall_score = correct / total if total > 0 else 0.0

    # Compute mean metrics across all questions
    mean_metrics = {
        k: sum(v) / len(v)
        for k, v in metric_accumulator.items()
        if v
    }

    return LoCoMoSummary(
        total=total,
        correct=correct,
        score=overall_score,
        by_type=by_type,
        results=results,
        mean_metrics=mean_metrics,
    )
