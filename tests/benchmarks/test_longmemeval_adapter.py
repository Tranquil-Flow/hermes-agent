"""
Tests for the LongMemEval benchmark adapter.

These tests use synthetic data (no HuggingFace download needed).
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from benchmarks.longmemeval.adapter import (
    LongMemQuestion,
    LongMemResult,
    LongMemSummary,
    ingest_sessions_into_store,
    evaluate_question,
    run_longmemeval,
)


# ── Fixtures ──

def make_question(
    question_id: str = "q1",
    question_type: str = "single-session-user",
    question: str = "What programming language does the user prefer?",
    answer: str = "Python",
    haystack_sessions: list | None = None,
    answer_session_ids: list | None = None,
) -> LongMemQuestion:
    """Create a synthetic LongMemQuestion for testing."""
    if haystack_sessions is None:
        haystack_sessions = [
            [
                {"role": "user", "content": "I really love programming in Python.", "has_answer": True},
                {"role": "assistant", "content": "Python is a great choice!"},
            ]
        ]
    if answer_session_ids is None:
        answer_session_ids = ["session_0"]

    return LongMemQuestion(
        question_id=question_id,
        question_type=question_type,
        question=question,
        answer=answer,
        question_date="2024-01-15",
        haystack_dates=["2024-01-10"],
        haystack_session_ids=["session_0"],
        haystack_sessions=haystack_sessions,
        answer_session_ids=answer_session_ids,
    )


def make_mock_store():
    """Create a mock CognitiveBenchmarkAdapter."""
    store = MagicMock()
    store.reset.return_value = None
    store.store.return_value = None
    store.simulate_time.return_value = None
    store.recall.return_value = ["I really love programming in Python."]
    return store


class TestLongMemQuestion:
    def test_from_dict(self):
        d = {
            "question_id": "q1",
            "question_type": "temporal-reasoning",
            "question": "When did X happen?",
            "answer": "2024-01-01",
            "question_date": "2024-03-01",
            "haystack_dates": ["2024-01-01"],
            "haystack_session_ids": ["s1"],
            "haystack_sessions": [[{"role": "user", "content": "X happened."}]],
            "answer_session_ids": ["s1"],
        }
        q = LongMemQuestion.from_dict(d)
        assert q.question_id == "q1"
        assert q.question_type == "temporal-reasoning"
        assert q.answer == "2024-01-01"
        assert len(q.haystack_sessions) == 1

    def test_from_dict_missing_optional(self):
        d = {
            "question_id": "q2",
            "question_type": "multi-session",
            "question": "What?",
            "answer": "Something",
            "haystack_sessions": [],
            "answer_session_ids": [],
        }
        q = LongMemQuestion.from_dict(d)
        assert q.haystack_dates == []
        assert q.haystack_session_ids == []
        assert q.question_date == ""


class TestIngestSessionsIntoStore:
    def test_stores_all_session_messages(self):
        store = make_mock_store()
        q = make_question(
            haystack_sessions=[
                [
                    {"role": "user", "content": "I like Python."},
                    {"role": "assistant", "content": "That's great."},
                ]
            ],
            answer_session_ids=["session_0"],
        )
        count = ingest_sessions_into_store(store, q)
        assert count == 2
        assert store.store.call_count == 2

    def test_skips_empty_content(self):
        store = make_mock_store()
        q = make_question(
            haystack_sessions=[
                [
                    {"role": "user", "content": ""},
                    {"role": "user", "content": "Real content."},
                ]
            ]
        )
        count = ingest_sessions_into_store(store, q)
        assert count == 1  # Only one non-empty message

    def test_answer_session_gets_higher_importance(self):
        store = make_mock_store()
        # Session 0 is the answer session
        q = make_question(
            haystack_sessions=[
                [{"role": "user", "content": "Answer session fact."}],
                [{"role": "user", "content": "Non-answer session fact."}],
            ],
            answer_session_ids=["session_0"],
        )
        q.haystack_session_ids = ["session_0", "session_1"]

        ingest_sessions_into_store(store, q)

        # Check importance values
        calls = store.store.call_args_list
        assert len(calls) == 2

        # Answer session fact should have higher importance (0.8)
        answer_call_importance = calls[0][1]["importance"]
        non_answer_call_importance = calls[1][1]["importance"]
        assert answer_call_importance > non_answer_call_importance

    def test_multiple_sessions_simulate_time(self):
        store = make_mock_store()
        q = make_question(
            haystack_sessions=[
                [{"role": "user", "content": "Session 1 fact."}],
                [{"role": "user", "content": "Session 2 fact."}],
                [{"role": "user", "content": "Session 3 fact."}],
            ]
        )
        q.haystack_session_ids = ["s1", "s2", "s3"]
        q.answer_session_ids = ["s1"]

        ingest_sessions_into_store(store, q)

        # 2 time advances (between 3 sessions)
        assert store.simulate_time.call_count == 2

    def test_single_session_no_time_advance(self):
        store = make_mock_store()
        q = make_question()
        ingest_sessions_into_store(store, q)
        store.simulate_time.assert_not_called()


class TestEvaluateQuestion:
    def test_correct_answer_detected(self):
        store = make_mock_store()
        store.recall.return_value = [
            "I really love programming in Python. It's my favourite language.",
            "I also know JavaScript.",
        ]
        q = make_question(answer="Python")

        mock_judge = MagicMock()
        mock_judge.judge_answer.return_value = MagicMock(correct=True)

        result = evaluate_question(store, q, mock_judge)
        assert result.correct is True
        assert result.question_id == "q1"
        assert result.recall_count == 2

    def test_wrong_answer_detected(self):
        store = make_mock_store()
        store.recall.return_value = ["I know JavaScript."]
        q = make_question(answer="Python")

        mock_judge = MagicMock()
        mock_judge.judge_answer.return_value = MagicMock(correct=False)

        result = evaluate_question(store, q, mock_judge)
        assert result.correct is False

    def test_empty_recall(self):
        store = make_mock_store()
        store.recall.return_value = []
        q = make_question(answer="Python")

        mock_judge = MagicMock()
        mock_judge.judge_answer.return_value = MagicMock(correct=False)

        result = evaluate_question(store, q, mock_judge)
        assert result.recalled == ""
        assert result.context == ""
        assert result.recall_count == 0

    def test_context_is_top_5_joined(self):
        store = make_mock_store()
        store.recall.return_value = [f"fact_{i}" for i in range(10)]
        q = make_question()

        mock_judge = MagicMock()
        mock_judge.judge_answer.return_value = MagicMock(correct=True)

        result = evaluate_question(store, q, mock_judge)
        # context should be top 5 facts
        assert result.context == "fact_0 | fact_1 | fact_2 | fact_3 | fact_4"


class TestRunLongmemeval:
    def _make_mock_backend_cls(self, recall_results: list[str] | None = None):
        """Create a mock backend class that returns controlled recall results."""
        if recall_results is None:
            recall_results = ["The user prefers Python."]

        mock_instance = MagicMock()
        mock_instance.reset.return_value = None
        mock_instance.store.return_value = None
        mock_instance.simulate_time.return_value = None
        mock_instance.recall.return_value = recall_results

        mock_cls = MagicMock(return_value=mock_instance)
        return mock_cls

    def test_empty_questions_returns_zero(self):
        mock_judge = MagicMock()
        summary = run_longmemeval([], mock_judge, backend_cls=self._make_mock_backend_cls())
        assert summary.total == 0
        assert summary.score == 0.0

    def test_all_correct(self):
        questions = [
            make_question(question_id=f"q{i}", question_type="single-session-user")
            for i in range(5)
        ]

        mock_judge = MagicMock()
        mock_judge.judge_answer.return_value = MagicMock(correct=True)

        summary = run_longmemeval(
            questions,
            mock_judge,
            backend_cls=self._make_mock_backend_cls(),
        )
        assert summary.total == 5
        assert summary.correct == 5
        assert summary.score == 1.0

    def test_all_wrong(self):
        questions = [make_question(question_id=f"q{i}") for i in range(3)]

        mock_judge = MagicMock()
        mock_judge.judge_answer.return_value = MagicMock(correct=False)

        summary = run_longmemeval(
            questions,
            mock_judge,
            backend_cls=self._make_mock_backend_cls(),
        )
        assert summary.total == 3
        assert summary.correct == 0
        assert summary.score == 0.0

    def test_by_type_aggregation(self):
        questions = [
            make_question(question_id="q1", question_type="temporal-reasoning"),
            make_question(question_id="q2", question_type="temporal-reasoning"),
            make_question(question_id="q3", question_type="multi-session"),
        ]

        # First 2 correct, last wrong
        call_count = [0]
        def judge_answer(q, gold, actual):
            result = MagicMock()
            result.correct = call_count[0] < 2
            call_count[0] += 1
            return result

        mock_judge = MagicMock()
        mock_judge.judge_answer.side_effect = judge_answer

        summary = run_longmemeval(
            questions,
            mock_judge,
            backend_cls=self._make_mock_backend_cls(),
        )

        assert summary.by_type["temporal-reasoning"]["correct"] == 2
        assert summary.by_type["temporal-reasoning"]["score"] == 1.0
        assert summary.by_type["multi-session"]["correct"] == 0
        assert summary.by_type["multi-session"]["score"] == 0.0

    def test_results_list_populated(self):
        questions = [make_question(question_id="q1")]

        mock_judge = MagicMock()
        mock_judge.judge_answer.return_value = MagicMock(correct=True)

        summary = run_longmemeval(
            questions,
            mock_judge,
            backend_cls=self._make_mock_backend_cls(),
        )
        assert len(summary.results) == 1
        assert summary.results[0].question_id == "q1"

    def test_new_store_created_per_question(self):
        """Each question should get its own fresh store."""
        questions = [make_question(question_id=f"q{i}") for i in range(3)]

        mock_judge = MagicMock()
        mock_judge.judge_answer.return_value = MagicMock(correct=False)

        mock_cls = self._make_mock_backend_cls()
        run_longmemeval(questions, mock_judge, backend_cls=mock_cls)

        # Backend class should be instantiated once per question
        assert mock_cls.call_count == 3


class TestLoadLongmemevalLocal:
    def test_load_from_json_list(self, tmp_path):
        import json

        data = [
            {
                "question_id": "q1",
                "question_type": "multi-session",
                "question": "What?",
                "answer": "42",
                "haystack_sessions": [[{"role": "user", "content": "42 is the answer."}]],
                "answer_session_ids": [],
            }
        ]
        f = tmp_path / "test.json"
        f.write_text(json.dumps(data))

        from benchmarks.longmemeval.adapter import load_longmemeval_local
        questions = load_longmemeval_local(str(f))
        assert len(questions) == 1
        assert questions[0].question_id == "q1"

    def test_sample_limit(self, tmp_path):
        import json

        data = [
            {
                "question_id": f"q{i}",
                "question_type": "multi-session",
                "question": "?",
                "answer": "a",
                "haystack_sessions": [],
                "answer_session_ids": [],
            }
            for i in range(10)
        ]
        f = tmp_path / "test.json"
        f.write_text(json.dumps(data))

        from benchmarks.longmemeval.adapter import load_longmemeval_local
        questions = load_longmemeval_local(str(f), sample=3)
        assert len(questions) == 3

    def test_type_filter(self, tmp_path):
        import json

        data = [
            {
                "question_id": "q1",
                "question_type": "temporal-reasoning",
                "question": "?",
                "answer": "a",
                "haystack_sessions": [],
                "answer_session_ids": [],
            },
            {
                "question_id": "q2",
                "question_type": "multi-session",
                "question": "?",
                "answer": "a",
                "haystack_sessions": [],
                "answer_session_ids": [],
            },
        ]
        f = tmp_path / "test.json"
        f.write_text(json.dumps(data))

        from benchmarks.longmemeval.adapter import load_longmemeval_local
        questions = load_longmemeval_local(str(f), question_type_filter="temporal-reasoning")
        assert len(questions) == 1
        assert questions[0].question_type == "temporal-reasoning"

    def test_file_not_found(self):
        from benchmarks.longmemeval.adapter import load_longmemeval_local
        with pytest.raises(FileNotFoundError):
            load_longmemeval_local("/nonexistent/path.json")
