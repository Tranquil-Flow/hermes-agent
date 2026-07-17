"""Regression coverage for persisted background-review harness recovery."""

import pytest

from agent.background_review import (
    _COMBINED_REVIEW_PROMPT,
    _MEMORY_REVIEW_PROMPT,
    _SKILL_REVIEW_PROMPT,
)
from hermes_state import (
    _is_background_review_harness_message,
    _strip_background_review_harness,
)


@pytest.mark.parametrize(
    "prompt",
    [
        _MEMORY_REVIEW_PROMPT,
        _SKILL_REVIEW_PROMPT,
        _COMBINED_REVIEW_PROMPT,
    ],
)
def test_current_prefixed_review_prompts_are_detected(prompt):
    """All three disclaimer-prefixed operational prompts are harnesses."""
    assert _is_background_review_harness_message({
        "role": "user",
        "content": prompt,
    })


def test_generic_system_note_is_not_misclassified():
    """Only exact review openings match; ordinary system-note text remains."""
    assert not _is_background_review_harness_message({
        "role": "user",
        "content": "[System Note] The user explicitly asked to preserve this text.",
    })


def test_combined_prefixed_harness_and_reply_are_stripped():
    messages = [
        {"role": "user", "content": "real user message"},
        {"role": "user", "content": _COMBINED_REVIEW_PROMPT},
        {"role": "assistant", "content": "curator reply"},
        {"role": "assistant", "content": "real assistant reply"},
    ]

    assert _strip_background_review_harness(messages) == [
        {"role": "user", "content": "real user message"},
        {"role": "assistant", "content": "real assistant reply"},
    ]
