"""Unit tests for MemoChain end-to-end QA evaluation helpers (no GPU / vLLM required)."""

from __future__ import annotations

import numpy as np
import pytest

from memochain.eval_qa import (
    MEMORY_TOKEN,
    PERSONA_PROMPT,
    build_closed_book_messages,
    build_memory_messages,
    choice_index,
    clean_generation,
    memory_placeholder,
    score_choices_from_vectors,
)


def test_clean_generation_strips_qwen_tokens() -> None:
    raw = "  <|im_start|>Berlin is the user's current city.<|im_end|> \n"
    assert clean_generation(raw) == "Berlin is the user's current city."


def test_memory_messages_reserve_exact_placeholder_count() -> None:
    msgs = build_memory_messages("Where do I live?", num_memories=16)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith(PERSONA_PROMPT)
    assert msgs[0]["content"].count(MEMORY_TOKEN) == 16
    assert memory_placeholder(4) == "\nMemory<|memory|><|memory|><|memory|><|memory|>Memory"

    cb = build_closed_book_messages("Where do I live?")
    assert MEMORY_TOKEN not in cb[0]["content"]


def test_choice_scoring_and_hard_subset() -> None:
    choices = [
        ["Paris", "Berlin", "Tokyo", ""],
        ["Python", "Rust", "Go"],
        ["Morning", "Night"],
    ]
    refs = ["Berlin", "Rust", "Night"]
    hard = [True, True, False]

    # Row 0: correct (Berlin=idx 1), Row 1: wrong (picks Python=idx 0), Row 2: correct (Night=idx 1)
    cvecs = [
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    ]
    pvecs = np.array(
        [
            [0.05, 0.99],  # -> idx 1 (Berlin, correct, hard=True)
            [0.95, 0.10],  # -> idx 0 (Python, wrong, hard=True)
            [0.10, 0.98],  # -> idx 1 (Night, correct, hard=False)
        ],
        dtype=np.float32,
    )

    metrics, selected = score_choices_from_vectors(pvecs, cvecs, choices, refs, hard)
    assert selected.tolist() == [1, 0, 1]
    assert pytest.approx(metrics["accuracy"]) == 2.0 / 3.0
    assert pytest.approx(metrics["accuracy_hard"]) == 0.5

    with pytest.raises(ValueError):
        choice_index(["a", "b"], "c")
