"""Unit tests for distillation-free end-to-end training (no Memorilla teacher)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from memochain.eval_qa import MEMORY_TOKEN
from memochain.student import MemoChainStudent, StudentConfig
from memochain.student_int import IntStudentWriter, quantize_embeddings_q14
from memochain.train_e2e import (
    build_training_example,
    compute_e2e_loss,
    export_student_artifacts,
    inject_memory_embeddings,
)


class TinyTokenizer:
    """Minimal word-level tokenizer supporting apply_chat_template + <|memory|>."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {
            "<pad>": 0,
            "<eos>": 1,
            MEMORY_TOKEN: 2,
        }
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.eos_token = "<eos>"

    def _tok_id(self, tok: str) -> int:
        if tok not in self.vocab:
            self.vocab[tok] = len(self.vocab) % 250 + 3
        return self.vocab[tok]

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    ) -> str:
        parts = []
        for m in messages:
            parts.append(f"{m['role']}: {m['content']}")
        if add_generation_prompt:
            parts.append("assistant:")
        return " ".join(parts)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Split <|memory|> tokens cleanly even when adjacent.
        spaced = text.replace(MEMORY_TOKEN, f" {MEMORY_TOKEN} ").replace(
            self.eos_token, f" {self.eos_token} "
        )
        return [self._tok_id(tok) for tok in spaced.split() if tok]


class TinyCausalLM(nn.Module):
    """Frozen tiny causal LM whose loss depends linearly on inputs_embeds."""

    def __init__(self, vocab_size: int = 256, hidden: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, vocab_size)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, inputs_embeds, attention_mask=None, labels=None, use_cache=False):
        # Mix preceding tokens (including injected memory slots) into every position
        # via a causal prefix mean so gradients flow from answer labels -> memory slots.
        h = torch.tanh(self.proj(inputs_embeds))
        csum = torch.cumsum(h, dim=1)
        counts = torch.arange(1, h.shape[1] + 1, device=h.device, dtype=h.dtype).view(1, -1, 1)
        ctx = csum / counts
        logits = self.head(ctx)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        class Out:
            pass

        out = Out()
        out.loss = loss
        out.logits = logits
        return out


def test_build_training_example_preserves_memory_tokens() -> None:
    tok = TinyTokenizer()
    ids, labels = build_training_example(
        tok,
        "What color is the sky? " + "extra " * 100,
        "Blue.",
        num_memories=4,
        memory_token_id=2,
        max_prompt_tokens=20,
    )
    assert ids.count(2) == 4
    # prompt positions masked to -100, answer positions supervised
    assert any(x == -100 for x in labels)
    assert any(x != -100 for x in labels)
    assert len(ids) == len(labels)


def test_e2e_gradients_flow_to_writer_and_reader_without_teacher() -> None:
    torch.manual_seed(0)
    tok = TinyTokenizer()
    lm = TinyCausalLM(vocab_size=256, hidden=32)

    cfg = StudentConfig(embedding_dim=64, output_dim=32, num_memories=4, initial_beta=0.3)
    student = MemoChainStudent(cfg)

    docs = torch.randn(2, 6, 64)
    pad = torch.zeros(2, 6, dtype=torch.bool)
    q_emb = torch.randn(2, 64)
    q_txt = ["What did the user choose?", "Which port is used?"]
    a_txt = ["dark theme", "port 9000"]

    loss = compute_e2e_loss(
        student,
        lm,
        lm.embed,
        tok,
        memory_token_id=2,
        docs=docs,
        padding=pad,
        q_embeds=q_emb,
        q_texts=q_txt,
        a_texts=a_txt,
        quantize=True,
    )
    assert torch.isfinite(loss)
    loss.backward()

    # Gradients MUST flow all the way through frozen LM -> reader -> STE Q14 -> Delta-rule writer.
    for name in ("key_proj.weight", "val_proj.weight", "beta_logits", "decay_logits", "reader_slots.weight", "reader_out.weight"):
        p = dict(student.named_parameters())[name]
        assert p.grad is not None, f"missing grad on {name}"
        assert float(p.grad.abs().sum().item()) > 0.0, f"zero grad on {name}"
    # Frozen LM stayed frozen
    assert all(p.grad is None for p in lm.parameters())


def test_e2e_loss_decreases_over_steps_and_exports_q14_writer(tmp_path) -> None:
    torch.manual_seed(42)
    tok = TinyTokenizer()
    lm = TinyCausalLM(vocab_size=256, hidden=32)

    # Use full 2560 embed_dim on export check so IntStudentWriter validates wire format.
    cfg = StudentConfig(embedding_dim=2560, output_dim=32, num_memories=4, initial_beta=0.3)
    student = MemoChainStudent(cfg)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-3)

    docs = torch.randn(2, 4, 2560) * 0.05
    pad = torch.zeros(2, 4, dtype=torch.bool)
    q_emb = torch.randn(2, 2560) * 0.05
    q_txt = ["Where is the config?", "Which database?"]
    a_txt = ["settings json", "postgres"]

    losses = []
    for _ in range(8):
        opt.zero_grad()
        loss = compute_e2e_loss(
            student, lm, lm.embed, tok, 2, docs, pad, q_emb, q_txt, a_txt, quantize=True
        )
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"

    info = export_student_artifacts(student, tmp_path / "ckpt")
    assert (tmp_path / "ckpt" / "student.pt").exists()
    assert (tmp_path / "ckpt" / "weights.bin").exists()
    assert (tmp_path / "ckpt" / "reader.npz").exists()
    assert len(info["weights_digest"]) == 64

    # Verify the exported Q14 int16 writer folds cleanly into the 22,528-byte state
    writer = IntStudentWriter.from_bytes((tmp_path / "ckpt" / "weights.bin").read_bytes())
    assert writer.weights_digest() == info["weights_digest"]
    q14_docs = quantize_embeddings_q14(docs[0].numpy())
    state, head = writer.fold_sequence(q14_docs)
    assert state.nbytes == 22528
    assert head.step == 4
