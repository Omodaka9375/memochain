# Your Chat History Fits in 22 KB (and Can Prove It)

*November 2026 - 5 minute read*

AI agents that work with you for months have a memory problem, and it is
not the one you think. The problem is not that they forget. It is that
*remembering* is structurally expensive: either you re-send the whole
history every turn (tokens, latency, cost - all growing without bound),
or you shard it into a vector store and hope the right chunk surfaces at
the right moment. Neither is honest: the first is unbounded, the second
is hit-or-miss, and neither can *prove* anything about what it remembers.

We built MemoChain around a different question: **what is the smallest
memory artifact that absorbs an unbounded stream in O(1) per write, is
portable as a single file, and carries a proof that it was written
honestly?**

## Mina, but for memory

The Mina blockchain keeps its entire history verifiable in a constant
~22 KB by recursively proving each new block against the previous
proof. We applied the same shape to conversation memory:

```text
Mina:      new_chain = fold(old_chain, block)  + recursive zk proof
MemoChain: new_state = fold(old_state, doc)   + receipt / zk proof
```

The state is exactly 22,528 bytes: 11 heads x 32 keys x 32 values of Q14
fixed-point int16, updated by a Delta rule. One document in, one integer
matrix update (0.45 ms on a laptop CPU core), the state never grows.
And because the fold is pure integer arithmetic - no floats anywhere -
the same input sequence produces the same 256-bit digest on every
machine. That is what turns a neural memory into a cryptographic
commitment, and what makes it portable to a zkVM: a compressed STARK
proves the fold, and anyone verifies in ~28 ms that a 22 KB capsule was
honestly computed from a committed input sequence under committed
writer weights - without seeing the raw history.

## No teacher, no distillation

The fun part is how the memory trains itself. A frozen LLM's own
cross-entropy loss on real answers flows backward through injected
memory tokens, through the reader, and through a straight-through
quantizer into the Delta-rule writer. No teacher model, no distilled
memory module - the LLM literally teaches the memory to be useful to
itself.

On held-out PersonaMem-style QA (4,237 questions, 30 unseen users, 200
documents each, frozen Qwen3-8B decoder):

| Method | Memory/user | Accuracy |
|---|---|---|
| Closed-book | 0 | 58.18% |
| Memorilla teacher (KV cache) | 4.10 MB | 64.76% |
| **MemoChain 22 KB (self-trained)** | **22,528 B** | **64.69%** |

The self-trained capsule matches the 388 MB cross-attention memory
teacher exactly, at 181.8x smaller memory, with zero teacher weights in
training. Hard questions: 30.84% -> 38.16%, same as the teacher.

## The results we did not expect

**Capacity saturates at 10 KB.** We swept the state from 10 KB to 88 KB
with identical training recipes. Memory gain moved by +0.007 nats -
noise. On gist-style QA the bottleneck is not associative capacity but
the read/write projections and the decoder's 16-token memory budget.
The 22 KB "Mina-sized" state is comfortably past the knee, which is a
nice bit of cosmic alignment.

**It remembers real coding work.** We trained the same capsule on one
author's real developer-terminal sessions (241 sessions, zero future
leakage, secrets scrubbed). Coding recall doubled (+0.94 -> +1.76 nats,
100% of held-out sessions improved), and a joint multi-domain model kept
96.6% of its original QA gain. Qualitatively: asked about a held-out
session, the capsule reconstructs concrete facts - the sshd_config
edits, the authorized_keys-to-credentials JSON schema, the start.sh
launcher - where the personal-chat-only model hallucinates a generic
project.

**One honest caveat we publish.** Swapping the real capsule for a
zeroed or wrong-session capsule moves choice-accuracy by only 1-2 pp on
this benchmark, even though the NLL effect is large. The
nearest-neighbor choice scorer saturates. We headline NLL gains, we
publish the audit, and we flag the reader's question-bypass as open
work. Memory for agents has enough vaporware already; it deserves
numbers that survive scrutiny.

## It runs live

The stack behind this text is a slim (~55 MB RAM, zero extra GPU) proxy
supporting both OpenAI `/v1/chat/completions` and native Anthropic
`/v1/messages` with BYOK key pass-through, running in front of local
servers (vLLM, llama.cpp, Ollama) or cloud routers (OpenRouter, direct
Anthropic). Conversations start with zero overhead; past an 80,000-token
window the oldest turns fold into the 22 KB capsule and SHA-256 receipt
chain, and an optional background queue replays every fold inside the SP1
zkVM. Tampering with the capsule is a hash mismatch, not a silent
corruption.

One practical lesson from daily use: a 22 KB neural state preserves gist
and preferences, while a coding session also needs exact verbatim recall
of ports, hashes, and paths - and hosting a second 18 GB neural reader
model defeats the point of a lightweight proxy. So the default `hybrid`
mode pairs the verifiable 22 KB capsule with a reader-free recall layer (a
rolling bullet fact sheet from the active backend plus an int8-quantized,
FIFO-capped SQLite store for top-K verbatim snippets; outside the ZK
circuit, stated plainly in the paper's honesty audit). Two containers
sharing one session volume carry the same `x-session-id` across local and
cloud models. On a 120-turn A/B replay metered by the backend's own
`usage.prompt_tokens`, total input tokens dropped from 7,086,360 to
674,016 - a 90% reduction.

## What is next

The capacity knee should move under verbatim-recall regimes - that is a
one-command experiment now. Capsule merging (two agents, two capsules,
one combined proof) is algebraically plausible and unverified.
Cryptographic unlearning - proving a memory was built *without* a
revoked document - is the most direct application of the receipt chain,
and it is unbuilt.

The paper, the full experimental log (including negative results), the
Rust and zkVM implementations, the training pipeline, the savings
benchmark, and the live proxy are all in the repository. The core and its
76 Python tests (plus 14 Rust tests) run on a laptop CPU; training fits a
single 12 GB GPU; proofs run on Linux/ARM.

Memory for long-horizon agents can be constant-size, portable, and
honest at the same time. That is the whole thesis - in 22 KB.
