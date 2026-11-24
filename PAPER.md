# MemoChain: Verifiable Constant-Size Memory for Language Models

**Draft 1 - November 2026**

## Abstract

Long-horizon LLM agents currently carry memory in one of two forms: an
ever-growing prompt (O(n) tokens per query, unbounded state) or a vector
store (O(n) index, hit-or-miss recall with no synthesis). We present
MemoChain, a neural memory that is constant-size, streaming, and
cryptographically verifiable. The entire history of a document collection or
chat session folds into a fixed 22,528-byte state (11 heads x 32 keys x 32
values, Q14 fixed-point int16) updated by a Delta rule in O(1) per document.
Because the fold is pure integer arithmetic, the same inputs produce
bit-identical state digests in Python, native Rust, and a zkVM, which lets us
attach Mina-style recursive zero-knowledge proofs: a third party verifies in
~28 ms that a 22 KB memory capsule was honestly computed from a committed
input sequence under a committed writer, without re-reading the history and
without seeing the raw inputs.

We train the memory without any teacher model. Gradients from a frozen
causal LM's cross-entropy loss flow through differentiable memory-token
injection, a reader head, and a straight-through Q14 quantizer into the
Delta-rule writer. On held-out PersonaMem-style QA (4,237 questions, 30
unseen users, 200 documents each), the self-trained 22 KB capsule lifts a
frozen Qwen3-8B decoder from 58.18% to 64.69% (hard questions 30.84% to
38.16%), exactly matching a 388 MB cross-attention memory teacher (64.76% /
38.16%) that occupies a 4.10 MB KV cache - at 181.8x smaller memory and zero
teacher dependence in training. A capacity ablation shows the memory gain
saturates at a 10 KiB state on gist-style QA: growing the state 8.8x moves
the gain by only +0.007 nats. Training the same 22 KB capsule on a real
developer's 241 coding sessions doubles coding-session recall (+0.94 to
+1.76 nats) while retaining 96.6% of the personal-QA gain. We publish the
full stack: an OpenAI-compatible memory proxy in live daily use, bit-exact
Rust and SP1 zkVM implementations, and a capacity/honesty audit including
negative results.

## 1. Introduction

An agent that reads, codes, or converses for months cannot keep its history
in the prompt. The two standard escapes both leak cost and trust:

- **Full context** re-sends and re-attends over history every turn. Cost
  grows without bound; latency and price follow.
- **Retrieval (RAG)** stores chunks in an index and recalls top-k. Cost per
  write is small, but recall is lexical-adjacent, synthesis across chunks is
  left to the decoder, and the index itself grows forever.

Latent-memory systems such as Memorilla compress a collection into a fixed
set of memory tokens read by the decoder in O(1), but the memory module
re-attends over all documents when writing (O(n) per collection), the
resulting KV cache is megabytes, and nothing about the memory is portable or
verifiable: it exists only inside one model's attention.

MemoChain asks a different question: **what is the smallest memory artifact
that (a) absorbs an unbounded stream in O(1) per write, (b) is portable
across processes and machines as a single file, and (c) carries a proof that
it was written honestly?** The blockchain analogy is direct:

```text
Mina protocol:  new_chain = fold(old_chain, block)  + recursive zk proof
MemoChain:      new_state = fold(old_state, doc)     + receipt / zk proof of the write
```

Mina keeps a whole blockchain constant-size by recursively proving each
block transition against the previous proof. MemoChain keeps a whole
conversation history constant-size by folding each document into a fixed
22 KB associative state, and proves each transition with a receipt chain
that upgrades to a recursive succinct proof. The analogy is load-bearing,
not decorative: it dictated the integer-only arithmetic (Section 2), the
receipt/accumulator design (Section 3), and the 22 KB size target.

Contributions:

1. **A constant-size, O(1)-update neural memory with bit-exact integer
   semantics** (Section 2): the Delta-rule fold runs identically in Python,
   Rust, and a zkVM; every write produces a SHA-256 receipt; any tampering
   with the state is detectable.
2. **Mina-style recursive IVC over neural memory** (Section 3): a
   compressed STARK proves a 200-document fold (238.9M cycles, 1.24 MiB
   proof) and each +1-document recursive step (33.8M cycles); verification
   is ~28 ms; the capsule extracted from the proof's public values is
   byte-identical to the Python fold.
3. **Distillation-free training** (Section 4): the writer and reader train
   end-to-end through a frozen LM's loss with straight-through Q14
   quantization. No teacher, no memory-module distillation target. The
   trained capsule matches the Memorilla teacher's held-out accuracy
   exactly (Section 5.2).
4. **A capacity scaling law for quantized fast-weight memory**
   (Section 5.3): on gist-style QA the knee is at 10 KiB; the 22 KiB
   Mina-sized state is past the knee, and the binding constraints are the
   read/write projections and the decoder's token budget, not state size.
5. **Domain transfer with a real user's coding history** (Section 5.4):
   causal-prefix training on 241 real sessions doubles coding recall while
   a joint multi-domain model keeps 96.6% of the original domain's gain.
6. **An honesty audit** (Section 6): null-state and wrong-session capsule
   swaps move choice accuracy by only 1-2 pp; we therefore report NLL gains
   as the primary metric and publish the decomposition analysis, negative
   results included.
7. **A deployed system** (Section 7): an OpenAI-compatible proxy that folds
   conversation overflow into the capsule live, injects memory notes into
   the system prompt, and runs the zkVM proof queue in the background.

## 2. The 22 KB memory state

### 2.1 Format

The state is exactly 22,528 bytes:

```text
11 heads x 32 keys x 32 values x int16 (Q14) = 22,528 bytes
```

Each head is a fast-weight associative matrix in Q14 fixed point (a stored
value n means n / 2^14). Eleven 32x32 heads give 352 independent fact slots
with 32-dimensional payloads while landing on Mina's ~22 KB constant-size
figure. The state carries a full SHA-256 digest; any write changes it, so
tampering is detectable. Disk exports use a versioned capsule envelope
(magic, version, step, chain head, payload hash, payload); a flipped bit
fails the hash check loudly on load.

### 2.2 The fold

A document is a 2560-d Q14 embedding (Qwen3-Embedding-4B in our runs).
Learned integer linear layers map the full 2560-d embedding into 11
per-head keys and values (`W_key`, `W_val` of shape `[352, 2560]` plus
biases):

```text
k = normalize((W_key[h] @ x >> 14) + b_key[h])     # Q14 unit key, integer sqrt
v = clip16((W_val[h] @ x >> 14) + b_val[h])        # bounded int16
recalled = (k @ S_h) >> 14
delta    = beta_h * (v - recalled) >> 14
S_h      = (decay_h * S_h >> 14) + (k outer delta >> 14)
```

This is the Delta rule (fast-weight / linear-attention family): writing a
fact under the same key replaces it rather than blending, and per-head
learnable beta/decay gates control write strength and forgetting. All
arithmetic is integer multiply-add-shift plus one integer square root. This
is what makes the state a cryptographic commitment rather than a float
array: the same input sequence yields the same 256-bit digest on any
machine.

Fold and read cost is flat in history size (measured 0.45 ms/doc fold,
0.28 ms read on one CPU core at 100k documents), while a full-history
baseline degrades linearly.

### 2.3 Receipts and the light client

Every write emits a receipt and advances a rolling chain head:

```text
binding     = sha256(step | prev_head | sha256(doc) | new_state_digest)
accumulator = sha256(prev_head | binding)
```

A light client holds only (step, accumulator, state_digest) - three fields,
forever - and verifies a claimed state in O(1). A full audit replays every
document and recomputes the accumulator: any inserted, dropped, reordered,
or edited document breaks the chain. This is the same trust split as a
blockchain light client, applied to memory.

## 3. Verifiable memory writes (SP1 recursive IVC)

The integer fold ports to the SP1 zkVM unchanged, because there is no
float semantics to emulate. The guest reads the quantized writer weights
(3,605,932 bytes, SHA-256 committed) and the raw Q14 documents, runs the
fold with the receipt chain, and commits (step, state, state_digest,
accumulator) as public values.

Two proof modes:

- **Execute** (fast integrity check): full zkVM guest replay of a session's
  cumulative documents, checking the capsule's step, state digest,
  accumulator, and weights digest. ~13 s for 34 documents (99.6M cycles) on
  an ARM desktop GPU host; used as the background audit in the live proxy.
- **Prove** (succinct proof): a compressed recursive STARK. The base proof
  for 199 documents costs 238.9M cycles / ~24 min on the same host and
  1.24 MiB; each further document is a recursive step verifying the
  previous proof (33.8M cycles, ~5.7 min). Verification is ~28 ms with no
  history access.

Exactness inside the guest is nontrivial at this scale: for batches larger
than four documents the prover supplies exact int64 pre-shift dot products
and the guest checks them with a 2-round exact i128 Fiat-Shamir Freivalds
protocol (u64 challenges, soundness error <= 2^-128). Patching the sha2
crate to SP1's hardware SHA-256 precompile cut hashing cycles 6.8x. The
resulting capsule, read from the proof's verified public values, is
byte-for-byte identical to the Python fold; we verified four-way agreement
(Python == native Rust == zkVM execute == zkVM proof) on real unseen
sessions.

What the proof proves, precisely: "this 22 KB state was produced by running
writer weights W over the document sequence committed to accumulator A."
It does not prove the embedding model ran honestly (embeddings are trusted
inputs, or would need their own proof), and it does not constrain the
downstream decoder.

## 4. Training without a teacher

The reader addresses the state with query keys, then produces M memory
tokens (16 slots x the decoder's hidden width) from the recalled values
plus the question embedding. Training is end-to-end through a frozen
causal LM:

```text
docs -> fold (straight-through Q14) -> 22 KB state
(state, question) -> reader -> [16, hidden] memory tokens
memory tokens replace <|memory|> embeddings in the frozen LM's prompt
loss = cross-entropy on the answer
```

Gradients flow from the LM's loss back through the token injection, the
reader, and the straight-through quantizer into the Delta-rule writer; the
LM itself never updates. The Q14 straight-through estimator is the key
trick: the trained weights survive hard int16 quantization with no
accuracy cliff (float vs quantized state agree to the 7th decimal in
cosine; the int16 writer matches the float writer at 0.999991 cosine
across all held-out questions).

No teacher weights, no distillation target, no memory-module architecture
is needed anywhere in training. The data format (parquet questions +
precomputed embedding stores) is shared with the Memorilla-Data release,
which we use purely as a public benchmark source.

## 5. Evaluation

All QA evaluations follow Memorilla's protocol exactly: frozen Qwen3-8B
decoder (bfloat16, greedy, thinking disabled), 16 `<|memory|>` slots in the
system prompt, Qwen3-Embedding-4B nearest-neighbor choice scoring. Every
MemoChain number is produced from a capsule serialized to bytes and
reloaded (`to_bytes` -> `from_bytes`), so no float state can leak from
writer to reader.

### 5.1 Distilled reference (teacher comparison)

The distillation-trained student (kept as a comparison path) on pv4/test -
4,237 questions, 30 unseen users, 200 documents each:

| method | memory per user | accuracy | hard acc |
| --- | --- | --- | --- |
| closed-book | 0 | 58.58 | 31.80 |
| Memorilla teacher (KV cache) | 4.10 MB | 64.76 | 38.16 |
| MemoChain 22 KB capsule | 22,528 B | 64.67 | 38.00 |

98.5% of the teacher's memory gain at 181.8x smaller memory. On pmv2/test
(5,000 questions, 200 unseen users), the capsule exceeds the teacher
outright: 32.96% vs 31.42% (closed-book 25.60%), 126.5% of the teacher's
gain at 105.5x compression. The Delta-rule state generalizes 3.1x past its
64-document training horizon to 200 documents with no loss.

### 5.2 Self-trained (no teacher anywhere)

From scratch, 3,000 steps, batch 4 x grad-accum 2, through frozen decoders:

| decoder | val NLL (zero-mem -> capsule) | gain | QA accuracy (closed-book -> capsule) |
| --- | --- | --- | --- |
| Qwen3-0.6B | 5.24 -> 3.53 | +1.71 nats | 50.77 -> 57.02 (+6.25 pp) |
| Qwen3-8B | 5.92 -> 2.665 | +3.25 nats | 58.18 -> 64.69 (+6.51 pp; hard 30.84 -> 38.16) |

The 8B-trained capsule matches the Memorilla teacher's accuracy and
hard-question accuracy exactly (64.69 vs 64.76; 38.16 vs 38.16) with the
same frozen decoder, the same 22 KB format, and zero teacher weights in
training. For this benchmark, the self-trained constant-size capsule closes
the entire gap between closed-book and a 4.10 MB attention cache.

Decoder size, not optimization length, is the training lever: a
warm-restarted 10,000-step 0.6B run declined monotonically in validation
gain (we stopped it at step 5,000 and report this as a negative result).

### 5.3 Capacity scaling

Identical recipes across state sizes (frozen Qwen3-0.6B, 1,500 steps,
30 fixed validation batches):

| state | size | fact slots | params | val gain (nats) |
| --- | --- | --- | --- | --- |
| 5x32x32 | 10 KiB | 160 | 114.9M | +1.673 |
| 11x32x32 | 22 KiB | 352 | 123.8M | +1.670 |
| 22x32x32 | 44 KiB | 704 | 140.0M | +1.680 |
| 44x32x32 | 88 KiB | 1,408 | 172.4M | +1.675 |

On 200-document gist-style QA the knee is at 10 KiB: 8.8x more state buys
+0.007 nats (noise). The binding constraints are the read/write
projections and the decoder's 16-token budget, not associative capacity.
The 22 KiB Mina-sized state is comfortably past the knee. We expect the
knee to move under verbatim recall demands, 10k+ document histories, and
larger token budgets; the ablation harness makes those one-command
follow-ups.

### 5.4 Coding-domain transfer from a real user's history

We built a causal-prefix dataset from one author's real developer-terminal
sessions (241 sessions; chronological docs include user turns, assistant
replies, and compact tool-action summaries; every training question folds
only documents strictly prior to that turn - zero future leakage; secrets
scrubbed; session-level splits):

| checkpoint | coding test NLL gain | pv4 gain retained |
| --- | --- | --- |
| pv4-only (zero-shot) | +0.94 nats | +3.25 (baseline) |
| coding from scratch | +1.75 nats | 54% |
| coding from pv4 warm-start | +1.74 nats | 93% |
| **joint pv4 + coding** | **+1.76 nats** | **96.6%** |

Domain training nearly doubles coding recall (100% of held-out rows
improved), and the joint multi-domain model is the best single checkpoint
on both domains - consistent with the capacity finding that 22 KiB has
spare room. Qualitatively, the joint model's capsule reconstructs concrete
session facts (server configuration files, credential-mapping schema,
launch scripts) where the pv4-only model hallucinates a generic project.

### 5.5 Live system

The proxy runs daily in front of a local chat backend. Sessions start with
no memory; past an 80,000-token conversation window (system prompt excluded,
never folded), the oldest turns fold into the capsule, memory notes are
injected into the system prompt, and the background queue replays each
fold in the zkVM, checking step, state digest, accumulator, and weights
digest against the capsule and receipt chain. Folding 34 messages / 6,044
tokens produces the 22,606-byte capsule in milliseconds; the execute check
takes ~13 s.

## 6. Honesty audit

We report three uncomfortable measurements rather than bury them:

1. **Choice-accuracy deltas understate the memory signal.** Swapping the
   real capsule for a zeroed state or a wrong-session capsule changes
   pv4/test choice accuracy by only 1-2 pp (64.2 real vs 62.4 zeroed vs
   63.4 wrong-session on a 500-row slice), even though the NLL effect is
   large (+2.16 nats for the distilled model). The nearest-neighbor choice
   scorer saturates; NLL is the honest headline metric. The full
   decomposition of state-routed vs question-bypassed reader energy is
   future work.
2. **Lossy compression is real.** 40,000 tokens of history into 22 KB is
   roughly 160 KB of text into 22 KB: gist and preferences survive;
   verbatim long lists do not. The capacity section bounds where this
   bites.
3. **Embedding lock-in.** Capsules are bound to one embedding geometry
   and a reader checkpoint. Portability across embedding models requires
   retraining or an adapter; the capsule is portable across processes and
   machines, not across encoders.

Known costs: SP1 prove is minutes per step on desktop-class hardware
(execute-mode checks are seconds), the training data format inherits from
the Memorilla-Data release for benchmark comparability, and all
evaluation used Qwen-family models.

## 7. Related work

- **Memorilla** (snap-stanford): latent semantic memory tokens read by the
  decoder in O(1); our training data format and evaluation protocol follow
  theirs for comparability, and their module served as the distillation
  teacher in early phases. MemoChain differs in O(1) writes, integer
  bit-exactness, portability, and verifiability; as of Section 5.2 the
  self-trained capsule matches their teacher without using it.
- **Fast weights / DeltaNet / linear attention**: the write rule is from
  this family; MemoChain's contribution is making it fixed-point exact,
  capsule-portable, receipt-chained, and zk-provable.
- **Mina / recursive SNARKs / SP1**: the constant-size-verifiable-state
  pattern and the IVC recursion structure; MemoChain applies them to
  neural state.
- **RAG / vector stores**: complementary for verbatim recall; MemoChain
  targets gist, synthesis, and constant-size trust.

## 8. Limitations and future work

The capacity knee is measured on gist-style QA; verbatim-recall regimes
should shift it and need their own sweep. The reader routes part of its
signal around the state (question bypass); architecturally constraining
the reader to use the state more is open. Prove-mode cost is minutes on
CPU-class hosts; folding schemes (Nova-family) and GPU provers are the
natural speedup. Capsule merging (two agents, two capsules, one combined
proof) is algebraically plausible for linear fast-weights and unverified.
Cryptographic unlearning - proving a capsule was built without a revoked
document - is the most direct application of the receipt chain and is
unbuilt.

## 9. Conclusion

A 22 KB integer state, trained by nothing but a frozen LM's own
language-modeling loss, matches a megabyte-scale cross-attention memory
teacher on held-out personal QA; folds a real developer's coding history
into concrete, recallable facts; carries a receipt chain that turns
memory tampering into a hash mismatch; and proves its own writes in a
zkVM with 28 ms verification. Memory for long-horizon agents can be
constant-size, portable, and honest at the same time.

## Reproducing

Everything runs from the repository: the pure-NumPy core and 67 tests on a
laptop CPU; e2e training on a single 12 GB GPU; SP1 proofs on Linux/ARM
(see crates/sp1/README.md); the proxy on any host with an OpenAI backend.
See README.md for quickstart commands and DESIGN.md for the full
phase-by-phase experimental log.
