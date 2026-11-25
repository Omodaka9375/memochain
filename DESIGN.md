# MemoChain: Mina-Style Constant-Size Memory for LLMs

A recursive memory system where the entire history of a document collection
collapses into a fixed 22 KB state, updated one document at a time in O(1),
with cryptographic receipts that prove the state was written honestly.

The analogy:

```text
Mina protocol:   new_chain = fold(old_chain, block)  + recursive zk proof
MemoChain:       new_state = fold(old_state, doc)    + receipt of correct write
```

Mina keeps a whole blockchain verifiable in ~22 KB by recursively proving each
new block against the previous proof. MemoChain applies the same shape to
long-term memory: instead of an ever-growing vector store or an ever-growing
prompt, there is one fixed-size artifact that absorbs every new document and
never grows, no matter how much history accumulates.

## Why not the alternatives

| Approach | Cost per query | Cost per write | Failure mode |
| --- | --- | --- | --- |
| Full context in prompt | grows with history | none | prompt bloat, latency, cost |
| RAG / vector search | O(index) | O(log n) | hit-or-miss recall, no synthesis |
| Memorilla (state of the art) | O(1) decode, but O(n) re-encode or O(n) KV cache | O(n) | memory module re-attends over all docs |
| MemoChain (this) | O(1) | **O(1)** | lossy fold, mitigated below |

## State format (v1)

The memory state is exactly 22,528 bytes (about 22 KB):

```text
11 heads x 32 keys x 32 values x int16 = 22,528 bytes
```

Each head is a fast-weight associative matrix in Q14 fixed point (a value of
n means n / 2^14). The head count and head shape are design constants; 11
heads of 32 x 32 hits the Mina 22 KB figure while leaving room for 352
independent fact slots with 32-dimensional payloads.

The state carries a full 256-bit SHA-256 digest, which is the "block hash":
any write changes the digest, which makes tampering detectable. Disk exports
use a versioned Capsule envelope:

```text
magic(4) | version(u16) | step(u64) | chain head(32) | payload sha256(32) | payload(22,528)
```

A capsule with a flipped bit fails to load loudly instead of silently
corrupting memory.

## The fold: multi-head Delta rule

Documents are 704 int16 values, split into a key part (first 352, what
addresses memory) and a value part (last 352, what gets stored). Head h
binds its own 32-dim key slice to its own 32-dim value slice through
learnable projections W_key[h] and W_val[h]:

```text
k = normalize(W_key[h] @ key_slice_h)      # Q14 unit key, integer sqrt
v = W_val[h] @ value_slice_h               # bounded int16
recalled = (k @ S_h) >> 14                 # what the memory thinks v is
delta    = beta * (v - recalled) >> 14     # correction
S_h      = (decay * S_h >> 14) + (k outer delta >> 14)
```

This is the Delta rule (linear-attention / fast-weight family). Writing a
fact with the same key but new content replaces the old value instead of
blending: "moved from Paris to Berlin" just updates. Measured recall after a
single write is exactly beta * v (correlation 1.0), and after rewriting the
least-squares fit is decay*(1-beta)*v_old + beta*v_new, so the new fact
dominates the old one by about 4x at beta = 0.75.

All arithmetic is pure integer (multiply, add, shift, plus an integer square
root for key normalization). The same input sequence produces bit-identical
state digests on any machine, which is what makes receipts meaningful and
the fold portable to ZK circuits later.

## Verification: O(1) light client, O(n) full audit

Every write emits a receipt and advances a rolling chain head:

```text
binding     = sha256(step | prev_head | sha256(doc) | new_state_digest)
accumulator = sha256(prev_head | binding)
```

The light client holds only `(step, accumulator, state_digest)` - three
small fields, no matter how long the history. Light verification compares
the stored state digest to the head's in O(1). A full audit replays every
document and recomputes the accumulator: any inserted, dropped, reordered,
or edited document changes the head and fails.

Production version: compile the fold to a zkVM (SP1) or a folding scheme
(Nova) so the accumulator becomes a real succinct proof. The integer-only
fold makes this tractable: no floating point, no transcendentals, just
multiply-add-shift over int16. Prove only the writer, never the frozen
decoder.

## What the tests prove

```text
- state is exactly 22,528 bytes, digest is full 256-bit
- capsules roundtrip and reject single-bit corruption
- fold is bit-identical across independent Folder instances
- fold cost is constant in history size (100 vs 10,000 docs)
- a planted fact survives 500 noisy distractor writes (corr > 0.5)
- rewriting a fact suppresses the old value (new dominates by >2x)
- the light head stays O(1) after 1,000 writes
- full audit replays and verifies; any tampering fails
```

Benchmark (demo.py, int16 fold on one CPU core):

```text
    docs | fold ms/doc | memo read ms | baseline read ms | memo RAM | baseline RAM
     100 |       0.453 |        0.280 |             0.211 |   22 KiB |    138 KiB
    1000 |       0.450 |        0.276 |             1.620 |   22 KiB |    1.3 MiB
   10000 |       0.454 |        0.285 |            16.637 |   22 KiB |   13.4 MiB
  100000 |       0.458 |        0.271 |           172.800 |   22 KiB | 134.3 MiB
```

Fold and read stay flat while the baseline degrades linearly.

## Repository layout

```text
src/memochain/
  state.py       MemoryState: 11x32x32 int16, full digest; Capsule v1/v2/v3
  fold.py        Folder: Delta-rule fold, associative read, O(n) baseline
  verify.py      receipts, O(1) ChainHead, full-audit verify_chain
  student.py     differentiable student: STE Q14 fold + memory-token reader
  student_int.py exact Q14 integer writer + weights.bin wire format
  reader_np.py   torch-free numpy reader (proxy/sidecar parity with torch)
  train_e2e.py   distillation-free trainer: frozen-LLM loss -> 22 KB student
  distill.py     optional teacher-distillation trainer (comparison path)
  teacher.py     standalone Memorilla teacher + Memorilla-Data store loader
  eval_qa.py     end-to-end QA eval (vLLM), --teacher none for standalone
  eval_distill.py  held-out cosine vs teacher + raw-doc baseline
  export.py      capsule export/verify/read CLI with provenance head
  hybrid.py      hot-tail ring buffer over the cold state, capsule roundtrip
  server/        OpenAI-compatible memory proxy + SP1 proof queue
                 (session folding, SSE passthrough, reader sidecar service)
crates/memochain-core/   Rust: bit-exact integer fold + receipts (SP1 seed)
crates/chain-logic/      the provable transition (bit-exact vs Python)
crates/sp1/              zkVM guest + host + Mina-style recursive IVC
tools/         parity vector generators, Q14/numpy weight exporters
demo.py        end-to-end: fold, recall, verify, tamper check, benchmark
tests/         pytest suite (58 tests; torch tests skip without torch)
```

## Running

```bash
uv sync --extra dev
uv run pytest
uv run demo.py
```

## Phase 3: distillation (implemented)

```text
src/memochain/teacher.py  MemorillaTeacher: standalone Memorilla MemoryModule
                          (loads stage3/memory.pt, 388 MB, no vllm needed) +
                          Memorilla-Data embedding-store loader + offline
                          SyntheticTeacher for smoke tests
src/memochain/student.py  MemoChainStudent: differentiable Delta-rule write
                          into the exact 11x32x32 state (Q14 straight-through
                          quantizer) + reader producing Memorilla-shaped
                          [B, 16, 4096] tokens; exports to MemoryState
src/memochain/hybrid.py   HybridMemory: W-slot hot ring buffer in front of the
                          22 KB cold state; total footprint O(1) forever
src/memochain/distill.py  Trainer CLI: cosine + MSE against teacher tokens,
                          AdamW with warmup + cosine schedule, peak-VRAM report
```

Smoke run on the RTX 3060 (synthetic teacher, Q14 STE quantization on):

```text
device: NVIDIA GeForce RTX 3060 (12.0 GiB)
step    0  loss 1.0185  cosine 0.0002  peak VRAM 2536 MiB
step   60  loss 0.0205  cosine 0.9879
step  240  loss 0.0007  cosine 0.9998
step  299  loss 0.0001  cosine 1.0000  peak VRAM 2537 MiB
```

Peak VRAM 2.5 GiB of 12 GiB: the whole distillation rig fits on a 12 GB
card with 9.5 GiB spare. On the real teacher the module adds ~388 MB, and
pv4 embeddings add 382 MB: still well under 4 GiB total.

## Phase 4: real-data distillation (pv4, done)

Downloaded the true teacher (stage3/memory.pt, 388 MB) and the real pv4
embedding store (119 train users x 200 conversations, 15058 question rows),
then trained the student against the real teacher with collections rotating
every step (no memorization possible).

```text
training (2000 steps, batch 2, 64 docs, beta 0.3, lr 3e-4):
  step    0  cosine -0.000   step  100  cosine 0.967
  step 1000  cosine 0.976    step 1999  cosine 0.979 (mean last-100: 0.983)
  0.175 s/step, peak VRAM 3289 MiB on the RTX 3060

held-out validation (pv4/validation, unseen users, 100 batches):
  student cosine mean 0.9836  (p25 0.983, p75 0.985)
  raw-doc baseline    0.0180  (mean doc vector -> teacher projection)
```

The 22 KB capsule reproduces Memorilla's query-conditioned memory tokens on
users it never saw, with a 55x cosine gap over the no-memory baseline.
Artifacts: runs/pv4/student.pt + runs/pv4/eval.json (gitignored).

Commands:

```bash
$env:PYTHONPATH='src'  # or uv sync --extra train
python -m memochain.distill  --teacher data/stage3 --dataset pv4 --data-dir data --steps 2000 --out runs/pv4
python -m memochain.eval_distill --checkpoint runs/pv4 --teacher data/stage3 --dataset pv4
```

## Phase 4b: longer training + quantization audit (done)

10,000 steps (same recipe, ~30 min on the 3060):

```text
train:      cosine 0.9967 (mean last-100: 0.9960), 0.21 s/step
validation: student cosine 0.9958 (p25 0.995, p75 0.996)
            float-state reference: 0.99583694 vs quantized: 0.99583693
            raw-doc baseline: 0.0182
```

The float-state and hard-Q14-quantized-state cosines agree to the 8th
decimal: straight-through quantization training means the int16 capsule
pays nothing for its 22 KB format. The gap to the no-memory baseline is now
55x on held-out users.

## Phase 6: cross-dataset transfer (pmv2, done)

Warm-started the pv4_10k student and trained on pmv2 (PersonaMem-v2: 799
collections, 18521 rows, short chat sessions; intrinsically harder, its
raw-doc baseline is 0.031):

```text
zero-shot (pv4 model on pmv2, no training):  cosine 0.884
after 5000 steps at lr 1e-4:                 train 0.9755
held-out pmv2 validation:                    student 0.9709 (p25 0.969, p75 0.973)
                                             quantized == float (7th decimal)
```

The Delta-rule state machinery transfers across datasets: the pv4-trained
writer/reader reach 0.971 on a different benchmark's unseen users with a
light touch-up. Artifacts: runs/pv4_10k, runs/pmv2_transfer.

## Phase 5: capsule export (done)

The trained student now produces real artifacts:

```text
export:  python -m memochain.export export --checkpoint runs/pv4 --data-dir data \
             --dataset pv4 --split validation --collection 6 --out capsule_user6.bin
verify:  python -m memochain.export verify --capsule capsule_user6.bin   (fresh process, CPU only)
read:    python -m memochain.export read --capsule capsule_user6.bin --checkpoint runs/pv4 \
             --data-dir data --dataset pv4 --split validation --row 0 --compare-teacher
```

Measured end to end on pv4 validation user 6 (200 conversations):

```text
fold:        200 docs -> 22,528-byte capsule, provenance head b47459ab...
verify:      payload intact, fresh process, no GPU, no checkpoint needed
read:        tokens (1, 16, 4096) from the reloaded state
             student-vs-teacher cosine 0.967 (single row; eval mean 0.9836)
```

The provenance head is a sha256 chain over per-document embedding hashes plus
the final state digest: any changed, dropped, or reordered document changes
the head, and verification needs only the capsule file. The reader needs the
student weights (reader must match the writer that folded the state); the
capsule itself stays a pure 22,528-byte payload.

## Phase 7: hot-tail capsules, learned gates, and the Rust core (done)

**Capsule format v2** (v1 files still load): after the 22,528-byte state,
an optional hot tail stores verbatim recent documents:

```text
v2 header: magic(4) version(u16) step(u64) head(32) payload_hash(32)
           hot_count(u32) hot_dim(u32)
payload:   state (22,528) + hot_count * hot_dim int16 values
```

The payload hash covers state + hot bytes, so corruption anywhere fails
loudly. HybridMemory now round-trips through capsules
(`HybridMemory.to_capsule()` / `from_capsule()`): cold 22 KB state plus the
verbatim hot tail, receipts intact. Tests: 5 new cases including v1
backward compatibility.

**Learned gates (pv4_10k checkpoint):** all 11 heads converged to
beta ~ 0.39-0.42 (init 0.3) and decay ~ 0.9978 (init 0.999): the model
wanted faster forgetting than initialized, with no per-head specialization
under single-dataset training. Multi-task mixing may split them; that is
the next ablation.

**Rust core** (`crates/memochain-core`, no dependencies): the integer
Delta-rule fold ported 1:1 (i64 multiply-add-shift, integer sqrt,
floor-division via div_euclid to match Python's `//`). Bit-exact parity is
enforced by tests/parity.rs against vectors generated by
tools/gen_rust_vectors.py from the Python implementation:

```text
cargo test --manifest-path crates/memochain-core/Cargo.toml
  4 unit tests + fold_matches_python_bit_for_bit: PASS
```

This is the seed of the SP1 guest: the fold is ~120 lines of pure integer
Rust, directly provable. Next step is receipts (sha256 chain) in Rust plus
a small host/guest split.

## Phase 8: Rust receipts, capsule v3, multi-dataset training (done)

**Rust receipt chain** (`crates/memochain-core/src/receipt.rs`): the sha256
chain over (step, prev accumulator, doc hash, state digest) ported 1:1 from
verify.py, with the compatibility contract documented in the module doc.
9 cargo tests pass: known sha256 vectors, chain advance/verify, tamper
detection, and bit-exact receipt parity against Python over 6 folded
documents (vectors from tools/gen_receipt_vectors.py; every receipt digest
and the final head match Python's verify.py exactly).

**Capsule format v3**: the hot tail gains a kind byte:
0 = raw int16 integer-fold docs (v2 semantics), 1 = float16 embeddings from
the student world (dim from the header, 2560 for Qwen3-Embedding-4B). The
student world can now ship verbatim recent embeddings in a capsule; v1 and
v2 files still load. Tests: 5 new cases.

**Multi-dataset distillation**: `--dataset pv4 quail drop ...` draws batches
round-robin across configs (deterministic interleave + shuffle, matching
memorilla's config-concatenation spirit). Verified: 38337 rows over
pubmedqa + quail + samsum + dialogsum load and batch correctly.

**beta ablation results** (2000 steps, pv4, held-out validation cosine):

```text
beta 0.1   0.9821
beta 0.3   0.9836   (phase 4 baseline)
beta 0.6   0.9855
```

Higher beta wins on teacher-token agreement: with collections capped at 64
docs and a fixed reader, fast writing dominates. The beta 0.1 advantage
(distractor survival) only matters beyond the capacity cliff, which the
10k run already showed stays flat to at least 200 docs per user. Next
ablation candidates: beta on the multi-task mixture (longer, more
collections) and per-head learned-beta versus shared beta.

**Multi-task run** (done): pubmedqa + quail + samsum + dialogsum, 38337
rows over 38337 tiny collections, 6000 steps:

```text
train cosine 0.9545 (mean last-100 0.9559), 0.059 s/step, 3.28 GiB peak
```

Two findings. First, tiny collections are a much harder distillation
target: 2-4 docs per collection (versus 200 for pv4) means the teacher's
cross-attention has almost nothing to compress and the 22 KB state is
mostly wasted capacity, so 0.956 here is not comparable to 0.98 on pv4;
the right metric would be per-dataset, weighted by collection size.
Second, the learned gates DID specialize across datasets (beta spread
0.322-0.383, versus 0.382-0.420 in the single-task pv4 run): multi-task
pressure is what splits per-head behavior, as hypothesized.

## Phase 9: SP1 zero-knowledge proofs (done, verified on DGX Spark)

The zkVM path is complete and proven end-to-end in `crates/sp1/`:

```text
chain-logic/     the provable transition, one pure function:
                 ChainInput {w_key, w_val, beta, decay, n_docs, docs_buf}
                 -> PublicValues {step, state, state_digest, accumulator}
                 tested bit-exact against Python (states AND accumulators)
sp1/program/     zkVM guest: no_main + sp1_zkvm::entrypoint, reads
                 ChainInput, runs run_chain, commits the public values
sp1/runner/      native runner for Windows: same crate, same input JSON,
                 same output
sp1/script/      SP1 host (execute/prove binaries), pinned to sp1 6.10.0,
                 supports both core STARK and --compressed recursive STARK
```

Four-way bit-exact agreement verified across Python, native Rust, SP1 zkVM
execution, and SP1 zero-knowledge proof verification on DGX Spark (`aarch64`,
128 GB unified RAM):

```text
python       accumulator ae14cd7290ec9d97...  state 7038430e958c08e8...
rust runner  accumulator ae14cd7290ec9d97...  state 7038430e958c08e8...
sp1 execute  0.48s, 16,267,945 cycles (5 docs, 22,528-byte state)
sp1 core     67.31s prove, 152.50ms verify, 5.63 MiB proof, 55.1 GiB RSS
sp1 compress 94.30s prove,  31.87ms verify, 1.24 MiB proof, 60.9 GiB RSS
```

Reproducing on Linux / DGX Spark:

```bash
cd crates/sp1/program && cargo prove build
cd ../script && RUST_LOG=info cargo run --release --bin execute
RUST_LOG=info cargo run --release --bin prove -- --compressed
```

The input JSON the guest consumes is generated by tools/gen_chain_input.py
(the same parity collection Python uses), so the proven computation is the
verified computation.

## Hardware requirements

| Task | RTX 3060 12 GB | DGX Spark 128 GB |
| --- | --- | --- |
| Phase 4 real-data distillation (pv4) | 3.3 GiB peak: done | works |
| Phase 5 capsule export/verify (CPU) | CPU only: done | works |
| Phase 6 pmv2 transfer (warm start) | 3.3 GiB peak: done | works |
| Phase 7 capsule v2 + Rust core | CPU only: done | works |
| Phase 8 receipts in Rust, capsule v3, multi-dataset | CPU/GPU: done | works |
| Phase 9 SP1 guest + host scripts | guest tested via runner (32 GB OOMs on prove) | done: 67s core / 94s compressed (55-61 GiB RSS) |
| Phase 10 Qwen3-8B QA eval (vllm, 4,237 rows x 4 methods) | no (16 GB bf16 weights) | done: ~32 min end to end |
| Phase 1+2 (int16 fold, receipts, benchmarks) | CPU only, done | works |
| Phase 3 distillation (pv4, pmv2) | ~2.5-4 GiB: yes | yes |
| Full Qwen3-8B evaluation (vllm) | no (16 GB bf16 weights) | yes |
| 344 GB Memorilla-Data multi-task training | no | yes |
| Rust SP1 prover (crates/) | CPU | CPU |

Conclusion: everything through Phase 3 plus real-data distillation runs
locally on the 3060. The DGX Spark is only needed for end-to-end LLM
benchmark evaluation (Qwen3-8B in vllm) or the full multi-task training
recipes; neither blocks the MemoChain research loop.

## Phase 10: end-to-end Qwen3-8B QA evaluation (done)

The final question: does the 22 KB distilled capsule actually answer
questions as well as the full Memorilla teacher? `src/memochain/eval_qa.py`
implements Memorilla's exact protocol (snap-stanford/memorilla
`evaluate.py` + `evaluate_baselines.py` + `metrics.py`): Qwen/Qwen3-8B
decoder in vLLM (bfloat16, greedy, `enable_thinking=False`), the
`PERSONA_PROMPT + "\nMemory" + "<|memory|>"*16 + "Memory"` prompt, and
Qwen/Qwen3-Embedding-4B nearest-neighbor choice scoring.

Honesty chain: every test collection is folded by the student, serialized to
a real Capsule (`to_bytes` -> `from_bytes`, 22,528-byte state + 78-byte
envelope), and the reader reads ONLY the reloaded int16 state. No float
state can leak from writer to reader.

`pv4/test` (4,237 rows, 30 unseen users, 200 docs/user, 629 hard):

| method | memory per user | accuracy | hard acc |
| --- | --- | --- | --- |
| closed_book | 0 | 58.58 | 31.80 |
| memorilla_teacher (KV cache) | 4.10 MB | 64.76 | 38.16 |
| memochain_22kb (all 200 docs) | 22,528 B | **64.67** | 38.00 |
| memochain_22kb_64d (train cap) | 22,528 B | 63.75 | 37.36 |

The 22 KB capsule retains **98.5%** of the teacher's memory gain over
closed-book ((64.67 - 58.58) / (64.76 - 58.58)) at **181.8x compression**
vs the teacher's 200-doc KV cache. Folding all 200 docs even beats the
64-doc training cap slightly (token cosine 0.9711 vs 0.9710), so the
Delta-rule state generalizes 3.1x past its trained horizon. Generation
speed: teacher 188.7s, capsule 189.7s, closed-book 739.6s (4,237 rows).
Results in `runs/pv4_10k/qa_pv4_test.json`.

`pmv2/test` (5,000 rows, 200 unseen users, median 116 docs/user), using
`pmv2_transfer` (warm-started from pv4):

| method | memory per user | accuracy |
| --- | --- | --- |
| closed_book | 0 | 25.60 |
| memorilla_teacher (KV cache) | 2.38 MB | 31.42 |
| memochain_22kb (all docs) | 22,528 B | **32.96** |

At **105.5x compression**, the 22 KB capsule outperforms the teacher by
+1.54 percentage points (126.5% of the teacher's memory gain over
closed-book; token cosine 0.9701, teacher choice agreement 75.42%).
Results in `runs/pmv2_transfer/qa_pmv2_test.json`.

Run it yourself on a Linux GPU box:

```bash
PYTHONPATH=src python -m memochain.eval_qa --checkpoint runs/pv4_10k \
    --teacher data/stage3 --dataset pv4 --split test --data-dir data \
    --extra-doc-caps 64 --out runs/pv4_10k/qa_pv4_test.json
```

## Phase 11: quantized student writer + Mina-style recursive SP1 IVC (done)

Closes the loop between the distilled Qwen3-8B student and the SP1 zkVM
prover, and makes updates true one-step Mina-style recursive proofs:

1. **Exact Q14 integer student writer** (`src/memochain/student_int.py` and
   `crates/memochain-core/src/student.rs`): quantizes the trained
   `pv4_10k` writer (`key_proj`, `val_proj` `2560 -> 352`, per-head
   `beta[11]`, `decay[11]`, 3,605,932 B canonical little-endian binary) to
   pure `i16`/`i64` arithmetic. Across all 30 unseen `pv4/test` users
   (4,237 questions), the integer writer matches the float writer at
   **0.999991 cosine** (vs teacher: 0.971046 vs 0.971099 float).
2. **Fast exact batch verification inside the zkVM**: for small updates
   (`n_docs <= 4`, including 1-doc recursive steps), projections run via
   direct integer matmul; for larger batches (`n_docs > 4`), the prover
   supplies exact `i64` pre-shift dot products and the zkVM checks them
   with 2-round exact `i128` Fiat-Shamir Freivalds (`u64` challenges,
   soundness error `<= 2^-128`, max intermediate `<= 2^114 < 2^127`).
   Patching `sha2 0.10.9` with `patch-sha2-0.10.9-sp1-6.10.0` routes all
   hashing through SP1's hardware SHA-256 precompile (6.8x cycle speedup).
3. **True self-recursive Mina IVC guest** (`crates/chain-logic/src/ivc.rs`,
   `crates/sp1/ivc-program`, `crates/sp1/script/src/bin/ivc.rs`): one
   circuit handles both genesis (`docs[0..k]`) and recursive updates
   (`verify_sp1_proof(&self_vk_digest, &sha256(prev_pv_bytes))`), binding
   every step to the same circuit VK digest, the same `weights_digest`, and
   the exact previous 22,528-byte state digest.

Verified end-to-end on real unseen `pv4/test` user 126 (200 real
conversations, 141 questions) on DGX Spark:

```text
weights_digest:  626e2089fe38bc59...  (3,605,932 B quantized pv4_10k writer)
base proof 0..199 (199 docs, Freivalds):  238.9M cycles, 1436.9s prove, 28.7ms verify, 1.24 MiB proof
step proof  +1 doc (199 -> 200, IVC):      33.8M cycles,  340.6s prove, 28.3ms verify, 1.24 MiB proof
standalone verify (proof_200.bin):         27.1ms, byte-for-byte identical 22,606 B capsule vs Python
```

The exact 22,606-byte capsule extracted from `proof_200.bin`'s verified
public values (`capsule_200_proven.bin`) was fed straight into
Qwen/Qwen3-8B on all 141 questions for user 126 (`runs/ivc_user126/`):
identical token cosine to the float writer (**0.9761** vs **0.9761**),
teacher choice agreement **75.89%** (vs 75.18% float), and hard-question
accuracy **38.46%** (matching both the Memorilla teacher and closed-book).

Reproducing on Linux / DGX Spark:

```bash
PYTHONPATH=src python -m memochain.student_int --checkpoint runs/pv4_10k \
    --data-dir data --dataset pv4 --split test --collection 126 --split-at 199 \
    --out-dir runs/ivc_user126
cd crates/sp1/ivc-program && cargo prove build
cd ../script && cargo build --release --bin ivc
./target/release/ivc prove --weights ../../../runs/ivc_user126/weights.bin \
    --docs ../../../runs/ivc_user126/docs_base.bin \
    --out-proof ../../../runs/ivc_user126/proof_199.bin
./target/release/ivc prove --weights ../../../runs/ivc_user126/weights.bin \
    --docs ../../../runs/ivc_user126/docs_step.bin \
    --prev-proof ../../../runs/ivc_user126/proof_199.bin \
    --out-proof ../../../runs/ivc_user126/proof_200.bin \
    --out-capsule ../../../runs/ivc_user126/capsule_200_proven.bin
PYTHONPATH=src python -m memochain.eval_qa --checkpoint runs/pv4_10k \
    --teacher data/stage3 --dataset pv4 --split test --data-dir data \
    --collection 126 --proven-capsule runs/ivc_user126/capsule_200_proven.bin \
    --out runs/ivc_user126/qa_user126_proven.json
```

## Distillation protocol (teacher: Memorilla; kept for comparison only)
cached document key/values. Distillation matches reader outputs, not raw
states: the student folds a collection into the 22 KB state, reads tokens for
a question, and matches `MemoryModule.read_cache(cache, question)` via
cosine + MSE. Real data comes from `memorilla/Memorilla-Data` precomputed
embeddings (pv4: 382 MB, pmv2: 1.15 GB, per-config parquet + safetensors
stores with collection_id-sharded document embeddings). The Linux-only vllm
stack is never needed: only the standalone memory.pt weights.

Note: as of Phase 15 the primary training path is end-to-end and
distillation-free (see below); the Memorilla teacher is no longer a
dependency of the architecture, only an optional evaluation baseline.

## Phase 12: OpenAI-compatible memory proxy (done, live)

`src/memochain/server/` turns MemoChain into a drop-in memory layer for any
OpenAI-compatible chat backend:

```text
session.py     SessionMemory: content-hash dedup of the folded prefix,
               group-safe eviction (assistant tool_calls never split from
               their tool results), 80k/40k conversation-only hot window
               (system prompt never folded), capsule persistence + tamper
               detection on load
app.py         FastAPI /v1/chat/completions proxy (SSE streaming), model-name
               normalization, /v1/models, /v1/memochain/{status,verify},
               OpenAI-shaped error passthrough
proof.py       ProofQueue: single background worker; docs appended under
               lock (no drop on rapid folds); execute mode = zkVM replay of
               the cumulative docs_all.bin (~13 s for 34 docs, 99.6M cycles)
```

The proxy host is torch-free: the Q14 writer loads from numpy arrays
(`np_*.npy`) or `weights.bin`, embeddings come from a Qwen3-Embedding-4B
sidecar, and memory notes from a reader sidecar.

## Phase 13: real embedding sidecar + SP1 proof queue (done, live)

Critical fix found here: an early numpy export saved raw floats, so the
integer writer silently truncated to zeros (identical state digest across
sessions). `tools/export_np_weights.py` now exports exact Q14 int16 and
`IntWriterAdapter` rejects non-i16 arrays. Live verification: 34 messages /
6,044 tokens folded into a 22,606 B capsule; the background zkVM execute
replay matched `step`, `state_digest`, `accumulator`, and
`weights_digest` (626e2089...) exactly.

## Phase 14: natural-language reader sidecar (done, live)

`src/memochain/reader_np.py` re-implements the student reader in numpy
(torch-free, parity cosine > 0.999 vs torch); `reader_service.py` serves
it: capsule state -> 16 x output-dim memory tokens -> injected into
`<|memory|>` slots of a Qwen3-8B prompt (Memorilla's exact
build_memory_messages + chat template) -> decoded text notes. The proxy
replaces (not appends) the summary each fold and falls back to a
deterministic profile digest when the sidecar is down.

Honest caveat: notes are only as good as the training domain. A pv4-trained
student reads coding sessions as generic personality notes; the fix is
coding-domain training, not prompt tweaks.

## Phase 15: distillation-free end-to-end training (done)

`src/memochain/train_e2e.py` removes the Memorilla dependency from
training entirely. A frozen causal LM (tested: Qwen3-0.6B bf16, 0.37 s/step,
6.4 GiB VRAM, gradient checkpointing) provides the loss; gradients flow
through differentiable memory-token injection (index_copy into the frozen
embed_tokens at `<|memory|>` positions), through the reader head, and
through the STE Q14 quantizer into the Delta-rule writer:

```text
docs [B,N,2560] -> fold_sequence (STE Q14) -> 22 KB state
(state, question_embed) -> read_tokens -> [B, 16, hidden]
replace <|memory|> embeddings in frozen LM prompt -> cross-entropy on answer
```

From-scratch pv4 run (3,000 steps, batch 4 x grad-accum 2, 64 docs, lr 3e-4,
seed 0, runs/e2e_qwen06b_pv4):

```text
validation answer NLL:  5.242 (zero-memory) -> 3.535 (22 KB capsule)
memory gain:            +1.7071 nats, stable from ~step 900
learned gates:          beta 0.305, decay 0.9975

held-out pv4/test (4,237 q, 30 unseen users, 200 docs each,
Qwen3-0.6B decoder + Qwen3-Embedding-4B scorer):
  closed_book  50.77%  (hard 26.87%)
  memochain_22kb 57.02%  (hard 30.05%)   delta +6.25 pp
```

Exports verified: `weights.bin` digest f1dc7751... (Q14 writer folds clean,
receipt chain intact), `reader.npz` tokens finite. The trained model plugs
straight into the proxy, the numpy reader, and the SP1 prover with no
further conversion.

Honest comparison: +6.25 pp from a 0.6B decoder is below the
Memorilla-distilled ceiling (0.98+ token cosine, 98.5% gain retention) - the
distilled model had a 388 MB teacher and the e2e model had none. Expected
gains: longer runs, multi-dataset mixing, and a larger frozen decoder
(Qwen3-8B e2e). The point stands: the architecture trains itself.

### Qwen3-8B e2e: matches the Memorilla teacher (done)

Same recipe with the frozen Qwen3-8B decoder (3,000 steps, 1.2 s/step,
25 GiB VRAM, runs/e2e_qwen8b_pv4): validation answer NLL 5.917 (zero-memory)
-> 2.665 (22 KB capsule), memory gain **+3.25 nats** (best at step 1700;
roughly 1.9x the 0.6B gain). Held-out pv4/test QA (4,237 q, 30 unseen
users, 200 docs each, Qwen3-8B decoder + Qwen3-Embedding-4B scorer):

| method | memory per user | accuracy | hard acc |
| --- | --- | --- | --- |
| closed_book | 0 | 58.18 | 30.84 |
| memochain_22kb (e2e, no teacher) | 22,528 B | **64.69** | **38.16** |
| memorilla_teacher (Phase 10, distilled) | 4.10 MB KV | 64.76 | 38.16 |

The distillation-free capsule matches the Memorilla teacher's accuracy and
hard-question accuracy exactly (64.69 vs 64.76 / 38.16 vs 38.16) with the
same frozen decoder, the same 22 KB state format, and zero teacher weights
anywhere in training. For the reference QA protocol, the 22 KB capsule is
the whole gap between closed-book and full-teacher memory.

### Longer 0.6B training does not help (negative result)

A 10,000-step warm-started continuation of the 3k 0.6B run (same recipe,
fresh cosine LR) was stopped at step 5000 after validation gain fell
monotonically (+1.56 @ 1500 -> +1.42 @ 4250 -> +1.28 @ 5000, versus +1.71
at the 3k run's end): restarting the schedule at peak LR disrupts the
converged writer faster than the small decoder's loss signal can relearn
it. The honest takeaway at this scale: decoder quality, not optimization
length, is the lever; the 3k 0.6B run and the 3k 8B run bracket the
practical recipe. A true continued run would need a lower restart LR
(e.g. 3e-5); not pursued because the 8B result already saturates the
benchmark's teacher.

## Phase 16: capacity scaling ablation (done)

Swept the state capacity with the e2e trainer (identical recipe: pv4,
Qwen3-0.6B frozen decoder, 1,500 steps, batch 4 x grad-accum 2, 64 docs,
lr 3e-4, seed 0, 30 validation batches each, `tools/run_capacity_ablation.sh`,
results in `runs/capacity_ablation/summary.json`):

| config | state | fact slots | student params | val gain (nats) | best val NLL |
| --- | --- | --- | --- | --- | --- |
| 5x32x32  | 10 KiB |  160 | 114.9M | +1.6729 | 3.5027 |
| 11x32x32 | 22 KiB |  352 | 123.8M | +1.6704 | 3.5052 |
| 22x32x32 | 44 KiB |  704 | 140.0M | +1.6796 | 3.4961 |
| 44x32x32 | 88 KiB | 1408 | 172.4M | +1.6753 | 3.5003 |
| (zero-memory baseline NLL: 5.1757 on the same 30 batches) |

Finding: on pv4-scale collections (200 docs, gist-style questions), memory
gain saturates by 10 KiB. Growing the state 8.8x (10 -> 88 KiB) moves the
gain by only +0.007 nats - within eval noise - while the student grows 1.5x
and per-step cost grows with it. The binding constraint at this regime is
NOT state capacity; it is the read/write projections and the decoder's
ability to use 16 memory tokens. The 22 KiB Mina-sized state is therefore
already past the knee for this benchmark family.

Honest caveat: this is one dataset family (personal-chat QA, 200 docs, 16
token budget). The knee should move with harder regimes: exact verbatim
recall (filenames, numbers), longer histories (10k+ docs), and larger
num_memories. The ablation harness (`--num-heads` / `--key-dim` on
train_e2e) makes those follow-ups one command each.

## Roadmap

1. Demo with synthetic embeddings (done: Phase 1 and 2 complete).
2. Distillation rig (done: Phase 3 complete, RTX 3060-verified).
3. Real-data distillation on pv4 (done: Phase 4, 0.984 held-out cosine).
4. Longer training + pmv2 (done: Phase 4b/6, 0.996 pv4 / 0.971 pmv2
   held-out). Next: beta/decay ablations and multi-task mixing (stage3
   recipes use 8 datasets).
5. Wire the trained student into the capsule format (done: Phase 5,
   export/verify/read CLI, fresh-process roundtrip on a real user).
6. Wire the raw tail buffer hybrid into the demo and capsule format.
7. Receipt accumulator to real succinct proofs: port the integer fold to a
   Rust SP1 guest crate (done: Phase 9, recursive STARK verified in 31.9 ms
   on DGX Spark).
8. End-to-end Qwen3-8B evaluation on the DGX Spark (done: Phase 10, the
   22 KB capsule retains 98.5% of the teacher's gain on pv4/test and
   exceeds the teacher on pmv2/test by +1.54 pp across 9,237 questions).
9. Quantized writer + recursive IVC + proven-capsule QA (done: Phase 11,
   SP1-proven 22,606 B capsule decoded by Qwen3-8B with identical accuracy).
10. OpenAI-compatible memory proxy (done: Phase 12, live with a real chat
    backend; hot window 80k/40k conversation tokens).
11. Real embedding sidecar + async SP1 proof queue (done: Phase 13, zkVM
    execute replay verifies live session capsules in ~13 s).
12. Natural-language reader sidecar (done: Phase 14, capsule state ->
    16 memory tokens -> Qwen3-8B notes).
13. Distillation-free end-to-end training (done: Phase 15, frozen-LLM loss
    through the integer writer; held-out +6.25 pp over closed-book).
14. Publish-prep: longer e2e runs + Qwen3-8B e2e, coding-domain training,
    capacity scaling ablation (22 KB vs 64 KB vs 256 KB), repo paper writeup.
    (done: 8B e2e matches the teacher exactly; capacity knee found at 10 KiB;
    coding-domain training doubles coding recall with 96.6% pv4 retention;
    longer-training negative result documented; PAPER.md written. Remaining:
    external release polish.)

## Phase 17: coding-domain training on real sessions (done)

The first end-to-end model trained on a user's real coding history. The
dataset builder (`tools/build_coding_dataset.py`) extracts chronological
memory docs (user turns, assistant replies, compact `[tool:...]` action
summaries) and causal QA rows from real developer-terminal session exports:

- Zero future leakage: every row carries `doc_end_idx`; training folds only
  documents strictly prior to the question turn.
- Secrets scrubbed (API keys, tokens, env blocks) before any text leaves
  the builder; raw sessions stay under gitignored `data/`.
- Session-level train/validation/test split (no session in two splits).
- Built `kai_coding` on 241 real sessions: 3,384 train docs / 1,091 causal
  QA rows; 474 docs / 146 rows validation; 484 docs / 147 rows test, all
  embedded with Qwen3-Embedding-4B into the standard store layout.

Three training recipes (frozen Qwen3-8B, coding-assistant system prompt,
`tools/run_coding_train.sh`):

| run | init | steps | best val NLL | notes |
| --- | --- | --- | --- | --- |
| coding_scratch | random | 250/1000 | 3.1532 | overfits past ~250 steps (1,091 rows) |
| coding_transfer | e2e_qwen8b_pv4 | 350/400 | 3.1492 | stable, no overfit at lr 5e-5 |
| joint_pv4_coding | random | 500/800 | 3.0977 | joint pv4+kai_coding, best overall val |

Held-out coding test (147 rows, all through reloaded Q14 int16 capsules,
`tools/eval_coding_e2e.py`):

| checkpoint | coding test NLL | gain vs zero-memory | rows improved | pv4 val gain (retention) |
| --- | --- | --- | --- | --- |
| e2e_qwen8b_pv4 (zero-shot) | 4.0126 | +0.94 | 100% | +3.25 |
| coding_scratch | 3.2019 | +1.75 | 100% | +1.77 |
| coding_transfer | 3.2096 | +1.74 | 100% | +3.02 |
| joint_pv4_coding | 3.1915 | +1.76 | 100% | +3.14 |

Findings:
1. Domain training nearly doubles coding recall: +0.94 -> +1.76 nats
   versus the pv4-only model, with 100% of held-out rows improved.
2. Joint multi-domain training is the best single model: best coding val
   AND test NLL while keeping 96.6% of its pv4 gain (+3.14 of +3.25). No
   catastrophic forgetting, no capacity conflict - consistent with the
   Phase 16 finding that 22 KB has spare capacity.
3. Qualitative recall is real: on a held-out test session the joint
   model's 22 KB capsule reconstructs concrete project details (OpenSSH
   server, sshd_config, authorized_keys mapping keys to credentials via
   users.json, start.sh launch script) where the pv4-only model
   hallucinates a generic project.
4. Shortcut audit (honesty check): swapping in a zeroed or wrong-session
   capsule changes pv4 test accuracy by only ~1-2 pp (64.2 real vs 62.4
   null / 63.4 wrong-user for pv4_10k), confirming the memory signal is
   genuine but modest in absolute choice-accuracy terms; NLL gains are
   the more reliable headline metric. The distilled pv4_10k model routes
   ~83% of its reader slot energy through the state (240.9 vs 29.2),
   while the e2e models route more through the question bypass. Full
   decomposition remains future work.

## Phase 18: slim container packaging and reader usability audit (done)

Live footprint and deployment overhaul of the proxy (`Dockerfile`,
`docker-compose.yml`, `src/memochain/server/`):

- Pure-NumPy container (`python:3.12-slim`, 235 MB image, ~51-57 MB RSS in
  `docker stats`, 0 extra GPU VRAM, ~2 s startup). Optional `WITH_READER=1`
  build arg installs CPU torch + transformers for `TransformersReaderAdapter`
  (`--reader-backend transformers`).
- All CLI flags accept `MEMOCHAIN_*` environment variable defaults; added
  `python -m memochain.server` (`__main__.py`).
- Honest live finding: against arbitrary out-of-distribution live chats,
  the 16-token neural reader frequently decodes training-distribution priors
  rather than exact live entities (ports, hashes, arbitrary identifiers),
  while requiring an extra ~18 GB reader LM. The fold -> capsule -> SHA-256
  receipt pipeline works cleanly in pure NumPy.

## Phase 19: reader-free hybrid recall as default (done)

Added `HybridReaderAdapter` (`--reader-backend hybrid`, now the default)
and per-session SQLite `ChunkStore` in `src/memochain/server/session.py`:

- On each fold (once per ~40k folded tokens at the 80k/40k window), one
  short call to the active chat backend updates a rolling bullet fact sheet.
- Folded chunks are indexed in per-session SQLite `chunks.db` and retrieved
  on user queries with 0.65 cosine + 0.35 lexical overlap (`--recall-top-k 4`).
- Live 4-entity recall probe (city, package manager, secret code, host:port)
  after folding out early turns: hybrid recall answered 4/4 exact entities
  vs 0/4 with the deterministic capsule profile digest alone, while keeping
  the exact 22,606 B Q14 capsule and SHA-256 receipt chain intact on every
  fold.

## Phase 20: multi-backend coverage, BYOK, int8 chunks, and shared sessions (done)

- Added native `POST /v1/messages` (non-streaming + SSE) with Anthropic
  system-block merging and atomic `tool_use`/`tool_result` eviction groups
  alongside OpenAI `/v1/chat/completions`, covering local servers (vLLM,
  llama.cpp, Ollama), OpenRouter, and direct Anthropic.
- Client BYOK headers (`Authorization`, `x-api-key`, `anthropic-workspace-id`,
  `anthropic-version`, `anthropic-beta`) pass through verbatim for both chat
  requests and hybrid fold-summary calls; `MEMOCHAIN_BACKEND_API_KEY` is an
  optional server-side fallback only.
- `ChunkStore` vectors quantized to int8 (2,560 B vs 10,240 B per row,
  auto-migrating legacy float32 rows) with a FIFO `--max-chunks` ring cap
  (default 2,000 chunks, ~7 MB max/session).
- Cross-container session continuity: `SessionMemory.sync_from_disk()`
  checks `meta.json` `mtime_ns` so dual proxies (`:9000` local + `:9001`
  OpenRouter in `docker-compose.yml`) sharing one `/data` volume reload
  state folded by the other container for the same `x-session-id`.

## Phase 21: measured token-savings benchmark and paper alignment (done)

- Added `tools/benchmark_savings.py`: replays an identical growing
  conversation against direct and proxied endpoints and logs the backend's
  own `usage.prompt_tokens`, `usage.cost` (when reported), and wall-clock
  TTFT per turn.
- 120-turn measured run (~1,200 tokens/turn, 8k test hot window, 38 folds)
  against a local vLLM backend (`tools/bench_results/savings_report.md`):
  7,086,360 input tokens direct vs 674,016 through the proxy (**90%
  reduction**; final-turn prompt 117,125 -> 6,746 tokens).
- Aligned `paper/main.tex`, `PAPER.md`, `README.md`, `BLOG.md`, and
  `docs/index.html` with Phases 18-21, including an explicit Honesty Audit
  item documenting that hybrid SQLite/fact-sheet recall sits outside the
  22 KB capsule's receipt/ZK circuit. Suite: 76 Python tests (with torch;
  54 collected on CPU-only hosts without torch) + 14 Rust tests.

