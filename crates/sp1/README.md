# MemoChain SP1 workspace

Proves that a 22 KB memory write was performed honestly: the guest folds a
document collection through the integer Delta-rule with the sha256 receipt
chain and commits `(step, state, state_digest, accumulator)` as public
values. A light client accepts the write by checking the accumulator against
its stored head - no history re-read.

```text
crates/
  chain-logic/   THE provable logic (shared): fold + receipts + IVC step.
                 Tested bit-exact against Python (states + accumulators).
  memochain-core/  the integer fold + sha256 receipts (no deps except sha2)
  sp1/
    program/     the zkVM guest (no_main + sp1_zkvm entrypoint), calls
                 chain_logic::run_chain and commits PublicValues
    ivc-program/ the recursive IVC guest: genesis fold or +batch step on
                 top of a verified previous proof (verify_sp1_proof)
    runner/      native Windows runner: same chain-logic, same inputs,
                 same outputs - the pre-proof development loop
    script/      the SP1 host (execute/prove + ivc execute/prove/verify),
                 Linux/macOS only
```

## Status

- chain-logic: cargo tests, bit-exact with Python (states + accumulators).
- runner: builds and runs on Windows; verified 3-way agreement
  (Python == memochain-core == runner) on the parity input:
  accumulator `ae14cd72...`, state digest `7038430e...`.
- ivc-program/script: verified end-to-end on DGX Spark (Phase 11, see
  DESIGN.md): 200-doc base proof 238.9M cycles / 1.24 MiB compressed STARK,
  +1-doc recursive step 33.8M cycles, verify ~28 ms; the proven capsule was
  byte-identical to the Python fold.
- program/script: version-pinned to sp1 6.10.0, but SP1 does not run
  natively on Windows (Linux/macOS only). On a Linux machine or
  the DGX Spark:

```bash
# one-time: install sp1up toolchain (see docs.succinct.xyz install page)
cargo prove build            # inside crates/sp1/program (or via build.rs)
cd crates/sp1/script
RUST_LOG=info cargo run --release --bin execute   # fast, no proof
RUST_LOG=info cargo run --release --bin prove     # generates + verifies proof
SP1_PROVER=network cargo run --release --bin prove  # hosted prover
```

## Input/output format

ChainInput JSON (tools/gen_chain_input.py generates it):

```json
{
  "w_key": [int16...],       // 11*32*32 fold weights
  "w_val": [int16...],       // 11*32*32 fold weights
  "beta": 12288,             // Q14
  "decay": 16368,            // Q14
  "docs": [[704 int16]...]   // the collection
}
```

PublicValues (committed by the guest, printed by the runner):

```json
{
  "step": 5,
  "state": [11264 int16...],
  "state_digest": "7038430e...",
  "accumulator": "ae14cd72..."
}
```

## Why this design

The guest contains no crypto primitives of its own beyond sha2; all logic
is integer multiply-add-shift over i64 plus one integer sqrt, so cycles are
dominated by the 11 small matvecs per document rather than float emulation.
Proving one write proves the state and the receipt chain; the light client
property from DESIGN.md carries over unchanged.
