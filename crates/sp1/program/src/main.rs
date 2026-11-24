//! MemoChain SP1 guest: prove a memory write.
//!
//! Input (serde over SP1 stdin): ChainInput { fold weights, documents }.
//! Execution: the exact integer Delta-rule fold with the receipt chain from
//! chain-logic (the same code tested bit-exact against Python).
//! Output (public values): step count, final 22,528-byte state, state
//! digest, and the receipt-chain accumulator - everything a light client
//! needs to accept the write without re-reading history.

#![no_main]
sp1_zkvm::entrypoint!(main);

use chain_logic::{run_chain, ChainInput};
use serde::{Deserialize, Serialize};

/// Output committed to public values.
#[derive(Serialize, Deserialize)]
struct GuestOut {
    step: u64,
    state: Vec<i16>,
    state_digest: String,
    accumulator: String,
}

pub fn main() {
    let input: ChainInput = sp1_zkvm::io::read();
    let out = run_chain(&input).expect("chain failed inside zkVM");
    assert_eq!(out.state.len(), chain_logic::STATE_PARAMS);

    sp1_zkvm::io::commit(&out.step);
    sp1_zkvm::io::commit(&out.state);
    sp1_zkvm::io::commit(&out.state_digest);
    sp1_zkvm::io::commit(&out.accumulator);
    let _ = GuestOut { step: out.step, state: out.state, state_digest: out.state_digest, accumulator: out.accumulator };
}
