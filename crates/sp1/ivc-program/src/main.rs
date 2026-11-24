//! MemoChain Mina-style recursive IVC guest (SP1 6.10.0).
//!
//! Proves the EXACT quantized MemoChainStudent writer (2560 -> 11x32x32 Q14 int16)
//! with true self-recursion:
//!   - Genesis (`prev_pv_bytes.is_empty()`): folds docs[0..k] from zero state.
//!   - Recursive update (`!prev_pv_bytes.is_empty()`): verifies the previous
//!     compressed SP1 proof inside the zkVM (`verify_sp1_proof`), confirms the
//!     previous proof came from THIS EXACT circuit VK and used the SAME writer
//!     weights, then folds ONLY the new document(s) (e.g. +1 conversation).
//!
//! Public values committed: exact `bincode::serialize(&IvcPublicValues)` bytes.

#![no_main]
sp1_zkvm::entrypoint!(main);

use chain_logic::ivc::{run_ivc_step, IvcPublicValues};
use sha2::{Digest, Sha256};

pub fn main() {
    let self_vk_digest: [u32; 8] = sp1_zkvm::io::read();
    let weights_raw: Vec<u8> = sp1_zkvm::io::read_vec();
    let docs_raw: Vec<u8> = sp1_zkvm::io::read_vec();
    let proj_raw: Vec<u8> = sp1_zkvm::io::read_vec();
    let prev_pv_bytes: Vec<u8> = sp1_zkvm::io::read_vec();

    let prev_pv: Option<IvcPublicValues> = if prev_pv_bytes.is_empty() {
        None
    } else {
        let pv_digest: [u8; 32] = Sha256::digest(&prev_pv_bytes).into();
        sp1_zkvm::lib::verify::verify_sp1_proof(&self_vk_digest, &pv_digest);
        let decoded: IvcPublicValues =
            bincode::deserialize(&prev_pv_bytes).expect("failed to decode previous IvcPublicValues");
        Some(decoded)
    };

    let proj_opt = if proj_raw.is_empty() {
        None
    } else {
        Some(proj_raw.as_slice())
    };

    let out = run_ivc_step(
        self_vk_digest,
        &weights_raw,
        &docs_raw,
        proj_opt,
        prev_pv.as_ref(),
    )
    .expect("IVC step failed inside zkVM");

    let encoded = bincode::serialize(&out).expect("failed to encode IvcPublicValues");
    sp1_zkvm::io::commit_slice(&encoded);
}
