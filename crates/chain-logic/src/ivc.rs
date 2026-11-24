//! Incremental verifiable computation (Mina-style IVC) over the quantized student writer.
//!
//! One guest program handles BOTH:
//!   - Genesis (`prev_pv == None`): folds docs[0..k] from zero state into a 22,528-byte state
//!   - Recursive step (`prev_pv == Some(prev)`): verifies the prior compressed SP1 proof
//!     inside the zkVM (`verify_sp1_proof`), asserts identical circuit VK digest, identical
//!     trained writer weights digest, and matching 22 KB state digest, then folds ONLY the
//!     new document(s) (e.g. +1 conversation) to advance `step`, `state`, and `accumulator`.

use memochain_core::receipt::{state_digest, ChainHead};
use memochain_core::student::StudentWriter;
use memochain_core::{zero_state, STATE_BYTES, STATE_PARAMS};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Canonical public values committed by `crates/sp1/ivc-program` (bincode-serialized).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IvcPublicValues {
    /// Self verifying-key digest [u32; 8] — binds every recursive step to this exact guest.
    pub vk_digest: [u32; 8],
    /// SHA-256 hex digest of the canonical quantized StudentWriter weights (3,605,932 B).
    pub weights_digest: String,
    /// Total documents folded from genesis (0 -> step).
    pub step: u64,
    /// SHA-256 hex digest of the final 22,528-byte int16 state.
    pub state_digest: String,
    /// Rolling SHA-256 receipt-chain accumulator over all documents 1..=step.
    pub accumulator: String,
    /// Final 22,528-byte Q14 int16 memory state (11,264 i16 values, head-major).
    pub state: Vec<i16>,
}

/// Pure IVC state transition shared by native Rust tests and the SP1 zkVM guest.
///
/// When `prev_pv` is `Some(prev)`, the caller (the zkVM guest) MUST have already
/// verified the SP1 proof committing to `prev` with `vk_digest == self_vk_digest`.
pub fn run_ivc_step(
    self_vk_digest: [u32; 8],
    weights_raw: &[u8],
    docs_raw: &[u8],
    proj_raw: Option<&[u8]>,
    prev_pv: Option<&IvcPublicValues>,
) -> Result<IvcPublicValues, String> {
    let writer = StudentWriter::from_bytes(weights_raw)?;
    let weights_digest = writer.weights_digest();

    let (start_state, start_head) = match prev_pv {
        None => (zero_state(), None),
        Some(prev) => {
            if prev.vk_digest != self_vk_digest {
                return Err("recursive VK digest mismatch (cannot mix different circuits)".into());
            }
            if prev.weights_digest != weights_digest {
                return Err("recursive weights_digest mismatch (cannot swap writer weights mid-chain)".into());
            }
            if prev.state.len() != STATE_PARAMS {
                return Err(format!("prev state len {} != {}", prev.state.len(), STATE_PARAMS));
            }
            if state_digest(&prev.state) != prev.state_digest {
                return Err("prev state does not match prev state_digest".into());
            }
            (
                prev.state.clone(),
                Some(ChainHead {
                    step: prev.step,
                    accumulator: prev.accumulator.clone(),
                    state_digest: prev.state_digest.clone(),
                }),
            )
        }
    };

    let (new_state, head) = writer.fold_batch(
        &start_state,
        start_head.as_ref(),
        docs_raw,
        proj_raw,
        &weights_digest,
    )?;

    Ok(IvcPublicValues {
        vk_digest: self_vk_digest,
        weights_digest,
        step: head.step,
        state_digest: head.state_digest,
        accumulator: head.accumulator,
        state: new_state,
    })
}

fn hex_to_32(hex: &str) -> Result<[u8; 32], String> {
    if hex.len() != 64 {
        return Err(format!("expected 64 hex chars, got {}", hex.len()));
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
            .map_err(|e| format!("bad hex byte at {i}: {e}"))?;
    }
    Ok(out)
}

/// Encode `IvcPublicValues` as a MemoChain Capsule v1 binary file (22,606 bytes),
/// directly readable by `src/memochain/state.py::Capsule.from_bytes`.
pub fn encode_capsule_v1(pv: &IvcPublicValues) -> Result<Vec<u8>, String> {
    if pv.state.len() != STATE_PARAMS {
        return Err(format!("state len {} != {}", pv.state.len(), STATE_PARAMS));
    }
    let mut payload = Vec::with_capacity(STATE_BYTES);
    for &v in &pv.state {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    let payload_hash: [u8; 32] = Sha256::digest(&payload).into();
    let head_bytes = hex_to_32(&pv.accumulator)?;

    let mut out = Vec::with_capacity(78 + STATE_BYTES);
    out.extend_from_slice(b"MCN1");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&pv.step.to_le_bytes());
    out.extend_from_slice(&head_bytes);
    out.extend_from_slice(&payload_hash);
    out.extend_from_slice(&payload);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use memochain_core::student::{compute_projections_raw, EMBED_DIM, WEIGHTS_BYTES};
    use memochain_core::{KEY_PART, N_HEADS};

    fn synth_weights(seed: i64) -> Vec<u8> {
        let mut raw = vec![0u8; WEIGHTS_BYTES];
        let mut set = |off: usize, v: i16| {
            let b = v.to_le_bytes();
            raw[off] = b[0];
            raw[off + 1] = b[1];
        };
        for r in 0..KEY_PART {
            for c in 0..EMBED_DIM {
                let v = (((r as i64 * 131 + c as i64 * 17 + seed) % 31) - 15) as i16;
                set((r * EMBED_DIM + c) * 2, v);
                set(
                    (KEY_PART * EMBED_DIM + KEY_PART + r * EMBED_DIM + c) * 2,
                    ((v as i64 * 3) % 29) as i16,
                );
            }
        }
        let beta_off = (KEY_PART * EMBED_DIM * 2 + KEY_PART * 2) * 2;
        for h in 0..N_HEADS {
            set(beta_off + h * 2, 6554);
            set(beta_off + (N_HEADS + h) * 2, 16350);
        }
        raw
    }

    fn synth_docs(n: usize) -> Vec<u8> {
        let mut raw = Vec::with_capacity(n * EMBED_DIM * 2);
        for i in 0..n {
            for c in 0..EMBED_DIM {
                let v = ((((i + 1) * 73 + c * 19) % 51) as i64 - 25) as i16;
                raw.extend_from_slice(&v.to_le_bytes());
            }
        }
        raw
    }

    #[test]
    fn ivc_genesis_plus_recursive_step_matches_monolithic() {
        let vk = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let w_raw = synth_weights(0);
        let writer = StudentWriter::from_bytes(&w_raw).unwrap();
        let docs = synth_docs(6);
        let proj_all = compute_projections_raw(&writer, &docs);

        // Monolithic 6-doc fold
        let mono = run_ivc_step(vk, &w_raw, &docs, Some(&proj_all), None).unwrap();
        assert_eq!(mono.step, 6);

        // Mina IVC: base proof of 5 docs (Freivalds) + 1-doc recursive step (direct matmul)
        let docs_5 = &docs[..5 * EMBED_DIM * 2];
        let proj_5 = compute_projections_raw(&writer, docs_5);
        let pv_5 = run_ivc_step(vk, &w_raw, docs_5, Some(&proj_5), None).unwrap();
        assert_eq!(pv_5.step, 5);

        let doc_6 = &docs[5 * EMBED_DIM * 2..];
        let pv_6 = run_ivc_step(vk, &w_raw, doc_6, None, Some(&pv_5)).unwrap();
        assert_eq!(pv_6, mono);

        // Weight swap or VK swap mid-chain MUST fail
        let bad_w = synth_weights(99);
        assert!(run_ivc_step(vk, &bad_w, doc_6, None, Some(&pv_5)).is_err());
        let bad_vk = [9u32; 8];
        assert!(run_ivc_step(bad_vk, &w_raw, doc_6, None, Some(&pv_5)).is_err());

        // Capsule encoding produces exact 22,606-byte v1 envelope
        let cap = encode_capsule_v1(&pv_6).unwrap();
        assert_eq!(cap.len(), 78 + STATE_BYTES);
        assert_eq!(&cap[..4], b"MCN1");
    }
}
