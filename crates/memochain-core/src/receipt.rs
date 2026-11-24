//! Receipt chain: tamper detection over the fold, bit-compatible with
//! src/memochain/verify.py.
//!
//! Every write emits a receipt binding (step, previous accumulator, doc
//! hash, new state digest) and advances a rolling accumulator, so a light
//! verifier needs only the latest 22 KB state, the step count, and one
//! 256-bit accumulator hex string.
//!
//! Compatibility contract with the Python side:
//!   doc_hash      = sha256(doc.as_int16_le_bytes()).hexdigest()          (64 hex)
//!   state_digest  = sha256(state.as_int16_le_bytes()).hexdigest()        (64 hex)
//!   binding       = sha256("{step}|{prev_head}|{doc}|{state}")           (64 hex)
//!   accumulator   = sha256(prev_head_ascii + binding_ascii)              (64 hex)
//! All hashes are lowercase hex; prev_head for step 1 is 64 '0's.

use sha2::{Digest, Sha256};

/// sha256 of raw bytes, lowercase hex.
pub fn sha256_hex(data: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut hasher = Sha256::new();
    hasher.update(data);
    let out = hasher.finalize();
    let mut buf = [0u8; 64];
    for (i, &b) in out.iter().enumerate() {
        buf[i * 2] = HEX[(b >> 4) as usize];
        buf[i * 2 + 1] = HEX[(b & 0xf) as usize];
    }
    String::from_utf8(buf.to_vec()).unwrap()
}

/// Hash of one document's int16 little-endian bytes.
pub fn doc_hash(doc: &[i16]) -> String {
    let mut bytes = Vec::with_capacity(doc.len() * 2);
    for &v in doc {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    sha256_hex(&bytes)
}

/// Hash of the flat int16 state (little-endian bytes).
pub fn state_digest(state: &[i16]) -> String {
    let mut bytes = Vec::with_capacity(state.len() * 2);
    for &v in state {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    sha256_hex(&bytes)
}

/// One write's receipt plus the new rolling head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    pub step: u64,
    pub prev_head: String,
    pub doc_digest: String,
    pub new_head: String,
    pub digest: String,
}

/// The O(1) verification artifact a light client holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainHead {
    pub step: u64,
    pub accumulator: String,
    pub state_digest: String,
}

/// Fold one document with a receipt. Returns (new_state, receipt, new_head).
/// genesis head must be the previous ChainHead (None for step 1).
pub fn write_with_receipt(
    folder: &crate::Folder,
    state: &[i16],
    doc: &[i16],
    prev: Option<&ChainHead>,
) -> (Vec<i16>, Receipt, ChainHead) {
    let step = prev.map_or(1, |p| p.step + 1);
    let prev_head = prev.map_or("0".repeat(64), |p| p.accumulator.clone());
    let d_hash = doc_hash(doc);
    let new_state = folder.fold(state, doc);
    let new_state_hex = state_digest(&new_state);

    let binding = sha256_hex(format!("{step}|{prev_head}|{d_hash}|{new_state_hex}").as_bytes());
    let accumulator = sha256_hex(format!("{prev_head}{binding}").as_bytes());

    let receipt = Receipt {
        step,
        prev_head: prev_head.clone(),
        doc_digest: d_hash,
        new_head: new_state_hex.clone(),
        digest: binding,
    };
    let head = ChainHead {
        step,
        accumulator,
        state_digest: new_state_hex,
    };
    (new_state, receipt, head)
}

/// Light check: stored state matches the chain head's digest.
pub fn verify_head(state: &[i16], head: &ChainHead) -> bool {
    state_digest(state) == head.state_digest
}

/// Full audit: replay every document and confirm the final head.
pub fn verify_chain(folder: &crate::Folder, docs: &[Vec<i16>], head: &ChainHead) -> bool {
    let mut state = crate::zero_state();
    let mut running: Option<ChainHead> = None;
    for doc in docs {
        let (new_state, _r, h) = write_with_receipt(folder, &state, doc, running.as_ref());
        state = new_state;
        running = Some(h);
    }
    match running {
        Some(r) => r.step == head.step
            && r.accumulator == head.accumulator
            && r.state_digest == head.state_digest,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{zero_state, Folder};

    fn test_folder() -> Folder {
        // Deterministic small weights (not the seeded python vectors).
        let mut w_key = Vec::new();
        for h in 0..crate::N_HEADS {
            let mut m = Vec::new();
            for i in 0..crate::KEY_DIM * crate::KEY_DIM {
                m.push((((h * 131 + i * 17) % 23) as i64 - 11) as i16);
            }
            w_key.push(m);
        }
        let mut w_val = Vec::new();
        for h in 0..crate::N_HEADS {
            let mut m = Vec::new();
            for i in 0..crate::VAL_DIM * crate::KEY_DIM {
                m.push((((h * 257 + i * 29) % 19) as i64 - 9) as i16);
            }
            w_val.push(m);
        }
        Folder { w_key, w_val, beta: 12288, decay: 16368 }
    }

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn chain_advances_and_verifies() {
        let folder = test_folder();
        let doc: Vec<i16> = (0..crate::DOC_DIM as i16).collect();
        let state = zero_state();

        let (s1, _r1, h1) = write_with_receipt(&folder, &state, &doc, None);
        assert_eq!(h1.step, 1);
        assert_eq!(h1.accumulator.len(), 64);
        assert!(verify_head(&s1, &h1));

        let (s2, _r2, h2) = write_with_receipt(&folder, &s1, &doc, Some(&h1));
        assert_eq!(h2.step, 2);
        assert_ne!(h1.accumulator, h2.accumulator);
        assert!(verify_head(&s2, &h2));
        assert!(!verify_head(&s1, &h2)); // old state fails new head
    }

    #[test]
    fn tampering_breaks_the_audit() {
        let folder = test_folder();
        let docs: Vec<Vec<i16>> = (0..4)
            .map(|i| {
                let mut d = vec![0i16; crate::DOC_DIM];
                d[i] = 1000 + i as i16;
                d
            })
            .collect();

        let mut state = zero_state();
        let mut head: Option<ChainHead> = None;
        for doc in &docs {
            let (s, _r, h) = write_with_receipt(&folder, &state, doc, head.as_ref());
            state = s;
            head = Some(h);
        }
        assert!(verify_chain(&folder, &docs, head.as_ref().unwrap()));

        // flip one document
        let mut bad = docs.clone();
        bad[2][5] += 3;
        assert!(!verify_chain(&folder, &bad, head.as_ref().unwrap()));

        // drop one document
        let mut shorter = docs.clone();
        shorter.pop();
        assert!(!verify_chain(&folder, &shorter, head.as_ref().unwrap()));
    }
}
