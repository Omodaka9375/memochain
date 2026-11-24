//! The provable MemoChain transition, as one pure function.
//!
//! This is the exact logic the SP1 guest executes (`crates/sp1/program`) and
//! the same logic native runners execute (`crates/sp1/runner`). It folds a
//! document collection through the integer Delta-rule with a receipt chain
//! and produces the public values a verifier checks.
//!
//! Compatibility contract (mirrors src/memochain/verify.py exactly):
//!   doc_hash      = sha256(doc int16 little-endian bytes)             (64 hex)
//!   state_digest  = sha256(state int16 little-endian bytes)           (64 hex)
//!   binding       = sha256("{step}|{prev_acc}|{doc_hash}|{state}")
//!   accumulator   = sha256(prev_acc_ascii + binding_ascii)
//! Genesis accumulator is 64 '0' chars.

use memochain_core::receipt::{write_with_receipt, ChainHead};
use memochain_core::{Folder, N_HEADS};
use serde::{Deserialize, Serialize};

pub mod ivc;

pub const STATE_PARAMS: usize = memochain_core::STATE_PARAMS; // 11,264
pub const DOC_DIM: usize = memochain_core::DOC_DIM; // 704
pub const MAX_DOCS: usize = 256;

/// Guest input: the fold weights and the documents to fold.
///
/// Arrays are fixed-size so the zkVM guest stays allocation-free and the
/// prover's memory footprint is bounded and known in advance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainInput {
    /// Flattened [N_HEADS][KEY_DIM][KEY_DIM] row-major int16 weights.
    pub w_key: Vec<i16>,
    /// Flattened [N_HEADS][VAL_DIM][KEY_DIM] row-major int16 weights.
    pub w_val: Vec<i16>,
    /// Q14 write strength.
    pub beta: i64,
    /// Q14 per-write decay.
    pub decay: i64,
    /// Number of valid docs in `docs_buf`.
    pub n_docs: usize,
    /// Document buffer: n_docs docs of DOC_DIM values each (rest ignored).
    pub docs_buf: Vec<i16>,
}

impl ChainInput {
    pub fn docs(&self) -> impl Iterator<Item = &[i16]> {
        self.docs_buf.chunks(DOC_DIM).take(self.n_docs)
    }

    pub fn from_vecs(w_key: Vec<i16>, w_val: Vec<i16>, beta: i64, decay: i64, docs: Vec<Vec<i16>>) -> Self {
        let n_docs = docs.len();
        let docs_buf = docs.into_iter().flatten().collect();
        Self { w_key, w_val, beta, decay, n_docs, docs_buf }
    }
}

/// Public values committed by the guest and checked by any verifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicValues {
    /// Number of documents folded (= chain step).
    pub step: u64,
    /// Final 22,528-byte state as int16 values.
    pub state: Vec<i16>,
    /// sha256 hex of the final state (the "block hash").
    pub state_digest: String,
    /// Rolling receipt-chain accumulator (the light-client head).
    pub accumulator: String,
}

/// Validate + rebuild the Folder from flat weights.
pub fn folder_from_input(input: &ChainInput) -> Result<Folder, String> {
    let k_len = N_HEADS * memochain_core::KEY_DIM * memochain_core::KEY_DIM;
    let v_len = N_HEADS * memochain_core::VAL_DIM * memochain_core::KEY_DIM;
    if input.w_key.len() != k_len {
        return Err(format!("w_key must have {k_len} values, got {}", input.w_key.len()));
    }
    if input.w_val.len() != v_len {
        return Err(format!("w_val must have {v_len} values, got {}", input.w_val.len()));
    }
    if input.n_docs == 0 || input.n_docs > MAX_DOCS {
        return Err(format!("n_docs must be 1..={MAX_DOCS}, got {}", input.n_docs));
    }
    if input.docs_buf.len() != input.n_docs * DOC_DIM {
        return Err(format!(
            "docs_buf must have n_docs*{DOC_DIM} values, got {}",
            input.docs_buf.len()
        ));
    }
    let chunk = |flat: &[i16], rows: usize, cols: usize| -> Vec<Vec<i16>> {
        (0..rows).map(|r| flat[r * cols..(r + 1) * cols].to_vec()).collect()
    };
    Ok(Folder {
        w_key: chunk(&input.w_key, N_HEADS, k_len / N_HEADS),
        w_val: chunk(&input.w_val, N_HEADS, v_len / N_HEADS),
        beta: input.beta,
        decay: input.decay,
    })
}

/// Fold every document with receipts; returns the public values.
pub fn run_chain(input: &ChainInput) -> Result<PublicValues, String> {
    let folder = folder_from_input(input)?;
    let mut state = memochain_core::zero_state();
    let mut head: Option<ChainHead> = None;
    for doc in input.docs() {
        let (s, _r, h) = write_with_receipt(&folder, &state, doc, head.as_ref());
        state = s;
        head = Some(h);
    }
    let head = head.ok_or_else(|| "no documents provided".to_string())?;
    Ok(PublicValues {
        step: head.step,
        state,
        state_digest: head.state_digest,
        accumulator: head.accumulator,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    /// Parse the parity vector file produced by tools/gen_rust_vectors.py.
    fn parse_sections(path: &Path) -> Vec<(String, Vec<String>)> {
        let text = fs::read_to_string(path).expect("parity.txt missing");
        let mut tokens = text.split_whitespace();
        let mut sections: Vec<(String, Vec<String>)> = Vec::new();
        let mut section = String::new();
        let mut count = 0usize;
        let mut values: Vec<String> = Vec::new();
        loop {
            let Some(tok) = tokens.next() else { break };
            if count == 0 {
                if !section.is_empty() {
                    sections.push((std::mem::take(&mut section), std::mem::take(&mut values)));
                }
                section = tok.to_string();
                count = tokens.next().expect("count").parse().unwrap();
            } else {
                values.push(tok.to_string());
                count -= 1;
            }
        }
        if !section.is_empty() {
            sections.push((section, values));
        }
        sections
    }

    /// The guest logic must reproduce Python bit-for-bit: every intermediate
    /// state and every receipt-chain accumulator from the generated vectors.
    #[test]
    fn run_chain_matches_python_vectors() {
        let data_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../memochain-core/tests/data");
        let sections = parse_sections(&data_dir.join("parity.txt"));
        let get = |name: &str| -> &Vec<String> {
            &sections
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("missing section {name}"))
                .1
        };

        // Accumulators: tokens are "N <count> <64-hex> ..." (raw hex preserves
        // leading zeros, unlike integer encoding).
        let acc_text = fs::read_to_string(data_dir.join("parity_accumulators.txt"))
            .expect("parity_accumulators.txt missing; run tools/gen_rust_vectors.py");
        let toks: Vec<&str> = acc_text.split_whitespace().collect();
        assert_eq!(toks[0], "N", "expected 'N <count>' header");
        let n_acc: usize = toks[1].parse().unwrap();
        assert_eq!(toks.len(), 2 + n_acc, "accumulator count mismatch");
        let accumulators: Vec<String> = toks[2..].iter().map(|s| s.to_string()).collect();

        let w_key: Vec<i16> = get("W_KEY").iter().map(|s| s.parse::<i64>().unwrap() as i16).collect();
        let w_val: Vec<i16> = get("W_VAL").iter().map(|s| s.parse::<i64>().unwrap() as i16).collect();
        let docs: Vec<i16> = get("DOCS").iter().map(|s| s.parse::<i64>().unwrap() as i16).collect();
        let n_docs: usize = get("N_DOCS")[0].parse().unwrap();
        let doc_dim: usize = get("DOC_DIM")[0].parse().unwrap();
        assert_eq!(doc_dim, DOC_DIM);

        let input = ChainInput::from_vecs(
            w_key,
            w_val,
            get("BETA")[0].parse().unwrap(),
            get("DECAY")[0].parse().unwrap(),
            (0..n_docs)
                .map(|i| docs[i * doc_dim..(i + 1) * doc_dim].to_vec())
                .collect(),
        );

        let out = run_chain(&input).expect("run_chain failed");
        assert_eq!(out.step as usize, n_docs);
        assert_eq!(out.state.len(), STATE_PARAMS);

        // Every intermediate state must match, not just the final one:
        // replay step by step with the same functions the guest calls.
        let folder = folder_from_input(&input).unwrap();
        let mut state = memochain_core::zero_state();
        let mut head: Option<ChainHead> = None;
        for i in 0..n_docs {
            let (s, _r, h) =
                write_with_receipt(&folder, &state, &input.docs_buf[i * DOC_DIM..(i + 1) * DOC_DIM], head.as_ref());
            state = s;
            let acc_here = h.accumulator.clone();
            head = Some(h);
            let expected_state: Vec<i16> = get(&format!("STATE_{i}"))
                .iter()
                .map(|s| s.parse::<i64>().unwrap() as i16)
                .collect();
            assert_eq!(state, expected_state, "state mismatch after doc {i}");
            // Accumulator parity: from parity_accumulators.txt (raw hex tokens,
            // generated together with the state vectors).
            if let Some(acc) = accumulators.get(i) {
                assert_eq!(&acc_here, acc, "accumulator mismatch after doc {i}");
            }
        }

        // Final values must equal the replay and the vectors.
        assert_eq!(out.state, state);
        let final_head = head.expect("at least one doc");
        assert_eq!(out.accumulator, final_head.accumulator);
        assert_eq!(out.accumulator, accumulators[n_acc - 1]);
        let expected_final: Vec<i16> = get(&format!("STATE_{}", n_docs - 1))
            .iter()
            .map(|s| s.parse::<i64>().unwrap() as i16)
            .collect();
        assert_eq!(out.state, expected_final);
    }

    #[test]
    fn rejects_bad_input() {
        let input = ChainInput {
            w_key: vec![0; 10],
            w_val: vec![],
            beta: 1,
            decay: 1,
            n_docs: 0,
            docs_buf: vec![],
        };
        assert!(folder_from_input(&input).is_err());
        let good_weights = ChainInput {
            w_key: vec![0; 11 * 32 * 32],
            w_val: vec![0; 11 * 32 * 32],
            beta: 1,
            decay: 1,
            n_docs: 1,
            docs_buf: vec![0; DOC_DIM - 1],
        };
        assert!(folder_from_input(&good_weights).is_err());
        let out = run_chain(&ChainInput {
            w_key: vec![0; 11 * 32 * 32],
            w_val: vec![0; 11 * 32 * 32],
            beta: 1,
            decay: 1,
            n_docs: 0,
            docs_buf: vec![],
        });
        assert!(out.is_err());
    }

    #[test]
    fn empty_fold_fails_cleanly() {
        let input = ChainInput {
            w_key: vec![0; 11 * 32 * 32],
            w_val: vec![0; 11 * 32 * 32],
            beta: 16384,
            decay: 16384,
            n_docs: 0,
            docs_buf: vec![],
        };
        assert!(run_chain(&input).is_err());
    }
}
