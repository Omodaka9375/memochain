//! Exact Q14 fixed-point writer for the distilled MemoChainStudent,
//! bit-identical with src/memochain/student_int.py.
//!
//! Layout matches the trained Qwen3-Embedding-4B -> 11x32x32 student:
//!   - Input document embedding: 2560 i16 values in Q14 fixed point
//!   - key_proj: [352, 2560] i16 weights + [352] i16 bias
//!   - val_proj: [352, 2560] i16 weights + [352] i16 bias
//!   - Per-head gates: beta[11] i16, decay[11] i16 (in [0, 16384])
//!
//! For small batches (n_docs <= FREIVALDS_THRESHOLD, e.g. 1-doc Mina IVC steps),
//! projections are computed directly inside the zkVM. For larger batches
//! (e.g. 199 docs in a base proof), the prover supplies exact int64 pre-shift
//! dot products and the verifier checks them with 4 independent exact i128
//! Fiat-Shamir Freivalds rounds (soundness error <= 2^-128, zero float, zero
//! overflow: max intermediate <= 2^82 << 2^127).

use crate::receipt::{sha256_hex, state_digest, ChainHead, Receipt};
use crate::{clip16, unit_key, HEAD_PARAMS, KEY_DIM, KEY_PART, N_HEADS, Q_SHIFT, STATE_PARAMS, VAL_DIM};
use sha2::{Digest, Sha256};

pub const EMBED_DIM: usize = 2560;
pub const FREIVALDS_THRESHOLD: usize = 4;
pub const FREIVALDS_ROUNDS: usize = 2; // 2 independent u64 challenges => <= 2^-128 soundness error

pub const WEIGHTS_BYTES: usize = KEY_PART * EMBED_DIM * 2 // w_key
    + KEY_PART * 2                                         // b_key
    + KEY_PART * EMBED_DIM * 2                             // w_val
    + KEY_PART * 2                                         // b_val
    + N_HEADS * 2                                          // beta
    + N_HEADS * 2;                                         // decay (3,605,932 B)

/// Zero-copy view over canonical little-endian quantized student weights.
pub struct StudentWriter<'a> {
    pub raw: &'a [u8],
}

impl<'a> StudentWriter<'a> {
    pub fn from_bytes(raw: &'a [u8]) -> Result<Self, String> {
        if raw.len() != WEIGHTS_BYTES {
            return Err(format!(
                "weights length {} != expected {}",
                raw.len(),
                WEIGHTS_BYTES
            ));
        }
        Ok(Self { raw })
    }

    pub fn weights_digest(&self) -> String {
        sha256_hex(self.raw)
    }

    #[inline]
    fn read_i16(&self, byte_offset: usize) -> i16 {
        i16::from_le_bytes([self.raw[byte_offset], self.raw[byte_offset + 1]])
    }

    #[inline]
    fn w_key_off() -> usize {
        0
    }
    #[inline]
    fn b_key_off() -> usize {
        KEY_PART * EMBED_DIM * 2
    }
    #[inline]
    fn w_val_off() -> usize {
        Self::b_key_off() + KEY_PART * 2
    }
    #[inline]
    fn b_val_off() -> usize {
        Self::w_val_off() + KEY_PART * EMBED_DIM * 2
    }
    #[inline]
    fn beta_off() -> usize {
        Self::b_val_off() + KEY_PART * 2
    }
    #[inline]
    fn decay_off() -> usize {
        Self::beta_off() + N_HEADS * 2
    }

    pub fn b_key(&self, r: usize) -> i64 {
        self.read_i16(Self::b_key_off() + r * 2) as i64
    }

    pub fn b_val(&self, r: usize) -> i64 {
        self.read_i16(Self::b_val_off() + r * 2) as i64
    }

    pub fn beta(&self, h: usize) -> i64 {
        self.read_i16(Self::beta_off() + h * 2) as i64
    }

    pub fn decay(&self, h: usize) -> i64 {
        self.read_i16(Self::decay_off() + h * 2) as i64
    }

    #[inline]
    fn row_dot(&self, matrix_base: usize, row: usize, x: &[i16]) -> i64 {
        let row_bytes = &self.raw[matrix_base + row * EMBED_DIM * 2..matrix_base + (row + 1) * EMBED_DIM * 2];
        let mut acc = 0i64;
        for (wb, &xv) in row_bytes.chunks_exact(2).zip(x.iter()) {
            let w = i16::from_le_bytes([wb[0], wb[1]]) as i64;
            acc += w * (xv as i64);
        }
        acc
    }

    /// Compute exact pre-shift dot products (dot_k[352], dot_v[352]) for one 2560-d doc.
    pub fn project_dots(&self, doc: &[i16]) -> ([i64; KEY_PART], [i64; KEY_PART]) {
        assert_eq!(doc.len(), EMBED_DIM);
        let mut dk = [0i64; KEY_PART];
        let mut dv = [0i64; KEY_PART];
        let wk_base = Self::w_key_off();
        let wv_base = Self::w_val_off();
        for r in 0..KEY_PART {
            dk[r] = self.row_dot(wk_base, r, doc);
            dv[r] = self.row_dot(wv_base, r, doc);
        }
        (dk, dv)
    }

    /// Fold one document given its exact pre-shift dot products [352], [352].
    pub fn fold_from_dots(&self, state: &[i16], dot_k: &[i64], dot_v: &[i64]) -> Vec<i16> {
        assert_eq!(state.len(), STATE_PARAMS);
        assert_eq!(dot_k.len(), KEY_PART);
        assert_eq!(dot_v.len(), KEY_PART);

        let mut tokens: Vec<i64> = state.iter().map(|&x| x as i64).collect();
        let mut raw_k_h = [0i64; KEY_DIM];
        let mut raw_v_h = [0i16; VAL_DIM];

        for h in 0..N_HEADS {
            for i in 0..KEY_DIM {
                let r = h * KEY_DIM + i;
                raw_k_h[i] = (dot_k[r] >> Q_SHIFT) + self.b_key(r);
                raw_v_h[i] = clip16((dot_v[r] >> Q_SHIFT) + self.b_val(r));
            }
            let uk = unit_key(&raw_k_h);
            if uk.iter().all(|&x| x == 0) {
                continue;
            }
            let base = h * HEAD_PARAMS;
            let beta_h = self.beta(h);
            let decay_h = self.decay(h);

            let mut recalled = [0i64; VAL_DIM];
            for v in 0..VAL_DIM {
                let mut acc = 0i64;
                for kk in 0..KEY_DIM {
                    acc += uk[kk] as i64 * tokens[base + kk * VAL_DIM + v];
                }
                recalled[v] = acc >> Q_SHIFT;
            }
            let mut delta = [0i64; VAL_DIM];
            for v in 0..VAL_DIM {
                delta[v] = (beta_h * (raw_v_h[v] as i64 - recalled[v])) >> Q_SHIFT;
            }
            for kk in 0..KEY_DIM {
                let k_val = uk[kk] as i64;
                for v in 0..VAL_DIM {
                    let idx = base + kk * VAL_DIM + v;
                    let decayed = (decay_h * tokens[idx]) >> Q_SHIFT;
                    let update = (k_val * delta[v]) >> Q_SHIFT;
                    tokens[idx] = decayed + update;
                }
            }
        }
        tokens.iter().map(|&x| clip16(x)).collect()
    }

    /// Verify exact integer matrix products dot_k == X @ W_key^T and dot_v == X @ W_val^T
    /// using 2-round exact i128 Fiat-Shamir Freivalds with uniform u64 challenges
    /// (soundness error <= (2^-64)^2 = 2^-128; max intermediate <= 2^114 << 2^127,
    /// single pass over the 3.6 MB weight matrices).
    pub fn verify_projections_freivalds(
        &self,
        weights_digest_hex: &str,
        docs_raw: &[u8],
        proj_raw: &[u8],
        n_docs: usize,
    ) -> Result<(), String> {
        let expected_docs = n_docs * EMBED_DIM * 2;
        if docs_raw.len() != expected_docs {
            return Err(format!("docs_raw len {} != {}", docs_raw.len(), expected_docs));
        }
        let expected_proj = n_docs * KEY_PART * 8 * 2;
        if proj_raw.len() != expected_proj {
            return Err(format!("proj_raw len {} != {}", proj_raw.len(), expected_proj));
        }

        // Fiat-Shamir transcript commits weights, docs, and claimed projections.
        let mut h = Sha256::new();
        h.update(b"memochain-freivalds-v2|");
        h.update(weights_digest_hex.as_bytes());
        h.update(b"|");
        h.update(Sha256::digest(docs_raw));
        h.update(b"|");
        h.update(Sha256::digest(proj_raw));
        let seed: [u8; 32] = h.finalize().into();

        // Derive 2 independent uniform u64 challenge vectors r0, r1 (length n_docs).
        let mut r = [vec![0u64; n_docs], vec![0u64; n_docs]];
        for trial in 0..FREIVALDS_ROUNDS {
            let mut pos = 0usize;
            let mut block = 0u32;
            while pos < n_docs {
                let mut ch = Sha256::new();
                ch.update(seed);
                ch.update((trial as u32).to_le_bytes());
                ch.update(block.to_le_bytes());
                let digest: [u8; 32] = ch.finalize().into();
                for chunk in digest.chunks_exact(8) {
                    if pos < n_docs {
                        r[trial][pos] = u64::from_le_bytes(chunk.try_into().unwrap());
                        pos += 1;
                    }
                }
                block += 1;
            }
        }

        // Compute vx[0] = r0^T @ X and vx[1] = r1^T @ X in a single pass over docs_raw.
        // Max |vx[t][c]| <= 256 * 2^64 * 32768 = 2^87 << 2^127.
        let mut vx0 = vec![0i128; EMBED_DIM];
        let mut vx1 = vec![0i128; EMBED_DIM];
        for i in 0..n_docs {
            let r0 = r[0][i] as i128;
            let r1 = r[1][i] as i128;
            let doc_slice = &docs_raw[i * EMBED_DIM * 2..(i + 1) * EMBED_DIM * 2];
            for (c, xb) in doc_slice.chunks_exact(2).enumerate() {
                let x = i16::from_le_bytes([xb[0], xb[1]]) as i128;
                vx0[c] += r0 * x;
                vx1[c] += r1 * x;
            }
        }

        let half = n_docs * KEY_PART * 8;
        let dot_k_bytes = &proj_raw[..half];
        let dot_v_bytes = &proj_raw[half..];
        let wk_base = Self::w_key_off();
        let wv_base = Self::w_val_off();

        // Single pass over W_key and W_val verifying both independent u64 trials at once.
        // Max |(vx @ W^T)[row]| <= 2560 * 2^87 * 32768 <= 2^114 << 2^127.
        for row in 0..KEY_PART {
            let mut lhs_k0 = 0i128;
            let mut lhs_k1 = 0i128;
            let mut lhs_v0 = 0i128;
            let mut lhs_v1 = 0i128;
            let wk_row = &self.raw[wk_base + row * EMBED_DIM * 2..wk_base + (row + 1) * EMBED_DIM * 2];
            let wv_row = &self.raw[wv_base + row * EMBED_DIM * 2..wv_base + (row + 1) * EMBED_DIM * 2];

            for (((wb_k, wb_v), &v0), &v1) in wk_row
                .chunks_exact(2)
                .zip(wv_row.chunks_exact(2))
                .zip(vx0.iter())
                .zip(vx1.iter())
            {
                let wk = i16::from_le_bytes([wb_k[0], wb_k[1]]) as i128;
                let wv = i16::from_le_bytes([wb_v[0], wb_v[1]]) as i128;
                lhs_k0 += v0 * wk;
                lhs_k1 += v1 * wk;
                lhs_v0 += v0 * wv;
                lhs_v1 += v1 * wv;
            }

            let mut rhs_k0 = 0i128;
            let mut rhs_k1 = 0i128;
            let mut rhs_v0 = 0i128;
            let mut rhs_v1 = 0i128;
            for i in 0..n_docs {
                let r0 = r[0][i] as i128;
                let r1 = r[1][i] as i128;
                let off = (i * KEY_PART + row) * 8;
                let ck = i64::from_le_bytes(dot_k_bytes[off..off + 8].try_into().unwrap()) as i128;
                let cv = i64::from_le_bytes(dot_v_bytes[off..off + 8].try_into().unwrap()) as i128;
                rhs_k0 += r0 * ck;
                rhs_k1 += r1 * ck;
                rhs_v0 += r0 * cv;
                rhs_v1 += r1 * cv;
            }

            if lhs_k0 != rhs_k0 || lhs_k1 != rhs_k1 {
                return Err(format!("Freivalds key mismatch at row {row}"));
            }
            if lhs_v0 != rhs_v0 || lhs_v1 != rhs_v1 {
                return Err(format!("Freivalds val mismatch at row {row}"));
            }
        }
        Ok(())
    }

    /// Fold one batch of docs starting from (initial_state, initial_head).
    /// Uses direct exact matmul for n_docs <= 4 and 4-round i128 Freivalds for n_docs > 4.
    pub fn fold_batch(
        &self,
        initial_state: &[i16],
        initial_head: Option<&ChainHead>,
        docs_raw: &[u8],
        proj_raw: Option<&[u8]>,
        weights_digest_hex: &str,
    ) -> Result<(Vec<i16>, ChainHead), String> {
        if initial_state.len() != STATE_PARAMS {
            return Err(format!("state len {} != {}", initial_state.len(), STATE_PARAMS));
        }
        if docs_raw.is_empty() || docs_raw.len() % (EMBED_DIM * 2) != 0 {
            return Err(format!("invalid docs_raw length {}", docs_raw.len()));
        }
        let n_docs = docs_raw.len() / (EMBED_DIM * 2);

        let use_freivalds = n_docs > FREIVALDS_THRESHOLD;
        if use_freivalds {
            let proj = proj_raw.ok_or("proj_raw required when n_docs > 4")?;
            self.verify_projections_freivalds(weights_digest_hex, docs_raw, proj, n_docs)?;
        }

        let mut state = initial_state.to_vec();
        let mut step = initial_head.map_or(0u64, |h| h.step);
        let mut accumulator = initial_head.map_or_else(|| "0".repeat(64), |h| h.accumulator.clone());
        let mut doc_i16 = vec![0i16; EMBED_DIM];
        let mut dk = [0i64; KEY_PART];
        let mut dv = [0i64; KEY_PART];
        let half = n_docs * KEY_PART * 8;

        for i in 0..n_docs {
            let doc_bytes = &docs_raw[i * EMBED_DIM * 2..(i + 1) * EMBED_DIM * 2];
            let d_hash = sha256_hex(doc_bytes);

            if use_freivalds {
                let proj = proj_raw.unwrap();
                for r in 0..KEY_PART {
                    let off = (i * KEY_PART + r) * 8;
                    dk[r] = i64::from_le_bytes(proj[off..off + 8].try_into().unwrap());
                    dv[r] = i64::from_le_bytes(proj[half + off..half + off + 8].try_into().unwrap());
                }
            } else {
                for c in 0..EMBED_DIM {
                    doc_i16[c] = i16::from_le_bytes([doc_bytes[c * 2], doc_bytes[c * 2 + 1]]);
                }
                let (pdk, pdv) = self.project_dots(&doc_i16);
                dk = pdk;
                dv = pdv;
            }

            state = self.fold_from_dots(&state, &dk, &dv);
            let s_digest = state_digest(&state);
            step += 1;
            let binding = sha256_hex(format!("{step}|{accumulator}|{d_hash}|{s_digest}").as_bytes());
            accumulator = sha256_hex(format!("{accumulator}{binding}").as_bytes());
            let _ = Receipt {
                step,
                prev_head: String::new(),
                doc_digest: d_hash,
                new_head: s_digest.clone(),
                digest: binding,
            };
        }

        let final_digest = state_digest(&state);
        Ok((
            state,
            ChainHead {
                step,
                accumulator,
                state_digest: final_digest,
            },
        ))
    }
}

/// Helper for host-side precomputation of exact int64 projections for Freivalds.
pub fn compute_projections_raw(writer: &StudentWriter<'_>, docs_raw: &[u8]) -> Vec<u8> {
    assert_eq!(docs_raw.len() % (EMBED_DIM * 2), 0);
    let n_docs = docs_raw.len() / (EMBED_DIM * 2);
    let mut dk_all = Vec::with_capacity(n_docs * KEY_PART * 8);
    let mut dv_all = Vec::with_capacity(n_docs * KEY_PART * 8);
    let mut doc_i16 = vec![0i16; EMBED_DIM];
    for i in 0..n_docs {
        let slice = &docs_raw[i * EMBED_DIM * 2..(i + 1) * EMBED_DIM * 2];
        for c in 0..EMBED_DIM {
            doc_i16[c] = i16::from_le_bytes([slice[c * 2], slice[c * 2 + 1]]);
        }
        let (dk, dv) = writer.project_dots(&doc_i16);
        for &v in &dk {
            dk_all.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &dv {
            dv_all.extend_from_slice(&v.to_le_bytes());
        }
    }
    dk_all.extend_from_slice(&dv_all);
    dk_all
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zero_state;

    fn synthetic_weights() -> Vec<u8> {
        let mut raw = vec![0u8; WEIGHTS_BYTES];
        let mut write_i16 = |off: usize, v: i16| {
            let b = v.to_le_bytes();
            raw[off] = b[0];
            raw[off + 1] = b[1];
        };
        for r in 0..KEY_PART {
            for c in 0..EMBED_DIM {
                let v = (((r * 131 + c * 17) % 41) as i64 - 20) as i16;
                write_i16((r * EMBED_DIM + c) * 2, v);
                let v2 = (((r * 257 + c * 29) % 37) as i64 - 18) as i16;
                write_i16(StudentWriter::w_val_off() + (r * EMBED_DIM + c) * 2, v2);
            }
            write_i16(StudentWriter::b_key_off() + r * 2, (r as i16) - 100);
            write_i16(StudentWriter::b_val_off() + r * 2, 50 - (r as i16));
        }
        for h in 0..N_HEADS {
            write_i16(StudentWriter::beta_off() + h * 2, 6500 + (h as i16) * 10);
            write_i16(StudentWriter::decay_off() + h * 2, 16350);
        }
        raw
    }

    fn synthetic_docs(n: usize) -> Vec<u8> {
        let mut raw = Vec::with_capacity(n * EMBED_DIM * 2);
        for i in 0..n {
            for c in 0..EMBED_DIM {
                let v = ((((i + 1) * 97 + c * 31) % 61) as i64 - 30) as i16;
                raw.extend_from_slice(&v.to_le_bytes());
            }
        }
        raw
    }

    #[test]
    fn direct_and_freivalds_are_bit_exact_and_catch_tampering() {
        let w_raw = synthetic_weights();
        let writer = StudentWriter::from_bytes(&w_raw).unwrap();
        let w_digest = writer.weights_digest();
        let docs = synthetic_docs(6);
        let proj = compute_projections_raw(&writer, &docs);

        // Batch of 6 uses Freivalds; compare against 6 sequential 1-doc direct steps!
        let (s_batch, h_batch) = writer
            .fold_batch(&zero_state(), None, &docs, Some(&proj), &w_digest)
            .unwrap();

        let mut s_seq = zero_state();
        let mut h_seq: Option<ChainHead> = None;
        for i in 0..6 {
            let one = &docs[i * EMBED_DIM * 2..(i + 1) * EMBED_DIM * 2];
            let (ns, nh) = writer
                .fold_batch(&s_seq, h_seq.as_ref(), one, None, &w_digest)
                .unwrap();
            s_seq = ns;
            h_seq = Some(nh);
        }
        assert_eq!(s_batch, s_seq);
        assert_eq!(h_batch, h_seq.unwrap());

        // Flip 1 single LSB in dot_k -> Freivalds MUST reject!
        let mut bad_proj = proj.clone();
        bad_proj[123] ^= 1;
        let err = writer.fold_batch(&zero_state(), None, &docs, Some(&bad_proj), &w_digest);
        assert!(err.is_err(), "tampered projection must fail Freivalds");
    }
}
