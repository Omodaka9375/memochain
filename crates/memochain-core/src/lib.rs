//! MemoChain core: the integer Delta-rule fold, bit-exact with src/memochain/fold.py.
//!
//! The state is 11 heads x 32 keys x 32 values of int16 in Q14 fixed point
//! (22,528 bytes total). Every operation is multiply-add-shift over i64 plus
//! an integer square root, so the same inputs produce the same 22,528 bytes
//! on every machine and the transition is directly provable in a zkVM.
//!
//! Layout matches the Python reference: a document is 704 int16 values
//! (key part 0..352, value part 352..704); head h binds its 32-value key
//! slice through W_key[h] and its 32-value value slice through W_val[h].
//! The state is head-major: head h occupies [h * 1024, (h + 1) * 1024),
//! row-major (key index, value index).

pub const N_HEADS: usize = 11;
pub const KEY_DIM: usize = 32;
pub const VAL_DIM: usize = 32;
pub const KEY_PART: usize = N_HEADS * KEY_DIM; // 352
pub const DOC_DIM: usize = 2 * KEY_PART; // 704
pub const HEAD_PARAMS: usize = KEY_DIM * VAL_DIM; // 1024
pub const STATE_PARAMS: usize = N_HEADS * HEAD_PARAMS; // 11264
pub const STATE_BYTES: usize = STATE_PARAMS * 2; // 22528

pub const Q_SHIFT: i64 = 14;
pub const Q_ONE: i64 = 1 << Q_SHIFT; // 16384

pub mod receipt;
pub mod student;

const I16_MAX: i64 = 32767;
const I16_MIN: i64 = -32768;

/// Deterministic integer square root (Newton, integer only).
pub fn isqrt(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// Floor division with a positive divisor (matches Python's `//`).
#[inline]
fn floor_div(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

#[inline]
fn clip16(x: i64) -> i16 {
    x.clamp(I16_MIN, I16_MAX) as i16
}

/// Scale a key to Q14 unit norm using integer math only.
pub fn unit_key(k: &[i64]) -> Vec<i16> {
    let norm = isqrt(k.iter().map(|&x| x * x).sum());
    if norm == 0 {
        return vec![0i16; k.len()];
    }
    k.iter()
        .map(|&x| clip16(floor_div(x * Q_ONE, norm)))
        .collect()
}

/// The fold operator: per-head key/value projections plus Delta-rule updates.
#[derive(Clone)]
pub struct Folder {
    /// [N_HEADS][KEY_DIM][KEY_DIM], row-major.
    pub w_key: Vec<Vec<i16>>,
    /// [N_HEADS][VAL_DIM][KEY_DIM], row-major.
    pub w_val: Vec<Vec<i16>>,
    /// Q14 write strength.
    pub beta: i64,
    /// Q14 per-write decay.
    pub decay: i64,
}

impl Folder {
    /// Project a doc into per-head unit keys (Q14) and bounded values.
    /// Returns (keys, values), each [N_HEADS] x vec of KEY_DIM / VAL_DIM.
    pub fn project(&self, doc: &[i16]) -> (Vec<Vec<i16>>, Vec<Vec<i16>>) {
        assert_eq!(doc.len(), DOC_DIM, "doc must have {} values", DOC_DIM);
        let mut keys: Vec<Vec<i16>> = Vec::with_capacity(N_HEADS);
        let mut vals: Vec<Vec<i16>> = Vec::with_capacity(N_HEADS);
        for h in 0..N_HEADS {
            let k_slice: Vec<i64> = doc[h * KEY_DIM..(h + 1) * KEY_DIM]
                .iter()
                .map(|&x| x as i64)
                .collect();
            let v_off = KEY_PART + h * VAL_DIM;
            let v_slice: Vec<i64> = doc[v_off..v_off + VAL_DIM]
                .iter()
                .map(|&x| x as i64)
                .collect();

            let raw_k: Vec<i64> = (0..KEY_DIM)
                .map(|i| {
                    self.w_key[h][i * KEY_DIM..(i + 1) * KEY_DIM]
                        .iter()
                        .zip(&k_slice)
                        .map(|(&w, &x)| w as i64 * x)
                        .sum::<i64>()
                        >> 4
                })
                .collect();
            let raw_v: Vec<i64> = (0..VAL_DIM)
                .map(|i| {
                    self.w_val[h][i * KEY_DIM..(i + 1) * KEY_DIM]
                        .iter()
                        .zip(&v_slice)
                        .map(|(&w, &x)| w as i64 * x)
                        .sum::<i64>()
                        >> 8
                })
                .collect();

            keys.push(unit_key(&raw_k));
            vals.push(raw_v.iter().map(|&x| clip16(x)).collect());
        }
        (keys, vals)
    }

    /// Fold one document into the state (flat, head-major, len STATE_PARAMS).
    /// Returns the new state; the input state is left untouched.
    pub fn fold(&self, state: &[i16], doc: &[i16]) -> Vec<i16> {
        assert_eq!(state.len(), STATE_PARAMS, "state must have {} values", STATE_PARAMS);
        let (keys, vals) = self.project(doc);
        let mut tokens: Vec<i64> = state.iter().map(|&x| x as i64).collect();

        for h in 0..N_HEADS {
            let k = &keys[h];
            if k.iter().all(|&x| x == 0) {
                continue; // zero key: nothing addressable, leave the head alone
            }
            let base = h * HEAD_PARAMS;
            // s is row-major [KEY_DIM][VAL_DIM]
            let recalled: Vec<i64> = (0..VAL_DIM)
                .map(|v| {
                    let mut acc = 0i64;
                    for kk in 0..KEY_DIM {
                        acc += k[kk] as i64 * tokens[base + kk * VAL_DIM + v];
                    }
                    acc >> Q_SHIFT
                })
                .collect();
            let d: Vec<i64> = vals[h]
                .iter()
                .zip(&recalled)
                .map(|(&v, &r)| (self.beta * (v as i64 - r)) >> Q_SHIFT)
                .collect();

            let decayed: Vec<i64> = tokens[base..base + HEAD_PARAMS]
                .iter()
                .map(|&x| (self.decay * x) >> Q_SHIFT)
                .collect();

            for kk in 0..KEY_DIM {
                for v in 0..VAL_DIM {
                    let update = (k[kk] as i64 * d[v]) >> Q_SHIFT;
                    tokens[base + kk * VAL_DIM + v] = decayed[kk * VAL_DIM + v] + update;
                }
            }
        }

        tokens.iter().map(|&x| clip16(x)).collect()
    }
}

/// Convenience: zero state.
pub fn zero_state() -> Vec<i16> {
    vec![0i16; STATE_PARAMS]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_is_exactly_22kb() {
        assert_eq!(STATE_BYTES, 22_528);
        assert_eq!(STATE_PARAMS, 11_264);
        assert_eq!(DOC_DIM, 704);
    }

    #[test]
    fn isqrt_matches_reference() {
        for n in [0i64, 1, 2, 3, 4, 15, 16, 17, 16383, 16384, 16385, 1_000_000_000] {
            let r = isqrt(n);
            assert!(r * r <= n, "isqrt({n})={r} too big");
            assert!((r + 1) * (r + 1) > n, "isqrt({n})={r} too small");
        }
        assert_eq!(isqrt(-5), 0);
    }

    #[test]
    fn unit_key_normalizes() {
        let k = vec![3i64, 4, 0];
        let uk = unit_key(&k);
        let norm_sq: i64 = uk.iter().map(|&x| x as i64 * x as i64).sum();
        // Q14 unit norm: sum of squares ~ Q_ONE^2
        assert!((norm_sq - Q_ONE * Q_ONE).abs() < 4 * Q_ONE, "norm_sq={norm_sq}");
        assert_eq!(unit_key(&[0, 0, 0]), vec![0, 0, 0]);
    }

    #[test]
    fn fold_of_zero_doc_is_identity_through_decay() {
        // A doc with zero key part has all-zero keys -> heads untouched.
        let folder = Folder {
            w_key: vec![vec![0i16; KEY_DIM * KEY_DIM]; N_HEADS],
            w_val: vec![vec![0i16; VAL_DIM * KEY_DIM]; N_HEADS],
            beta: (0.75 * Q_ONE as f64) as i64,
            decay: Q_ONE, // decay 1.0: state must be unchanged
        };
        let state = zero_state();
        let doc = vec![0i16; DOC_DIM];
        assert_eq!(folder.fold(&state, &doc), state);
    }
}
