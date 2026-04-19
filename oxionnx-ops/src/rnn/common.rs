//! Shared activation types and mathematical helpers used by all RNN kernels.

// ── Activation ──────────────────────────────────────────────────────────────

/// Supported activation functions for RNN gates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Activation {
    Sigmoid,
    Tanh,
    Relu,
}

impl Activation {
    pub(super) fn apply(self, x: f32) -> f32 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Relu => x.max(0.0),
        }
    }

    pub(super) fn from_name(s: &str) -> Self {
        match s {
            "Sigmoid" => Activation::Sigmoid,
            "Relu" => Activation::Relu,
            _ => Activation::Tanh,
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Compute A @ B^T where A is [m, k] and B is [n, k], result is [m, n].
pub(super) fn matmul_2d_a_bt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for kk in 0..k {
                s += a[i * k + kk] * b[j * k + kk];
            }
            out[i * n + j] = s;
        }
    }
    out
}

/// Check whether processing step `t` is valid for batch element `b`.
///
/// For forward: valid when `t < sequence_lens[b]`.
/// For reverse: the reversed input processes original timestep `(seq_len-1-t)`,
/// which is valid when `(seq_len-1-t) < sequence_lens[b]`, i.e. `t >= seq_len - lens[b]`.
pub(super) fn step_is_valid(
    t: usize,
    b: usize,
    seq_len: usize,
    sequence_lens: Option<&[usize]>,
    is_reverse: bool,
) -> bool {
    match sequence_lens {
        None => true,
        Some(lens) => {
            let len_b = if b < lens.len() { lens[b] } else { seq_len };
            if is_reverse {
                len_b >= seq_len || t >= (seq_len - len_b)
            } else {
                t < len_b
            }
        }
    }
}
