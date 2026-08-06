//! Shared activation types and mathematical helpers used by all RNN kernels.

use oxionnx_core::OnnxError;

// ── Activation ──────────────────────────────────────────────────────────────

/// The activation functions ONNX permits in the `activations` attribute of
/// `RNN`, `GRU` and `LSTM`.
///
/// See <https://onnx.ai/onnx/operators/onnx__LSTM.html> — the list is normative
/// and an activation outside it is a malformed model, not a silently-defaulted
/// `Tanh`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ActivationKind {
    Relu,
    Tanh,
    Sigmoid,
    Affine,
    LeakyRelu,
    ThresholdedRelu,
    ScaledTanh,
    HardSigmoid,
    Elu,
    Softsign,
    Softplus,
}

impl ActivationKind {
    /// `(alpha, beta)` defaults, taken from the corresponding ONNX operators.
    const fn default_params(self) -> (f32, f32) {
        match self {
            ActivationKind::Affine => (1.0, 0.0),
            ActivationKind::LeakyRelu => (0.01, 0.0),
            ActivationKind::ThresholdedRelu => (1.0, 0.0),
            ActivationKind::ScaledTanh => (1.0, 1.0),
            ActivationKind::HardSigmoid => (0.2, 0.5),
            ActivationKind::Elu => (1.0, 0.0),
            _ => (0.0, 0.0),
        }
    }

    /// Parse an ONNX activation name.
    ///
    /// ONNX spells these in PascalCase; any casing is accepted so that exports
    /// written as `"sigmoid"` still load, but a genuinely unknown name is a hard
    /// error rather than a silent fallback.
    fn from_name(name: &str) -> Result<Self, OnnxError> {
        let lowered = name.to_ascii_lowercase();
        let kind = match lowered.as_str() {
            "relu" => ActivationKind::Relu,
            "tanh" => ActivationKind::Tanh,
            "sigmoid" => ActivationKind::Sigmoid,
            "affine" => ActivationKind::Affine,
            "leakyrelu" => ActivationKind::LeakyRelu,
            "thresholdedrelu" => ActivationKind::ThresholdedRelu,
            "scaledtanh" => ActivationKind::ScaledTanh,
            "hardsigmoid" => ActivationKind::HardSigmoid,
            "elu" => ActivationKind::Elu,
            "softsign" => ActivationKind::Softsign,
            "softplus" => ActivationKind::Softplus,
            _ => {
                return Err(OnnxError::Unsupported(format!(
                    "RNN activation '{name}' is not a valid ONNX RNN activation; \
                     expected one of Relu, Tanh, Sigmoid, Affine, LeakyRelu, \
                     ThresholdedRelu, ScaledTanh, HardSigmoid, Elu, Softsign, Softplus"
                )))
            }
        };
        Ok(kind)
    }
}

/// An ONNX RNN activation together with its `alpha` / `beta` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Activation {
    kind: ActivationKind,
    alpha: f32,
    beta: f32,
}

impl Activation {
    /// `Sigmoid` with default parameters (the ONNX default gate activation `f`).
    pub(super) const SIGMOID: Activation = Activation::of(ActivationKind::Sigmoid);
    /// `Tanh` with default parameters (the ONNX default activation `g` / `h`).
    pub(super) const TANH: Activation = Activation::of(ActivationKind::Tanh);

    /// Build an activation with the ONNX default `alpha` / `beta` for its kind.
    const fn of(kind: ActivationKind) -> Self {
        let params = kind.default_params();
        Self {
            kind,
            alpha: params.0,
            beta: params.1,
        }
    }

    /// Parse `name`, overriding `alpha` / `beta` when the model supplied them.
    ///
    /// Returns [`OnnxError::Unsupported`] for a name outside the ONNX RNN
    /// activation list.
    pub(super) fn parse(
        name: &str,
        alpha: Option<f32>,
        beta: Option<f32>,
    ) -> Result<Self, OnnxError> {
        let mut act = Activation::of(ActivationKind::from_name(name)?);
        if let Some(a) = alpha {
            act.alpha = a;
        }
        if let Some(b) = beta {
            act.beta = b;
        }
        Ok(act)
    }

    pub(super) fn apply(self, x: f32) -> f32 {
        match self.kind {
            ActivationKind::Relu => x.max(0.0),
            ActivationKind::Tanh => x.tanh(),
            ActivationKind::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationKind::Affine => self.alpha * x + self.beta,
            ActivationKind::LeakyRelu => {
                if x >= 0.0 {
                    x
                } else {
                    self.alpha * x
                }
            }
            ActivationKind::ThresholdedRelu => {
                if x > self.alpha {
                    x
                } else {
                    0.0
                }
            }
            ActivationKind::ScaledTanh => self.alpha * (self.beta * x).tanh(),
            ActivationKind::HardSigmoid => (self.alpha * x + self.beta).clamp(0.0, 1.0),
            ActivationKind::Elu => {
                if x >= 0.0 {
                    x
                } else {
                    self.alpha * (x.exp() - 1.0)
                }
            }
            ActivationKind::Softsign => x / (1.0 + x.abs()),
            // Numerically stable `ln(exp(x) + 1)`.
            ActivationKind::Softplus => x.max(0.0) + (-x.abs()).exp().ln_1p(),
        }
    }
}

/// Resolve one activation slot from the `activations` / `activation_alpha` /
/// `activation_beta` attribute lists.
///
/// `index` is the flat position of the slot within the attribute lists
/// (`direction_index * activations_per_direction + slot`), which is how ONNX
/// orders them ("the values are consumed in the order of activation functions").
pub(super) fn resolve_activation(
    activations: Option<&[&str]>,
    alphas: &[f32],
    betas: &[f32],
    index: usize,
    default: Activation,
) -> Result<Activation, OnnxError> {
    // A missing slot (short or absent list) keeps the ONNX default for that gate;
    // only a *named* activation outside the ONNX list is an error.
    let Some(name) = activations
        .and_then(|a| a.get(index))
        .filter(|n| !n.is_empty())
    else {
        return Ok(default);
    };
    Activation::parse(name, alphas.get(index).copied(), betas.get(index).copied())
}

// ── Direction ───────────────────────────────────────────────────────────────

/// Validate an ONNX RNN `direction` attribute against the three values the
/// spec permits: `"forward"`, `"reverse"`, `"bidirectional"`
/// (<https://onnx.ai/onnx/operators/onnx__LSTM.html>,
/// `onnx__GRU.html`).
///
/// Every `num_dir` / `is_reverse` computation in `lstm.rs` and `gru.rs` reads
/// `direction` with `direction == "bidirectional"` / `direction == "reverse"`
/// comparisons that treat anything else as `"forward"` -- so an unrecognized
/// string (a typo, wrong case, `"sideways"`) previously ran silently as plain
/// forward instead of reporting an error, the same class of bug
/// [`ActivationKind::from_name`] above exists to prevent for `activations`.
///
/// Call this once, before computing `num_dir`, at `lstm_into_seq_major` /
/// `gru_into_seq_major` -- the single core both kernels' `execute`,
/// `execute_into_slots`, and F16/BF16 typed (`rnn_typed`) dispatch paths all
/// funnel through, regardless of how the registry layer reads the raw
/// attribute string.
pub(super) fn validate_direction(op: &str, direction: &str) -> Result<(), OnnxError> {
    match direction {
        "forward" | "reverse" | "bidirectional" => Ok(()),
        other => Err(OnnxError::Unsupported(format!(
            "{op}: direction '{other}' is not one of \"forward\", \"reverse\", \"bidirectional\""
        ))),
    }
}

// ── Optional ONNX RNN attributes ────────────────────────────────────────────

/// Optional ONNX attributes shared by `RNN`, `GRU` and `LSTM` that the plain
/// kernel entry points leave at their defaults.
///
/// Kept as a separate struct so the historical `lstm()` / `gru()` /
/// `simple_rnn()` signatures stay source-compatible.
#[derive(Clone, Copy, Debug)]
pub struct RnnExtras<'a> {
    /// ONNX `clip`: bounds the *input of every activation* to `[-clip, +clip]`.
    /// `f32::INFINITY` (the default) means "no clip".
    pub clip: f32,
    /// ONNX `layout`: `0` = `[seq, batch, ...]` (default), `1` = `[batch, seq, ...]`.
    pub layout: i64,
    /// ONNX `activation_alpha`, consumed positionally alongside `activations`.
    pub activation_alpha: &'a [f32],
    /// ONNX `activation_beta`, consumed positionally alongside `activations`.
    pub activation_beta: &'a [f32],
}

impl Default for RnnExtras<'_> {
    fn default() -> Self {
        Self {
            clip: f32::INFINITY,
            layout: 0,
            activation_alpha: &[],
            activation_beta: &[],
        }
    }
}

/// Clamp an activation input to `[-clip, +clip]`.
///
/// `clip` is loop-invariant, so the `is_finite` test is hoisted out of the gate
/// loops and the no-clip path (the overwhelmingly common one) costs nothing.
#[inline(always)]
pub(super) fn clip_val(x: f32, clip: f32) -> f32 {
    if clip.is_finite() {
        x.clamp(-clip, clip)
    } else {
        x
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Compute A @ B^T where A is `[m, k]` and B is `[n, k]`, result is `[m, n]`.
///
/// This is the gate projection shared by every RNN kernel: `x_t @ W^T` and
/// `h @ R^T` for LSTM/GRU/RNN, called once per timestep per direction. Both
/// calls already cover *every* gate in one shot -- `W`/`R` are pre-sliced to
/// the full `[gate_count * hidden_size, *]` block before `matmul_2d_a_bt` is
/// called (see `lstm_one_direction`/`gru_one_direction`), so `n` here is
/// already `gate_count * hidden_size`. What was slow was the primitive
/// itself: a hand-rolled scalar triple loop with a serial per-(i,j)
/// accumulator chain that the compiler cannot vectorise.
///
/// Delegates to `matrixmultiply::sgemm`, expressing `B^T` as a stride swap
/// (row-stride 1, column-stride `k`) instead of a physical transpose: sgemm
/// reads `B` exactly as stored, so this is zero-copy on both operands.
/// Measured against the scalar loop above (kept only as documentation of
/// what this replaces -- see the before/after note in the PR/report) at
/// realistic RNN scale (`k`/`n` in the hundreds to low thousands, as
/// `input_size`/`hidden_size` always are in real exported models):
/// 1.3x-1.9x faster even at `m == 1` (a streaming/online-inference batch
/// size, and every existing LSTM/GRU test's batch size), rising to
/// 8x-16x by `m == 8`. The only workload where sgemm measured *slower* was
/// pathologically tiny `k` (single digits) with `m == 1` -- e.g. `k=1`,
/// `n=4`, an order of magnitude below any real `hidden_size`/`input_size` --
/// where sgemm's setup cost isn't amortised; that regression is nanoseconds
/// in absolute terms and only reachable from a toy unit test, never a real
/// model, so there is no size-gated fallback here: every call goes through
/// sgemm.
#[allow(unsafe_code)] // matrixmultiply::sgemm requires unsafe
pub(super) fn matmul_2d_a_bt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    if m == 0 || k == 0 || n == 0 {
        // Nothing to contract; also sidesteps handing sgemm a zero-length
        // dimension. The scalar loop this replaces produces the same
        // all-zero (or zero-length) result here: with `k == 0` every `s`
        // stays `0.0`, and `m == 0` / `n == 0` just make `out` empty.
        return out;
    }
    // Safety: `a` holds exactly `m * k` elements and `b` holds exactly `n *
    // k` elements -- every RNN call site slices `x_t`/`h` and `W`/`R` to
    // precisely those lengths, and `validate_rnn_shapes` (this module)
    // rejects a model whose `W`/`R`/`X`/initial-state tensors are too small
    // before any kernel runs. `out` is freshly allocated to exactly `m * n`
    // elements. Given `rsa=k, csa=1` for `a` (`[m,k]` row-major) and
    // `rsb=1, csb=k` for `b` (reads it as a `[k,n]` matrix, i.e. `B^T`,
    // without moving any bytes), the highest offset sgemm touches in `a` is
    // `(m-1)*k + (k-1) = m*k-1`, in `b` is `(k-1) + (n-1)*k = n*k-1`, and in
    // `out` (`rsc=n, csc=1`) is `(m-1)*n + (n-1) = m*n-1` -- exactly the last
    // valid index of each slice, so every access stays in bounds.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            1,
            k as isize,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
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

// ── Shape validation ────────────────────────────────────────────────────────

/// Per-direction slice sizes shared by the three kernels.
pub(super) struct DirSizes {
    pub w: usize,
    pub r: usize,
    pub b: usize,
    pub h: usize,
}

/// Everything [`validate_rnn_shapes`] needs to bound-check one RNN invocation.
pub(super) struct RnnShapeCheck<'a> {
    /// Operator name, used only in error messages.
    pub op: &'a str,
    pub x: &'a oxionnx_core::Tensor,
    pub w: &'a oxionnx_core::Tensor,
    pub r: &'a oxionnx_core::Tensor,
    pub b: Option<&'a oxionnx_core::Tensor>,
    /// `initial_h` and (for LSTM) `initial_c`.
    pub initial_states: &'a [Option<&'a oxionnx_core::Tensor>],
    pub hidden_size: usize,
    /// Number of gates: 1 for `RNN`, 3 for `GRU`, 4 for `LSTM`.
    pub gates: usize,
    pub num_dir: usize,
}

/// Validate that every model-supplied tensor is large enough for `num_dir`
/// directions, so the kernels can slice without bounds panics.
pub(super) fn validate_rnn_shapes(check: RnnShapeCheck<'_>) -> Result<DirSizes, OnnxError> {
    let RnnShapeCheck {
        op,
        x,
        w,
        r,
        b,
        initial_states,
        hidden_size,
        gates,
        num_dir,
    } = check;
    if x.ndim() != 3 {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: X must be 3D [seq_length, batch_size, input_size], got {:?}",
            x.shape
        )));
    }
    if hidden_size == 0 {
        return Err(OnnxError::InvalidModel(format!(
            "{op}: hidden_size must be greater than 0"
        )));
    }
    let (seq_len, batch, input_size) = (x.shape[0], x.shape[1], x.shape[2]);
    let x_elems = seq_len
        .checked_mul(batch)
        .and_then(|v| v.checked_mul(input_size))
        .ok_or_else(|| OnnxError::ShapeMismatch(format!("{op}: X shape overflows usize")))?;
    if x.data.len() < x_elems {
        return Err(OnnxError::ShapeMismatch(format!(
            "{op}: X holds {} elements but shape {:?} needs {x_elems}",
            x.data.len(),
            x.shape
        )));
    }

    let gate_rows = gates.checked_mul(hidden_size).ok_or_else(|| {
        OnnxError::ShapeMismatch(format!("{op}: gates * hidden_size overflows usize"))
    })?;
    let sizes = DirSizes {
        w: gate_rows
            .checked_mul(input_size)
            .ok_or_else(|| OnnxError::ShapeMismatch(format!("{op}: W size overflows usize")))?,
        r: gate_rows
            .checked_mul(hidden_size)
            .ok_or_else(|| OnnxError::ShapeMismatch(format!("{op}: R size overflows usize")))?,
        b: 2 * gate_rows,
        h: batch.checked_mul(hidden_size).ok_or_else(|| {
            OnnxError::ShapeMismatch(format!("{op}: batch * hidden_size overflows usize"))
        })?,
    };

    let check = |name: &str, have: usize, need: usize| -> Result<(), OnnxError> {
        if have < need {
            Err(OnnxError::ShapeMismatch(format!(
                "{op}: input {name} holds {have} elements but {num_dir} direction(s) \
                 with hidden_size={hidden_size} need {need}"
            )))
        } else {
            Ok(())
        }
    };
    check("W", w.data.len(), num_dir * sizes.w)?;
    check("R", r.data.len(), num_dir * sizes.r)?;
    if let Some(bt) = b {
        check("B", bt.data.len(), num_dir * sizes.b)?;
    }
    for state in initial_states.iter().flatten() {
        check("initial_h/initial_c", state.data.len(), num_dir * sizes.h)?;
    }

    Ok(sizes)
}

/// [W2-perf-misc / a6-12] Correctness of the `matrixmultiply::sgemm`-based
/// `matmul_2d_a_bt`.
///
/// The risk this guards against is specific: `A @ B^T` is expressed as an
/// `sgemm` call with `B`'s transpose folded into a stride swap (`rsb=1,
/// csb=k` instead of a physical transpose). Swap `rsb`/`csb` back
/// (`rsb=k, csb=1`, i.e. accidentally compute `A @ B`) and, for a *square*
/// `k == n` case, the shapes still line up and the call still returns a
/// finite, plausible-looking `[m, n]` result -- just the wrong one. Every
/// case below therefore deliberately uses `m`, `k`, `n` all different from
/// each other, and non-uniform/non-symmetric data (no constant rows, no
/// `A == B`), so a swapped-stride bug cannot hide behind an accidental
/// square-matrix or symmetric-data coincidence.
#[cfg(test)]
mod matmul_2d_a_bt_tests {
    use super::*;

    /// Independent reference: the exact scalar triple loop `matmul_2d_a_bt`
    /// used before this change (not shared with the code under test).
    fn reference_matmul_2d_a_bt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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

    /// Deterministic, non-uniform pseudo-random `f32`s in roughly `[-1, 1]`
    /// (a multiplicative hash of the index, not a linear function of it, so
    /// no row/column is a scalar multiple of another).
    fn det_vals(n: usize, seed: u64) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = (i as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed)
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    >> 40;
                (h % 20_001) as f32 / 10_000.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn matches_reference_across_asymmetric_shapes() {
        // Every (m, k, n) below has three pairwise-distinct values, spanning
        // both `m < k < n` and `m > k`, `k > n`, etc., plus the exact
        // shapes real LSTM/GRU gate projections use (`k`=input_size or
        // hidden_size, `n`=gate_count*hidden_size).
        let shapes: &[(usize, usize, usize)] = &[
            (1, 2, 3),
            (2, 3, 5),
            (3, 5, 2),
            (5, 2, 3),
            (4, 7, 44),  // LSTM x_t @ W^T shape from the end-to-end test below
            (4, 11, 44), // LSTM h @ R^T shape from the end-to-end test below
            (6, 17, 5),
            (9, 3, 40),
            (1, 1, 1),
            (1, 100, 1),
            (1, 1, 100),
        ];
        for &(m, k, n) in shapes {
            let a = det_vals(m * k, 11);
            let b = det_vals(n * k, 97);
            let got = matmul_2d_a_bt(&a, &b, m, k, n);
            let want = reference_matmul_2d_a_bt(&a, &b, m, k, n);
            assert_eq!(got.len(), want.len(), "m={m} k={k} n={n}");
            for (idx, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                // sgemm's blocked accumulation reassociates the sum over `k`
                // relative to the naive left-to-right reference, so this is a
                // tolerance check, not exact equality. Measured max relative
                // error across every shape/index here is ~2e-7 (1-2 ULP of
                // f32, i.e. exactly what reassociating a sub-100-term sum
                // predicts) -- 1e-5 keeps ~50x headroom above that measured
                // value for platform-dependent SIMD-width/blocking
                // differences in `matrixmultiply`, while still being far
                // tighter than a transposition/stride bug could hide under
                // (that shows up as a completely different value, not a
                // rounding-level difference). This meets the brief's stated
                // "parity within 1e-5" contract with margin, not just barely.
                let diff = (g - w).abs();
                let scale = w.abs().max(1.0);
                assert!(
                    diff <= 1e-5 * scale,
                    "m={m} k={k} n={n} idx={idx}: got {g}, want {w} (diff {diff})"
                );
            }
        }
    }

    #[test]
    fn zero_sized_dims_return_empty_or_zero_without_panic() {
        for &(m, k, n) in &[(0usize, 5, 5), (5, 0, 5), (5, 5, 0), (0, 0, 0)] {
            let a = vec![0.0f32; m * k];
            let b = vec![0.0f32; n * k];
            let out = matmul_2d_a_bt(&a, &b, m, k, n);
            assert_eq!(out.len(), m * n);
            assert!(out.iter().all(|&v| v == 0.0));
        }
    }

    /// End-to-end: drive the sgemm-based projection through the full LSTM
    /// kernel (not just the isolated primitive) with `batch = 5` (`>= 4`,
    /// so this is genuinely exercising `sgemm`, not a degenerate case) and
    /// `input_size = 7`, `hidden_size = 11` (so `4*hidden_size = 44`) --
    /// three pairwise-distinct sizes for both gate projections
    /// (`x_t @ W^T` is `[5,7]x[44,7]^T`, `h @ R^T` is `[5,11]x[44,11]^T`).
    /// Reference computed independently in NumPy/float64 (ONNX gate order
    /// i, o, f, c; no peephole; default Sigmoid/Tanh/Tanh activations):
    ///
    /// ```python
    /// import numpy as np
    /// def sigmoid(x): return 1.0/(1.0+np.exp(-x))
    /// H = np.zeros((batch, hidden_size)); C = np.zeros((batch, hidden_size))
    /// Wi,Wo,Wf,Wc = np.split(W, 4); Ri,Ro,Rf,Rc = np.split(R, 4)
    /// Wbi,Wbo,Wbf,Wbc = np.split(Wb, 4); Rbi,Rbo,Rbf,Rbc = np.split(Rb, 4)
    /// for t in range(seq_len):
    ///     Xt = X[t]
    ///     it = sigmoid(Xt@Wi.T + H@Ri.T + Wbi + Rbi)
    ///     ft = sigmoid(Xt@Wf.T + H@Rf.T + Wbf + Rbf)
    ///     ct_cand = np.tanh(Xt@Wc.T + H@Rc.T + Wbc + Rbc)
    ///     C = ft*C + it*ct_cand
    ///     ot = sigmoid(Xt@Wo.T + H@Ro.T + Wbo + Rbo)
    ///     H = ot*np.tanh(C)
    /// ```
    #[test]
    fn lstm_end_to_end_matches_numpy_reference_asymmetric_batch5() {
        let batch = 5usize;
        let input_size = 7usize;
        let hidden_size = 11usize;
        let seq_len = 3usize;
        let gate4 = 4 * hidden_size;

        let x_data: Vec<f32> = vec![
            -1.496702,
            1.190017,
            0.8767363,
            0.5634556,
            0.2501749,
            -0.06310583,
            -0.3763865,
            -0.6896672,
            -1.002948,
            -1.316229,
            1.370491,
            1.05721,
            0.7439292,
            0.4306485,
            0.1173678,
            -0.1959129,
            -0.5091936,
            -0.8224743,
            -1.135755,
            -1.449036,
            1.237684,
            0.9244029,
            0.6111222,
            0.2978415,
            -0.01543919,
            -0.3287199,
            -0.6420006,
            -0.9552813,
            -1.268562,
            1.418157,
            1.104877,
            0.7915959,
            0.4783152,
            0.1650345,
            -0.1482462,
            -0.4615269,
            -0.7748076,
            -1.088088,
            -1.401369,
            1.28535,
            0.9720696,
            0.6587888,
            0.3455081,
            0.03222744,
            -0.2810533,
            -0.594334,
            -0.9076147,
            -1.220895,
            1.465824,
            1.152543,
            0.8392625,
            0.5259818,
            0.2127011,
            -0.1005796,
            -0.4138603,
            -0.727141,
            -1.040422,
            -1.353702,
            1.333017,
            1.019736,
            0.7064555,
            0.3931748,
            0.07989407,
            -0.2333866,
            -0.5466673,
            -0.859948,
            -1.173229,
            -1.486509,
            1.20021,
            0.8869291,
            0.5736484,
            0.2603677,
            -0.05291296,
            -0.3661937,
            -0.6794744,
            -0.9927551,
            -1.306036,
            1.380684,
            1.067403,
            0.7541221,
            0.4408414,
            0.1275607,
            -0.18572,
            -0.4990007,
            -0.8122814,
            -1.125562,
            -1.438843,
            1.247876,
            0.9345958,
            0.6213151,
            0.3080344,
            -0.005246328,
            -0.318527,
            -0.6318077,
            -0.9450884,
            -1.258369,
            1.42835,
            1.115069,
            0.8017887,
            0.488508,
            0.1752273,
            -0.1380534,
            -0.4513341,
            -0.7646148,
            -1.077895,
        ];
        let w_data: Vec<f32> = vec![
            -0.5973618,
            0.4773259,
            0.3520136,
            0.2267013,
            0.101389,
            -0.02392325,
            -0.1492355,
            -0.2745478,
            -0.3998601,
            -0.5251724,
            0.5495153,
            0.4242031,
            0.2988908,
            0.1735785,
            0.04826621,
            -0.07704607,
            -0.2023583,
            -0.3276706,
            -0.4529829,
            -0.5782952,
            0.4963925,
            0.3710802,
            0.245768,
            0.1204557,
            -0.0048566,
            -0.1301689,
            -0.2554812,
            -0.3807934,
            -0.5061057,
            0.568582,
            0.4432697,
            0.3179574,
            0.1926451,
            0.06733287,
            -0.05797941,
            -0.1832917,
            -0.308604,
            -0.4339163,
            -0.5592285,
            0.5154592,
            0.3901469,
            0.2648346,
            0.1395223,
            0.01421005,
            -0.1111022,
            -0.2364145,
            -0.3617268,
            -0.4870391,
            0.5876486,
            0.4623364,
            0.3370241,
            0.2117118,
            0.08639952,
            -0.03891276,
            -0.164225,
            -0.2895373,
            -0.4148496,
            -0.5401619,
            0.5345258,
            0.4092136,
            0.2839013,
            0.158589,
            0.03327671,
            -0.09203558,
            -0.2173479,
            -0.3426601,
            -0.4679724,
            -0.5932847,
            0.481403,
            0.3560907,
            0.2307785,
            0.1054662,
            -0.01984611,
            -0.1451584,
            -0.2704707,
            -0.395783,
            -0.5210952,
            0.5535925,
            0.4282802,
            0.3029679,
            0.1776556,
            0.05234336,
            -0.07296892,
            -0.1982812,
            -0.3235935,
            -0.4489058,
            -0.574218,
            0.5004697,
            0.3751574,
            0.2498451,
            0.1245328,
            -0.0007794544,
            -0.1260917,
            -0.251404,
            -0.3767163,
            -0.5020286,
            0.5726591,
            0.4473469,
            0.3220346,
            0.1967223,
            0.07141001,
            -0.05390227,
            -0.1792145,
            -0.3045268,
            -0.4298391,
            -0.5551514,
            0.5195363,
            0.394224,
            0.2689118,
            0.1435995,
            0.0182872,
            -0.1070251,
            -0.2323374,
            -0.3576496,
            -0.4829619,
            0.5917258,
            0.4664135,
            0.3411012,
            0.2157889,
            0.09047667,
            -0.03483562,
            -0.1601479,
            -0.2854602,
            -0.4107725,
            -0.5360847,
            0.538603,
            0.4132907,
            0.2879784,
            0.1626661,
            0.03735385,
            -0.08795843,
            -0.2132707,
            -0.338583,
            -0.4638953,
            -0.5892076,
            0.4854802,
            0.3601679,
            0.2348556,
            0.1095433,
            -0.01576896,
            -0.1410812,
            -0.2663935,
            -0.3917058,
            -0.5170181,
            0.5576696,
            0.4323573,
            0.3070451,
            0.1817328,
            0.05642051,
            -0.06889178,
            -0.1942041,
            -0.3195163,
            -0.4448286,
            -0.5701409,
            0.5045468,
            0.3792345,
            0.2539223,
            0.12861,
            0.003297692,
            -0.1220146,
            -0.2473269,
            -0.3726392,
            -0.4979514,
            0.5767363,
            0.451424,
            0.3261117,
            0.2007994,
            0.07548716,
            -0.04982512,
            -0.1751374,
            -0.3004497,
            -0.425762,
            -0.5510742,
            0.5236135,
            0.3983012,
            0.2729889,
            0.1476766,
            0.02236434,
            -0.1029479,
            -0.2282602,
            -0.3535725,
            -0.4788848,
            0.5958029,
            0.4704907,
            0.3451784,
            0.2198661,
            0.09455381,
            -0.03075847,
            -0.1560708,
            -0.281383,
            -0.4066953,
            -0.5320076,
            0.5426801,
            0.4173678,
            0.2920556,
            0.1667433,
            0.041431,
            -0.08388128,
            -0.2091936,
            -0.3345058,
            -0.4598181,
            -0.5851304,
            0.4895573,
            0.364245,
            0.2389327,
            0.1136205,
            -0.01169182,
            -0.1370041,
            -0.2623164,
            -0.3876287,
            -0.5129409,
            0.5617468,
            0.4364345,
            0.3111222,
            0.1858099,
            0.06049765,
            -0.06481463,
            -0.1901269,
            -0.3154392,
            -0.4407515,
            -0.5660638,
            0.508624,
            0.3833117,
            0.2579994,
            0.1326871,
            0.007374838,
            -0.1179374,
            -0.2432497,
            -0.368562,
            -0.4938743,
            0.5808134,
            0.4555011,
            0.3301889,
            0.2048766,
            0.0795643,
            -0.04574798,
            -0.1710603,
            -0.2963725,
            -0.4216848,
            -0.5469971,
            0.5276906,
            0.4023783,
            0.2770661,
            0.1517538,
            0.02644149,
            -0.09887079,
            -0.2241831,
            -0.3494954,
            -0.4748076,
            0.5998801,
            0.4745678,
            0.3492555,
            0.2239432,
            0.09863096,
            -0.02668132,
            -0.1519936,
            -0.2773059,
            -0.4026182,
            -0.5279304,
            0.5467573,
            0.421445,
            0.2961327,
            0.1708204,
            0.04550814,
            -0.07980414,
            -0.2051164,
            -0.3304287,
            -0.455741,
            -0.5810533,
            0.4936345,
            0.3683222,
            0.2430099,
            0.1176976,
            -0.00761467,
            -0.132927,
            -0.2582392,
            -0.3835515,
            -0.5088638,
            0.5658239,
            0.4405116,
            0.3151994,
            0.1898871,
            0.0645748,
            -0.06073748,
            -0.1860498,
            -0.311362,
            -0.4366743,
            -0.5619866,
            0.5127011,
            0.3873888,
            0.2620765,
            0.1367643,
            0.01145198,
            -0.1138603,
            -0.2391726,
            -0.3644849,
            -0.4897971,
            0.5848906,
            0.4595783,
            0.334266,
            0.2089537,
            0.08364145,
            -0.04167083,
            -0.1669831,
            -0.2922954,
            -0.4176077,
            -0.54292,
            0.5317678,
        ];
        let r_data: Vec<f32> = vec![
            -0.5960428,
            0.4786449,
            0.3533327,
            0.2280204,
            0.1027081,
            -0.02260418,
            -0.1479165,
            -0.2732287,
            -0.398541,
            -0.5238533,
            0.5508344,
            0.4255221,
            0.3002099,
            0.1748976,
            0.04958529,
            -0.07572699,
            -0.2010393,
            -0.3263516,
            -0.4516638,
            -0.5769761,
            0.4977116,
            0.3723993,
            0.247087,
            0.1217748,
            -0.003537524,
            -0.1288498,
            -0.2541621,
            -0.3794744,
            -0.5047866,
            0.5699011,
            0.4445888,
            0.3192765,
            0.1939642,
            0.06865194,
            -0.05666034,
            -0.1819726,
            -0.3072849,
            -0.4325972,
            -0.5579095,
            0.5167783,
            0.391466,
            0.2661537,
            0.1408414,
            0.01552913,
            -0.1097832,
            -0.2350954,
            -0.3604077,
            -0.48572,
            0.5889677,
            0.4636554,
            0.3383432,
            0.2130309,
            0.0877186,
            -0.03759368,
            -0.162906,
            -0.2882182,
            -0.4135305,
            -0.5388428,
            0.5358449,
            0.4105326,
            0.2852203,
            0.1599081,
            0.03459578,
            -0.0907165,
            -0.2160288,
            -0.3413411,
            -0.4666533,
            -0.5919656,
            0.4827221,
            0.3574098,
            0.2320975,
            0.1067853,
            -0.01852703,
            -0.1438393,
            -0.2691516,
            -0.3944639,
            -0.5197762,
            0.5549116,
            0.4295993,
            0.304287,
            0.1789747,
            0.05366244,
            -0.07164985,
            -0.1969621,
            -0.3222744,
            -0.4475867,
            -0.572899,
            0.5017887,
            0.3764765,
            0.2511642,
            0.1258519,
            0.0005396223,
            -0.1247727,
            -0.2500849,
            -0.3753972,
            -0.5007095,
            0.5739782,
            0.4486659,
            0.3233537,
            0.1980414,
            0.07272909,
            -0.05258319,
            -0.1778955,
            -0.3032078,
            -0.42852,
            -0.5538323,
            0.5208554,
            0.3955431,
            0.2702308,
            0.1449186,
            0.01960628,
            -0.105706,
            -0.2310183,
            -0.3563306,
            -0.4816429,
            0.5930449,
            0.4677326,
            0.3424203,
            0.217108,
            0.09179574,
            -0.03351654,
            -0.1588288,
            -0.2841411,
            -0.4094534,
            -0.5347657,
            0.5399221,
            0.4146098,
            0.2892975,
            0.1639852,
            0.03867293,
            -0.08663935,
            -0.2119516,
            -0.3372639,
            -0.4625762,
            -0.5878885,
            0.4867992,
            0.361487,
            0.2361747,
            0.1108624,
            -0.01444989,
            -0.1397622,
            -0.2650744,
            -0.3903867,
            -0.515699,
            0.5589887,
            0.4336764,
            0.3083641,
            0.1830519,
            0.05773958,
            -0.0675727,
            -0.192885,
            -0.3181973,
            -0.4435095,
            -0.5688218,
            0.5058659,
            0.3805536,
            0.2552413,
            0.129929,
            0.004616768,
            -0.1206955,
            -0.2460078,
            -0.3713201,
            -0.4966324,
            0.5780554,
            0.4527431,
            0.3274308,
            0.2021185,
            0.07680624,
            -0.04850605,
            -0.1738183,
            -0.2991306,
            -0.4244429,
            -0.5497552,
            0.5249325,
            0.3996203,
            0.274308,
            0.1489957,
            0.02368342,
            -0.1016289,
            -0.2269411,
            -0.3522534,
            -0.4775657,
            0.597122,
            0.4718097,
            0.3464975,
            0.2211852,
            0.09587289,
            -0.02943939,
            -0.1547517,
            -0.280064,
            -0.4053762,
            -0.5306885,
            0.5439992,
            0.4186869,
            0.2933746,
            0.1680624,
            0.04275007,
            -0.08256221,
            -0.2078745,
            -0.3331868,
            -0.4584991,
            -0.5838113,
            0.4908764,
            0.3655641,
            0.2402518,
            0.1149395,
            -0.01037274,
            -0.135685,
            -0.2609973,
            -0.3863096,
            -0.5116219,
            0.5630659,
            0.4377536,
            0.3124413,
            0.187129,
            0.06181673,
            -0.06349555,
            -0.1888078,
            -0.3141201,
            -0.4394324,
            -0.5647447,
            0.509943,
            0.3846308,
            0.2593185,
            0.1340062,
            0.008693914,
            -0.1166184,
            -0.2419306,
            -0.3672429,
            -0.4925552,
            0.5821325,
            0.4568202,
            0.3315079,
            0.2061957,
            0.08088338,
            -0.0444289,
            -0.1697412,
            -0.2950535,
            -0.4203657,
            -0.545678,
            0.5290097,
            0.4036974,
            0.2783851,
            0.1530728,
            0.02776057,
            -0.09755171,
            -0.222864,
            -0.3481763,
            -0.4734886,
            -0.5988008,
            0.4758869,
            0.3505746,
            0.2252623,
            0.09995003,
            -0.02536225,
            -0.1506745,
            -0.2759868,
            -0.4012991,
            -0.5266114,
            0.5480763,
            0.4227641,
            0.2974518,
            0.1721395,
            0.04682722,
            -0.07848506,
            -0.2037973,
            -0.3291096,
            -0.4544219,
            -0.5797342,
            0.4949535,
            0.3696413,
            0.244329,
            0.1190167,
            -0.006295593,
            -0.1316079,
            -0.2569202,
            -0.3822324,
            -0.5075447,
            0.567143,
            0.4418307,
            0.3165184,
            0.1912062,
            0.06589387,
            -0.05941841,
            -0.1847307,
            -0.310043,
            -0.4353553,
            -0.5606675,
            0.5140202,
            0.3887079,
            0.2633956,
            0.1380833,
            0.01277106,
            -0.1125412,
            -0.2378535,
            -0.3631658,
            -0.4884781,
            0.5862097,
            0.4608974,
            0.3355851,
            0.2102728,
            0.08496053,
            -0.04035175,
            -0.165664,
            -0.2909763,
            -0.4162886,
            -0.5416009,
            0.5330868,
            0.4077746,
            0.2824623,
            0.15715,
            0.03183771,
            -0.09347457,
            -0.2187868,
            -0.3440991,
            -0.4694114,
            -0.5947237,
            0.479964,
            0.3546517,
            0.2293395,
            0.1040272,
            -0.0212851,
            -0.1465974,
            -0.2719097,
            -0.3972219,
            -0.5225342,
            0.5521535,
            0.4268412,
            0.3015289,
            0.1762166,
            0.05090437,
            -0.07440791,
            -0.1997202,
            -0.3250325,
            -0.4503448,
            -0.575657,
            0.4990307,
            0.3737184,
            0.2484061,
            0.1230938,
            -0.002218447,
            -0.1275307,
            -0.252843,
            -0.3781553,
            -0.5034676,
            0.5712201,
            0.4459079,
            0.3205956,
            0.1952833,
            0.06997102,
            -0.05534126,
            -0.1806535,
            -0.3059658,
            -0.4312781,
            -0.5565904,
            0.5180973,
            0.3927851,
            0.2674728,
            0.1421605,
            0.01684821,
            -0.1084641,
            -0.2337764,
            -0.3590886,
            -0.4844009,
            0.5902868,
            0.4649745,
            0.3396622,
            0.21435,
            0.08903767,
            -0.03627461,
            -0.1615869,
            -0.2868992,
            -0.4122115,
            -0.5375237,
            0.537164,
            0.4118517,
            0.2865394,
            0.1612271,
            0.03591486,
            -0.08939742,
            -0.2147097,
            -0.340022,
            -0.4653343,
            -0.5906465,
            0.4840412,
            0.3587289,
            0.2334166,
            0.1081043,
            -0.01720795,
            -0.1425202,
            -0.2678325,
            -0.3931448,
            -0.5184571,
            0.5562306,
            0.4309184,
            0.3056061,
            0.1802938,
            0.05498151,
            -0.07033077,
            -0.195643,
            -0.3209553,
            -0.4462676,
            -0.5715799,
            0.5031078,
            0.3777955,
            0.2524833,
            0.127171,
            0.001858699,
            -0.1234536,
            -0.2487659,
            -0.3740781,
            -0.4993904,
            0.5752973,
            0.449985,
            0.3246727,
            0.1993604,
            0.07404817,
            -0.05126412,
            -0.1765764,
            -0.3018887,
            -0.427201,
            -0.5525132,
            0.5221745,
            0.3968622,
            0.2715499,
            0.1462376,
            0.02092535,
            -0.1043869,
            -0.2296992,
            -0.3550115,
            -0.4803238,
            0.5943639,
            0.4690517,
            0.3437394,
            0.2184271,
            0.09311482,
            -0.03219746,
            -0.1575097,
            -0.282822,
            -0.4081343,
            -0.5334466,
            0.5412411,
            0.4159288,
            0.2906166,
            0.1653043,
            0.03999201,
            -0.08532028,
            -0.2106326,
            -0.3359448,
            -0.4612571,
            -0.5865694,
            0.4881183,
            0.362806,
            0.2374938,
            0.1121815,
            -0.01313081,
            -0.1384431,
            -0.2637554,
            -0.3890677,
            -0.5143799,
            0.5603078,
            0.4349955,
            0.3096832,
            0.1843709,
            0.05905866,
            -0.06625362,
            -0.1915659,
            -0.3168782,
            -0.4421905,
            -0.5675027,
            0.507185,
            0.3818727,
            0.2565604,
            0.1312481,
            0.005935845,
            -0.1193764,
            -0.2446887,
            -0.370001,
            -0.4953133,
            0.5793744,
            0.4540622,
            0.3287499,
            0.2034376,
            0.07812531,
        ];
        let wb_data: Vec<f32> = vec![
            -0.2973618,
            0.239982,
            0.1773259,
            0.1146697,
            0.05201359,
            -0.01064255,
            -0.07329869,
            -0.1359548,
            -0.198611,
            -0.2612671,
            0.2760767,
            0.2134206,
            0.1507645,
            0.08810832,
            0.02545218,
            -0.03720396,
            -0.0998601,
            -0.1625162,
            -0.2251724,
            -0.2878285,
            0.2495153,
            0.1868592,
            0.1242031,
            0.06154692,
            -0.001109224,
            -0.06376536,
            -0.1264215,
            -0.1890776,
            -0.2517338,
            0.2856101,
            0.2229539,
            0.1602978,
            0.09764165,
            0.03498551,
            -0.02767063,
            -0.09032677,
            -0.1529829,
            -0.2156391,
            -0.2782952,
            0.2590487,
            0.1963925,
            0.1337364,
            0.07108024,
            0.008424103,
        ];
        let rb_data: Vec<f32> = vec![
            -0.2967023,
            0.2406416,
            0.1779854,
            0.1153293,
            0.05267313,
            -0.009983012,
            -0.07263915,
            -0.1352953,
            -0.1979514,
            -0.2606076,
            0.2767363,
            0.2140801,
            0.151424,
            0.08876786,
            0.02611172,
            -0.03654442,
            -0.09920056,
            -0.1618567,
            -0.2245128,
            -0.287169,
            0.2501749,
            0.1875187,
            0.1248626,
            0.06220646,
            -0.0004496852,
            -0.06310583,
            -0.125762,
            -0.1884181,
            -0.2510742,
            0.2862696,
            0.2236135,
            0.1609573,
            0.09830119,
            0.03564505,
            -0.02701109,
            -0.08966723,
            -0.1523234,
            -0.2149795,
            -0.2776357,
            0.2597082,
            0.1970521,
            0.1343959,
            0.07173978,
            0.009083641,
        ];
        assert_eq!(x_data.len(), seq_len * batch * input_size);
        assert_eq!(w_data.len(), gate4 * input_size);
        assert_eq!(r_data.len(), gate4 * hidden_size);
        assert_eq!(wb_data.len(), gate4);
        assert_eq!(rb_data.len(), gate4);

        let x = oxionnx_core::Tensor::new(x_data, vec![seq_len, batch, input_size]);
        let w = oxionnx_core::Tensor::new(w_data, vec![1, gate4, input_size]);
        let r = oxionnx_core::Tensor::new(r_data, vec![1, gate4, hidden_size]);
        let b_data: Vec<f32> = wb_data.into_iter().chain(rb_data).collect();
        let b = oxionnx_core::Tensor::new(b_data, vec![1, 2 * gate4]);

        let (_y, y_h, y_c) = super::super::lstm::lstm(
            &x,
            &w,
            &r,
            Some(&b),
            None,
            None,
            None,
            None,
            hidden_size,
            "forward",
            None,
        )
        .expect("lstm failed");

        assert_eq!(y_h.shape, vec![1, batch, hidden_size]);
        assert_eq!(y_c.shape, vec![1, batch, hidden_size]);

        let want_y_h: Vec<f32> = vec![
            0.08396056,
            -0.07528658,
            -0.1399045,
            0.3223879,
            0.007571316,
            -0.02749948,
            0.0504562,
            0.2779833,
            0.01205736,
            7.89034e-05,
            0.2154894,
            0.2480232,
            -0.07472045,
            -0.09932221,
            -0.04501352,
            0.2049515,
            -0.0719112,
            -0.000470815,
            0.4323986,
            0.01274286,
            -0.0467005,
            0.1967289,
            -0.06351372,
            0.4658329,
            0.03104618,
            -0.04647236,
            -0.1383195,
            0.4179056,
            0.02786143,
            0.01843909,
            -0.001043038,
            0.1938661,
            0.02137832,
            -0.06491567,
            -0.08595544,
            0.4018675,
            -0.08009584,
            -0.09632344,
            -0.182788,
            0.3068817,
            0.00374079,
            0.01031929,
            0.3744507,
            0.08544672,
            0.1714861,
            -0.08519244,
            -0.1249437,
            0.2910502,
            0.03256371,
            -0.05953014,
            0.08018234,
            0.323088,
            0.01432208,
            -0.03711265,
            0.3374503,
        ];
        let want_y_c: Vec<f32> = vec![
            0.1139995,
            -0.312816,
            -0.4229896,
            0.4102926,
            0.01720661,
            -0.15033,
            0.09364367,
            0.4851436,
            0.06015265,
            0.0002621345,
            0.2437337,
            0.2814043,
            -0.2823884,
            -0.3918282,
            -0.06383679,
            0.4367546,
            -0.2141876,
            -0.001159514,
            0.7023081,
            0.06871039,
            -0.1323966,
            0.2990523,
            -0.2066092,
            0.5310664,
            0.06777529,
            -0.1818672,
            -0.3773246,
            0.4780698,
            0.1131903,
            0.1178773,
            -0.00177672,
            0.2324255,
            0.06638879,
            -0.1862834,
            -0.1672248,
            0.4576156,
            -0.2817016,
            -0.2918049,
            -0.3660789,
            0.502082,
            0.01930496,
            0.06157491,
            0.495606,
            0.2309162,
            0.2243108,
            -0.3679764,
            -0.3907806,
            0.3883339,
            0.0714515,
            -0.2458716,
            0.1810791,
            0.5790006,
            0.06400839,
            -0.1117885,
            0.4196225,
        ];

        // Measured max delta here (recurrent over 3 timesteps, batch=5,
        // input=7, hidden=11, through sigmoid/tanh) is ~2.4e-7 -- again ~1-2
        // ULP of f32, comfortably inside the brief's 1e-5 parity requirement
        // with ~40x headroom for cross-platform libm/SIMD differences.
        for (i, (&g, &w)) in y_h.data.iter().zip(want_y_h.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-5,
                "Y_h[{i}]: got {g}, want {w} (delta {})",
                (g - w).abs()
            );
        }
        for (i, (&g, &w)) in y_c.data.iter().zip(want_y_c.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-5,
                "Y_c[{i}]: got {g}, want {w} (delta {})",
                (g - w).abs()
            );
        }
    }
}
