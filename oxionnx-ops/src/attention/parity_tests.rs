//! Numerical parity tests for the sgemm/rayon attention kernels.
//!
//! Two kinds of check:
//!
//! 1. **numpy references** — constants produced by `python3`/NumPy in float64
//!    and rounded to f32 (the generating snippet is quoted above each block),
//!    pinning the kernels to the mathematical definition rather than to a
//!    previous Rust implementation.
//! 2. **path parity** — the parallel `(batch × head)` path, the `Q_BLOCK`
//!    query tiling and the `sgemm` path must agree with the serial /
//!    single-row / scalar paths that the same code falls back to for small
//!    shapes.
//!
//! Tolerance is `1e-5` relative-and-absolute (`|a-b| <= tol * (1 + |b|)`).
//! Routing the attention matmuls through `matrixmultiply::sgemm` and reducing
//! over `(batch, head)` in parallel reassociates floating-point additions, so
//! bit-exactness is not expected (and was never guaranteed across the
//! `simd` / non-`simd` builds either).

// The reference constants below are pasted verbatim from NumPy's float64
// print-out so they can be diffed against the generating snippet above each
// block; keeping digits f32 cannot represent is deliberate.
#![allow(clippy::excessive_precision)]

use super::core::{rotary_embedding, sdpa_into, sdpa_output_shape};
use super::variants::grouped_query_attention;
use super::{multi_head_attention, scaled_dot_product_attention};
use oxionnx_core::Tensor;

/// `sin(i * a + b)` in f64 then narrowed — identical to the NumPy generator
/// used to produce the reference constants below.
fn gen(n: usize, a: f64, b: f64) -> Vec<f32> {
    (0..n).map(|i| ((i as f64) * a + b).sin() as f32).collect()
}

fn tensor(shape: &[usize], a: f64, b: f64) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::new(gen(n, a, b), shape.to_vec())
}

#[track_caller]
fn assert_all_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= tol * (1.0 + w.abs()),
            "{label}: idx {i}: got {g}, want {w} (diff {})",
            (g - w).abs()
        );
    }
}

const TOL: f32 = 1e-5;

// ── SDPA vs NumPy ────────────────────────────────────────────────────────────

/// ```text
/// q = sin(arange(1*2*5*4)*0.31 + 0.10).reshape(1,2,5,4)
/// k = sin(arange(1*2*5*4)*0.17 + 0.20).reshape(1,2,5,4)
/// v = sin(arange(1*2*5*4)*0.23 + 0.30).reshape(1,2,5,4)
/// softmax((q @ k.transpose(0,1,3,2)) / sqrt(4)) @ v
/// ```
#[test]
fn sdpa_matches_numpy() {
    let q = tensor(&[1, 2, 5, 4], 0.31, 0.10);
    let k = tensor(&[1, 2, 5, 4], 0.17, 0.20);
    let v = tensor(&[1, 2, 5, 4], 0.23, 0.30);
    let want = [
        4.86299187e-01,
        4.46433246e-01,
        3.83054972e-01,
        2.99502224e-01,
        5.75431049e-01,
        5.20099819e-01,
        4.37376410e-01,
        3.31617594e-01,
        2.87236661e-01,
        2.25947455e-01,
        1.52758271e-01,
        7.15237111e-02,
        -1.25917524e-01,
        -1.92261890e-01,
        -2.48480335e-01,
        -2.91612029e-01,
        2.24884730e-02,
        -2.29210090e-02,
        -6.71232790e-02,
        -1.07790358e-01,
        3.72679085e-01,
        3.71008843e-01,
        3.49798590e-01,
        3.10165435e-01,
        5.08873880e-01,
        4.45162445e-01,
        3.58005524e-01,
        2.51993507e-01,
        1.50087371e-01,
        1.99713081e-01,
        2.38820449e-01,
        2.65349805e-01,
        -8.81438404e-02,
        5.00487909e-02,
        1.85605496e-01,
        3.11386883e-01,
        -9.25656036e-03,
        1.19257852e-01,
        2.41491273e-01,
        3.51006031e-01,
    ];
    let out = scaled_dot_product_attention(&q, &k, &v, None, None).expect("sdpa");
    assert_eq!(out.shape, vec![1, 2, 5, 4]);
    assert_all_close(&out.data, &want, TOL, "sdpa vs numpy");
}

/// Same inputs as [`sdpa_matches_numpy`] plus a `triu(-inf, 1)` additive mask.
#[test]
fn sdpa_causal_matches_numpy() {
    let q = tensor(&[1, 2, 5, 4], 0.31, 0.10);
    let k = tensor(&[1, 2, 5, 4], 0.17, 0.20);
    let v = tensor(&[1, 2, 5, 4], 0.23, 0.30);
    let mut mask_data = vec![0.0f32; 25];
    for i in 0..5 {
        for j in 0..5 {
            if j > i {
                mask_data[i * 5 + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Tensor::new(mask_data, vec![5, 5]);
    let want = [
        2.95520216e-01,
        5.05533338e-01,
        6.88921452e-01,
        8.36025953e-01,
        7.46950448e-01,
        8.47259164e-01,
        9.02945161e-01,
        9.11075473e-01,
        7.15184808e-01,
        7.40857363e-01,
        7.27511048e-01,
        6.75848722e-01,
        4.27428424e-01,
        4.20669615e-01,
        3.91755342e-01,
        3.42208356e-01,
        2.24884730e-02,
        -2.29210090e-02,
        -6.71232790e-02,
        -1.07790358e-01,
        -9.82452631e-01,
        -9.14060473e-01,
        -7.97527313e-01,
        -6.38990581e-01,
        -7.94378817e-01,
        -6.74257398e-01,
        -5.18624663e-01,
        -3.35677445e-01,
        -3.81487221e-01,
        -2.28643909e-01,
        -6.37585744e-02,
        1.04484767e-01,
        -1.24325261e-01,
        2.72515193e-02,
        1.77393034e-01,
        3.18191767e-01,
        -9.25656036e-03,
        1.19257852e-01,
        2.41491273e-01,
        3.51006031e-01,
    ];
    let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None).expect("sdpa causal");
    assert_all_close(&out.data, &want, TOL, "sdpa causal vs numpy");
}

// ── MHA vs NumPy ─────────────────────────────────────────────────────────────

/// Packed QKV + bias, out-projection + bias; batch 1, seq 6, embed 8, 2 heads.
///
/// ```text
/// x  = sin(arange(48)*0.19 + 0.05).reshape(1,6,8)
/// w  = sin(arange(192)*0.07 + 0.11).reshape(24,8)
/// b  = sin(arange(24)*0.13 + 0.17)
/// ow = sin(arange(64)*0.29 + 0.23).reshape(8,8)
/// ob = sin(arange(8)*0.37 + 0.41)
/// ```
#[test]
fn mha_with_projections_matches_numpy() {
    let x = tensor(&[1, 6, 8], 0.19, 0.05);
    let w = tensor(&[24, 8], 0.07, 0.11);
    let bias = tensor(&[24], 0.13, 0.17);
    let ow = tensor(&[8, 8], 0.29, 0.23);
    let ob = tensor(&[8], 0.37, 0.41);
    let want = [
        1.73863087e+01,
        -1.80983758e+00,
        -1.26517897e+01,
        2.19882660e+01,
        -1.40760803e+01,
        2.48698056e-01,
        1.62275925e+01,
        -2.07727737e+01,
        1.67854042e+01,
        -1.74086809e+00,
        -1.21448298e+01,
        2.12287598e+01,
        -1.35485086e+01,
        2.89592206e-01,
        1.56443176e+01,
        -2.00191841e+01,
        -1.30081215e+01,
        3.30199790e+00,
        1.07797499e+01,
        -1.50399466e+01,
        1.29289417e+01,
        4.93050575e-01,
        -1.11102667e+01,
        1.62200909e+01,
        -1.30415773e+01,
        3.31035805e+00,
        1.08018179e+01,
        -1.50783653e+01,
        1.29592047e+01,
        4.90247846e-01,
        -1.11367121e+01,
        1.62589149e+01,
        1.71934700e+01,
        -1.71319413e+00,
        -1.25905895e+01,
        2.18082600e+01,
        -1.38920937e+01,
        1.78092733e-01,
        1.61397781e+01,
        -2.05825558e+01,
        1.73996868e+01,
        -1.82352173e+00,
        -1.26465273e+01,
        2.19947815e+01,
        -1.40902195e+01,
        2.61439681e-01,
        1.62243748e+01,
        -2.07811337e+01,
    ];
    let out = multi_head_attention(
        &x,
        &x,
        &x,
        Some(&w),
        Some(&bias),
        Some(&ow),
        Some(&ob),
        None,
        2,
    )
    .expect("mha");
    assert_eq!(out.shape, vec![1, 6, 8]);
    assert_all_close(&out.data, &want, TOL, "mha vs numpy");
}

// ── GQA vs NumPy ─────────────────────────────────────────────────────────────

/// 4 query heads sharing 2 KV heads.
///
/// ```text
/// q = sin(arange(80)*0.27 + 0.19).reshape(1,4,5,4)
/// k = sin(arange(40)*0.15 + 0.37).reshape(1,2,5,4)
/// v = sin(arange(40)*0.23 + 0.53).reshape(1,2,5,4)
/// ```
#[test]
fn gqa_matches_numpy() {
    let q = tensor(&[1, 4, 5, 4], 0.27, 0.19);
    let k = tensor(&[1, 2, 5, 4], 0.15, 0.37);
    let v = tensor(&[1, 2, 5, 4], 0.23, 0.53);
    let want = [
        4.21104878e-01,
        3.57632428e-01,
        2.75324464e-01,
        1.78515956e-01,
        4.94479150e-01,
        4.17402178e-01,
        3.18341732e-01,
        2.02515155e-01,
        3.22862327e-01,
        2.49569803e-01,
        1.63133115e-01,
        6.81046695e-02,
        -3.42448652e-02,
        -9.50054675e-02,
        -1.50762409e-01,
        -1.98579118e-01,
        -1.27155825e-01,
        -1.79381549e-01,
        -2.22159728e-01,
        -2.53237396e-01,
        1.71269625e-01,
        1.21366560e-01,
        6.50714412e-02,
        5.34919556e-03,
        4.53232318e-01,
        3.86270940e-01,
        2.98965722e-01,
        1.95914835e-01,
        4.82401550e-01,
        4.04671252e-01,
        3.05628061e-01,
        1.90488279e-01,
        2.62965649e-01,
        1.91953391e-01,
        1.10831462e-01,
        2.38723550e-02,
        -8.33892599e-02,
        -1.42427087e-01,
        -1.93963677e-01,
        -2.35284761e-01,
        3.14025134e-01,
        4.08222347e-01,
        4.80919629e-01,
        5.28288245e-01,
        2.61487603e-01,
        3.11670125e-01,
        3.45437855e-01,
        3.61012340e-01,
        1.73246875e-01,
        1.72984287e-01,
        1.63611129e-01,
        1.45621002e-01,
        1.20006934e-01,
        1.28393322e-01,
        1.30017623e-01,
        1.24794230e-01,
        1.71830148e-01,
        2.34179497e-01,
        2.84195274e-01,
        3.19243282e-01,
        2.83261895e-01,
        3.81228894e-01,
        4.59117621e-01,
        5.12825906e-01,
        3.11461687e-01,
        4.00821537e-01,
        4.69071239e-01,
        5.12616277e-01,
        2.44686335e-01,
        2.83653051e-01,
        3.07680547e-01,
        3.15503389e-01,
        1.60092488e-01,
        1.56339556e-01,
        1.44352660e-01,
        1.24763109e-01,
        1.18803427e-01,
        1.35209024e-01,
        1.44493565e-01,
        1.46168008e-01,
    ];
    let out = grouped_query_attention(&q, &k, &v, 2, None, None).expect("gqa");
    assert_eq!(out.shape, vec![1, 4, 5, 4]);
    assert_all_close(&out.data, &want, TOL, "gqa vs numpy");
}

// ── RoPE vs NumPy ────────────────────────────────────────────────────────────

/// Rotary embedding with *supplied* cos/sin caches — the path that used to
/// deep-copy both tables on every call and now borrows them.
///
/// ```text
/// x  = sin(arange(48)*0.21 + 0.13).reshape(1,2,4,6)
/// cc = sin(arange(24)*0.09 + 0.02).reshape(8,3)
/// sc = sin(arange(24)*0.11 + 0.31).reshape(8,3)
/// out[..,i]      = x0*cc[p,i] - x1*sc[p,i]
/// out[..,half+i] = x1*cc[p,i] + x0*sc[p,i]
/// ```
#[test]
fn rope_with_supplied_caches_matches_numpy() {
    let x = tensor(&[1, 2, 4, 6], 0.21, 0.13);
    let cc = tensor(&[8, 3], 0.09, 0.02);
    let sc = tensor(&[8, 3], 0.11, 0.31);
    let pos = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4]);
    let want = [
        -2.07568929e-01,
        -2.99746126e-01,
        -3.63577247e-01,
        5.33235222e-02,
        2.26537392e-01,
        4.47926700e-01,
        -2.56658167e-01,
        -1.68059647e-01,
        -4.91468385e-02,
        8.45045447e-01,
        9.74553406e-01,
        1.02857316e+00,
        3.64542216e-01,
        4.69274312e-01,
        5.40116251e-01,
        3.16085696e-01,
        3.84722464e-02,
        -2.90854454e-01,
        4.36420441e-01,
        3.26363653e-01,
        1.82580054e-01,
        -1.39665782e+00,
        -1.61394620e+00,
        -1.74758327e+00,
        1.23789884e-01,
        2.38045361e-02,
        -9.50267985e-02,
        -2.82962739e-01,
        -3.49833667e-01,
        -3.35575670e-01,
        -3.76809448e-01,
        -4.39114213e-01,
        -4.62284356e-01,
        2.87818193e-01,
        5.47539830e-01,
        8.28504562e-01,
        -2.12863132e-01,
        -8.34432766e-02,
        6.51157424e-02,
        1.28836286e+00,
        1.35311198e+00,
        1.32459748e+00,
        4.86304253e-01,
        5.60754359e-01,
        5.89683950e-01,
        3.26398134e-01,
        -2.58353036e-02,
        -4.06014472e-01,
    ];
    let out = rotary_embedding(&x, &pos, Some(&cc), Some(&sc), 10000.0).expect("rope");
    assert_eq!(out.shape, vec![1, 2, 4, 6]);
    assert_all_close(&out.data, &want, TOL, "rope vs numpy");
}

/// Supplying the caches must not mutate them (they are now borrowed, not
/// cloned — a regression here would mean the kernel is writing through the
/// borrow).
#[test]
fn rope_does_not_disturb_supplied_caches() {
    let x = tensor(&[1, 2, 4, 6], 0.21, 0.13);
    let cc = tensor(&[8, 3], 0.09, 0.02);
    let sc = tensor(&[8, 3], 0.11, 0.31);
    let cc_before = cc.data.clone();
    let sc_before = sc.data.clone();
    let pos = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4]);
    let _ = rotary_embedding(&x, &pos, Some(&cc), Some(&sc), 10000.0).expect("rope");
    assert_eq!(cc.data, cc_before);
    assert_eq!(sc.data, sc_before);
}

// ── Path parity: parallel vs serial, tiled vs untiled ────────────────────────

/// The rayon `(batch × head)` path must produce exactly what running each
/// head through the serial path produces — in particular the mask must be
/// re-broadcast per `(batch, head)` inside every task.
#[test]
fn sdpa_parallel_matches_per_head_serial() {
    let (batch, heads, seq_q, seq_k, d) = (4, 4, 16, 16, 8);
    let q = tensor(&[batch, heads, seq_q, d], 0.13, 0.05);
    let k = tensor(&[batch, heads, seq_k, d], 0.17, 0.11);
    let v = tensor(&[batch, heads, seq_k, d], 0.19, 0.23);
    // [batch, 1, seq_q, seq_k] — broadcast over the head axis.
    let mask = tensor(&[batch, 1, seq_q, seq_k], 0.07, 0.29);

    let full = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None).expect("sdpa full");

    for b in 0..batch {
        for h in 0..heads {
            let bh = b * heads + h;
            let qs = q.data[bh * seq_q * d..(bh + 1) * seq_q * d].to_vec();
            let ks = k.data[bh * seq_k * d..(bh + 1) * seq_k * d].to_vec();
            let vs = v.data[bh * seq_k * d..(bh + 1) * seq_k * d].to_vec();
            let ms = mask.data[b * seq_q * seq_k..(b + 1) * seq_q * seq_k].to_vec();
            let one = scaled_dot_product_attention(
                &Tensor::new(qs, vec![1, 1, seq_q, d]),
                &Tensor::new(ks, vec![1, 1, seq_k, d]),
                &Tensor::new(vs, vec![1, 1, seq_k, d]),
                Some(&Tensor::new(ms, vec![1, 1, seq_q, seq_k])),
                None,
            )
            .expect("sdpa one head");
            assert_all_close(
                &full.data[bh * seq_q * d..(bh + 1) * seq_q * d],
                &one.data,
                TOL,
                &format!("parallel vs serial head b={b} h={h}"),
            );
        }
    }
}

/// `seq_q` past the `Q_BLOCK` tile boundary: causal masking and mask rows must
/// still line up with the *global* query index, not the tile-local one.
#[test]
fn sdpa_query_tiling_matches_row_at_a_time() {
    let (seq_q, seq_k, d) = (150usize, 40usize, 8usize);
    let q = tensor(&[1, 1, seq_q, d], 0.09, 0.02);
    let k = tensor(&[1, 1, seq_k, d], 0.13, 0.21);
    let v = tensor(&[1, 1, seq_k, d], 0.11, 0.31);
    let mut mask_data = vec![0.0f32; seq_q * seq_k];
    for (i, row) in mask_data.chunks_exact_mut(seq_k).enumerate() {
        for (j, m) in row.iter_mut().enumerate() {
            if j > i {
                *m = f32::NEG_INFINITY;
            } else {
                *m = ((i * seq_k + j) as f64 * 0.017).sin() as f32;
            }
        }
    }
    let mask = Tensor::new(mask_data.clone(), vec![seq_q, seq_k]);
    let full = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None).expect("tiled");

    for i in 0..seq_q {
        let q_row = q.data[i * d..(i + 1) * d].to_vec();
        let m_row = mask_data[i * seq_k..(i + 1) * seq_k].to_vec();
        let one = scaled_dot_product_attention(
            &Tensor::new(q_row, vec![1, 1, 1, d]),
            &k,
            &v,
            Some(&Tensor::new(m_row, vec![1, seq_k])),
            None,
        )
        .expect("single row");
        assert_all_close(
            &full.data[i * d..(i + 1) * d],
            &one.data,
            TOL,
            &format!("query tile row {i}"),
        );
    }
}

/// `is_causal` (the ONNX `Attention-23` flag) across a `Q_BLOCK` boundary must
/// match the equivalent explicit `-inf` upper-triangular mask.
#[test]
fn sdpa_is_causal_matches_explicit_mask_across_tiles() {
    let (seq, d) = (100usize, 6usize);
    let q = tensor(&[1, 2, seq, d], 0.07, 0.13);
    let k = tensor(&[1, 2, seq, d], 0.11, 0.17);
    let v = tensor(&[1, 2, seq, d], 0.05, 0.23);
    let mut mask_data = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            mask_data[i * seq + j] = f32::NEG_INFINITY;
        }
    }
    let mask = Tensor::new(mask_data, vec![seq, seq]);
    let explicit = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None).expect("explicit");

    let (shape, len) = sdpa_output_shape(&q, &k, &v);
    let mut flagged = vec![0.0f32; len];
    let got_shape = sdpa_into(&q, &k, &v, None, None, true, &mut flagged).expect("is_causal");
    assert_eq!(got_shape, shape);
    assert_all_close(&flagged, &explicit.data, TOL, "is_causal vs explicit mask");
}

// ── Degenerate shapes must not panic ─────────────────────────────────────────

#[test]
fn sdpa_degenerate_shapes_do_not_panic() {
    let v_full = tensor(&[1, 1, 3, 4], 0.3, 0.1);
    // seq_k == 0: nothing to attend to.
    let q = tensor(&[1, 1, 2, 4], 0.1, 0.2);
    let k = Tensor::new(vec![], vec![1, 1, 0, 4]);
    let v = Tensor::new(vec![], vec![1, 1, 0, 4]);
    let out = scaled_dot_product_attention(&q, &k, &v, None, None).expect("empty keys");
    assert_eq!(out.shape, vec![1, 1, 2, 4]);
    assert!(out.data.iter().all(|x| *x == 0.0));

    // d_v == 0.
    let v0 = Tensor::new(vec![], vec![1, 1, 3, 0]);
    let k3 = tensor(&[1, 1, 3, 4], 0.2, 0.3);
    let out = scaled_dot_product_attention(&q, &k3, &v0, None, None).expect("zero d_v");
    assert_eq!(out.shape, vec![1, 1, 2, 0]);

    // seq_q == 0.
    let q0 = Tensor::new(vec![], vec![1, 1, 0, 4]);
    let out = scaled_dot_product_attention(&q0, &k3, &v_full, None, None).expect("zero seq_q");
    assert_eq!(out.shape, vec![1, 1, 0, 4]);

    // d_k == 0 — the default scale is `1/sqrt(0) = +inf`, so every score is
    // `0 * inf = NaN` and the result is NaN throughout. That is the historical
    // behaviour; what matters here is that it stays a value, not a panic.
    let q_dk0 = Tensor::new(vec![], vec![1, 1, 2, 0]);
    let k_dk0 = Tensor::new(vec![], vec![1, 1, 3, 0]);
    let out = scaled_dot_product_attention(&q_dk0, &k_dk0, &v_full, None, None).expect("zero d_k");
    assert_eq!(out.shape, vec![1, 1, 2, 4]);
    // With an explicit finite scale the scores are all zero, so attention
    // degenerates to a uniform average over V.
    let out = scaled_dot_product_attention(&q_dk0, &k_dk0, &v_full, None, Some(1.0))
        .expect("zero d_k, explicit scale");
    for d in 0..4 {
        let mean = (v_full.data[d] + v_full.data[4 + d] + v_full.data[8 + d]) / 3.0;
        for row in 0..2 {
            assert!(
                (out.data[row * 4 + d] - mean).abs() <= 1e-5 * (1.0 + mean.abs()),
                "zero d_k row {row} dim {d}: {} vs {mean}",
                out.data[row * 4 + d]
            );
        }
    }
}

/// An undersized output buffer is a typed error, never an out-of-bounds panic.
#[test]
fn sdpa_into_rejects_short_output_buffer() {
    let q = tensor(&[1, 2, 3, 4], 0.1, 0.2);
    let k = tensor(&[1, 2, 3, 4], 0.2, 0.3);
    let v = tensor(&[1, 2, 3, 4], 0.3, 0.4);
    let mut too_small = vec![0.0f32; 5];
    let err = sdpa_into(&q, &k, &v, None, None, false, &mut too_small).expect_err("short buffer");
    assert!(
        format!("{err}").contains("output buffer"),
        "unexpected error: {err}"
    );
}

/// Q with a smaller batch than K/V used to make the kernel write more slices
/// than the derived output shape described, tripping `Tensor::new`'s length
/// assertion — a panic on caller-supplied shapes. `sdpa_output_shape` and
/// `sdpa_into` now derive the output shape from the NumPy broadcast of
/// Q/K/V's leading dims (not Q alone), so a Q batch of 1 broadcasting against
/// a larger K/V batch is a normal, successful call — the historical
/// `[1]`-lead case is exactly the "single axis" case `lead_is_tileable`
/// always accepts.
#[test]
fn sdpa_broadcasts_undersized_q_batch_against_kv_instead_of_erroring() {
    let q1 = tensor(&[1, 5, 6], 0.11, 0.03);
    let k = tensor(&[4, 5, 6], 0.13, 0.07);
    let v = tensor(&[4, 5, 6], 0.17, 0.09);

    let broadcast =
        scaled_dot_product_attention(&q1, &k, &v, None, None).expect("q batch 1 broadcasts");
    assert_eq!(broadcast.shape, vec![4, 5, 6]);

    // Equal batches still work (unchanged from before this fix).
    let q4 = tensor(&[4, 5, 6], 0.11, 0.03);
    let out = scaled_dot_product_attention(&q4, &k, &v, None, None).expect("equal batches");
    assert_eq!(out.shape, vec![4, 5, 6]);

    // The defining property of broadcasting: reusing Q's single batch for
    // every one of K/V's 4 batches must equal running the same SDPA with Q
    // *explicitly* tiled 4x — not just a plausible-looking shape. `q4`'s data
    // is a differently-generated sequence (see `tensor`/`gen`), so this tile
    // is built by hand from `q1`'s own data rather than via `tensor(..)`.
    let mut q1_tiled_data = Vec::with_capacity(4 * 5 * 6);
    for _ in 0..4 {
        q1_tiled_data.extend_from_slice(&q1.data);
    }
    let q1_tiled = Tensor::new(q1_tiled_data, vec![4, 5, 6]);
    let tiled_reference = scaled_dot_product_attention(&q1_tiled, &k, &v, None, None)
        .expect("explicit-tile reference");
    assert_all_close(
        &broadcast.data,
        &tiled_reference.data,
        TOL,
        "q-batch broadcast vs explicit tile",
    );
}

/// A leading-shape combination the flat batch index cannot correctly tile —
/// e.g. `[2, 1]` broadcasting against `[2, 3]`, where Q's broadcast axis is
/// *inner* to its own matching axis — must be a typed error, not a silently
/// wrong answer. `b % q_batch` would read `q`'s batch-0 slice for 2 of the 3
/// K/V rows in group 1 instead of broadcasting the intended row.
#[test]
fn sdpa_rejects_a_leading_shape_the_flat_batch_index_cannot_tile() {
    let q = tensor(&[2, 1, 5, 6], 0.11, 0.03);
    let k = tensor(&[2, 3, 5, 6], 0.13, 0.07);
    let v = tensor(&[2, 3, 5, 6], 0.17, 0.09);
    let err = scaled_dot_product_attention(&q, &k, &v, None, None)
        .expect_err("[2,1] vs [2,3] is not tileable by a flat modulo index");
    assert!(
        matches!(err, oxionnx_core::OnnxError::ShapeMismatch(_)),
        "expected ShapeMismatch, got {err:?}"
    );
    assert!(
        format!("{err}").contains("broadcast"),
        "unexpected error: {err}"
    );
}

/// A tensor whose buffer is shorter than its shape must be a typed error.
#[test]
fn rope_rejects_undersized_input() {
    let short = Tensor::new(vec![0.0f32; 6], vec![6]);
    // Reshape into a [2, 4] claim over a 6-element buffer via a hand-built
    // tensor: `Tensor::new` enforces the invariant, so build it by mutating a
    // valid tensor's data instead.
    let mut bad = Tensor::new(vec![0.0f32; 8], vec![1, 2, 4]);
    bad.data.truncate(5);
    let pos = Tensor::new(vec![0.0, 1.0], vec![2]);
    let err = rotary_embedding(&bad, &pos, None, None, 10000.0).expect_err("short input");
    assert!(
        format!("{err}").contains("element(s)"),
        "unexpected error: {err}"
    );
    drop(short);
}
