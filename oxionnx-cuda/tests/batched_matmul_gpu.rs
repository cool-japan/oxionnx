//! Batched `MatMul` / `Gemm` end to end through this crate's *real* dispatch,
//! on a real device.
//!
//! Run with:
//!
//! ```text
//! OXIONNX_CUDA=1 cargo test -p oxionnx-cuda --features gpu-tests --release \
//!     --test batched_matmul_gpu
//! ```
//!
//! # What this covers that nothing else did
//!
//! `oxicuda-blas` has its own tests for `gemm_strided_batched`, and this
//! crate's `matmul_shape_sweep_gpu` has its own for the 2-D case. Neither
//! exercises the thing that is actually easy to get wrong here: the
//! *bookkeeping* in `try_cuda_dispatch`'s MatMul arm that turns a pair of ONNX
//! tensor shapes into a batch count, a per-operand batch count, two batch
//! strides, and an output shape — under numpy broadcasting, where an operand
//! may legitimately carry no batch dimension at all, or a batch dimension of
//! exactly 1, and must then be *reused* for every slice rather than advanced.
//!
//! A per-slice loop and a strided-batch launch fail differently when that
//! bookkeeping is wrong, and both fail *silently*: the output has exactly the
//! right shape either way, filled with the product of the wrong slices. So
//! every case below pins the numbers, not just the shape, against an
//! independently-computed CPU expectation.
//!
//! The cases are chosen so that a wrong slice pairing cannot coincidentally
//! produce the right answer: each batch slice is scaled by a distinct decade,
//! so pairing slice *i* of A with slice *j* of B lands in a different order of
//! magnitude than the correct pairing rather than merely a different value.
//!
//! # Contract, not implementation
//!
//! Nothing here names a batched-GEMM entry point. These are properties of
//! `try_cuda_dispatch`'s public behaviour — they held for the per-slice loop
//! this crate started with, they must hold for the strided-batch dispatch that
//! replaced it, and they must keep holding for whatever replaces that.

use std::collections::HashMap;

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::{OnnxError, Tensor};
use oxionnx_cuda::context::Activation;
use oxionnx_cuda::{try_cuda_dispatch, CudaContext};

/// Acquire a device, bypassing the `OXIONNX_CUDA` env-var gate, or `None`
/// when no CUDA driver / device is present.
///
/// `Activation::Enabled` is the embedder-opt-in path (the env gate is policy,
/// tested separately in `context::tests`, and orthogonal to what these tests
/// prove). Returning `None` rather than panicking is the OxiCUDA convention
/// (`oxicuda-blas`'s `src/gpu_tests.rs` / `tests/gemm_shape_sweep_gpu.rs`):
/// each test skips, so `--all-features` stays green on a CPU-only host.
fn device() -> Option<CudaContext> {
    CudaContext::try_new_with(Activation::Enabled)
}

fn matmul_node(inputs: &[&str]) -> Node {
    Node {
        op: OpKind::MatMul,
        name: "batched".to_string(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: vec!["y".to_string()],
        attrs: Attributes::default(),
    }
}

/// The shape of one batched matrix multiplication, as the ONNX broadcast rules
/// leave it: `batch` output slices, of which operand A supplies `a_batches`
/// (either `1` or `batch`) and operand B supplies `b_batches`.
#[derive(Clone, Copy)]
struct Shape {
    batch: usize,
    a_batches: usize,
    b_batches: usize,
    m: usize,
    k: usize,
    n: usize,
}

/// The CPU expectation: plain row-major `A @ B` per batch slice, with each
/// operand's slice chosen by the same numpy broadcast rule the dispatch arm
/// applies (`i % operand_batches`).
///
/// Written out longhand rather than reusing anything from the crate under
/// test, so that a bug in the dispatch's own broadcast bookkeeping cannot be
/// mirrored into the expectation.
fn cpu_batched_matmul(a: &[f32], b: &[f32], shape: Shape) -> Vec<f32> {
    let Shape {
        batch,
        a_batches,
        b_batches,
        m,
        k,
        n,
    } = shape;
    let mut out = vec![0.0_f32; batch * m * n];
    for slice in 0..batch {
        let a_base = (slice % a_batches) * m * k;
        let b_base = (slice % b_batches) * k * n;
        let c_base = slice * m * n;
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0_f64;
                for inner in 0..k {
                    acc += f64::from(a[a_base + row * k + inner])
                        * f64::from(b[b_base + inner * n + col]);
                }
                out[c_base + row * n + col] = acc as f32;
            }
        }
    }
    out
}

/// Distinct, non-degenerate values whose per-slice *scale* differs by a decade,
/// so a mis-paired slice is off by an order of magnitude rather than by a
/// plausible-looking few percent.
fn slice_scaled(batch: usize, per_slice: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut data = Vec::with_capacity(batch * per_slice);
    for slice in 0..batch {
        // 1, 10, 100, ... per slice.
        let scale = 10.0_f32.powi(slice as i32);
        for _ in 0..per_slice {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = f64::from((state >> 32) as u32) / 4_294_967_296.0;
            data.push((unit as f32 + 0.5) * scale);
        }
    }
    data
}

/// Assert element-wise agreement with a relative tolerance, reporting the
/// worst offender rather than the first.
///
/// Relative, because the decade scaling above deliberately spans several orders
/// of magnitude across slices: a fixed absolute epsilon would be vacuous on the
/// large slices and impossible on the small ones.
fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let mut worst = (0usize, 0.0_f32);
    for (index, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let rel = (g - w).abs() / w.abs().max(1e-6);
        if rel > worst.1 {
            worst = (index, rel);
        }
    }
    assert!(
        worst.1 < 1e-4,
        "{what}: worst relative error {:.3e} at index {} (GPU {}, CPU {})",
        worst.1,
        worst.0,
        got[worst.0],
        want[worst.0],
    );
}

/// Dispatch and unwrap, insisting the node was *claimed*.
///
/// A decline is a legitimate outcome for a configuration this crate does not
/// accelerate — but every shape below is one it does, so `Ok(None)` here means
/// batched MatMul silently stopped being accelerated, which is precisely the
/// regression these tests exist to catch and which no numeric assertion would
/// notice (the CPU would compute the right answer one frame up).
fn dispatch_claimed(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    ctx: &CudaContext,
    what: &str,
) -> Vec<Tensor> {
    let claimed: Result<Option<Vec<Tensor>>, OnnxError> =
        try_cuda_dispatch(node, weights, intermediates, ctx);
    claimed
        .unwrap_or_else(|e| panic!("{what}: dispatch hard-errored: {e}"))
        .unwrap_or_else(|| {
            panic!("{what}: batched MatMul was declined (Ok(None)), not accelerated")
        })
}

/// Both operands carry the same real batch dimension: slice *i* of A must meet
/// slice *i* of B, and no other pairing.
#[test]
fn batch_of_both_operands_pairs_slices_index_for_index() {
    let Some(ctx) = device() else {
        eprintln!(
            "no CUDA device present, skipping batch_of_both_operands_pairs_slices_index_for_index"
        );
        return;
    };
    let (batch, m, k, n) = (5usize, 7usize, 9usize, 6usize);

    let a = slice_scaled(batch, m * k, 0x0BAD_C0DE_1234_5678);
    let b = slice_scaled(batch, k * n, 0x0FEE_1DEA_8765_4321);

    let mut intermediates = HashMap::new();
    intermediates.insert("a".to_string(), Tensor::new(a.clone(), vec![batch, m, k]));
    intermediates.insert("b".to_string(), Tensor::new(b.clone(), vec![batch, k, n]));

    let outputs = dispatch_claimed(
        &matmul_node(&["a", "b"]),
        &HashMap::new(),
        &intermediates,
        &ctx,
        "batch x batch",
    );

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, vec![batch, m, n]);
    let want = cpu_batched_matmul(
        &a,
        &b,
        Shape {
            batch,
            a_batches: batch,
            b_batches: batch,
            m,
            k,
            n,
        },
    );
    assert_close(&outputs[0].data, &want, "batch x batch");
}

/// B has no batch dimension at all (`[k, n]`): every slice of A must be
/// multiplied by that same B — a stride-0 read on the B operand, not an
/// advance through a buffer that only holds one slice.
///
/// This is the case a strided-batch dispatch gets wrong most naturally: giving
/// B the same stride as A walks straight off the end of a `k*n` allocation and
/// reads whatever the pool left there.
#[test]
fn unbatched_second_operand_is_reused_for_every_slice() {
    let Some(ctx) = device() else {
        eprintln!(
            "no CUDA device present, skipping unbatched_second_operand_is_reused_for_every_slice"
        );
        return;
    };
    let (batch, m, k, n) = (4usize, 5usize, 8usize, 3usize);

    let a = slice_scaled(batch, m * k, 0x1111_2222_3333_4444);
    let b = slice_scaled(1, k * n, 0x5555_6666_7777_8888);

    let mut intermediates = HashMap::new();
    intermediates.insert("a".to_string(), Tensor::new(a.clone(), vec![batch, m, k]));
    intermediates.insert("b".to_string(), Tensor::new(b.clone(), vec![k, n]));

    let outputs = dispatch_claimed(
        &matmul_node(&["a", "b"]),
        &HashMap::new(),
        &intermediates,
        &ctx,
        "batch x unbatched",
    );

    assert_eq!(outputs[0].shape, vec![batch, m, n]);
    let want = cpu_batched_matmul(
        &a,
        &b,
        Shape {
            batch,
            a_batches: batch,
            b_batches: 1,
            m,
            k,
            n,
        },
    );
    assert_close(&outputs[0].data, &want, "batch x unbatched");
}

/// The mirror image: A is `[m, k]` and B is `[batch, k, n]`. The output still
/// carries the batch dimension, and A is the stride-0 operand.
#[test]
fn unbatched_first_operand_is_reused_for_every_slice() {
    let Some(ctx) = device() else {
        eprintln!(
            "no CUDA device present, skipping unbatched_first_operand_is_reused_for_every_slice"
        );
        return;
    };
    let (batch, m, k, n) = (3usize, 6usize, 4usize, 5usize);

    let a = slice_scaled(1, m * k, 0x9999_AAAA_BBBB_CCCC);
    let b = slice_scaled(batch, k * n, 0xDDDD_EEEE_FFFF_0000);

    let mut intermediates = HashMap::new();
    intermediates.insert("a".to_string(), Tensor::new(a.clone(), vec![m, k]));
    intermediates.insert("b".to_string(), Tensor::new(b.clone(), vec![batch, k, n]));

    let outputs = dispatch_claimed(
        &matmul_node(&["a", "b"]),
        &HashMap::new(),
        &intermediates,
        &ctx,
        "unbatched x batch",
    );

    assert_eq!(outputs[0].shape, vec![batch, m, n]);
    // `batch` output slices, from A's single slice (`a_batches = 1`, so A is
    // reused) against each of B's `batch` slices.
    let want = cpu_batched_matmul(
        &a,
        &b,
        Shape {
            batch,
            a_batches: 1,
            b_batches: batch,
            m,
            k,
            n,
        },
    );
    assert_close(&outputs[0].data, &want, "unbatched x batch");
}

/// An explicit leading `1` (`[1, k, n]`) broadcasts exactly like an absent
/// batch dimension does, and the *output* shape follows numpy: `[2, 3, m, n]`
/// against a rank-3 `[1, k, n]`, so the multi-dimensional batch prefix survives
/// the round trip through a flat batch count.
#[test]
fn multi_dimensional_batch_prefix_survives_and_broadcasts() {
    let Some(ctx) = device() else {
        eprintln!("no CUDA device present, skipping multi_dimensional_batch_prefix_survives_and_broadcasts");
        return;
    };
    let (b0, b1, m, k, n) = (2usize, 3usize, 4usize, 5usize, 3usize);
    let batch = b0 * b1;

    let a = slice_scaled(batch, m * k, 0x0102_0304_0506_0708);
    let b = slice_scaled(1, k * n, 0x090A_0B0C_0D0E_0F10);

    let mut intermediates = HashMap::new();
    intermediates.insert("a".to_string(), Tensor::new(a.clone(), vec![b0, b1, m, k]));
    intermediates.insert("b".to_string(), Tensor::new(b.clone(), vec![1, k, n]));

    let outputs = dispatch_claimed(
        &matmul_node(&["a", "b"]),
        &HashMap::new(),
        &intermediates,
        &ctx,
        "[2,3,m,k] x [1,k,n]",
    );

    assert_eq!(
        outputs[0].shape,
        vec![b0, b1, m, n],
        "the broadcast batch prefix must be preserved, not flattened",
    );
    let want = cpu_batched_matmul(
        &a,
        &b,
        Shape {
            batch,
            a_batches: batch,
            b_batches: 1,
            m,
            k,
            n,
        },
    );
    assert_close(&outputs[0].data, &want, "[2,3,m,k] x [1,k,n]");
}

/// A batched `Gemm` with `transB=1` and a bias: the transpose is applied per
/// operand batch slice (B holds only `b_batches` slices, not `batch` of them),
/// and the bias is broadcast across every row of every slice.
///
/// `Gemm` with a batch dimension is not spec-conformant ONNX, but the dispatch
/// arm accepts it uniformly with `MatMul`, so the combination of *transpose*,
/// *broadcast* and *bias* has to keep agreeing with the CPU on every slice.
#[test]
fn batched_gemm_with_transposed_broadcast_operand_and_bias() {
    let Some(ctx) = device() else {
        eprintln!("no CUDA device present, skipping batched_gemm_with_transposed_broadcast_operand_and_bias");
        return;
    };
    let (batch, m, k, n) = (3usize, 4usize, 6usize, 5usize);

    let a = slice_scaled(batch, m * k, 0xFEED_FACE_CAFE_BEEF);
    // Stored transposed: `[n, k]`, so `transB=1` makes it the `[k, n]` operand.
    let b_t = slice_scaled(1, n * k, 0xBEEF_CAFE_FACE_FEED);
    let bias: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 1000.0).collect();
    let alpha = 0.5_f32;
    let beta = 2.0_f32;

    let mut node = Node {
        op: OpKind::Gemm,
        name: "batched_gemm".to_string(),
        inputs: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        outputs: vec!["y".to_string()],
        attrs: Attributes::default(),
    };
    node.attrs.ints.insert("transB".to_string(), 1);
    node.attrs.floats.insert("alpha".to_string(), alpha);
    node.attrs.floats.insert("beta".to_string(), beta);

    let mut intermediates = HashMap::new();
    intermediates.insert("a".to_string(), Tensor::new(a.clone(), vec![batch, m, k]));
    intermediates.insert("b".to_string(), Tensor::new(b_t.clone(), vec![n, k]));
    intermediates.insert("c".to_string(), Tensor::new(bias.clone(), vec![n]));

    let outputs = dispatch_claimed(&node, &HashMap::new(), &intermediates, &ctx, "batched gemm");

    assert_eq!(outputs[0].shape, vec![batch, m, n]);

    // Untranspose on the host, independently of the crate's own helper.
    let mut b = vec![0.0_f32; k * n];
    for row in 0..n {
        for col in 0..k {
            b[col * n + row] = b_t[row * k + col];
        }
    }
    let mut want = cpu_batched_matmul(
        &a,
        &b,
        Shape {
            batch,
            a_batches: batch,
            b_batches: 1,
            m,
            k,
            n,
        },
    );
    for (index, value) in want.iter_mut().enumerate() {
        *value = *value * alpha + beta * bias[index % n];
    }
    assert_close(&outputs[0].data, &want, "batched gemm");
}

/// A weight (graph initializer) operand must produce the same numbers on the
/// tenth dispatch as on the first.
///
/// This is the residency invariant stated as a *behavioural* claim rather than
/// a cache-internals one: whatever the dispatch does to avoid re-uploading
/// invariant bytes, the answer may not drift. It would not: a cache that served
/// the wrong buffer, or a pooled scratch buffer handed out while still in use,
/// shows up here as a differing slice — and only here, because a single-shot
/// test can never observe a stale-cache bug at all.
///
/// The activation changes between iterations, so a dispatch that accidentally
/// cached the *activation* (which is not invariant) also fails, rather than
/// passing because every iteration happened to be identical.
#[test]
fn repeated_dispatch_with_a_resident_weight_stays_numerically_stable() {
    let Some(ctx) = device() else {
        eprintln!("no CUDA device present, skipping repeated_dispatch_with_a_resident_weight_stays_numerically_stable");
        return;
    };
    let (batch, m, k, n) = (2usize, 8usize, 12usize, 6usize);

    let b = slice_scaled(1, k * n, 0x2468_ACE0_1357_9BDF);
    let mut weights = HashMap::new();
    weights.insert("w".to_string(), Tensor::new(b.clone(), vec![k, n]));

    let node = matmul_node(&["a", "w"]);

    for iteration in 0..10 {
        let a = slice_scaled(batch, m * k, 0x1357_9BDF_2468_ACE0 ^ iteration as u64);
        let mut intermediates = HashMap::new();
        intermediates.insert("a".to_string(), Tensor::new(a.clone(), vec![batch, m, k]));

        let outputs = dispatch_claimed(
            &node,
            &weights,
            &intermediates,
            &ctx,
            &format!("resident weight, iteration {iteration}"),
        );
        let want = cpu_batched_matmul(
            &a,
            &b,
            Shape {
                batch,
                a_batches: batch,
                b_batches: 1,
                m,
                k,
                n,
            },
        );
        assert_close(
            &outputs[0].data,
            &want,
            &format!("resident weight, iteration {iteration}"),
        );
    }
}

/// Interleaving *different* shapes through the same context must not let one
/// dispatch's buffers leak into another's.
///
/// A size-classed buffer pool hands the same underlying allocation to
/// successive dispatches, and a pooled buffer is generally *larger* than the
/// tensor in it. Every failure mode that creates — a stale tail read as data, a
/// buffer recycled while a kernel still reads it, an output length taken from
/// the allocation rather than the request — produces wrong numbers on the
/// *second* shape while the first still passes in isolation.
#[test]
fn interleaved_shapes_through_one_context_do_not_contaminate_each_other() {
    let Some(ctx) = device() else {
        eprintln!("no CUDA device present, skipping interleaved_shapes_through_one_context_do_not_contaminate_each_other");
        return;
    };

    // Deliberately different in every dimension *and* in batch rank, so a
    // recycled buffer is never coincidentally the right size.
    let cases: [(usize, usize, usize, usize); 4] = [
        (1, 16, 32, 8),
        (6, 3, 5, 7),
        (2, 33, 17, 9),
        (3, 1, 129, 4), // k > any earlier m*k, forcing a fresh, larger class
    ];

    for round in 0..4 {
        for (case, &(batch, m, k, n)) in cases.iter().enumerate() {
            let a = slice_scaled(
                batch,
                m * k,
                0xABCD_0000_0000_0000 ^ (case as u64) << 8 ^ round,
            );
            let b = slice_scaled(
                batch,
                k * n,
                0x1234_0000_0000_0000 ^ (case as u64) << 8 ^ round,
            );

            let mut intermediates = HashMap::new();
            intermediates.insert("a".to_string(), Tensor::new(a.clone(), vec![batch, m, k]));
            intermediates.insert("b".to_string(), Tensor::new(b.clone(), vec![batch, k, n]));

            let what = format!("round {round}, case {case} ({batch}x{m}x{k}x{n})");
            let outputs = dispatch_claimed(
                &matmul_node(&["a", "b"]),
                &HashMap::new(),
                &intermediates,
                &ctx,
                &what,
            );
            assert_eq!(outputs[0].shape, vec![batch, m, n], "{what}: wrong shape");
            let want = cpu_batched_matmul(
                &a,
                &b,
                Shape {
                    batch,
                    a_batches: batch,
                    b_batches: batch,
                    m,
                    k,
                    n,
                },
            );
            assert_close(&outputs[0].data, &want, &what);
        }
    }
}

/// The residency claim, stated as an assertion instead of a benchmark number:
/// **a steady-state dispatch uploads zero weight bytes and allocates no device
/// memory.**
///
/// This is the one property the whole buffer-pool/weight-cache design exists
/// to produce, and it is invisible to every other test here — they all check
/// *numbers*, and numbers stay right whether or not anything is being reused.
/// A regression that quietly stopped keying weights (or that thrashed one
/// identity between two conflicting labels) would leave every other test in
/// this file passing while the frame paid full price.
///
/// The three counters are asserted for what each one proves:
///
/// * `weight_bytes_uploaded == 0` — the initializer crossed the bus during
///   warm-up and has not since. This is the claim.
/// * `weight_hits > 0` — and it was *served from the cache*, rather than
///   uploading zero bytes because nobody asked for it at all.
/// * `pool_allocs == 0` — the activation and output buffers came from the
///   pool. A steady-state frame that still calls `cuMemAlloc` has a size class
///   churning, which the timings would show but nothing would name.
///
/// # Pinned to the ordinary launch path, on purpose
///
/// The subject here is the *pool and residency* machinery, so this pins
/// `set_graph_capture(false)` rather than inheriting whatever
/// `OXIONNX_CUDA_GRAPH` happens to say. A graph-backed dispatch deliberately
/// does not use the scratch pool at all — it replays against buffers the
/// recording owns, which is what makes the recorded addresses stable (see
/// `graph_cache`'s pointer-stability section) — so `pool_hits` is legitimately
/// zero there. Without the pin, running this suite with graphs on would fail
/// this test for doing exactly what it is supposed to do.
#[test]
fn a_steady_state_dispatch_uploads_no_weight_bytes_and_allocates_nothing() {
    let Some(ctx) = device() else {
        eprintln!("no CUDA device present, skipping a_steady_state_dispatch_uploads_no_weight_bytes_and_allocates_nothing");
        return;
    };
    ctx.set_graph_capture(false);
    let (batch, m, k, n) = (2usize, 16usize, 24usize, 8usize);

    let b = slice_scaled(1, k * n, 0xFACE_B00C_0BAD_F00D);
    let mut weights = HashMap::new();
    weights.insert("w".to_string(), Tensor::new(b.clone(), vec![k, n]));
    let node = matmul_node(&["a", "w"]);

    // Two warm-up dispatches: the first uploads the weight and allocates the
    // pool's buffers, the second returns those buffers so the third can reuse
    // them. Measuring from the third is what makes "steady state" mean steady
    // state rather than "the second call".
    for iteration in 0..2u64 {
        let a = slice_scaled(batch, m * k, 0xC0DE_0000_0000_0000 ^ iteration);
        let mut intermediates = HashMap::new();
        intermediates.insert("a".to_string(), Tensor::new(a, vec![batch, m, k]));
        dispatch_claimed(&node, &weights, &intermediates, &ctx, "warm-up");
    }

    assert!(
        ctx.is_weight_resident("w"),
        "the initializer must be on the device after warm-up, or there is nothing to reuse",
    );

    let warm = ctx.cache_counters();
    let a = slice_scaled(batch, m * k, 0xC0DE_0000_0000_0002);
    let mut intermediates = HashMap::new();
    intermediates.insert("a".to_string(), Tensor::new(a, vec![batch, m, k]));
    dispatch_claimed(&node, &weights, &intermediates, &ctx, "steady state");
    let delta = ctx.cache_counters().since(warm);

    assert_eq!(
        delta.weight_bytes_uploaded, 0,
        "a steady-state dispatch re-uploaded {} weight bytes -- the residency cache is not \
         being hit (counters: {delta:?})",
        delta.weight_bytes_uploaded,
    );
    assert!(
        delta.weight_hits > 0,
        "zero weight bytes uploaded, but also zero cache hits -- the operand is not being keyed \
         at all rather than being served from the cache (counters: {delta:?})",
    );
    assert_eq!(
        delta.pool_allocs, 0,
        "a steady-state dispatch still called cuMemAlloc {} time(s) -- the buffer pool is not \
         serving this shape (counters: {delta:?})",
        delta.pool_allocs,
    );
    assert!(
        delta.pool_hits > 0,
        "no pooled buffers were reused, so nothing was pooled (counters: {delta:?})",
    );

    // Releasing the caches must actually release them, and must leave the next
    // dispatch correct rather than merely fast.
    let freed = ctx.release_device_caches();
    assert!(freed > 0, "the caches reported holding nothing to release");
    assert!(!ctx.is_weight_resident("w"));
    let a = slice_scaled(batch, m * k, 0xC0DE_0000_0000_0003);
    let mut intermediates = HashMap::new();
    intermediates.insert("a".to_string(), Tensor::new(a.clone(), vec![batch, m, k]));
    let outputs = dispatch_claimed(
        &node,
        &weights,
        &intermediates,
        &ctx,
        "after releasing the caches",
    );
    let want = cpu_batched_matmul(
        &a,
        &b,
        Shape {
            batch,
            a_batches: batch,
            b_batches: 1,
            m,
            k,
            n,
        },
    );
    assert_close(&outputs[0].data, &want, "after releasing the caches");
}

/// A weight consumed *both* ways by two nodes of one graph — once as-is, once
/// transposed — must not have one form served for the other.
///
/// The transpose of a matrix is a different byte sequence under the same
/// initializer name, so a cache keyed on the name alone would hand the
/// `transB=1` node the untransposed bytes (or vice versa) and return a
/// correctly-shaped, entirely wrong answer. Only reachable when both nodes run
/// against the same context, which is exactly what a session does.
#[test]
fn one_initializer_consumed_both_transposed_and_not_keeps_the_two_forms_apart() {
    let Some(ctx) = device() else {
        eprintln!("no CUDA device present, skipping one_initializer_consumed_both_transposed_and_not_keeps_the_two_forms_apart");
        return;
    };
    let (m, k, n) = (5usize, 4usize, 4usize);

    // Square-ish and deliberately asymmetric, so the transpose really differs.
    let w: Vec<f32> = (0..(k * n)).map(|i| (i as f32 + 1.0) * 1.5).collect();
    let mut weights = HashMap::new();
    weights.insert("w".to_string(), Tensor::new(w.clone(), vec![k, n]));

    let a = slice_scaled(1, m * k, 0x0DDB_A11B_0BAD_CAFE);
    let mut intermediates = HashMap::new();
    intermediates.insert("a".to_string(), Tensor::new(a.clone(), vec![m, k]));

    let plain = matmul_node(&["a", "w"]);
    let mut transposed = Node {
        op: OpKind::Gemm,
        name: "gemm_t".to_string(),
        inputs: vec!["a".to_string(), "w".to_string()],
        outputs: vec!["y".to_string()],
        attrs: Attributes::default(),
    };
    transposed.attrs.ints.insert("transB".to_string(), 1);
    transposed.attrs.floats.insert("beta".to_string(), 0.0);

    let mut w_t = vec![0.0_f32; k * n];
    for row in 0..n {
        for col in 0..k {
            w_t[col * n + row] = w[row * k + col];
        }
    }

    let unbatched = Shape {
        batch: 1,
        a_batches: 1,
        b_batches: 1,
        m,
        k,
        n,
    };
    let want_plain = cpu_batched_matmul(&a, &w, unbatched);
    let want_transposed = cpu_batched_matmul(&a, &w_t, unbatched);

    // Interleaved and repeated: one round would let a name-only cache pass by
    // never being asked for the second form.
    for round in 0..3 {
        let got = dispatch_claimed(&plain, &weights, &intermediates, &ctx, "plain");
        assert_close(&got[0].data, &want_plain, &format!("plain, round {round}"));

        let got = dispatch_claimed(&transposed, &weights, &intermediates, &ctx, "transposed");
        assert_close(
            &got[0].data,
            &want_transposed,
            &format!("transposed, round {round}"),
        );
    }
}

/// A batch large enough that a per-slice host round trip and a single batched
/// launch are unmistakably different amounts of work, at a slice size small
/// enough that the launch overhead dominates.
///
/// Correctness only — the timing claim lives in `examples/dispatch_bench.rs` —
/// but a batched launch that silently computed only the first slice, or that
/// stopped after a `u32`-truncated count, fails here and nowhere else.
#[test]
fn a_large_batch_of_small_slices_computes_every_slice() {
    let Some(ctx) = device() else {
        eprintln!(
            "no CUDA device present, skipping a_large_batch_of_small_slices_computes_every_slice"
        );
        return;
    };
    let (batch, m, k, n) = (64usize, 4usize, 4usize, 4usize);

    // Scale by slice index rather than by decade here: 10^63 overflows f32.
    let mut a = Vec::with_capacity(batch * m * k);
    let mut b = Vec::with_capacity(batch * k * n);
    for slice in 0..batch {
        for element in 0..(m * k) {
            a.push((slice + 1) as f32 + element as f32 * 0.25);
        }
        for element in 0..(k * n) {
            b.push(1.0 / (slice + 1) as f32 + element as f32 * 0.125);
        }
    }

    let mut intermediates = HashMap::new();
    intermediates.insert("a".to_string(), Tensor::new(a.clone(), vec![batch, m, k]));
    intermediates.insert("b".to_string(), Tensor::new(b.clone(), vec![batch, k, n]));

    let outputs = dispatch_claimed(
        &matmul_node(&["a", "b"]),
        &HashMap::new(),
        &intermediates,
        &ctx,
        "batch=64",
    );
    assert_eq!(outputs[0].shape, vec![batch, m, n]);
    let want = cpu_batched_matmul(
        &a,
        &b,
        Shape {
            batch,
            a_batches: batch,
            b_batches: batch,
            m,
            k,
            n,
        },
    );
    assert_close(&outputs[0].data, &want, "batch=64");

    // Every slice must be distinct: a launch that computed slice 0 and then
    // copied it `batch` times would pass a loose element-wise comparison only
    // if the expectation were also wrong, so pin the distinctness directly.
    let first = &outputs[0].data[..m * n];
    let last = &outputs[0].data[(batch - 1) * m * n..];
    assert!(
        first
            .iter()
            .zip(last.iter())
            .any(|(x, y)| (x - y).abs() > 1e-3),
        "the first and last batch slices are identical -- the batch was not really computed",
    );
}
