//! CUDA graph capture / replay through this crate's *real* dispatch, on a real
//! device.
//!
//! Run with:
//!
//! ```text
//! cargo test -p oxionnx-cuda --features gpu-tests --release \
//!     --test graph_cache_gpu -- --test-threads=1
//! ```
//!
//! # What has to be true for graph replay to be allowed to exist
//!
//! A replayed CUDA graph is not "the same computation again" in any sense the
//! compiler can check: it is a recording of *device addresses* and launch
//! geometry, replayed against whatever those addresses hold now. Every way
//! that can go wrong produces a correctly-shaped tensor full of wrong numbers
//! — the exact failure mode this crate's shadow verification exists for, and
//! the exact failure mode a benchmark cannot see. So the properties below are
//! asserted against numbers, never against shapes or "it did not error":
//!
//! 1. **Replay equals the ordinary path, bit for bit.** Not "within a
//!    tolerance": the recorded launches *are* the ordinary launches (see
//!    `matmul::issue_gemm`, which both paths call), so the same inputs must
//!    produce the identical `f32` bit pattern. Anything less would mean the
//!    recording is not what it claims to be.
//! 2. **Replay tracks new input.** A graph holds pointers, not values. If a
//!    later frame's activation did not reach the recorded address, replay
//!    would keep returning the first frame's answer — plausible, stable, and
//!    completely wrong.
//! 3. **Two nodes of identical shape over different weights stay separate.**
//!    The sharpest available way to break a graph cache: same `m`/`k`/`n`,
//!    different resident weight. If the key did not carry the weight's device
//!    address, the second node would replay the first node's recording.
//! 4. **The toggle is honest in both directions**, so a bisect against it
//!    means something.
//!
//! Every test skips (rather than fails) when no CUDA device is present — this
//! crate's convention, see `batched_matmul_gpu.rs`.

use std::collections::HashMap;

use oxionnx_core::graph::{Attributes, Node, OpKind};
use oxionnx_core::Tensor;
use oxionnx_cuda::context::Activation;
use oxionnx_cuda::{try_cuda_dispatch, CudaContext};

/// Acquire a device, bypassing the `OXIONNX_CUDA` env-var gate, or `None` when
/// no CUDA driver / device is present.
fn device() -> Option<CudaContext> {
    CudaContext::try_new_with(Activation::Enabled)
}

fn matmul_node(inputs: &[&str]) -> Node {
    Node {
        op: OpKind::MatMul,
        name: "graph_cache_test".to_string(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: vec!["y".to_string()],
        attrs: Attributes::default(),
    }
}

/// Deterministic pseudo-random data (the LCG this crate's on-device tests
/// share), so a failure is reproducible and no value is degenerate.
fn pseudo_random(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = f64::from((state >> 32) as u32) / 4_294_967_296.0;
            (unit * 2.0 - 1.0) as f32
        })
        .collect()
}

/// Dispatch once, returning the single output tensor's data, or `None` if the
/// node was declined.
fn dispatch(
    ctx: &CudaContext,
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
) -> Option<Vec<f32>> {
    match try_cuda_dispatch(node, weights, intermediates, ctx) {
        Ok(Some(out)) => out.first().map(|t| t.data.clone()),
        Ok(None) => None,
        Err(e) => panic!("dispatch failed: {e}"),
    }
}

/// `[1, k] @ [k, n]` with the right operand as a named graph initializer —
/// the shape class the face pipeline repeats (ArcFace's head, InSwapper's
/// AdaIN projections) and the one `oxicuda-blas` routes through its split-K
/// path, which is the path that could not be captured at all until its
/// reduction workspace stopped being allocated per call.
fn skinny_case(
    weight_name: &str,
    k: usize,
    n: usize,
    a_seed: u64,
    b_seed: u64,
) -> (Node, HashMap<String, Tensor>, HashMap<String, Tensor>) {
    let mut weights = HashMap::new();
    weights.insert(
        weight_name.to_string(),
        Tensor::new(pseudo_random(k * n, b_seed), vec![k, n]),
    );
    let mut intermediates = HashMap::new();
    intermediates.insert(
        "a".to_string(),
        Tensor::new(pseudo_random(k, a_seed), vec![1, k]),
    );
    (matmul_node(&["a", weight_name]), weights, intermediates)
}

#[test]
fn replay_is_bit_identical_to_the_ordinary_launch_path() {
    let Some(ctx) = device() else {
        eprintln!("no CUDA device -- skipping");
        return;
    };
    let (node, weights, inter) = skinny_case("w_replay", 512, 2048, 11, 22);

    // Ordinary path, twice: the first call makes the weight resident, the
    // second is the steady state the graph path will be compared against.
    ctx.set_graph_capture(false);
    let Some(_) = dispatch(&ctx, &node, &weights, &inter) else {
        eprintln!("dispatch declined this shape -- skipping");
        return;
    };
    let expected = dispatch(&ctx, &node, &weights, &inter).expect("second ordinary dispatch");

    // Graph path: the first call records (and replays what it recorded), so
    // both of these must match.
    ctx.set_graph_capture(true);
    let recorded = dispatch(&ctx, &node, &weights, &inter).expect("recording dispatch");
    let replayed = dispatch(&ctx, &node, &weights, &inter).expect("replaying dispatch");

    let (total, poisoned) = ctx.graph_stats();
    assert_eq!(
        (total, poisoned),
        (1, 0),
        "the skinny GEMM shape must record exactly one graph and poison none; \
         a poisoned key here means capture failed and this test proved nothing",
    );

    assert_eq!(
        recorded.len(),
        expected.len(),
        "recording dispatch changed the output length"
    );
    // Bit-for-bit, deliberately: see this file's header.
    assert_eq!(
        recorded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "the recording dispatch's own result differs from the ordinary path",
    );
    assert_eq!(
        replayed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "a replayed graph differs from the ordinary launch path",
    );
}

#[test]
fn replay_tracks_a_new_activation_rather_than_repeating_the_recorded_one() {
    let Some(ctx) = device() else {
        eprintln!("no CUDA device -- skipping");
        return;
    };
    let (node, weights, first_frame) = skinny_case("w_frames", 512, 1024, 31, 41);

    ctx.set_graph_capture(false);
    if dispatch(&ctx, &node, &weights, &first_frame).is_none() {
        eprintln!("dispatch declined this shape -- skipping");
        return;
    }
    ctx.set_graph_capture(true);
    let frame_one = dispatch(&ctx, &node, &weights, &first_frame).expect("frame 1");

    // A different activation, everything else identical: exactly what the next
    // video frame looks like.
    let mut second_frame = HashMap::new();
    second_frame.insert(
        "a".to_string(),
        Tensor::new(pseudo_random(512, 999), vec![1, 512]),
    );
    let frame_two = dispatch(&ctx, &node, &weights, &second_frame).expect("frame 2");

    assert_ne!(
        frame_one, frame_two,
        "replay returned the recorded frame's answer for a new activation -- the new input never \
         reached the address the graph baked in",
    );

    // ...and the new answer is the *right* one, not merely a different one.
    ctx.set_graph_capture(false);
    let expected = dispatch(&ctx, &node, &weights, &second_frame).expect("frame 2, ordinary path");
    assert_eq!(
        frame_two.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "replay tracked the new activation but computed the wrong answer from it",
    );
}

#[test]
fn two_identical_shapes_over_different_weights_do_not_share_a_recording() {
    let Some(ctx) = device() else {
        eprintln!("no CUDA device -- skipping");
        return;
    };
    // Same m/k/n, same activation, two different resident weights. If the
    // cache key did not carry each weight's device address, the second node
    // would replay the first node's recording and return the first weight's
    // product — right shape, right magnitude, wrong numbers.
    let (node_a, weights_a, inter) = skinny_case("w_left", 512, 1024, 7, 101);
    let mut weights_b = HashMap::new();
    weights_b.insert(
        "w_right".to_string(),
        Tensor::new(pseudo_random(512 * 1024, 202), vec![512, 1024]),
    );
    let node_b = matmul_node(&["a", "w_right"]);

    // Ordinary path first, both to make the weights resident and to capture
    // the expectations.
    ctx.set_graph_capture(false);
    let Some(expect_a) = dispatch(&ctx, &node_a, &weights_a, &inter) else {
        eprintln!("dispatch declined this shape -- skipping");
        return;
    };
    let expect_b = dispatch(&ctx, &node_b, &weights_b, &inter).expect("ordinary dispatch, right");
    assert_ne!(
        expect_a, expect_b,
        "the two weights must produce different products for this test to mean anything",
    );

    ctx.set_graph_capture(true);
    // Record both, then replay both, interleaved — the order a real graph
    // would visit them in.
    let _ = dispatch(&ctx, &node_a, &weights_a, &inter).expect("record left");
    let _ = dispatch(&ctx, &node_b, &weights_b, &inter).expect("record right");
    let replay_a = dispatch(&ctx, &node_a, &weights_a, &inter).expect("replay left");
    let replay_b = dispatch(&ctx, &node_b, &weights_b, &inter).expect("replay right");

    let (total, poisoned) = ctx.graph_stats();
    assert_eq!(
        (total, poisoned),
        (2, 0),
        "two distinct weights over one shape must record TWO graphs; one would mean the key \
         collapses them",
    );
    assert_eq!(
        replay_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expect_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "the left node's replay does not match its ordinary result",
    );
    assert_eq!(
        replay_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expect_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "the right node's replay does not match its ordinary result -- it very likely replayed \
         the left node's recording",
    );
}

#[test]
fn a_batched_strided_dispatch_records_and_replays_exactly() {
    let Some(ctx) = device() else {
        eprintln!("no CUDA device -- skipping");
        return;
    };
    // `GemmPlan::StridedBatch`: `gemm_strided_batched` is a host-side loop of
    // one launch per batch element, so this records the most nodes of any
    // dispatch in the crate — and is where a graph that silently dropped
    // launches would show up as a partially-computed output.
    let (batch, m, k, n) = (6usize, 32usize, 64usize, 48usize);
    let mut inter = HashMap::new();
    inter.insert(
        "a".to_string(),
        Tensor::new(pseudo_random(batch * m * k, 5), vec![batch, m, k]),
    );
    inter.insert(
        "b".to_string(),
        Tensor::new(pseudo_random(batch * k * n, 6), vec![batch, k, n]),
    );
    let node = matmul_node(&["a", "b"]);

    ctx.set_graph_capture(false);
    let Some(expected) = dispatch(&ctx, &node, &HashMap::new(), &inter) else {
        eprintln!("dispatch declined this shape -- skipping");
        return;
    };
    assert_eq!(expected.len(), batch * m * n);

    ctx.set_graph_capture(true);
    let _ = dispatch(&ctx, &node, &HashMap::new(), &inter).expect("record");
    let replayed = dispatch(&ctx, &node, &HashMap::new(), &inter).expect("replay");
    let (_, poisoned) = ctx.graph_stats();
    assert_eq!(poisoned, 0, "the strided-batch dispatch failed to record");
    assert_eq!(
        replayed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "a replayed strided-batch graph differs from the ordinary launch path",
    );
}

#[test]
fn the_toggle_actually_gates_recording() {
    let Some(ctx) = device() else {
        eprintln!("no CUDA device -- skipping");
        return;
    };
    let (node, weights, inter) = skinny_case("w_toggle", 512, 512, 13, 17);

    ctx.set_graph_capture(false);
    assert!(!ctx.graph_capture_enabled());
    if dispatch(&ctx, &node, &weights, &inter).is_none() {
        eprintln!("dispatch declined this shape -- skipping");
        return;
    }
    // Several more dispatches with the toggle off must record nothing at all.
    for _ in 0..3 {
        let _ = dispatch(&ctx, &node, &weights, &inter);
    }
    assert_eq!(
        ctx.graph_stats(),
        (0, 0),
        "dispatches recorded a graph while capture was switched off",
    );

    ctx.set_graph_capture(true);
    assert!(ctx.graph_capture_enabled());
    let _ = dispatch(&ctx, &node, &weights, &inter);
    assert_eq!(
        ctx.graph_stats().0,
        1,
        "switching capture on did not record",
    );

    // Switching back off must keep serving correct results from the ordinary
    // path, with the recording left intact rather than torn down.
    ctx.set_graph_capture(false);
    let after_off =
        dispatch(&ctx, &node, &weights, &inter).expect("ordinary dispatch after toggle");
    ctx.set_graph_capture(true);
    let after_on = dispatch(&ctx, &node, &weights, &inter).expect("replay after toggle back on");
    assert_eq!(
        ctx.graph_stats().0,
        1,
        "toggling off and on again re-recorded an already-recorded key",
    );
    assert_eq!(
        after_off.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        after_on.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "the two paths diverge across a toggle",
    );
}

#[test]
fn a_declined_first_dispatch_leaves_the_cache_clean() {
    let Some(ctx) = device() else {
        eprintln!("no CUDA device -- skipping");
        return;
    };
    // A node `try_cuda_dispatch` declines outright (`Reshape` has no CUDA
    // arm). Nothing may be recorded for it, and nothing may be poisoned:
    // a decline is not a capture failure, and conflating the two would fill
    // the cache with keys for ops that never reach it.
    let node = Node {
        op: OpKind::Reshape,
        name: "declined".to_string(),
        inputs: vec!["a".to_string(), "shape".to_string()],
        outputs: vec!["y".to_string()],
        attrs: Attributes::default(),
    };
    let mut inter = HashMap::new();
    inter.insert("a".to_string(), Tensor::new(vec![1.0, 2.0], vec![2]));
    inter.insert("shape".to_string(), Tensor::new(vec![2.0], vec![1]));

    ctx.set_graph_capture(true);
    assert!(dispatch(&ctx, &node, &HashMap::new(), &inter).is_none());
    assert_eq!(
        ctx.graph_stats(),
        (0, 0),
        "a declined op must not enter the graph cache in any state",
    );
}
