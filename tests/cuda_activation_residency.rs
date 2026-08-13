//! Activation residency for the CUDA execution provider, end to end through
//! `Session::run`, on a real device.
//!
//! Run with:
//!
//! ```text
//! OXIONNX_CUDA=1 cargo test --features cuda --release \
//!     --test cuda_activation_residency -- --test-threads=1
//! ```
//!
//! # What this file proves, and why counters rather than wall clock
//!
//! Residency is not a speed claim, it is a *traffic* claim: a chain of
//! CUDA-claimed nodes should upload its input once, download its output once,
//! and block the host once — not once per node. Wall clock would prove that
//! badly (it varies with a shared GPU, with clocks, with the pool's warm state)
//! and would not distinguish "the transfers are gone" from "the transfers got
//! faster". `CacheCounters` counts the copies at the copies, and the fences at
//! the fences, so the assertions below are exact integers rather than
//! thresholds.
//!
//! Four claims, one test each:
//!
//! 1. **A five-node all-claimed chain costs exactly one upload, one download
//!    and one fence.** The headline. Before residency it was five of each.
//! 2. **A CPU island in the middle costs exactly one extra round trip.** A
//!    value *no* consumer can bind is disqualified by the plan before the run,
//!    so its producer reads it back rather than the island discovering the
//!    problem — one crossing either way, decided in advance.
//! 3. **A node that declines at *dispatch* time reads its resident operand back
//!    exactly once.** The case the plan cannot foresee, and the reason there is
//!    a runtime materialisation point at all: the host copy is memoized into
//!    the run state so a second CPU consumer finds it there.
//! 4. **A steady-state frame allocates nothing and returns every activation.**
//!    The failure residency newly makes possible is an allocation taken out of
//!    the pool and never handed back.
//!
//! Every test also compares its output against the same graph run with
//! `OXIONNX_CUDA_RESIDENCY` off, element for element, because a residency bug's
//! signature is a plausible-looking wrong answer (a stale buffer, a recycled
//! allocation read after release), not a crash.
//!
//! Each test skips when no CUDA device is present, per the OxiCUDA convention.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use oxionnx::graph::{Attributes, Graph, Node, OpKind};
use oxionnx::{OptLevel, Session, SessionBuilder, Tensor};
use oxionnx_cuda::residency::CacheCounters;

/// Elements per activation in the chains below.
///
/// Above `RESIDENT_DISPATCH_FLOOR` (256) so the two-tier gate admits every
/// node, and above the 16 KiB `gpu_threshold_bytes` used here (65 536 elements
/// = 256 KiB) so the *transferring* tier admits the first node too — otherwise
/// the chain would never start on the device and the test would pass
/// vacuously.
const CHAIN_ELEMENTS: usize = 65_536;

/// Bytes one activation occupies on the bus.
const CHAIN_BYTES: u64 = (CHAIN_ELEMENTS * std::mem::size_of::<f32>()) as u64;

fn node(op: OpKind, name: &str, inputs: &[&str], outputs: &[&str]) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: Attributes::default(),
    }
}

/// A graph of `nodes` from `x` to `y`, built as a session with CUDA eligible
/// for every node above the 16 KiB floor `oxiface` uses in production.
///
/// Returns `None` when the session came up without a CUDA context — no driver,
/// no device, or `OXIONNX_CUDA` unset — which is the skip condition for every
/// test here.
fn session_for(nodes: Vec<Node>) -> Option<Session> {
    let graph = Graph {
        nodes,
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        ..Default::default()
    };
    let session = SessionBuilder::new()
        // No fusion: these chains exist to be *chains*, and the optimizer would
        // happily rewrite `Relu -> Relu` into something with fewer node
        // boundaries — which is exactly the thing being counted.
        .with_optimization_level(OptLevel::None)
        .with_parallel_execution(false)
        .with_op_placement(oxionnx::execution_providers::OpPlacement::Auto {
            gpu_threshold_bytes: 16_384,
        })
        .build_from_graph(graph, HashMap::new())
        .expect("build the test session");
    session.cuda_cache_counters()?;
    Some(session)
}

/// An input whose values are distinct per element, so a chain that dropped or
/// duplicated a node's work produces a visibly different answer rather than a
/// coincidentally-equal one.
fn chain_input() -> Tensor {
    let data: Vec<f32> = (0..CHAIN_ELEMENTS)
        .map(|i| ((i % 97) as f32) * 0.125 - 6.0)
        .collect();
    Tensor::new(data, vec![CHAIN_ELEMENTS])
}

/// Run once to warm every session-lifetime cache (PTX modules, the buffer
/// pool), then snapshot, run again, and return the *second* run's counters
/// beside its output.
///
/// The first run is not the interesting one: it JIT-compiles kernels and grows
/// the pool, so its numbers describe session setup rather than a frame. Every
/// production claim in this workspace is about the steady state, and so is
/// every assertion below.
fn steady_state_run(session: &Session, input: &Tensor) -> (Vec<f32>, CacheCounters) {
    let mut inputs: HashMap<&str, Tensor> = HashMap::new();
    inputs.insert("x", input.clone());
    let _warm = session.run(&inputs).expect("warm-up run");

    let before = session
        .cuda_cache_counters()
        .expect("a session with a CUDA context");
    let out = session.run(&inputs).expect("measured run");
    let after = session
        .cuda_cache_counters()
        .expect("a session with a CUDA context");

    let y = out.get("y").expect("graph output y").data.clone();
    (y, after.since(before))
}

/// The same graph, run with residency switched off through
/// [`CUDA_RESIDENCY_ENV_VAR`](oxionnx::session::run::sequential::CUDA_RESIDENCY_ENV_VAR).
///
/// Building a *separate process* would be the airtight way to do this, but the
/// switch is read per run (not cached in a `OnceLock`, deliberately, for
/// exactly this reason), so setting it around a run in a single-threaded test
/// is sufficient and far cheaper.
///
/// # Safety
///
/// `std::env::set_var` is `unsafe` from Rust 2024 because another thread
/// reading the environment concurrently is UB. These tests are run with
/// `--test-threads=1` (see the module docs) and touch no other thread, so no
/// concurrent reader exists.
fn run_without_residency(session: &Session, input: &Tensor) -> (Vec<f32>, CacheCounters) {
    unsafe { std::env::set_var("OXIONNX_CUDA_RESIDENCY", "0") };
    let result = steady_state_run(session, input);
    unsafe { std::env::remove_var("OXIONNX_CUDA_RESIDENCY") };
    result
}

/// Elementwise `Relu`, on the host, for the expectations below.
fn relu(data: &[f32]) -> Vec<f32> {
    data.iter().map(|v| v.max(0.0)).collect()
}

// ───────────────────────────────────────────────────────────────────────────
// 1. The all-claimed chain
// ───────────────────────────────────────────────────────────────────────────

/// Five consecutive CUDA-claimed nodes move the graph input up once, the graph
/// output down once, and block the host once.
///
/// The pre-residency behaviour was five uploads, five downloads and five
/// fences — one complete round trip per node — which is the 1327 MB/frame and
/// 237 fences/frame the transfer audit measured across the real models.
#[test]
fn a_five_node_claimed_chain_costs_one_upload_one_download_and_one_fence() {
    let Some(session) = session_for(vec![
        node(OpKind::Relu, "n0", &["x"], &["h0"]),
        node(OpKind::Relu, "n1", &["h0"], &["h1"]),
        node(OpKind::Relu, "n2", &["h1"], &["h2"]),
        node(OpKind::Relu, "n3", &["h2"], &["h3"]),
        node(OpKind::Relu, "n4", &["h3"], &["y"]),
    ]) else {
        eprintln!("no CUDA device: skipping");
        return;
    };
    assert_eq!(
        session.cuda_streams_unified(),
        Some(true),
        "residency needs one queue; a split-stream context declines it entirely",
    );

    let input = chain_input();
    let (resident_out, counters) = steady_state_run(&session, &input);

    assert_eq!(
        counters.host_to_device_bytes,
        CHAIN_BYTES,
        "exactly the graph input crosses up: {} bytes, i.e. {} activations' worth",
        counters.host_to_device_bytes,
        counters.host_to_device_bytes / CHAIN_BYTES,
    );
    assert_eq!(
        counters.device_to_host_bytes,
        CHAIN_BYTES,
        "exactly the graph output crosses down: {} bytes, i.e. {} activations' worth",
        counters.device_to_host_bytes,
        counters.device_to_host_bytes / CHAIN_BYTES,
    );
    assert_eq!(
        counters.stream_syncs, 1,
        "one fence, at the one host-visible result",
    );
    assert_eq!(
        counters.device_handoffs, 4,
        "the four intermediate outputs stay on the device; only `y` is read back",
    );
    assert_eq!(
        counters.resident_activation_binds, 4,
        "the four nodes after the first bind their operand in place",
    );
    assert_eq!(
        counters.weight_bytes_uploaded, 0,
        "no initializers in this graph, and none invented",
    );

    // ...and the numbers are the same ones the fenced path produces.
    let expected = relu(&input.data);
    assert_eq!(
        resident_out, expected,
        "five Relus is one Relu, elementwise"
    );

    let (fenced_out, fenced) = run_without_residency(&session, &input);
    assert_eq!(
        fenced_out, resident_out,
        "residency must not change a single element",
    );
    assert_eq!(
        fenced.stream_syncs, 5,
        "with residency off the pre-residency cost is back: one fence per node",
    );
    assert_eq!(
        fenced.host_to_device_bytes,
        CHAIN_BYTES * 5,
        "and one upload per node",
    );
    assert_eq!(
        fenced.device_to_host_bytes,
        CHAIN_BYTES * 5,
        "and one download per node",
    );
}

// ───────────────────────────────────────────────────────────────────────────
// 2. The CPU island
// ───────────────────────────────────────────────────────────────────────────

/// A CUDA-incapable node in the middle of a claimed chain costs **one**
/// read-back, not one per consumer of the value it needs.
///
/// `Erf` has no CUDA arm at all (`is_supported_op` reports `false`), so `n2`
/// runs on the host and `h1` has to be materialised for it. What is being
/// pinned here is that the materialisation happens exactly once and that the
/// chain re-enters the device afterwards rather than staying on the host for
/// the rest of the run.
#[test]
fn a_cpu_island_materialises_its_operand_exactly_once() {
    let Some(session) = session_for(vec![
        node(OpKind::Relu, "n0", &["x"], &["h0"]),
        node(OpKind::Relu, "n1", &["h0"], &["h1"]),
        node(OpKind::Erf, "island", &["h1"], &["h2"]),
        node(OpKind::Relu, "n3", &["h2"], &["h3"]),
        node(OpKind::Relu, "n4", &["h3"], &["y"]),
    ]) else {
        eprintln!("no CUDA device: skipping");
        return;
    };

    let input = chain_input();
    let (resident_out, counters) = steady_state_run(&session, &input);

    // Up: the graph input, plus `h2` re-entering the device after the island.
    assert_eq!(
        counters.host_to_device_bytes,
        CHAIN_BYTES * 2,
        "the graph input and the island's result, and nothing else",
    );
    // Down: `h1` for the island, plus the graph output.
    assert_eq!(
        counters.device_to_host_bytes,
        CHAIN_BYTES * 2,
        "`h1` read back once for the CPU node, `y` read back once for the caller",
    );
    assert_eq!(
        counters.stream_syncs, 2,
        "one fence per host-visible value: the materialisation and the output",
    );
    // Two, not three, and the missing one is the point: `h1` has exactly one
    // consumer, `Erf`, which has no CUDA arm — so under either keep policy
    // nothing can bind it and the plan disqualifies it *before the run starts*.
    // `h1` is therefore read back by its own producer (`n1`, dispatched with
    // host placement) rather than kept resident and materialised afterwards.
    // One crossing either way; decided once, in advance, rather than discovered
    // at the consumer. (A value with a capable consumer *and* an incapable one
    // is the case the relaxed policy exists for, and it is cheaper — see
    // `KeepPolicy`'s arithmetic.)
    assert_eq!(
        counters.device_handoffs, 2,
        "`h0` and `h3` stay resident; `h1` is planned host-side for the island's sake, \
         `h2` is produced on the host, and `y` is read back for the caller",
    );

    let expected = {
        let after_relu = relu(&input.data);
        let after_erf: Vec<f32> = after_relu.iter().map(|v| erf(*v)).collect();
        relu(&after_erf)
    };
    assert_eq!(resident_out.len(), expected.len());
    for (index, (got, want)) in resident_out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-5,
            "element {index}: got {got}, expected {want}",
        );
    }

    let (fenced_out, _) = run_without_residency(&session, &input);
    assert_eq!(
        fenced_out, resident_out,
        "a CPU island must not change what residency computes",
    );
}

/// A node that clears the *plan's* capability check but declines at dispatch
/// time reads its resident operand back exactly once, through the
/// materialisation path.
///
/// This is the other half of the island story, and the half that needs a
/// runtime convergence point rather than a graph rule. `Softmax` accepts a
/// resident operand in slot 0, so the plan keeps `h1` on the device — but
/// `cuda_softmax` declines any row wider than 1024, which the `[32, 2048]`
/// shape below is. The node therefore falls through to the CPU operator with
/// its only copy of `h1` in a device buffer, and
/// `materialize_resident_cuda_inputs` is what stops that being a missing
/// tensor.
///
/// "Exactly once" is the claim the memoisation makes: the host copy goes into
/// the run state, so a second CPU consumer of the same value finds it there.
#[test]
fn a_runtime_decline_reads_its_resident_operand_back_exactly_once() {
    let Some(session) = session_for(vec![
        node(OpKind::Relu, "n0", &["x"], &["h0"]),
        node(OpKind::Relu, "n1", &["h0"], &["h1"]),
        // Rows of 2048 — wider than `cuda_softmax`'s 1024-element kernel limit,
        // so this declines *after* the plan has already kept `h1` resident.
        node(OpKind::Softmax, "wide", &["h1"], &["h2"]),
        node(OpKind::Relu, "n3", &["h2"], &["y"]),
    ]) else {
        eprintln!("no CUDA device: skipping");
        return;
    };

    let input = Tensor::new(chain_input().data, vec![32, 2048]);
    let (resident_out, counters) = steady_state_run(&session, &input);

    assert_eq!(
        counters.device_handoffs, 2,
        "`h0` and `h1` are kept — the plan cannot know `Softmax` will decline on width",
    );
    assert_eq!(
        counters.device_to_host_bytes,
        CHAIN_BYTES * 2,
        "`h1` materialised once for the CPU softmax, `y` read back once for the caller",
    );
    assert_eq!(
        counters.host_to_device_bytes,
        CHAIN_BYTES * 2,
        "the graph input, and `h2` re-entering the device for the final Relu",
    );
    assert_eq!(
        counters.stream_syncs, 2,
        "one fence for the materialisation, one for the graph output",
    );

    let (fenced_out, _) = run_without_residency(&session, &input);
    assert_eq!(
        fenced_out.len(),
        resident_out.len(),
        "a runtime decline must not change the output's extent",
    );
    for (index, (got, want)) in resident_out.iter().zip(fenced_out.iter()).enumerate() {
        assert_eq!(
            got, want,
            "element {index}: materialising a resident operand must be bit-identical to \
             never having made it resident",
        );
    }
    // Softmax rows sum to one, and a following Relu leaves non-negative values
    // alone — a property check that would catch a materialisation reading the
    // wrong buffer even if both paths were wrong in the same way.
    for (row, chunk) in resident_out.chunks(2048).enumerate() {
        let sum: f32 = chunk.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "row {row} sums to {sum}, not 1: the softmax did not see the values it should have",
        );
    }
}

/// The error function, to the accuracy the ONNX `Erf` kernel is checked at.
///
/// Abramowitz & Stegun 7.1.26 — enough to confirm the island ran the *host*
/// kernel between two device ones, which is what this file is about; the
/// operator's own accuracy is `oxionnx-ops`' business.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_) * t) + 1.421_413_7) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

// ───────────────────────────────────────────────────────────────────────────
// 3. Steady state
// ───────────────────────────────────────────────────────────────────────────

/// Repeated frames neither grow the pool nor leak an activation.
///
/// The failure this rules out is the one residency makes possible: an
/// allocation taken out of the pool for a resident value and never handed back,
/// which shows as a per-frame `pool_allocs` that refuses to fall to zero. In
/// the steady state every buffer a frame takes is one a previous frame
/// returned.
#[test]
fn repeated_frames_allocate_nothing_and_return_every_activation() {
    let Some(session) = session_for(vec![
        node(OpKind::Relu, "n0", &["x"], &["h0"]),
        node(OpKind::Relu, "n1", &["h0"], &["h1"]),
        node(OpKind::Relu, "n2", &["h1"], &["y"]),
    ]) else {
        eprintln!("no CUDA device: skipping");
        return;
    };

    let input = chain_input();
    let mut inputs: HashMap<&str, Tensor> = HashMap::new();
    inputs.insert("x", input);
    // Two warm-up frames: the first grows the pool, the second proves it has
    // stopped growing before anything is asserted.
    for _ in 0..2 {
        session.run(&inputs).expect("warm-up run");
    }

    let before = session.cuda_cache_counters().expect("a CUDA context");
    for _ in 0..8 {
        session.run(&inputs).expect("steady-state run");
    }
    let delta = session
        .cuda_cache_counters()
        .expect("a CUDA context")
        .since(before);

    assert_eq!(
        delta.pool_allocs, 0,
        "a steady-state frame must not ask the driver for memory: {} allocations over 8 frames",
        delta.pool_allocs,
    );
    assert_eq!(
        delta.activation_recycles, 16,
        "two resident activations per frame, all eight frames, all returned",
    );
    assert_eq!(delta.stream_syncs, 8, "one fence per frame");
    assert_eq!(
        delta.pool_evictions, 0,
        "nothing is dropped on the floor; every borrow comes back",
    );
}
