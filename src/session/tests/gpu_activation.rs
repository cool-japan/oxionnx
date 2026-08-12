//! End-to-end checks for run-scoped activation residency.
//!
//! Two claims, both device-backed, both skipped (loudly) on a machine with no
//! adapter:
//!
//! * **Bit identity.** Turning residency on changes where a value lives, never
//!   what it is. A `Conv -> Relu -> Conv -> Add` chain run both ways must
//!   produce the same `Vec<f32>`, compared exactly.
//! * **Less traffic.** The same chain must move strictly fewer bytes across the
//!   bus with residency on — otherwise the mechanism is bookkeeping.
//!
//! Plus the lifetime claim: when the run ends, every activation's buffer is
//! gone and the context's live-byte total is back at its resident-weight
//! baseline.
//!
//! # Why these chains
//!
//! The ops are chosen so that "same value" and "same bits" are the same
//! statement. `Conv` runs the identical implicit-GEMM kernel in both regimes,
//! so its accumulation order cannot change. `Add`, `Mul` and `LeakyRelu` are
//! promoted from the CPU to the GPU by residency — a real placement change —
//! but `select(alpha*x, x, x >= 0)`, IEEE-754 addition and IEEE-754
//! multiplication are exact and order-free, so both sides agree to the bit.
//!
//! Two graphs, because the optimizer folds one of them. `Conv -> Relu -> Conv
//! -> Add` is the mandated chain, and the Conv+Relu fusion pass
//! (`optimizer::fusion::conv::relu`) folds its `Relu` into the first
//! convolution's epilogue before the run loop ever sees it — so what executes
//! is `Conv(relu) -> Conv -> Add`, which exercises a resident convolution
//! result and a promoted binary operand but no standalone element-wise node.
//! [`a_standalone_elementwise_node_reaches_the_gpu_only_when_resident`] covers
//! that separately with `LeakyRelu`, which no fusion pass claims — and which is
//! the shape InSwapper's 57 memory-bound nodes actually have.
//!
//! That is a property of these ops, **not** of residency in general. Promoting
//! a reduction-based op (`OxiInstanceNorm`, `Softmax`, `LayerNorm`,
//! `ReduceMean`) from the CPU to the GPU changes the summation order and
//! therefore the low bits of the result — correct, but not identical. A chain
//! containing one belongs in a tolerance test, not this one.

use std::collections::HashMap;

use crate::execution_providers::OpPlacement;
use crate::graph::{Attributes, Graph, Node, OpKind};
use crate::session::gpu_residency::{run_stats, GpuRunStats};
use crate::tensor::Tensor;
use crate::Session;

/// Channels and spatial extent. 32 x 48 x 48 = 73_728 elements per activation,
/// which is above every kernel's own dispatch floor and large enough that the
/// transfer difference between the two regimes is unambiguous.
const C: usize = 32;
const HW: usize = 48;
const ELEMS: usize = C * HW * HW;

/// Deterministic, signed and non-monotonic: a flat fill would hide a buffer
/// bound at the wrong offset, and an all-positive one would make `Relu` the
/// identity.
fn fill(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u32).wrapping_mul(seed).wrapping_add(seed >> 3);
            ((x % 37) as f32) * 0.041 - 0.75
        })
        .collect()
}

fn conv_node(name: &str, input: &str, output: &str, weight: &str, bias: &str) -> Node {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("pads".to_string(), vec![1, 1, 1, 1]);
    attrs.int_lists.insert("strides".to_string(), vec![1, 1]);
    attrs.int_lists.insert("dilations".to_string(), vec![1, 1]);
    attrs.ints.insert("group".to_string(), 1);
    Node {
        op: OpKind::Conv,
        name: name.to_string(),
        inputs: vec![input.to_string(), weight.to_string(), bias.to_string()],
        outputs: vec![output.to_string()],
        attrs,
    }
}

/// `Conv -> Relu -> Conv -> Add`, every intermediate shaped `[1, C, HW, HW]`.
fn chain_graph() -> (Graph, HashMap<String, Tensor>) {
    let nodes = vec![
        conv_node("conv1", "x", "h1", "conv1.weight", "conv1.bias"),
        Node {
            op: OpKind::Relu,
            name: "relu".to_string(),
            inputs: vec!["h1".to_string()],
            outputs: vec!["h2".to_string()],
            attrs: Attributes::default(),
        },
        conv_node("conv2", "h2", "h3", "conv2.weight", "conv2.bias"),
        Node {
            op: OpKind::Add,
            name: "add".to_string(),
            inputs: vec!["h3".to_string(), "residual".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        },
    ];
    let graph = Graph {
        nodes,
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        ..Default::default()
    };

    let mut weights = HashMap::new();
    for (index, prefix) in ["conv1", "conv2"].iter().enumerate() {
        let seed = 13 + index as u32 * 7;
        weights.insert(
            format!("{prefix}.weight"),
            Tensor::new(fill(C * C * 3 * 3, seed), vec![C, C, 3, 3]),
        );
        weights.insert(
            format!("{prefix}.bias"),
            Tensor::new(fill(C, seed + 3), vec![C]),
        );
    }
    weights.insert(
        "residual".to_string(),
        Tensor::new(fill(ELEMS, 101), vec![1, C, HW, HW]),
    );
    (graph, weights)
}

/// A session with a device, `Auto` placement and a warm weight cache.
///
/// The warm-up run matters: the weight cache is session-lifetime, so measuring
/// a cold run against a warm one would credit residency with the initializer
/// uploads that weight residency already removed.
fn warm_session() -> Option<(Session, HashMap<&'static str, Tensor>)> {
    let (graph, weights) = chain_graph();
    let mut session = Session::from_graph(graph, weights).ok()?;
    if !pollster::block_on(session.enable_gpu_async()) {
        eprintln!("skip: no GPU adapter available");
        return None;
    }
    // The crate default is `CpuOnly`, under which wgpu is never offered a node
    // and this test would prove nothing.
    session.op_placement = OpPlacement::Auto {
        gpu_threshold_bytes: 65_536,
    };
    let mut inputs = HashMap::new();
    inputs.insert("x", Tensor::new(fill(ELEMS, 5), vec![1, C, HW, HW]));
    let _warm = pollster::block_on(session.run_gpu_async(&inputs)).ok()?;
    if run_stats().gpu_nodes == 0 {
        eprintln!("skip: the adapter declined every node of the chain");
        return None;
    }
    Some((session, inputs))
}

/// Every byte this run moved between host and device, in both directions.
fn transfer_bytes(session: &Session, uploads_before: u64, stats: &GpuRunStats) -> u64 {
    let uploaded = session
        .gpu
        .as_ref()
        .map_or(0, |ctx| ctx.uploaded_bytes().saturating_sub(uploads_before));
    uploaded
        .saturating_add(stats.readback_bytes)
        .saturating_add(stats.activation_readback_bytes)
}

/// Run the chain once and report `(outputs, stats, bytes moved)`.
fn measure(
    session: &Session,
    inputs: &HashMap<&'static str, Tensor>,
) -> (HashMap<String, Tensor>, GpuRunStats, u64) {
    let before = session.gpu.as_ref().map_or(0, |ctx| ctx.uploaded_bytes());
    let outputs = pollster::block_on(session.run_gpu_async(inputs)).expect("run");
    let stats = run_stats();
    let bytes = transfer_bytes(session, before, &stats);
    (outputs, stats, bytes)
}

/// The headline: identical values, strictly less traffic.
#[test]
fn residency_is_bit_identical_and_moves_fewer_bytes() {
    let Some((session, inputs)) = warm_session() else {
        return;
    };
    let Some(ctx) = session.gpu.as_ref() else {
        return;
    };

    ctx.set_activation_residency(false);
    let (off_outputs, off_stats, off_bytes) = measure(&session, &inputs);

    ctx.set_activation_residency(true);
    let (on_outputs, on_stats, on_bytes) = measure(&session, &inputs);

    assert!(
        !ctx.is_degraded(),
        "device degraded during the comparison: {:?}",
        ctx.last_error()
    );

    let off_y = off_outputs.get("y").expect("output with residency off");
    let on_y = on_outputs.get("y").expect("output with residency on");
    assert_eq!(off_y.shape, vec![1, C, HW, HW]);
    assert_eq!(on_y.shape, off_y.shape);
    // Exact. Every op in this chain is bit-stable across the placement change
    // the toggle causes — see the module docs for why that is a property of
    // these four ops and not of residency generally.
    assert_eq!(
        on_y.data, off_y.data,
        "residency changed a value; it may only change where the value lives",
    );

    // If the adapter declined the resident path entirely there is nothing to
    // compare, and saying so is better than asserting a tautology.
    if on_stats.resident_outputs == 0 {
        eprintln!(
            "skip: no node kept its output on the device (gpu_nodes={}, cpu_nodes={})",
            on_stats.gpu_nodes, on_stats.cpu_nodes
        );
        return;
    }

    assert_eq!(
        off_stats.resident_outputs, 0,
        "with the switch off nothing may stay on the device",
    );
    assert_eq!(off_stats.resident_operands, 0);
    assert!(
        on_bytes < off_bytes,
        "residency must move strictly fewer bytes: on={on_bytes} off={off_bytes} \
         (on: {} readback + {} activation readback; off: {} readback)",
        on_stats.readback_bytes,
        on_stats.activation_readback_bytes,
        off_stats.readback_bytes,
    );
    assert!(
        on_stats.activation_bytes_saved > 0,
        "a run that kept outputs resident must report the transfers it avoided",
    );
    assert!(
        on_stats.activation_peak_bytes > 0,
        "activations were held on the device, so the peak cannot be zero",
    );
    // The chain's whole point: `Relu` and `Add` carry `usize::MAX` floors while
    // transferring, so they only ever reach the GPU once their operands are
    // already there.
    assert!(
        on_stats.gpu_nodes > off_stats.gpu_nodes,
        "residency must open the memory-bound gate: on={} off={}",
        on_stats.gpu_nodes,
        off_stats.gpu_nodes,
    );
}

/// `Conv -> LeakyRelu -> Conv -> Mul`: the same chain with an element-wise node
/// no fusion pass folds away.
///
/// `LeakyRelu` carries `MEMORY_BOUND_TRANSFER_FLOOR` (`usize::MAX`), so it can
/// never reach the GPU while its operand has to be uploaded — it is one of the
/// 57 InSwapper nodes that decline at every size. Residency is the only thing
/// that opens that gate, which makes "it ran on the GPU" a direct observation
/// of the mechanism rather than an inference from byte counts.
fn standalone_chain_graph() -> (Graph, HashMap<String, Tensor>) {
    let mut leaky_attrs = Attributes::default();
    leaky_attrs.floats.insert("alpha".to_string(), 0.1);
    let nodes = vec![
        conv_node("conv1", "x", "h1", "conv1.weight", "conv1.bias"),
        Node {
            op: OpKind::LeakyRelu,
            name: "leaky".to_string(),
            inputs: vec!["h1".to_string()],
            outputs: vec!["h2".to_string()],
            attrs: leaky_attrs,
        },
        conv_node("conv2", "h2", "h3", "conv2.weight", "conv2.bias"),
        Node {
            op: OpKind::Mul,
            name: "mul".to_string(),
            inputs: vec!["h3".to_string(), "residual".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        },
    ];
    let graph = Graph {
        nodes,
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        ..Default::default()
    };
    (graph, chain_graph().1)
}

/// A standalone element-wise node runs on the GPU with residency on and on the
/// CPU with it off — and computes the same bits either way.
#[test]
fn a_standalone_elementwise_node_reaches_the_gpu_only_when_resident() {
    let (graph, weights) = standalone_chain_graph();
    let Ok(mut session) = Session::from_graph(graph, weights) else {
        return;
    };
    if !pollster::block_on(session.enable_gpu_async()) {
        eprintln!("skip: no GPU adapter available");
        return;
    }
    session.op_placement = OpPlacement::Auto {
        gpu_threshold_bytes: 65_536,
    };
    let mut inputs = HashMap::new();
    inputs.insert("x", Tensor::new(fill(ELEMS, 5), vec![1, C, HW, HW]));
    // Warm the session-lifetime weight cache so neither measurement is charged
    // for uploads the other has already paid.
    let _warm = pollster::block_on(session.run_gpu_async(&inputs)).expect("warm-up run");
    let Some(ctx) = session.gpu.as_ref() else {
        return;
    };

    ctx.set_activation_residency(false);
    let (off_outputs, off_stats, off_bytes) = measure(&session, &inputs);
    ctx.set_activation_residency(true);
    let (on_outputs, on_stats, on_bytes) = measure(&session, &inputs);
    assert!(
        !ctx.is_degraded(),
        "device degraded: {:?}",
        ctx.last_error()
    );

    let off_y = off_outputs.get("y").expect("output with residency off");
    let on_y = on_outputs.get("y").expect("output with residency on");
    assert_eq!(
        on_y.data, off_y.data,
        "promoting LeakyRelu and Mul onto the GPU must not change a single bit",
    );

    if on_stats.resident_outputs == 0 {
        eprintln!("skip: the adapter kept nothing on the device");
        return;
    }
    assert_eq!(
        off_stats.gpu_time_by_op.get("LeakyRelu"),
        None,
        "LeakyRelu carries a usize::MAX transfer floor; it must never dispatch \
         while its operand has to be uploaded",
    );
    assert!(
        on_stats.gpu_time_by_op.contains_key("LeakyRelu"),
        "with its operand already on the device, LeakyRelu must dispatch — that \
         gate opening is what removes InSwapper's conv/elementwise ping-pong",
    );
    assert!(
        on_stats.gpu_time_by_op.contains_key("Mul"),
        "Mul's small host operand is uploaded so the node dispatches in place",
    );
    assert!(
        on_bytes < off_bytes,
        "residency must move strictly fewer bytes: on={on_bytes} off={off_bytes}",
    );
}

/// Every activation's buffer is released at its last consumer, so a finished
/// run leaves the device holding exactly the resident weights plus a pool whose
/// entries are all reusable.
///
/// \[w4\] The pool clear below used to be a precaution and is now the mechanism:
/// a released activation is *recycled* into the pool rather than destroyed, so
/// its bytes are still live when the run ends. The property being asserted is
/// unchanged and no weaker — after the run, the only device bytes that are not
/// reclaimable are the session-lifetime weights — but it is now the pool, not
/// `TrackedBuffer::drop`, that holds them in the interim. Nothing is stranded:
/// `GpuBufferPool::reclaim_for` empties the pool before any allocation is
/// declined, which is what this clear does by hand.
#[test]
fn a_finished_run_returns_live_bytes_to_the_resident_weight_baseline() {
    let Some((session, inputs)) = warm_session() else {
        return;
    };
    let Some(ctx) = session.gpu.as_ref() else {
        return;
    };
    ctx.set_activation_residency(true);

    let (_outputs, stats, _bytes) = measure(&session, &inputs);
    if stats.resident_outputs == 0 {
        eprintln!("skip: no node kept its output on the device");
        return;
    }
    assert!(
        stats.activation_peak_bytes > 0,
        "the run held activations, so its peak must be recorded",
    );

    // Idle pooled buffers are live device bytes too, and they are not what this
    // assertion is about — they belong to the reusable-buffer pool, which
    // outlives any one run by design. Clearing them leaves exactly the bytes
    // nothing will ever release on its own.
    if let Ok(mut pool) = ctx.pool.lock() {
        pool.clear();
    }
    assert_eq!(
        ctx.live_gpu_bytes(),
        ctx.resident_bytes(),
        "a finished run must leave only the session-lifetime weights on the \
         device: live={} resident={} (peak activations this run: {})",
        ctx.live_gpu_bytes(),
        ctx.resident_bytes(),
        stats.activation_peak_bytes,
    );
    assert!(
        ctx.resident_bytes() > 0,
        "the chain has convolution weights, which must be resident",
    );
}

/// \[w4\] A released activation goes back to the reusable-buffer pool, the
/// buffers there are actually reused, and the pool never leaves its own
/// retention bounds.
///
/// The three halves are one claim. Recycling is only worth having if the
/// buffers are reused — an engine that pooled them and then allocated fresh
/// ones anyway would pay the memory and none of the saving — and it is only
/// affordable if the pooled total is bounded.
///
/// **Bounded is not the same as flat, and this asserts the true property.** A
/// graph that hands the pool more buffers per frame than it takes out walks the
/// count up until LRU eviction holds it at `max_buffers`; this chain does
/// exactly that (its `Add` promotes a host operand every frame, which is
/// uploaded outside the pool and recycled into it), so its pooled total climbs
/// over the first frames and then stops. Asserting flatness from frame one
/// would have been asserting a property this engine does not have — the first
/// draft of this test did, and failed. What is guaranteed is that the count and
/// the byte total never exceed the bounds the pool was constructed with, and
/// that once the count is at its bound the bytes stop moving.
#[test]
fn released_activations_are_reused_and_the_pool_stays_bounded() {
    let Some((session, inputs)) = warm_session() else {
        return;
    };
    let Some(ctx) = session.gpu.as_ref() else {
        return;
    };
    ctx.set_activation_residency(true);

    let (_, max_buffers) = ctx.pooled_buffers();
    let byte_budget = ctx.pool_byte_budget();
    assert!(
        max_buffers > 0 && byte_budget > 0,
        "the context's pool must have real retention bounds to test against",
    );

    // Enough frames to saturate the count bound, checking the bounds hold at
    // every step rather than only at the end.
    let frames = max_buffers * 2;
    let reuses_before = ctx.pool_reuses();
    let allocations_before = ctx.pool_allocations();
    let mut first_output: Option<Vec<f32>> = None;
    let mut pooled_bytes = Vec::with_capacity(frames);
    for frame in 0..frames {
        let (outputs, stats, _bytes) = measure(&session, &inputs);
        if frame == 0 && stats.resident_outputs == 0 {
            eprintln!("skip: no node kept its output on the device");
            return;
        }
        let y = outputs.get("y").map(|t| t.data.clone());
        match &first_output {
            None => first_output = y,
            Some(want) => assert_eq!(
                Some(want),
                y.as_ref(),
                "a recycled buffer holds the previous frame's bytes rather than \
                 the driver's zeroes, so two frames disagreeing at frame {frame} \
                 means some kernel is reading its output buffer before writing it",
            ),
        }
        let (held, cap) = session.gpu_pooled_buffers();
        assert!(
            held <= cap,
            "frame {frame}: the pool holds {held} buffers against a {cap} bound",
        );
        assert!(
            session.gpu_pooled_bytes() <= byte_budget,
            "frame {frame}: the pool holds {} B against a {byte_budget} B budget",
            session.gpu_pooled_bytes(),
        );
        assert!(
            session.gpu_pooled_bytes() <= session.gpu_live_bytes(),
            "frame {frame}: idle pooled bytes are a subset of the live total, \
             which is what makes clearing the pool a reclamation and not a leak",
        );
        pooled_bytes.push(session.gpu_pooled_bytes());
    }

    let reuses = session.gpu_pool_reuses().saturating_sub(reuses_before);
    let allocations = session
        .gpu_pool_allocations()
        .saturating_sub(allocations_before);
    assert!(
        reuses > allocations,
        "a warm session must serve most buffer requests from the pool, or \
         recycling is paying memory for nothing: {reuses} reused against \
         {allocations} allocated over {frames} frames",
    );

    // The second half of the run is past saturation, so the pooled total must
    // have stopped moving there. This is the run-over-run growth check, taken
    // where the claim is actually true.
    let tail = pooled_bytes.split_off(frames / 2);
    let first_tail = tail.first().copied().unwrap_or(0);
    assert!(
        tail.iter().all(|bytes| *bytes == first_tail),
        "past saturation the pooled total must be flat, but the last {} frames \
         read {tail:?}",
        tail.len(),
    );
}

/// Residency must survive being switched off and on again mid-session, and a
/// second resident run must agree with the first — the map is per-run, so a
/// value leaking across the boundary would show up here as a changed result.
#[test]
fn consecutive_resident_runs_agree_with_each_other() {
    let Some((session, inputs)) = warm_session() else {
        return;
    };
    let Some(ctx) = session.gpu.as_ref() else {
        return;
    };
    ctx.set_activation_residency(true);

    let (first, first_stats, _) = measure(&session, &inputs);
    let (second, second_stats, _) = measure(&session, &inputs);
    assert!(
        !ctx.is_degraded(),
        "device degraded: {:?}",
        ctx.last_error()
    );

    let first_y = first.get("y").expect("first output");
    let second_y = second.get("y").expect("second output");
    assert_eq!(
        first_y.data, second_y.data,
        "two identical resident runs must agree to the bit",
    );
    assert_eq!(
        first_stats.gpu_nodes, second_stats.gpu_nodes,
        "residency must not drift in what it accepts between frames",
    );
    assert_eq!(
        first_stats.resident_outputs, second_stats.resident_outputs,
        "the set of values kept on the device is a property of the graph",
    );
}

/// A resident value in front of a consumer that declines at run time is read
/// back exactly once, and the run still produces the right answer.
///
/// `Pad` with `mode="edge"` is the deterministic way to force that: the slot
/// table says `Pad` binds its input in place, so the convolution ahead of it
/// keeps its result on the device — and then the arm declines, because no WGSL
/// entry point implements edge padding (`pad_mode_for_gpu`). Every other route
/// into this path (a budget refusal, a device error) is by nature hard to
/// trigger on demand; this one is a property of the graph.
#[test]
fn a_consumer_that_declines_reads_its_resident_operand_back_once() {
    let mut pad_attrs = Attributes::default();
    pad_attrs
        .strings
        .insert("mode".to_string(), "edge".to_string());
    let nodes = vec![
        conv_node("conv1", "x", "h1", "conv1.weight", "conv1.bias"),
        Node {
            op: OpKind::Pad,
            name: "pad".to_string(),
            inputs: vec!["h1".to_string(), "pads".to_string()],
            outputs: vec!["y".to_string()],
            attrs: pad_attrs,
        },
    ];
    let graph = Graph {
        nodes,
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        ..Default::default()
    };
    let mut weights = chain_graph().1;
    weights.insert(
        "pads".to_string(),
        Tensor::new(vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], vec![8]),
    );

    let Ok(mut session) = Session::from_graph(graph, weights) else {
        return;
    };
    if !pollster::block_on(session.enable_gpu_async()) {
        eprintln!("skip: no GPU adapter available");
        return;
    }
    session.op_placement = OpPlacement::Auto {
        gpu_threshold_bytes: 65_536,
    };
    let mut inputs = HashMap::new();
    inputs.insert("x", Tensor::new(fill(ELEMS, 5), vec![1, C, HW, HW]));
    let _warm = pollster::block_on(session.run_gpu_async(&inputs)).expect("warm-up run");
    let Some(ctx) = session.gpu.as_ref() else {
        return;
    };

    ctx.set_activation_residency(false);
    let (off_outputs, off_stats, _) = measure(&session, &inputs);
    ctx.set_activation_residency(true);
    let (on_outputs, on_stats, _) = measure(&session, &inputs);
    assert!(
        !ctx.is_degraded(),
        "device degraded: {:?}",
        ctx.last_error()
    );

    let off_y = off_outputs.get("y").expect("output with residency off");
    let on_y = on_outputs.get("y").expect("output with residency on");
    assert_eq!(
        on_y.data, off_y.data,
        "a declining consumer must see exactly the bytes the GPU produced",
    );
    assert_eq!(
        off_stats.activation_readbacks, 0,
        "nothing is resident with the switch off, so nothing is read back late",
    );
    if on_stats.resident_outputs == 0 {
        eprintln!("skip: the convolution did not keep its result on the device");
        return;
    }
    assert_eq!(
        on_stats.activation_readbacks, 1,
        "the convolution's result is read back once, for the declining Pad",
    );
    assert_eq!(
        on_stats.activation_readback_bytes,
        (ELEMS * std::mem::size_of::<f32>()) as u64,
    );
    assert_eq!(
        on_stats.readbacks, 0,
        "no node read its own result back: the conv kept its result, and the \
         Pad ran on the CPU",
    );
}

// ---------------------------------------------------------------------------
// \[w4\] The shadowed-name path: one name, an initializer *and* a node output.
// ---------------------------------------------------------------------------

/// The name a model uses twice: once as an initializer, once as a node output.
///
/// Legal ONNX, and the only graph shape that reaches `RunActivations::insert`'s
/// *displaced* branch — the one Wave 4 routes into the reusable-buffer pool and
/// that Wave 1-3 dropped on the floor. `materialize_resident_inputs`'
/// `holds_node_output` check and `initializer_key`'s exist for this shape too.
const SHADOWED: &str = "w";

/// The initializer's value, held under [`SHADOWED`] until a node overwrites the
/// name.
///
/// **Positive and constant on purpose.** It is what makes a mix-up detectable
/// by sign alone — see
/// [`a_node_output_shadowing_a_promoted_initializer_wins_the_name`].
const SHADOW_INITIALIZER: f32 = 3.0;

/// `Conv -> Mul -> Conv -> LeakyRelu -> Abs -> Mul`, built so that one run
/// reaches `insert_promoted(SHADOWED, ..)` and then `insert_output(SHADOWED,
/// ..)`.
///
/// Node by node, with the residency consequence of each:
///
/// 0. `conv1(x)` -> `h1`. Compute-bound, dispatches at any tier, and `h1` is
///    keepable (consumed by a `Mul` slot, not a graph output) — so its result
///    stays in a device buffer.
/// 1. `mul_promote(h1, w)` -> `h2`. `Mul` carries `MEMORY_BOUND_TRANSFER_FLOOR`
///    while transferring, its slot 1 accepts a resident operand, and the host
///    operand `w` is exactly as large as the resident `h1` — the three
///    conditions `promote_operands_async` needs. **`w` is uploaded here**:
///    `insert_promoted(SHADOWED, ..)`.
/// 2. `conv2(h2)` -> `h3`, resident again, and read by two later nodes.
/// 3. `shadow(h3)` -> **`w`**. `LeakyRelu` is claimed by no fusion pass, so it
///    survives `OptLevel::All` as a standalone node; its output is keepable
///    (both consumers of `w` bind it in a resident-capable slot and `w` is not
///    a graph output), so it is produced straight onto the device. **This is
///    the displacement**: `insert_output(SHADOWED, ..)` finds the promoted
///    initializer still in the map and must dispose of it.
/// 4. `abs(h3)` -> `h4`. Its only job is to be a node the *last* consumer of
///    `w` depends on, so that consumer sorts after node 3 — see below.
/// 5. `mul_consume(w, h4)` -> `y`, the graph output. Reads `w` and must see
///    node 3's values, not the initializer's.
///
/// # Why the node order survives the topological sort
///
/// `Graph::topological_sort` is given `known` = every initializer name plus
/// every graph input, and it adds **no edge** for an input that is already
/// known. `w` is an initializer, so the `w` edges (node 3 -> node 1 and node 3
/// -> node 5) are invisible to it and cannot reorder anything. That is exactly
/// what this test needs — the promoting consumer *must* run before the
/// producer — but it is also why node 5 cannot depend on `w` alone: with its
/// only real edge invisible it would have in-degree 0 and be scheduled first.
/// `h4` is that real edge, and `abs` exists to provide it.
///
/// # Why both `w`s have the same shape
///
/// `dispatch_to_wgpu_async` validates a device-resident output against
/// `resolved_shapes`, and shape inference seeds `w` from the weight map. A
/// smaller, broadcast-shaped initializer (`[1, C, 1, 1]`, tempting because the
/// promotion would be cheaper) would make node 3 fail the check and return
/// `ShapeMismatch` instead of exercising anything.
fn shadowed_initializer_graph() -> (Graph, HashMap<String, Tensor>) {
    let mut leaky_attrs = Attributes::default();
    leaky_attrs.floats.insert("alpha".to_string(), 0.1);
    let nodes = vec![
        conv_node("conv1", "x", "h1", "conv1.weight", "conv1.bias"),
        Node {
            op: OpKind::Mul,
            name: "mul_promote".to_string(),
            inputs: vec!["h1".to_string(), SHADOWED.to_string()],
            outputs: vec!["h2".to_string()],
            attrs: Attributes::default(),
        },
        conv_node("conv2", "h2", "h3", "conv2.weight", "conv2.bias"),
        Node {
            op: OpKind::LeakyRelu,
            name: "shadow".to_string(),
            inputs: vec!["h3".to_string()],
            outputs: vec![SHADOWED.to_string()],
            attrs: leaky_attrs,
        },
        Node {
            op: OpKind::Abs,
            name: "abs".to_string(),
            inputs: vec!["h3".to_string()],
            outputs: vec!["h4".to_string()],
            attrs: Attributes::default(),
        },
        Node {
            op: OpKind::Mul,
            name: "mul_consume".to_string(),
            inputs: vec![SHADOWED.to_string(), "h4".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        },
    ];
    let graph = Graph {
        nodes,
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        ..Default::default()
    };

    let mut weights = HashMap::new();
    for (index, prefix) in ["conv1", "conv2"].iter().enumerate() {
        let seed = 13 + index as u32 * 7;
        weights.insert(
            format!("{prefix}.weight"),
            Tensor::new(fill(C * C * 3 * 3, seed), vec![C, C, 3, 3]),
        );
        weights.insert(
            format!("{prefix}.bias"),
            Tensor::new(fill(C, seed + 3), vec![C]),
        );
    }
    weights.insert(
        SHADOWED.to_string(),
        Tensor::new(vec![SHADOW_INITIALIZER; ELEMS], vec![1, C, HW, HW]),
    );
    (graph, weights)
}

/// The shadowed-name session: a device, `Auto` placement, a warm weight cache,
/// and a checked node order.
///
/// The order check is an assertion rather than a probe because every claim
/// below rests on it. If a future optimizer pass reordered `shadow` ahead of
/// `mul_promote`, or folded either away, the residency assertions would still
/// fail — but they would fail complaining about upload byte counts, which names
/// the symptom and not the cause.
fn shadowed_session() -> Option<(Session, HashMap<&'static str, Tensor>)> {
    let (graph, weights) = shadowed_initializer_graph();
    let mut session = Session::from_graph(graph, weights).ok()?;
    let order: Vec<(&str, &str)> = session
        .sorted_nodes
        .iter()
        .map(|node| (node.op.as_str(), node.name.as_str()))
        .collect();
    assert_eq!(
        order,
        vec![
            ("Conv", "conv1"),
            ("Mul", "mul_promote"),
            ("Conv", "conv2"),
            ("LeakyRelu", "shadow"),
            ("Abs", "abs"),
            ("Mul", "mul_consume"),
        ],
        "the shadowing node must execute at index 3, after the Mul that \
         promotes the initializer at index 1 and before the Mul that consumes \
         the node output at index 5; the optimizer changed the graph",
    );
    if !pollster::block_on(session.enable_gpu_async()) {
        eprintln!("skip: no GPU adapter available");
        return None;
    }
    session.op_placement = OpPlacement::Auto {
        gpu_threshold_bytes: 65_536,
    };
    let mut inputs = HashMap::new();
    inputs.insert("x", Tensor::new(fill(ELEMS, 5), vec![1, C, HW, HW]));
    // Warm the session-lifetime weight cache, so neither measurement below is
    // charged for uploads the other has already paid.
    let _warm = pollster::block_on(session.run_gpu_async(&inputs)).ok()?;
    Some((session, inputs))
}

/// Activations this graph holds at once at its widest point: at node 1 the
/// promoted `w`, the convolution result `h1` it was promoted alongside, and the
/// `Mul`'s own resident output `h2`.
const PEAK_ACTIVATIONS: usize = 3;

/// A node output that shadows an initializer takes the name from it, and the
/// initializer's promoted device copy is disposed of rather than leaked.
///
/// # How this run reaches the displaced branch
///
/// Three facts, each asserted, and together they leave no other reading:
///
/// * `activation_uploads == 1` with `activation_upload_bytes == ELEMS * 4`.
///   `promote_operands_async` is the only caller of `note_activation_upload`,
///   and in this graph it can fire exactly once: `Conv` returns early (it is
///   not blocked while transferring), `shadow` and `abs` have a single operand
///   which is already resident (no host candidates), and `mul_consume` finds
///   both its operands resident. So the one upload is `w` at node 1, and
///   `insert_promoted(SHADOWED, ..)` ran.
/// * `resident_outputs == 5` — nodes 0 to 4 each kept their result in a device
///   buffer, node 3's among them. Its output is named `w`, so
///   `insert_output(SHADOWED, ..)` ran.
/// * `w`'s last consumer is node 5, so `release_after` cannot have removed it
///   at node 1, 2 or 3. Nothing else removes from the map.
///
/// Therefore the `insert` at node 3 found the promoted value still there and
/// took the displaced branch.
///
/// # How this test fails in each broken world
///
/// * **Promotion never happens** (the operand rule changes, `Mul` stops
///   accepting a resident slot 1, the adapter declines node 1): the upload
///   count is 0, not 1, and the peak drops to two activations.
/// * **The shadowing output is not kept** (node 3 declines, or `w` stops being
///   keepable): `resident_outputs` is 4, not 5.
/// * **The consumer reads the initializer instead of the node output**: the
///   sign check below fails outright, and so does the bit-identity comparison
///   against the residency-off run.
/// * **The displaced buffer is leaked** (`insert` forgetting it rather than
///   disposing of it): caught by
///   [`repeated_shadowed_runs_neither_grow_the_pool_nor_leak_the_displaced_buffer`],
///   which is where the per-frame accounting lives.
#[test]
fn a_node_output_shadowing_a_promoted_initializer_wins_the_name() {
    let Some((session, inputs)) = shadowed_session() else {
        return;
    };
    let Some(ctx) = session.gpu.as_ref() else {
        return;
    };

    ctx.set_activation_residency(false);
    let (off_outputs, off_stats, _off_bytes) = measure(&session, &inputs);
    ctx.set_activation_residency(true);
    let (on_outputs, on_stats, _on_bytes) = measure(&session, &inputs);
    assert!(
        !ctx.is_degraded(),
        "device degraded during the comparison: {:?}",
        ctx.last_error()
    );

    let off_y = off_outputs.get("y").expect("output with residency off");
    let on_y = on_outputs.get("y").expect("output with residency on");
    assert_eq!(on_y.shape, vec![1, C, HW, HW]);

    // The value check first, because it is the one claim here that stands on
    // the resident run **alone** — a bug shared by both regimes would satisfy
    // the bit-identity comparison below and be caught only by this. It is also
    // in a form no accumulation order can blur: `h4` is `|h3|`, so it is
    // non-negative everywhere, and therefore
    //
    // * reading the **node output** gives `y = leaky(h3) * |h3|`, which is
    //   `-0.1 * h3^2` — strictly negative — wherever `h3 < 0`;
    // * reading the **initializer** would give `y = 3.0 * |h3|`, which is
    //   non-negative at *every* element.
    //
    // So a single negative element proves the consumer saw node 3's values;
    // requiring a real fraction of them proves the check has teeth (i.e. that
    // `h3` genuinely straddles zero) rather than resting on one denormal.
    let negatives = on_y.data.iter().filter(|value| **value < 0.0).count();
    assert!(
        negatives > ELEMS / 10,
        "the consumer of the shadowed name must see the node output \
         (leaky(h3) * |h3|, negative wherever h3 < 0), not the initializer \
         ({SHADOW_INITIALIZER} * |h3|, non-negative everywhere): only \
         {negatives} of {ELEMS} outputs are negative",
    );

    assert_eq!(
        on_y.data, off_y.data,
        "residency moved `w` onto the device and displaced a promoted operand \
         to do it; neither may change a single bit of the result",
    );

    assert_eq!(
        off_stats.activation_uploads, 0,
        "with the switch off nothing is promoted and nothing is resident",
    );
    assert_eq!(off_stats.resident_outputs, 0);

    assert_eq!(
        on_stats.gpu_nodes, 6,
        "every node of this graph dispatches once its operands are resident",
    );
    assert_eq!(
        on_stats.activation_uploads, 1,
        "exactly one host operand is promoted in this graph: the initializer \
         `{SHADOWED}` at node 1",
    );
    assert_eq!(
        on_stats.activation_upload_bytes,
        (ELEMS * std::mem::size_of::<f32>()) as u64,
        "and it is `{SHADOWED}`, whose size is the one that identifies it",
    );
    assert_eq!(
        on_stats.resident_outputs, 5,
        "nodes 0..=4 each keep their result on the device — including node 3, \
         whose output is named `{SHADOWED}` and therefore displaces the \
         promoted initializer",
    );
    assert_eq!(
        on_stats.resident_operands, 7,
        "slot by slot: `mul_promote` binds h1 and w, `conv2` binds h2, \
         `shadow` and `abs` bind h3, `mul_consume` binds w and h4",
    );
    // `>=` rather than `==`: a buffer taken from the reusable-buffer pool may
    // reserve more than it needs (`DeviceTensor::reserved_bytes`), and the peak
    // is measured in reserved bytes. The direction that matters is the lower
    // bound — three activations really were live at once, which is only true if
    // the promoted `w` was still held when node 1 produced `h2`.
    assert!(
        on_stats.activation_peak_bytes
            >= (PEAK_ACTIVATIONS * ELEMS * std::mem::size_of::<f32>()) as u64,
        "the promoted `{SHADOWED}`, `h1` and `h2` are live together at node 1, \
         so the peak cannot be under {PEAK_ACTIVATIONS} activations: {}",
        on_stats.activation_peak_bytes,
    );
}

/// Repeating the shadowed-name run neither grows the reusable-buffer pool past
/// its bounds nor leaves a byte behind that clearing the pool cannot reclaim.
///
/// # Why this is the leak check
///
/// The displaced buffer is one `[1, C, HW, HW]` allocation **per frame**. Three
/// worlds are distinguishable here, and only one passes:
///
/// * **Disposed into the pool** (what Wave 4 does): the pool receives one more
///   buffer per frame than this graph takes out of it — the promoted `w` is
///   uploaded fresh every frame by `upload_buffer`, which never draws on the
///   pool, and is recycled into it when node 3 displaces it. So the count
///   climbs to the pool's own `max_buffers` bound and LRU eviction pins it
///   there, the byte total stops moving with it, and clearing the pool returns
///   every one of those bytes.
/// * **Destroyed** (the pre-Wave-4 behaviour): the pool then returns as many
///   buffers per frame as it hands out and never reaches its count bound, so
///   the saturation assertion fails.
/// * **Leaked** (neither disposed nor destroyed): `pool.clear()` cannot reclaim
///   a buffer the pool never received, so the live total ends one allocation
///   per frame above the resident-weight baseline and the final assertion
///   fails.
///
/// The mechanics of the last assertion are
/// [`a_finished_run_returns_live_bytes_to_the_resident_weight_baseline`]'s: a
/// finished run leaves the device holding the session-lifetime weights plus a
/// pool that is by construction reclaimable, and clearing the pool by hand is
/// what `GpuBufferPool::reclaim_for` would do before declining an allocation.
#[test]
fn repeated_shadowed_runs_neither_grow_the_pool_nor_leak_the_displaced_buffer() {
    let Some((session, inputs)) = shadowed_session() else {
        return;
    };
    let Some(ctx) = session.gpu.as_ref() else {
        return;
    };
    ctx.set_activation_residency(true);

    // Read through the `Session` accessors rather than the context's own
    // methods. They are this wave's public window onto exactly the question
    // this test asks, and a delegation typo in one of them — `gpu_pooled_bytes`
    // wired to `live_gpu_bytes`, say — is invisible to the compiler and would
    // otherwise ship uncovered.
    let (_, max_buffers) = session.gpu_pooled_buffers();
    let byte_budget = session.gpu_pool_byte_budget();
    assert!(
        max_buffers > 0 && byte_budget > 0,
        "the context's pool must have real retention bounds to test against",
    );

    // Enough frames to walk the count bound and then sit on it, checking the
    // bounds at every step rather than only at the end.
    let frames = max_buffers * 2;
    let reuses_before = session.gpu_pool_reuses();
    let allocations_before = session.gpu_pool_allocations();
    let mut first_output: Option<Vec<f32>> = None;
    let mut pooled_bytes = Vec::with_capacity(frames);
    for frame in 0..frames {
        let (outputs, stats, _bytes) = measure(&session, &inputs);
        assert_eq!(
            stats.activation_uploads, 1,
            "frame {frame}: the initializer must be promoted on every frame, \
             or the frame did not reach the displaced path at all",
        );
        assert_eq!(
            stats.resident_outputs, 5,
            "frame {frame}: node 3 must keep its `{SHADOWED}` output on the \
             device on every frame",
        );
        let y = outputs.get("y").map(|tensor| tensor.data.clone());
        match &first_output {
            None => first_output = y,
            Some(want) => assert_eq!(
                Some(want),
                y.as_ref(),
                "frame {frame} disagrees with frame 0: a recycled buffer holds \
                 the previous frame's bytes rather than the driver's zeroes, so \
                 a displaced buffer handed back to the pool and then bound \
                 before it is fully written would show up here",
            ),
        }
        let (held, cap) = ctx.pooled_buffers();
        assert!(
            held <= cap,
            "frame {frame}: the pool holds {held} buffers against a {cap} bound",
        );
        assert!(
            ctx.pooled_gpu_bytes() <= byte_budget,
            "frame {frame}: the pool holds {} B against a {byte_budget} B budget",
            ctx.pooled_gpu_bytes(),
        );
        pooled_bytes.push(ctx.pooled_gpu_bytes());
    }

    let reuses = ctx.pool_reuses().saturating_sub(reuses_before);
    let allocations = ctx.pool_allocations().saturating_sub(allocations_before);
    assert!(
        reuses > allocations,
        "a warm session must serve most buffer requests from the pool: \
         {reuses} reused against {allocations} allocated over {frames} frames",
    );
    assert_eq!(
        session.gpu_pooled_buffers().0,
        max_buffers,
        "this graph gives the pool one more buffer per frame than it takes — \
         the displaced `{SHADOWED}` — so over {frames} frames the count must \
         reach its {max_buffers} bound. A displaced buffer that was destroyed \
         or leaked instead of recycled never arrives, and the pool never grows.",
    );

    // Past saturation the count is pinned by LRU eviction, so the byte total
    // must have stopped moving. This is the run-over-run growth check, taken
    // where the claim is true — see
    // `released_activations_are_reused_and_the_pool_stays_bounded` for why
    // flatness from frame one is not a property this engine has.
    let tail = pooled_bytes.split_off(frames / 2);
    let first_tail = tail.first().copied().unwrap_or(0);
    assert!(
        tail.iter().all(|bytes| *bytes == first_tail),
        "past saturation the pooled total must be flat, but the last {} frames \
         read {tail:?}",
        tail.len(),
    );

    if let Ok(mut pool) = ctx.pool.lock() {
        pool.clear();
    }
    assert_eq!(session.gpu_pooled_bytes(), 0, "the pool was just cleared",);
    assert_eq!(
        session.gpu_live_bytes(),
        ctx.resident_bytes(),
        "after {frames} runs the only device bytes left must be the \
         session-lifetime weights: live={} resident={}. One displaced buffer \
         leaked per frame would leave at least {} B more (measured: exactly \
         that plus the warm-up run's).",
        session.gpu_live_bytes(),
        ctx.resident_bytes(),
        frames * ELEMS * std::mem::size_of::<f32>(),
    );
    assert!(
        ctx.resident_bytes() > 0,
        "the graph has convolution weights, which must be resident",
    );
}

/// The displaced buffer goes to the reusable-buffer pool, byte for byte, at the
/// moment it is displaced — and the name then answers with the node output.
///
/// [`a_node_output_shadowing_a_promoted_initializer_wins_the_name`] proves the
/// path is *reached* by a real graph run, and
/// [`repeated_shadowed_runs_neither_grow_the_pool_nor_leak_the_displaced_buffer`]
/// proves nothing is left behind by repeating it. Neither can watch the handoff
/// itself, because it happens inside one node. This one does, by driving
/// [`RunActivations`] against the session's own device: the pool's byte total
/// must rise by exactly the displaced allocation's reserved size across the
/// `insert_output` call and nothing else.
///
/// That is the assertion that separates all three worlds rather than two.
/// Destroying the buffer (Wave 1-3) and leaking it both leave the pooled total
/// unchanged here; only recycling moves it, and only by that exact amount.
#[test]
fn displacing_a_promoted_operand_recycles_its_buffer_into_the_pool() {
    use crate::session::gpu_activations::RunActivations;
    use crate::session::gpu_dispatch::op_accepts_resident_slot;

    let Some((session, _inputs)) = shadowed_session() else {
        return;
    };
    let Some(ctx) = session.gpu.as_ref() else {
        return;
    };
    let (_, max_buffers) = ctx.pooled_buffers();
    assert!(
        max_buffers > 0 && ctx.pool_byte_budget() > 0,
        "the context's pool must have real retention bounds, or `return_buffer` \
         drops what it is handed and this test would read a destroyed buffer as \
         a leaked one",
    );

    let (graph, _weights) = shadowed_initializer_graph();
    let mut activations =
        RunActivations::new(true, &graph.nodes, &graph.output_names, |node, slot| {
            op_accepts_resident_slot(&node.op, slot)
        });
    assert!(
        activations.may_keep(SHADOWED),
        "both consumers of `{SHADOWED}` bind it in a resident-capable slot and \
         it is not a graph output, so a node may produce it onto the device",
    );

    // Two tensors of the same shape and very different values, so the second
    // assertion cannot pass by reading the wrong buffer.
    let shape = vec![1, C, HW, HW];
    let promoted_data = vec![SHADOW_INITIALIZER; ELEMS];
    let produced_data = fill(ELEMS, 29);
    assert_ne!(promoted_data, produced_data);

    let promoted = ctx
        .upload_device_tensor("displaced_test_promoted", &promoted_data, &shape)
        .expect("the device must accept a promoted operand of one activation");
    let displaced_bytes = promoted.reserved_bytes();
    activations.insert_promoted(SHADOWED, promoted, Some(ctx));
    assert!(
        !activations.holds_node_output(SHADOWED),
        "a promoted initializer is not a node output; `initializer_key` and \
         `materialize_resident_inputs` both branch on that",
    );

    let produced = ctx
        .upload_device_tensor("displaced_test_output", &produced_data, &shape)
        .expect("the device must accept a second activation");
    // Snapshot *after* the upload: `upload_buffer` does not draw on the pool
    // today, but a snapshot taken before it would read a pool one entry shorter
    // if that ever changed, and the delta below would be attributed to the
    // wrong thing.
    let pooled_before = ctx.pooled_gpu_bytes();
    activations.insert_output(SHADOWED, produced, Some(ctx));

    assert_eq!(
        ctx.pooled_gpu_bytes(),
        pooled_before + displaced_bytes,
        "displacing a resident value must hand its allocation to the pool, \
         exactly as releasing one at its last consumer does; destroying it \
         (Wave 1-3) or leaking it both leave the pooled total at \
         {pooled_before} B",
    );
    assert!(
        activations.holds_node_output(SHADOWED),
        "the name now answers for a node output, whatever the weight map holds",
    );

    let resident = activations
        .get(SHADOWED)
        .expect("the name must still resolve, to the displacing value");
    let read_back = pollster::block_on(oxionnx_gpu::read_device_tensor_async(ctx, resident))
        .expect("reading the surviving device buffer back");
    assert_eq!(
        read_back.data, produced_data,
        "the surviving buffer must be the node output's, not the promoted \
         initializer's",
    );

    // Leave the device as this test found it. Dropping the map destroys the
    // surviving activation (`RunActivations`' own `Drop` is the backstop that
    // does *not* recycle), and clearing the pool releases the displaced buffer
    // this test put there — so the live total is back where it started, which
    // the last assertion states rather than assumes.
    drop(activations);
    if let Ok(mut pool) = ctx.pool.lock() {
        pool.clear();
    }
    assert_eq!(
        ctx.live_gpu_bytes(),
        ctx.resident_bytes(),
        "both buffers this test uploaded are accounted for: one destroyed with \
         the map, one reclaimed from the pool it was displaced into",
    );
}
