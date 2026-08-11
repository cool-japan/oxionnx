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

/// Every activation's buffer is destroyed at its last consumer, so a finished
/// run leaves the device holding exactly the resident weights.
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
