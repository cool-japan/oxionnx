//! [w2-f16] Session-level checks for the opt-in half-precision compute path.
//!
//! The kernel-level quality, cache-keying and residency claims live in
//! `oxionnx-gpu/tests/w2_f16_compute.rs`, next to the kernels. What belongs
//! *here* is the claim only a whole session can make:
//!
//! * **With the toggle off, the engine is byte-identical to the tree that had
//!   no half-precision path at all.** Not "close", not "within tolerance" —
//!   the same `Vec<f32>`, compared exactly, on a multi-node convolution graph
//!   that ends in a real graph-output read-back.
//!
//! Plus the two supporting facts: the mode is off on a freshly built session,
//! and turning it on actually reaches the kernels (the result changes, and
//! changes within the quality gate) rather than being an accessor that sets a
//! flag nobody reads. That second one is the failure this file exists to catch:
//! a toggle that silently does nothing would pass every "off is unchanged"
//! assertion trivially.
//!
//! Skipped, loudly, on a machine with no adapter — and separately on an adapter
//! without `shader-f16`, where "on" is legitimately the same as "off".

use std::collections::HashMap;

use crate::execution_providers::OpPlacement;
use crate::graph::{Attributes, Graph, Node, OpKind};
use crate::session::gpu_residency::run_stats;
use crate::tensor::Tensor;
use crate::Session;

/// Wide enough that every convolution clears the 10 MFLOP dispatch gate
/// (`64 x 1024 x 576 = 37.7 MFLOP` each), small enough to stay quick.
const C: usize = 64;
const HW: usize = 32;
const ELEMS: usize = C * HW * HW;

/// Deterministic, signed, non-monotonic — the same shape of fill the
/// residency tests use, for the same reason: a flat or monotonic ramp hides
/// indexing bugs.
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

/// Four chained convolutions with a `Relu` between each pair — enough nodes
/// that a format mix-up anywhere in the chain would show up, and enough
/// distinct initializers (8) that the residency cache is genuinely exercised.
fn conv_chain() -> (Graph, HashMap<String, Tensor>) {
    let mut nodes = Vec::new();
    let mut weights = HashMap::new();
    let layers = 4usize;
    for layer in 0..layers {
        let input = if layer == 0 {
            "x".to_string()
        } else {
            format!("a{layer}")
        };
        let conv_out = format!("h{}", layer + 1);
        let name = format!("conv{}", layer + 1);
        nodes.push(conv_node(
            &name,
            &input,
            &conv_out,
            &format!("{name}.weight"),
            &format!("{name}.bias"),
        ));
        // The last convolution's output is the graph's, so it is read back.
        if layer + 1 < layers {
            nodes.push(Node {
                op: OpKind::Relu,
                name: format!("relu{}", layer + 1),
                inputs: vec![conv_out],
                outputs: vec![format!("a{}", layer + 1)],
                attrs: Attributes::default(),
            });
        }
        let seed = 13 + layer as u32 * 17;
        weights.insert(
            format!("{name}.weight"),
            // Scaled down: four chained 576-deep reductions otherwise grow the
            // activation magnitude past anything a real network produces.
            Tensor::new(
                fill(C * C * 3 * 3, seed)
                    .into_iter()
                    .map(|v| v * 0.05)
                    .collect(),
                vec![C, C, 3, 3],
            ),
        );
        weights.insert(
            format!("{name}.bias"),
            Tensor::new(fill(C, seed + 3), vec![C]),
        );
    }
    let graph = Graph {
        nodes,
        input_names: vec!["x".to_string()],
        output_names: vec![format!("h{layers}")],
        ..Default::default()
    };
    (graph, weights)
}

/// A session with a device, `Auto` placement and a warm weight cache.
fn warm_session() -> Option<(Session, HashMap<&'static str, Tensor>)> {
    let (graph, weights) = conv_chain();
    let mut session = Session::from_graph(graph, weights).ok()?;
    if !pollster::block_on(session.enable_gpu_async()) {
        println!("skip: no GPU adapter available");
        return None;
    }
    session.op_placement = OpPlacement::Auto {
        gpu_threshold_bytes: 65_536,
    };
    let mut inputs = HashMap::new();
    inputs.insert("x", Tensor::new(fill(ELEMS, 5), vec![1, C, HW, HW]));
    let _warm = pollster::block_on(session.run_gpu_async(&inputs)).ok()?;
    if run_stats().gpu_nodes == 0 {
        println!("skip: the adapter declined every node of the chain");
        return None;
    }
    Some((session, inputs))
}

/// Run the graph and return its single output's values.
fn run(session: &Session, inputs: &HashMap<&'static str, Tensor>) -> Vec<f32> {
    let outputs = pollster::block_on(session.run_gpu_async(inputs)).expect("run");
    let (_name, tensor) = outputs
        .into_iter()
        .next()
        .expect("the graph has exactly one output");
    tensor.data
}

/// PSNR in dB; infinite for a bit-identical pair.
fn psnr(got: &[f32], want: &[f32]) -> f64 {
    let peak = want.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
    let mse = want
        .iter()
        .zip(got)
        .map(|(&w, &g)| (f64::from(w) - f64::from(g)).powi(2))
        .sum::<f64>()
        / want.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (peak * peak / mse).log10()
}

/// A session that has never heard of half precision behaves exactly as it did
/// before half precision existed.
#[test]
fn a_fresh_session_has_half_precision_off() {
    let Some((session, _inputs)) = warm_session() else {
        return;
    };
    assert!(
        !session.f16_compute_enabled(),
        "a numerics-changing mode must never be on unless it was asked for"
    );
    println!(
        "  device supports shader-f16: {}",
        session.f16_compute_supported()
    );
}

/// **The mandate-5 test.** With the toggle off, the whole engine is
/// byte-identical — including after the f16 path has been exercised in between.
#[test]
fn the_toggle_off_engine_is_byte_identical() {
    let Some((session, inputs)) = warm_session() else {
        return;
    };

    // The reference: the toggle is never mentioned. This is what the tree
    // produced before this wave.
    let pristine = run(&session, &inputs);
    assert!(
        pristine.iter().any(|v| *v != 0.0),
        "a graph whose output is all zeros would make every comparison vacuous"
    );

    // An explicit `false` must be indistinguishable from never asking.
    assert!(!session.set_f16_compute(false));
    let explicit_off = run(&session, &inputs);
    assert_eq!(
        explicit_off, pristine,
        "an explicit set_f16_compute(false) changed the result"
    );

    if !session.f16_compute_supported() {
        println!("skip (partial): no shader-f16, so the on/off round trip is trivially equal");
        return;
    }

    // Exercise the f16 path, then come back. This is the interesting half: the
    // f32 weights must still be resident, in their own format slot, and must
    // still produce the same bits.
    assert!(session.set_f16_compute(true));
    let _half = run(&session, &inputs);
    assert!(!session.set_f16_compute(false));
    let back_off = run(&session, &inputs);
    assert_eq!(
        back_off, pristine,
        "running the f16 path changed what the f32 path produces — the weight \
         cache is serving bytes across formats"
    );
}

/// The other half of the same coin: turning it *on* must actually reach the
/// kernels. A toggle that quietly did nothing would pass the test above.
#[test]
fn turning_it_on_changes_the_result_and_stays_within_the_gate() {
    let Some((session, inputs)) = warm_session() else {
        return;
    };
    if !session.f16_compute_supported() {
        println!("skip: adapter does not support shader-f16");
        return;
    }
    let Some(ctx) = session.gpu.as_ref() else {
        return;
    };

    assert!(!session.set_f16_compute(false));
    let reference = run(&session, &inputs);
    let f32_resident = ctx.resident_bytes();

    assert!(session.set_f16_compute(true));
    assert!(session.f16_compute_enabled());
    let half = run(&session, &inputs);

    let db = psnr(&half, &reference);
    println!("  4-layer conv chain, f16 vs f32: PSNR {db:.1} dB");
    assert!(
        db.is_finite(),
        "the f16 run produced bit-identical output, so the toggle never reached \
         the kernels — this is the wrong-object failure this test exists for"
    );
    assert!(
        db >= 55.0,
        "PSNR {db:.1} dB across a 4-layer chain is below the 55 dB gate"
    );

    // Both formats are now resident for the same eight initializers.
    let both_resident = ctx.resident_bytes();
    assert!(
        both_resident > f32_resident,
        "the f16 copies must be additional allocations, not replacements \
         ({f32_resident} -> {both_resident})"
    );
    println!("  resident weight bytes: f32 only {f32_resident}, both {both_resident}");
    assert!(
        !ctx.is_degraded(),
        "device degraded during the comparison: {:?}",
        ctx.last_error()
    );
}
