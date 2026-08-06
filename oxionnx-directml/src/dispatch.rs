//! The per-node router.  **Platform-neutral** — and therefore unit-testable on Linux
//! against the declining stub backend, where the whole routing contract can be exercised
//! without a GPU.
//!
//! # The contract: DECLINED is not FAILED
//!
//! The version of this file that this one replaces did:
//!
//! ```ignore
//! kernels::matmul::dml_matmul(a, b, ctx).ok().map(|t| vec![t])
//! ```
//!
//! `.ok()` collapses **every** error — a failed shader compile, a removed device, a
//! dispatch failure, an out-of-memory — into `Ok(None)`, which the session then reads as
//! *"this provider does not do this op; use the CPU"*.  A total GPU failure became
//! indistinguishable from correct, expected behaviour: the user's "GPU-accelerated"
//! inference was silently exactly as slow as before, with no signal anywhere.  You could
//! delete the entire D3D12 device and the only observable difference would be the wall
//! clock.
//!
//! So the three outcomes are kept apart, and never conflated:
//!
//! | Kernel result | Meaning | Router |
//! |---|---|---|
//! | `Ok(t)` | the GPU computed it | `Ok(Some(vec![t]))` |
//! | `Err(`[`Declined`]`)` | **not ours** — this op/shape/dtype is outside what this backend expresses.  A *normal, expected* outcome. | `Ok(None)` + `debug!` |
//! | `Err(`[`ShapeMismatch`]`)` | **your model is broken** — the CPU operator would fail on the same inputs. | `Ok(None)` + `debug!`; the CPU op raises the real diagnostic |
//! | `Err(_)` | **the GPU broke** | `error!`, then `Ok(None)` — or `Err` under [`FailurePolicy::Strict`] |
//!
//! [`Declined`]: DirectMLError::Declined
//! [`ShapeMismatch`]: DirectMLError::ShapeMismatch
//!
//! Note what is *not* in that table: a genuine failure is never silent.  It falls back to
//! the CPU — so inference stays correct — but it says so, at `error!`, every time, because
//! "your GPU provider has been dead since process start" is not a debug-level fact.
//! `OXIONNX_DIRECTML_STRICT=1` turns it into a hard error for anyone who would rather the
//! run stopped.
//!
//! Only a *structural* problem escapes as `Err` unconditionally: an input tensor that is
//! simply not in the value map, which is a broken graph and which the CPU operator would
//! fail on too.
//!
//! # The op table must mirror [`crate::is_supported_op`]
//!
//! `is_supported_op` is what the session runner uses to decide which nodes to drag into its
//! **serial** GPU phase.  If this router claims an op that `is_supported_op` does not, the
//! node never reaches it.  If `is_supported_op` claims an op this router does not handle,
//! the node is serialised and then falls back to CPU anyway — a parallel CPU node has been
//! turned into a serial CPU node, which is a straight regression.
//!
//! The two **must** be edited together.  Today's table is exactly:
//! `MatMul`, `Gemm`, `Add`, `Sub`, `Mul`, `Div`, `Relu`, `Sigmoid`, `Tanh`, `Softmax`,
//! `ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`, `Conv`.

use std::collections::HashMap;

use oxionnx_core::{
    graph::{Node, OpKind},
    OnnxError, Tensor,
};

use crate::backend::Backend;
use crate::context::FailurePolicy;
use crate::error::DirectMLError;
use crate::kernels::{conv, elementwise, matmul, reduce, softmax};
use crate::plan::{BinaryOp, ReduceKind, UnaryOp};

/// Route a single ONNX node to the appropriate kernel.
///
/// # Errors
/// [`OnnxError::TensorNotFound`] when a required input is absent.  Under
/// [`FailurePolicy::Strict`], also whatever the kernel failed with.
pub(crate) fn route(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    backend: &Backend,
) -> core::result::Result<Option<Vec<Tensor>>, OnnxError> {
    route_with_policy(
        node,
        weights,
        intermediates,
        backend,
        FailurePolicy::current(),
    )
}

/// [`route`], with the failure policy injected.
///
/// Split out purely so the strict path is *testable*: `FailurePolicy::current()` caches a
/// read of the process environment in a `OnceLock`, which is exactly the right thing on the
/// dispatch path and exactly the wrong thing in a threaded test runner.  Both branches of
/// the declined-vs-failed contract are exercised in this module's tests against the
/// declining stub backend, on Linux, with no GPU.
///
/// # Errors
/// As [`route`].
pub(crate) fn route_with_policy(
    node: &Node,
    weights: &HashMap<String, Tensor>,
    intermediates: &HashMap<String, Tensor>,
    backend: &Backend,
    policy: FailurePolicy,
) -> core::result::Result<Option<Vec<Tensor>>, OnnxError> {
    let outcome = match &node.op {
        OpKind::MatMul => {
            let a = required(node, 0, weights, intermediates)?;
            let b = required(node, 1, weights, intermediates)?;
            matmul::dml_matmul(a, b, backend)
        }
        OpKind::Gemm => {
            let a = required(node, 0, weights, intermediates)?;
            let b = required(node, 1, weights, intermediates)?;
            // ONNX `Gemm`'s `C` is optional, and "absent" has two spellings: the node
            // declares fewer than three inputs, or it declares a third whose name is the
            // empty string.  Both must resolve to `None`.  A third input that is *named*
            // but not in the value map, by contrast, is a broken graph and must escape as
            // `TensorNotFound` — collapsing that into "no bias" would silently compute
            // `A·B` where the model said `A·B + C`.
            let c = optional(node, 2, weights, intermediates)?;
            matmul::dml_gemm(
                a,
                b,
                c,
                // ONNX defaults, and they are not symmetric: `alpha` and `beta` both
                // default to 1.0, so a bare `Gemm` node with a `C` input *does* add it.
                node.attrs.f("alpha", 1.0),
                node.attrs.f("beta", 1.0),
                node.attrs.i("transA", 0) != 0,
                node.attrs.i("transB", 0) != 0,
                backend,
            )
        }
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => {
            let op = match node.op {
                OpKind::Add => BinaryOp::Add,
                OpKind::Sub => BinaryOp::Sub,
                OpKind::Mul => BinaryOp::Mul,
                // Unreachable: the arm's own pattern lists exactly these four.  Spelled as
                // a total match anyway, because a `_ => unreachable!()` would be a `panic!`
                // in a crate that denies `clippy::panic`, and because the next person to
                // add `Pow` to the outer pattern must be *forced* to come here.
                _ => BinaryOp::Div,
            };
            let a = required(node, 0, weights, intermediates)?;
            let b = required(node, 1, weights, intermediates)?;
            elementwise::dml_binary(a, b, op, backend)
        }
        OpKind::Relu | OpKind::Sigmoid | OpKind::Tanh => {
            let op = match node.op {
                OpKind::Relu => UnaryOp::Relu,
                OpKind::Sigmoid => UnaryOp::Sigmoid,
                _ => UnaryOp::Tanh,
            };
            let a = required(node, 0, weights, intermediates)?;
            elementwise::dml_unary(a, op, backend)
        }
        OpKind::Softmax => {
            let a = required(node, 0, weights, intermediates)?;
            // ONNX opset-13 `Softmax` normalises a single axis, default `-1`.  (Opset < 13
            // defaulted to `1` and flattened everything after it into one big row; we read
            // the node and default to the modern `-1`, which the plan resolves against the
            // rank.)  A negative or out-of-range axis is the plan's to resolve or reject.
            softmax::dml_softmax(a, node.attrs.i("axis", -1), backend)
        }
        OpKind::ReduceSum | OpKind::ReduceMean | OpKind::ReduceMax | OpKind::ReduceMin => {
            let kind = match node.op {
                OpKind::ReduceSum => ReduceKind::Sum,
                OpKind::ReduceMean => ReduceKind::Mean,
                OpKind::ReduceMax => ReduceKind::Max,
                // Unreachable: the arm's pattern lists exactly these four.  A total match,
                // not a `_ => unreachable!()`, because this crate denies `clippy::panic` and
                // because adding `ReduceProd` to the outer pattern must force a visit here.
                _ => ReduceKind::Min,
            };
            let a = required(node, 0, weights, intermediates)?;
            // `axes` empty means "all axes" (ONNX); the plan reads that, together with any
            // multi-axis list, as a decline → the CPU kernel, which reduces correctly.
            // `keepdims` defaults to 1 (keep the reduced axis as a size-1 dim).
            reduce::dml_reduce(
                a,
                kind,
                node.attrs.ints("axes"),
                node.attrs.i("keepdims", 1) != 0,
                backend,
            )
        }
        OpKind::Conv => {
            let input = required(node, 0, weights, intermediates)?;
            let weight = required(node, 1, weights, intermediates)?;
            // ONNX `Conv`'s bias `B` is optional, with the same two spellings of "absent" as
            // `Gemm`'s `C`: fewer than three inputs, or a third named `""`.  A *named* third
            // input missing from the value map is a broken graph and escapes as
            // `TensorNotFound`, exactly as it does for `Gemm`.
            let bias = optional(node, 2, weights, intermediates)?;
            conv::dml_conv(
                input,
                weight,
                bias,
                // ONNX defaults: strides/dilations 1 (empty list), pads 0 (empty list),
                // group 1.  The kernel forwards the raw lists; the plan supplies the
                // defaults and range-checks the lengths.  `auto_pad` other than NOTSET makes
                // padding implicit, which the kernel declines rather than infer.
                node.attrs.ints("strides"),
                node.attrs.ints("pads"),
                node.attrs.ints("dilations"),
                node.attrs.i("group", 1),
                node.attrs.s("auto_pad"),
                backend,
            )
        }
        // Not in the table.  Not an error, not a failure, and not something to log about:
        // the session runner offers every node to every provider, so this is the common
        // case by a wide margin.
        _ => return Ok(None),
    };

    classify(node, outcome, policy)
}

/// Turn a kernel's `Result<Tensor>` into the router's three-way outcome.
///
/// This function *is* the declined-vs-failed contract.  Everything above it is plumbing.
fn classify(
    node: &Node,
    outcome: crate::error::Result<Tensor>,
    policy: FailurePolicy,
) -> core::result::Result<Option<Vec<Tensor>>, OnnxError> {
    match outcome {
        Ok(tensor) => Ok(Some(vec![tensor])),

        // DECLINED — "not ours".  Expected, correct, and cheap: the CPU kernel one line
        // away computes it properly.  `debug!`, because on a real graph this fires for
        // every 3-D MatMul and every broadcast Add, and an `info!` here would be noise.
        Err(DirectMLError::Declined(reason)) => {
            tracing::debug!(
                node = %node.name,
                op = ?node.op,
                %reason,
                "DirectML declined this node; running it on the CPU"
            );
            Ok(None)
        }

        // MALFORMED — the model is wrong.  Still `Ok(None)`, and still not promoted by
        // strict mode: the CPU operator is about to hit the identical inputs and raise a
        // diagnostic written by people who know what the op means.  Pre-empting it with a
        // DirectML-flavoured error would bury the user's actual bug under ours.
        Err(DirectMLError::ShapeMismatch(reason)) => {
            tracing::debug!(
                node = %node.name,
                op = ?node.op,
                %reason,
                "DirectML rejected this node's shapes; deferring to the CPU operator, which \
                 will raise the real error"
            );
            Ok(None)
        }

        // FAILED — the GPU broke.  This is the case the old `.ok()` swallowed.
        Err(error) => {
            tracing::error!(
                node = %node.name,
                op = ?node.op,
                %error,
                strict = matches!(policy, FailurePolicy::Strict),
                "DirectML kernel FAILED (this is a GPU/driver/shader fault, not a decline). \
                 Set {} to make this fatal.",
                crate::context::STRICT_ENV_VAR
            );
            match policy {
                FailurePolicy::Strict => Err(OnnxError::from(error)),
                // Inference stays correct — the CPU runs the node — but the failure has
                // been said out loud, which is the entire difference from the old code.
                FailurePolicy::Fallback => Ok(None),
            }
        }
    }
}

/// Resolve input `index`, which the op requires.
///
/// # Errors
/// [`OnnxError::TensorNotFound`] when the node does not declare that input at all, or
/// declares it under a name that is in neither map.
fn required<'a>(
    node: &Node,
    index: usize,
    weights: &'a HashMap<String, Tensor>,
    intermediates: &'a HashMap<String, Tensor>,
) -> core::result::Result<&'a Tensor, OnnxError> {
    let name = node.inputs.get(index).ok_or_else(|| {
        OnnxError::TensorNotFound(format!(
            "{} ({}): requires input {index}, but the node declares only {}",
            node.name,
            node.op.as_str(),
            node.inputs.len()
        ))
    })?;
    resolve(name, weights, intermediates).ok_or_else(|| {
        OnnxError::TensorNotFound(format!(
            "{name} (input {index} of {} ({}))",
            node.name,
            node.op.as_str()
        ))
    })
}

/// Resolve input `index`, which the op treats as optional.
///
/// `Ok(None)` when the input is absent — either not declared, or declared as `""`, which is
/// ONNX's spelling of "this optional input is omitted".
///
/// # Errors
/// [`OnnxError::TensorNotFound`] when the input *is* declared, under a non-empty name, and
/// that name is in neither map.  That is a broken graph, not an omitted optional.
fn optional<'a>(
    node: &Node,
    index: usize,
    weights: &'a HashMap<String, Tensor>,
    intermediates: &'a HashMap<String, Tensor>,
) -> core::result::Result<Option<&'a Tensor>, OnnxError> {
    match node.inputs.get(index) {
        None => Ok(None),
        Some(name) if name.is_empty() => Ok(None),
        Some(_) => required(node, index, weights, intermediates).map(Some),
    }
}

/// Look a tensor up in the intermediates first, then the weights.
///
/// An empty name is ONNX's spelling of "this optional input is absent", and must resolve to
/// `None` rather than to a spurious lookup miss.
pub(crate) fn resolve<'a>(
    name: &str,
    weights: &'a HashMap<String, Tensor>,
    intermediates: &'a HashMap<String, Tensor>,
) -> Option<&'a Tensor> {
    if name.is_empty() {
        None
    } else {
        intermediates.get(name).or_else(|| weights.get(name))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::resolve;
    use crate::context::FailurePolicy;
    use oxionnx_core::Tensor;
    use std::collections::HashMap;

    fn values(pairs: &[(&str, Tensor)]) -> HashMap<String, Tensor> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    fn tensor(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::new(
            (0..n)
                .map(|i| f32::from(u8::try_from(i % 7).unwrap_or(0)))
                .collect(),
            shape.to_vec(),
        )
    }

    // Note on coverage: the behavioural tests in `mod routing` below need a `Backend`, and
    // on Windows the only way to get one is to acquire a real device — so they are `cfg`'d
    // off there.  `route` itself is still type-checked for Windows regardless, because
    // `lib.rs::try_directml_dispatch` calls it unconditionally; the cross-target clippy run
    // therefore still covers the router's Windows monomorphisation.

    #[test]
    fn the_failure_policy_is_readable_without_panicking() {
        // Pins that `current()` is callable; the value depends on the ambient environment
        // and is deliberately not asserted.
        let _ = FailurePolicy::current();
    }

    #[test]
    fn resolve_prefers_intermediates_then_weights_then_nothing() {
        let weights = values(&[("w", tensor(&[1]))]);
        let intermediates = values(&[("i", tensor(&[1]))]);
        assert!(resolve("i", &weights, &intermediates).is_some());
        assert!(resolve("w", &weights, &intermediates).is_some());
        assert!(resolve("missing", &weights, &intermediates).is_none());
    }

    #[test]
    fn intermediates_shadow_weights() {
        let weights = values(&[("x", tensor(&[2]))]);
        let intermediates = values(&[("x", Tensor::new(vec![9.0, 9.0], vec![2]))]);
        let found = resolve("x", &weights, &intermediates).expect("x exists");
        assert_eq!(found.data, vec![9.0, 9.0], "the live value must win");
    }

    #[test]
    fn an_empty_name_resolves_to_nothing() {
        let weights = values(&[("", tensor(&[2]))]);
        assert!(
            resolve("", &weights, &HashMap::new()).is_none(),
            "\"\" is ONNX for 'omitted', not a tensor name to look up"
        );
    }

    /// The routing contract, driven end to end against the declining stub backend.
    ///
    /// `stub_backend::Backend` is the only `Backend` constructible without a GPU, which is
    /// exactly why it exists: it is what lets CI here — on Linux, with no device — assert
    /// that a decline becomes `Ok(None)` and never an `Err`.
    #[cfg(not(target_os = "windows"))]
    mod routing {
        use super::{tensor, values};
        use crate::backend::Backend;
        use crate::context::FailurePolicy;
        use crate::dispatch::route_with_policy;
        use oxionnx_core::{
            graph::{Attributes, Node, OpKind},
            OnnxError, Tensor,
        };
        use std::collections::HashMap;

        fn declining() -> Backend {
            Backend::declining_for_tests()
        }

        /// A node with the given op and input names, and no attributes.
        fn node(op: OpKind, inputs: &[&str]) -> Node {
            Node {
                op,
                name: "n0".into(),
                inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
                outputs: vec!["y".into()],
                attrs: Attributes::default(),
            }
        }

        /// A single-input `Reduce*` node carrying one explicit `axes` entry, so the plan
        /// resolves to a single axis and reaches the backend (rather than declining at the
        /// multi-axis guard before ever calling it).
        fn reduce_node(op: OpKind, axis: i64) -> Node {
            let mut attrs = Attributes::default();
            attrs.int_lists.insert("axes".into(), vec![axis]);
            Node {
                op,
                name: "r0".into(),
                inputs: vec!["a".into()],
                outputs: vec!["y".into()],
                attrs,
            }
        }

        #[test]
        fn an_unclaimed_op_is_declined_without_ever_touching_the_backend() {
            let n = node(OpKind::Identity, &["a"]);
            let out = route_with_policy(
                &n,
                &HashMap::new(),
                &HashMap::new(),
                &declining(),
                FailurePolicy::Strict, // even here: not-in-the-table is not a failure
            )
            .unwrap();
            assert!(out.is_none(), "an op we do not claim must be Ok(None)");
        }

        #[test]
        fn a_missing_required_input_is_a_structural_error() {
            let n = node(OpKind::MatMul, &["a", "b"]);
            let weights = values(&[("a", tensor(&[2, 3]))]);

            let err = route_with_policy(
                &n,
                &weights,
                &HashMap::new(),
                &declining(),
                FailurePolicy::Fallback,
            )
            .unwrap_err();

            assert!(
                matches!(err, OnnxError::TensorNotFound(_)),
                "a graph that references a tensor which does not exist is broken, and must \
                 NOT be swallowed into a CPU fallback; got {err:?}"
            );
            assert!(
                format!("{err}").contains('b'),
                "the error must name the missing input"
            );
        }

        #[test]
        fn a_node_declaring_too_few_inputs_is_a_structural_error() {
            let n = node(OpKind::MatMul, &["a"]);
            let weights = values(&[("a", tensor(&[2, 3]))]);
            let err = route_with_policy(
                &n,
                &weights,
                &HashMap::new(),
                &declining(),
                FailurePolicy::Fallback,
            )
            .unwrap_err();
            assert!(matches!(err, OnnxError::TensorNotFound(_)), "got {err:?}");
        }

        #[test]
        fn a_declining_backend_becomes_ok_none_and_never_an_err() {
            // THE contract, in one test.  Every claimed op, valid inputs, a backend that
            // declines: the router must produce `Ok(None)` — a correct, silent CPU
            // fallback — for all of them.
            let cases: Vec<(Node, HashMap<String, Tensor>)> = vec![
                (
                    node(OpKind::MatMul, &["a", "b"]),
                    values(&[("a", tensor(&[2, 3])), ("b", tensor(&[3, 4]))]),
                ),
                (
                    node(OpKind::Gemm, &["a", "b"]),
                    values(&[("a", tensor(&[2, 3])), ("b", tensor(&[3, 4]))]),
                ),
                (
                    node(OpKind::Add, &["a", "b"]),
                    values(&[("a", tensor(&[2, 3])), ("b", tensor(&[2, 3]))]),
                ),
                (
                    node(OpKind::Sub, &["a", "b"]),
                    values(&[("a", tensor(&[2, 3])), ("b", tensor(&[2, 3]))]),
                ),
                (
                    node(OpKind::Mul, &["a", "b"]),
                    values(&[("a", tensor(&[2, 3])), ("b", tensor(&[2, 3]))]),
                ),
                (
                    node(OpKind::Div, &["a", "b"]),
                    values(&[("a", tensor(&[2, 3])), ("b", tensor(&[2, 3]))]),
                ),
                (
                    node(OpKind::Relu, &["a"]),
                    values(&[("a", tensor(&[2, 3]))]),
                ),
                (
                    node(OpKind::Sigmoid, &["a"]),
                    values(&[("a", tensor(&[2, 3]))]),
                ),
                (
                    node(OpKind::Tanh, &["a"]),
                    values(&[("a", tensor(&[2, 3]))]),
                ),
                // Wave-4 ops.  Softmax over a valid axis, single-axis reduces, and a
                // well-formed 2-D Conv — each yields a plan the router feeds to the backend,
                // which declines, so all must land on `Ok(None)` too.
                (
                    node(OpKind::Softmax, &["a"]),
                    values(&[("a", tensor(&[2, 3]))]),
                ),
                (
                    reduce_node(OpKind::ReduceSum, 1),
                    values(&[("a", tensor(&[2, 3]))]),
                ),
                (
                    reduce_node(OpKind::ReduceMean, 1),
                    values(&[("a", tensor(&[2, 3]))]),
                ),
                (
                    reduce_node(OpKind::ReduceMax, 1),
                    values(&[("a", tensor(&[2, 3]))]),
                ),
                (
                    reduce_node(OpKind::ReduceMin, 1),
                    values(&[("a", tensor(&[2, 3]))]),
                ),
                (
                    node(OpKind::Conv, &["x", "w"]),
                    values(&[("x", tensor(&[1, 1, 5, 5])), ("w", tensor(&[1, 1, 3, 3]))]),
                ),
            ];

            for (n, weights) in cases {
                let out = route_with_policy(
                    &n,
                    &weights,
                    &HashMap::new(),
                    &declining(),
                    FailurePolicy::Fallback,
                );
                assert!(
                    matches!(out, Ok(None)),
                    "{:?}: a DECLINE must become Ok(None), got {out:?}",
                    n.op
                );
            }
        }

        #[test]
        fn strict_mode_does_not_promote_a_decline() {
            // The stub declines.  A decline is not a failure, so even STRICT mode must let
            // it fall through to the CPU.  If this ever starts returning `Err`, every
            // Linux user who set `OXIONNX_DIRECTML_STRICT=1` would see their inference
            // abort on the first `MatMul`.
            let n = node(OpKind::MatMul, &["a", "b"]);
            let weights = values(&[("a", tensor(&[2, 3])), ("b", tensor(&[3, 4]))]);
            let out = route_with_policy(
                &n,
                &weights,
                &HashMap::new(),
                &declining(),
                FailurePolicy::Strict,
            );
            assert!(
                matches!(out, Ok(None)),
                "STRICT must promote FAILURES, not DECLINES; got {out:?}"
            );
        }

        #[test]
        fn a_shape_mismatch_defers_to_the_cpu_operator_rather_than_pre_empting_it() {
            // `[2,3] · [4,5]` has no valid inner dimension.  `plan.rs` returns
            // `ShapeMismatch` — the model is malformed — and the router must still say
            // `Ok(None)`, because `oxionnx-ops`' MatMul is about to hit the same inputs and
            // raise a far better error than we can.
            let n = node(OpKind::MatMul, &["a", "b"]);
            let weights = values(&[("a", tensor(&[2, 3])), ("b", tensor(&[4, 5]))]);
            for policy in [FailurePolicy::Fallback, FailurePolicy::Strict] {
                let out = route_with_policy(&n, &weights, &HashMap::new(), &declining(), policy);
                assert!(
                    matches!(out, Ok(None)),
                    "{policy:?}: a malformed model is the CPU op's error to raise; got {out:?}"
                );
            }
        }

        #[test]
        fn a_broadcast_add_is_declined_not_mis_executed() {
            // `ElementwisePlan::binary` declines every non-identical shape pair, even
            // broadcastable ones, because the index-parallel shaders would read past the
            // end of the smaller operand and return a right-shaped tensor of garbage.
            let n = node(OpKind::Add, &["a", "b"]);
            let weights = values(&[("a", tensor(&[2, 3, 4])), ("b", tensor(&[1, 4]))]);
            let out = route_with_policy(
                &n,
                &weights,
                &HashMap::new(),
                &declining(),
                FailurePolicy::Strict,
            );
            assert!(matches!(out, Ok(None)), "got {out:?}");
        }

        #[test]
        fn an_empty_tensor_is_declined_rather_than_sent_to_a_zero_width_buffer() {
            // `[0, 128]` is routine after an empty batch, and `CreateCommittedResource`
            // with `Width = 0` fails outright.
            let n = node(OpKind::Relu, &["a"]);
            let weights = values(&[("a", tensor(&[0, 128]))]);
            let out = route_with_policy(
                &n,
                &weights,
                &HashMap::new(),
                &declining(),
                FailurePolicy::Strict,
            );
            assert!(matches!(out, Ok(None)), "got {out:?}");
        }

        #[test]
        fn gemm_reads_its_attributes_off_the_node() {
            // A `Gemm` whose `transB` is set makes `[2,3] · [4,3]ᵀ` valid.  Without reading
            // the attribute the plan would see `[2,3] · [4,3]`, an inner-dimension
            // mismatch, and return `ShapeMismatch` — which also routes to `Ok(None)`, so a
            // test that only checked the outcome would pass while the attribute was being
            // ignored.  Assert the *plan* instead, which is the thing that would be wrong.
            let mut attrs = Attributes::default();
            attrs.ints.insert("transB".into(), 1);
            attrs.floats.insert("alpha".into(), 0.5);
            attrs.floats.insert("beta".into(), 0.25);

            let n = Node {
                op: OpKind::Gemm,
                name: "g".into(),
                inputs: vec!["a".into(), "b".into(), "c".into()],
                outputs: vec!["y".into()],
                attrs,
            };
            assert_eq!(n.attrs.f("alpha", 1.0), 0.5);
            assert_eq!(n.attrs.f("beta", 1.0), 0.25);
            assert_eq!(n.attrs.i("transB", 0), 1);
            assert_eq!(n.attrs.i("transA", 0), 0);

            // And the plan those attributes produce is well-formed, so the router reaches
            // the backend (which then declines) rather than tripping on ShapeMismatch.
            let plan = crate::plan::MatMulPlan::gemm(
                &[2, 3],
                &[4, 3],
                Some(&[4]),
                n.attrs.f("alpha", 1.0),
                n.attrs.f("beta", 1.0),
                n.attrs.i("transA", 0) != 0,
                n.attrs.i("transB", 0) != 0,
            )
            .expect("transB makes [2,3] x [4,3] a valid Gemm");
            assert_eq!(plan.output_shape, vec![2, 4]);
            assert!(plan.trans_b);
            assert!(plan.has_bias());

            let weights = values(&[
                ("a", tensor(&[2, 3])),
                ("b", tensor(&[4, 3])),
                ("c", tensor(&[4])),
            ]);
            let out = route_with_policy(
                &n,
                &weights,
                &HashMap::new(),
                &declining(),
                FailurePolicy::Fallback,
            );
            assert!(matches!(out, Ok(None)), "got {out:?}");
        }

        #[test]
        fn gemm_treats_an_omitted_c_as_absent_but_a_dangling_c_as_broken() {
            // Two inputs: `C` omitted entirely.
            let two = node(OpKind::Gemm, &["a", "b"]);
            // Three inputs, the third named `""`: ONNX's spelling of "omitted".
            let empty_name = node(OpKind::Gemm, &["a", "b", ""]);
            let weights = values(&[("a", tensor(&[2, 3])), ("b", tensor(&[3, 4]))]);

            for n in [two, empty_name] {
                let out = route_with_policy(
                    &n,
                    &weights,
                    &HashMap::new(),
                    &declining(),
                    FailurePolicy::Fallback,
                );
                assert!(
                    matches!(out, Ok(None)),
                    "an omitted optional C must not be an error; got {out:?}"
                );
            }

            // Three inputs, the third named but absent from both maps: a broken graph.
            // Collapsing this into "no bias" would silently compute `A·B` where the model
            // said `A·B + C` — a plausible, wrong answer.
            let dangling = node(OpKind::Gemm, &["a", "b", "c"]);
            let err = route_with_policy(
                &dangling,
                &weights,
                &HashMap::new(),
                &declining(),
                FailurePolicy::Fallback,
            )
            .unwrap_err();
            assert!(matches!(err, OnnxError::TensorNotFound(_)), "got {err:?}");
        }

        #[test]
        fn softmax_reads_its_axis_off_the_node_and_defaults_to_minus_one() {
            // A wrong axis produces a right-shaped tensor of wrong numbers, so — as with the
            // `Gemm` attribute test — assert the *plan* the attribute produces, not just the
            // outcome (the stub declines every axis to the same `Ok(None)`).
            let mut attrs = Attributes::default();
            attrs.ints.insert("axis".into(), 0);
            let n = Node {
                op: OpKind::Softmax,
                name: "s".into(),
                inputs: vec!["a".into()],
                outputs: vec!["y".into()],
                attrs,
            };
            assert_eq!(n.attrs.i("axis", -1), 0, "an explicit axis must win");
            let plan = crate::plan::SoftmaxPlan::softmax(&[2, 3], n.attrs.i("axis", -1)).unwrap();
            assert_eq!(plan.axis, 0);
            assert_eq!(plan.inner, 3, "axis 0 of [2,3] has inner stride 3");

            // With the attribute absent, the router's default is the opset-13 `-1`.
            let bare = node(OpKind::Softmax, &["a"]);
            assert_eq!(bare.attrs.i("axis", -1), -1);
            let plan =
                crate::plan::SoftmaxPlan::softmax(&[2, 3], bare.attrs.i("axis", -1)).unwrap();
            assert_eq!(plan.axis, 1, "-1 resolves to the trailing axis");

            // And the whole node routes to a clean CPU fallback against the stub.
            let out = route_with_policy(
                &bare,
                &values(&[("a", tensor(&[2, 3]))]),
                &HashMap::new(),
                &declining(),
                FailurePolicy::Fallback,
            );
            assert!(matches!(out, Ok(None)), "got {out:?}");
        }

        #[test]
        fn a_multi_axis_reduce_is_declined_before_the_backend_even_under_strict() {
            // Empty `axes` over a rank-2 tensor is ONNX's "all axes" — a multi-axis reduce
            // the flat shader cannot index.  The plan declines it, so it must be `Ok(None)`
            // (CPU) under BOTH policies: a decline is never promoted to a hard error.
            let n = node(OpKind::ReduceSum, &["a"]); // no `axes` attribute at all
            let weights = values(&[("a", tensor(&[2, 3]))]);
            for policy in [FailurePolicy::Fallback, FailurePolicy::Strict] {
                let out = route_with_policy(&n, &weights, &HashMap::new(), &declining(), policy);
                assert!(
                    matches!(out, Ok(None)),
                    "{policy:?}: a multi-axis reduce is a decline, not a failure; got {out:?}"
                );
            }
        }

        #[test]
        fn a_conv_with_a_non_notset_auto_pad_is_declined_rather_than_guessed() {
            // SAME_UPPER makes the pads implicit.  The kernel refuses to infer them, so the
            // node declines to the CPU operator — even under strict mode, because a decline
            // is not a GPU failure.
            let mut attrs = Attributes::default();
            attrs.strings.insert("auto_pad".into(), "SAME_UPPER".into());
            let n = Node {
                op: OpKind::Conv,
                name: "c".into(),
                inputs: vec!["x".into(), "w".into()],
                outputs: vec!["y".into()],
                attrs,
            };
            let weights = values(&[("x", tensor(&[1, 1, 5, 5])), ("w", tensor(&[1, 1, 3, 3]))]);
            for policy in [FailurePolicy::Fallback, FailurePolicy::Strict] {
                let out = route_with_policy(&n, &weights, &HashMap::new(), &declining(), policy);
                assert!(
                    matches!(out, Ok(None)),
                    "{policy:?}: an implicit-pad Conv is declined to the CPU; got {out:?}"
                );
            }
        }

        #[test]
        fn conv_treats_an_omitted_bias_as_absent_but_a_dangling_bias_as_broken() {
            // Mirrors the `Gemm` optional-C contract for `Conv`'s optional bias `B`.
            let x = tensor(&[1, 1, 5, 5]);
            let w = tensor(&[1, 1, 3, 3]);
            let weights = values(&[("x", x), ("w", w)]);

            // Two inputs (bias omitted) and three with an empty-named third both mean "no
            // bias" and must route cleanly to the declining stub.
            for inputs in [vec!["x", "w"], vec!["x", "w", ""]] {
                let n = node(OpKind::Conv, &inputs);
                let out = route_with_policy(
                    &n,
                    &weights,
                    &HashMap::new(),
                    &declining(),
                    FailurePolicy::Fallback,
                );
                assert!(matches!(out, Ok(None)), "{inputs:?}: got {out:?}");
            }

            // A named-but-absent bias is a broken graph, not an omitted optional.
            let dangling = node(OpKind::Conv, &["x", "w", "b"]);
            let err = route_with_policy(
                &dangling,
                &weights,
                &HashMap::new(),
                &declining(),
                FailurePolicy::Fallback,
            )
            .unwrap_err();
            assert!(matches!(err, OnnxError::TensorNotFound(_)), "got {err:?}");
        }
    }
}
