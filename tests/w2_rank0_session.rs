//! Wave-3 `T3-rank0-migration`: rank-0 (scalar, shape `[]`) tensors end to end
//! through `Session::run`.
//!
//! `oxionnx-ops/tests/w2_rank0.rs` pins the operator-layer contract by calling
//! kernels directly. This file is the other half: it drives the same contract
//! through the real engine — graph construction, topological scheduling, output
//! slot preallocation, the `HashMap<String, Tensor>` returned to the caller —
//! because every one of those layers computes an element count from a shape, and
//! the empty shape is exactly where a `shape.iter().product()` and a
//! `shape[0]`-style access disagree.
//!
//! Two properties are worth stating up front, because they are what makes rank 0
//! more than a cosmetic difference:
//!
//! * **`Shape` is the observable.** A rank-0 tensor's shape vector is *empty* —
//!   a length-0 tensor, itself of shape `[0]`. Rank 1 `[1]` would report the
//!   length-1 vector `[1]`. Any `Reshape`/`Concat`/`Expand` driven by a `Shape`
//!   node therefore sees a different number of dimensions, which is how the
//!   distinction propagates into a model's actual output shape.
//! * **The rank-*producing* ops are the ones that had to change.** Rank 0 could
//!   already be *consumed* correctly (and could be produced by `Reshape` to an
//!   empty target). What Wave-3 fixed is `Squeeze`/`ReduceX`/`ArgMax`/`Size`/
//!   `Constant` promoting an emptied output shape back up to `[1]`.
//!
//! Reference values are NumPy's, whose rank-0 arrays implement the semantics
//! ONNX specifies. Computed with `python3`:
//!
//! ```text
//! np.squeeze(np.array([[[5.0]]])).shape                              -> ()
//! len(())                                                            -> 0
//! np.sum(np.arange(24).reshape(2,3,4), axis=(0,1,2), keepdims=False)  -> shape (), 276.0
//! np.arange(24).reshape(2,3,4).mean()                                -> shape (), 11.5
//! np.array(7.0) + np.arange(6).reshape(2,3)                          -> shape (2,3), [7..12]
//! np.argmax(np.array([3.,9.,4.]), axis=0)                            -> shape (), 1
//! ```

use std::collections::HashMap;

use oxionnx::{Attributes, Graph, Node, OpKind, OptLevel, Session, Tensor};

// ── helpers ─────────────────────────────────────────────────────────────────

fn node(op: OpKind, name: &str, inputs: &[&str], outputs: &[&str], attrs: Attributes) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs,
    }
}

fn int_attrs(pairs: &[(&str, i64)]) -> Attributes {
    let mut a = Attributes::default();
    for (k, v) in pairs {
        a.ints.insert((*k).to_string(), *v);
    }
    a
}

fn int_list_attrs(pairs: &[(&str, &[i64])]) -> Attributes {
    let mut a = Attributes::default();
    for (k, v) in pairs {
        a.int_lists.insert((*k).to_string(), v.to_vec());
    }
    a
}

/// Build and run a graph at a given optimization level.
///
/// The level is a parameter rather than a constant because it decides *which*
/// code path produces the answer: at `OptLevel::None` every node runs through
/// the operator registry, while a higher level may constant-fold a node away and
/// return a value the optimizer computed instead. Both must agree on rank.
fn run_at(
    level: OptLevel,
    nodes: Vec<Node>,
    input_names: &[&str],
    output_names: &[&str],
    weights: HashMap<String, Tensor>,
    feeds: Vec<(&str, Tensor)>,
) -> HashMap<String, Tensor> {
    let graph = Graph {
        nodes,
        input_names: input_names.iter().map(|s| (*s).to_string()).collect(),
        output_names: output_names.iter().map(|s| (*s).to_string()).collect(),
        ..Default::default()
    };
    let session = Session::builder()
        .with_optimization_level(level)
        .build_from_graph(graph, weights)
        .expect("build session");
    let feed: HashMap<&str, Tensor> = feeds.into_iter().collect();
    session.run(&feed).expect("run")
}

/// The empty shape, spelled once so the assertions below stay readable.
fn rank0_shape() -> Vec<usize> {
    Vec::new()
}

#[track_caller]
fn assert_rank0(t: &Tensor, value: f32, what: &str) {
    assert_eq!(t.shape, rank0_shape(), "{what}: must be rank 0 (shape [])");
    assert_eq!(t.data, vec![value], "{what}: value");
    assert_eq!(t.numel(), 1, "{what}: a scalar holds exactly one element");
}

/// A `Shape` node's output for a rank-0 input: the **empty** vector, which is
/// itself a length-0 tensor of shape `[0]`.
#[track_caller]
fn assert_empty_shape_vector(t: &Tensor, what: &str) {
    assert_eq!(t.shape, vec![0], "{what}: a length-0 vector has shape [0]");
    assert_eq!(t.data, Vec::<f32>::new(), "{what}: no axes to report");
}

// ═══════════════════════════════════════════════════════════════════════════
// 1 — rank 0 crossing the session boundary, in both directions
// ═══════════════════════════════════════════════════════════════════════════

/// A rank-0 tensor fed as a **session input** survives the run loop and comes
/// back out at rank 0, and the `Shape` node in between reports the empty vector.
///
/// This is the load-bearing plumbing check: `src/session/run/dispatch.rs` sizes
/// every output slot with `if shape.is_empty() { 1 } else { shape.product() }`,
/// so a slot for a rank-0 value has to be special-cased or it is sized 1 by
/// accident of the empty product — either way this test fails loudly if that
/// preallocation ever starts trusting a bare `shape[0]`.
#[test]
fn rank0_session_input_round_trips_and_shape_reports_the_empty_vector() {
    let out = run_at(
        OptLevel::None,
        vec![
            node(
                OpKind::Identity,
                "id",
                &["x"],
                &["same"],
                Attributes::default(),
            ),
            node(
                OpKind::Shape,
                "shp",
                &["x"],
                &["dims"],
                Attributes::default(),
            ),
        ],
        &["x"],
        &["same", "dims"],
        HashMap::new(),
        vec![("x", Tensor::rank0(7.0))],
    );

    assert_rank0(&out["same"], 7.0, "Identity of a rank-0 input");
    assert_empty_shape_vector(&out["dims"], "Shape of a rank-0 input");
}

/// The same value supplied as an **initializer** (a graph weight) rather than a
/// runtime input. Initializers travel a different path into the run loop —
/// `Session::build_from_graph`'s `weights` map, consulted by name — so rank 0
/// has to survive that one too.
///
/// `Add` here is doing real work: broadcasting a rank-0 operand against `[2,3]`
/// must produce `[2,3]`, *not* raise the rank. NumPy:
/// `np.array(7.0) + np.arange(6).reshape(2,3)` is `[[7,8,9],[10,11,12]]`.
#[test]
fn rank0_initializer_broadcasts_without_raising_rank() {
    let weights: HashMap<String, Tensor> = [("w".to_string(), Tensor::rank0(7.0))]
        .into_iter()
        .collect();

    let out = run_at(
        OptLevel::None,
        vec![
            node(
                OpKind::Add,
                "add",
                &["m", "w"],
                &["sum"],
                Attributes::default(),
            ),
            node(
                OpKind::Shape,
                "shp",
                &["w"],
                &["wdims"],
                Attributes::default(),
            ),
        ],
        &["m"],
        &["sum", "wdims"],
        weights,
        vec![(
            "m",
            Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]),
        )],
    );

    assert_eq!(out["sum"].shape, vec![2, 3], "rank-0 + [2,3] stays [2,3]");
    assert_eq!(out["sum"].data, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    assert_empty_shape_vector(&out["wdims"], "Shape of a rank-0 initializer");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2 — the Squeeze -> Shape flow (the finding's flagship consequence)
// ═══════════════════════════════════════════════════════════════════════════

/// `Squeeze` down to a scalar, then `Shape` — through `Session::run`.
///
/// This is the exact chain finding `[a0-21]` named. Squeezing `[1,1,1]` with no
/// `axes` attribute drops every size-1 axis and leaves rank 0, so the following
/// `Shape` node emits the empty vector. Before the migration `resolve_squeeze_shape`
/// promoted the emptied shape to `[1]` and this reported the length-1 vector
/// `[1]` instead — one dimension too many for anything downstream.
///
/// Run at both optimization levels: `OptLevel::None` pins the operator-registry
/// path, and the default level additionally covers whatever the optimizer does
/// with a fully-static subgraph (constant folding, shape inference), which must
/// not reintroduce the promotion.
#[test]
fn squeeze_to_scalar_then_shape_is_the_empty_vector_through_the_session() {
    for level in [OptLevel::None, OptLevel::All] {
        let out = run_at(
            level,
            vec![
                node(
                    OpKind::Squeeze,
                    "sq",
                    &["x"],
                    &["scalar"],
                    Attributes::default(),
                ),
                node(
                    OpKind::Shape,
                    "shp",
                    &["scalar"],
                    &["dims"],
                    Attributes::default(),
                ),
            ],
            &["x"],
            &["scalar", "dims"],
            HashMap::new(),
            vec![("x", Tensor::new(vec![5.0], vec![1, 1, 1]))],
        );

        assert_rank0(&out["scalar"], 5.0, &format!("Squeeze-all at {level:?}"));
        assert_empty_shape_vector(&out["dims"], &format!("Shape of it at {level:?}"));
    }
}

/// The rank-0 scalar a `Squeeze` produces is a usable value, not a dead end:
/// `Unsqueeze(axes=[0])` lifts it back to `[1]` (rank in + len(axes)), and
/// `Mul` broadcasts it across a matrix without raising the rank.
#[test]
fn a_squeezed_scalar_still_composes_downstream() {
    let out = run_at(
        OptLevel::None,
        vec![
            node(
                OpKind::Squeeze,
                "sq",
                &["x"],
                &["scalar"],
                Attributes::default(),
            ),
            node(
                OpKind::Unsqueeze,
                "unsq",
                &["scalar"],
                &["lifted"],
                int_list_attrs(&[("axes", &[0])]),
            ),
            node(
                OpKind::Mul,
                "mul",
                &["m", "scalar"],
                &["scaled"],
                Attributes::default(),
            ),
        ],
        &["x", "m"],
        &["lifted", "scaled"],
        HashMap::new(),
        vec![
            ("x", Tensor::new(vec![3.0], vec![1, 1])),
            ("m", Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])),
        ],
    );

    assert_eq!(
        out["lifted"].shape,
        vec![1],
        "Unsqueeze lifts rank 0 to [1]"
    );
    assert_eq!(out["lifted"].data, vec![3.0]);
    assert_eq!(out["scaled"].shape, vec![2, 2], "broadcast keeps [2,2]");
    assert_eq!(out["scaled"].data, vec![3.0, 6.0, 9.0, 12.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3 — reductions that consume the whole tensor
// ═══════════════════════════════════════════════════════════════════════════

/// `ReduceSum` over **every** axis with `keepdims=0` is a scalar graph output.
///
/// NumPy: `np.sum(np.arange(24).reshape(2,3,4), axis=(0,1,2), keepdims=False)`
/// has shape `()` and value `276.0`.
///
/// Both spellings of "reduce everything" are exercised, because they reach
/// different code: an explicit `axes` list and an omitted one (which the op
/// expands to `0..ndim`). Under the `simd` feature both then take the
/// `is_full_reduction` fast path, which returns its own hard-coded shape rather
/// than the one the general odometer computes — so this is the test that would
/// catch that fast path being migrated in only one of its two forms.
#[test]
fn reduce_sum_all_axes_keepdims0_is_a_rank0_session_output() {
    let x = Tensor::new((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]);

    for (label, attrs) in [
        ("explicit axes", {
            let mut a = int_attrs(&[("keepdims", 0)]);
            a.int_lists.insert("axes".to_string(), vec![0, 1, 2]);
            a
        }),
        ("omitted axes", int_attrs(&[("keepdims", 0)])),
    ] {
        let out = run_at(
            OptLevel::None,
            vec![
                node(OpKind::ReduceSum, "rs", &["x"], &["total"], attrs),
                node(
                    OpKind::Shape,
                    "shp",
                    &["total"],
                    &["dims"],
                    Attributes::default(),
                ),
            ],
            &["x"],
            &["total", "dims"],
            HashMap::new(),
            vec![("x", x.clone())],
        );

        assert_rank0(&out["total"], 276.0, &format!("ReduceSum ({label})"));
        assert_empty_shape_vector(&out["dims"], &format!("Shape of it ({label})"));
    }
}

/// `keepdims=1` is deliberately **unchanged** by the migration: every reduced
/// axis collapses to 1 and the rank is preserved. Asserting this next to the
/// test above is what makes that test about rank-0 rather than about "reductions
/// lost a dimension somewhere".
#[test]
fn reduce_sum_all_axes_keepdims1_keeps_the_input_rank() {
    let out = run_at(
        OptLevel::None,
        vec![node(
            OpKind::ReduceSum,
            "rs",
            &["x"],
            &["total"],
            int_attrs(&[("keepdims", 1)]),
        )],
        &["x"],
        &["total"],
        HashMap::new(),
        vec![(
            "x",
            Tensor::new((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]),
        )],
    );

    assert_eq!(out["total"].shape, vec![1, 1, 1]);
    assert_eq!(out["total"].data, vec![276.0]);
}

/// The other reduction kinds that carry their own full-reduction shortcut, and
/// `ArgMax`, whose `keepdims=0` on a 1-D input also empties the shape.
///
/// NumPy: `np.arange(24).reshape(2,3,4).mean()` is `11.5` at shape `()`;
/// `np.argmax(np.array([3.,9.,4.]), axis=0)` is `1` at shape `()`.
#[test]
fn reduce_mean_max_min_and_argmax_all_reach_rank0() {
    let x = Tensor::new((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]);

    for (op, expected, label) in [
        (OpKind::ReduceMean, 11.5, "ReduceMean"),
        (OpKind::ReduceMax, 23.0, "ReduceMax"),
        (OpKind::ReduceMin, 0.0, "ReduceMin"),
    ] {
        let out = run_at(
            OptLevel::None,
            vec![node(op, "r", &["x"], &["y"], int_attrs(&[("keepdims", 0)]))],
            &["x"],
            &["y"],
            HashMap::new(),
            vec![("x", x.clone())],
        );
        assert_rank0(&out["y"], expected, label);
    }

    let out = run_at(
        OptLevel::None,
        vec![node(
            OpKind::ArgMax,
            "am",
            &["v"],
            &["idx"],
            int_attrs(&[("axis", 0), ("keepdims", 0)]),
        )],
        &["v"],
        &["idx"],
        HashMap::new(),
        vec![("v", Tensor::new(vec![3.0, 9.0, 4.0], vec![3]))],
    );
    assert_rank0(&out["idx"], 1.0, "ArgMax(keepdims=0) on a 1-D input");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4 — ops whose output is a scalar by definition
// ═══════════════════════════════════════════════════════════════════════════

/// Opset-21 `Size` "outputs an int64 scalar", so its output is rank 0 whatever
/// the input rank is — including when the input is itself rank 0.
///
/// The count is the number of *elements*, so a rank-0 input gives 1: the empty
/// shape describes exactly one value.
#[test]
fn size_is_a_rank0_output_for_every_input_rank() {
    for (input, expected, label) in [
        (Tensor::rank0(7.0), 1.0, "Size of a rank-0 input"),
        (
            Tensor::new((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]),
            24.0,
            "Size of a [2,3,4] input",
        ),
    ] {
        let out = run_at(
            OptLevel::None,
            vec![
                node(OpKind::Size, "sz", &["x"], &["n"], Attributes::default()),
                node(
                    OpKind::Shape,
                    "shp",
                    &["n"],
                    &["dims"],
                    Attributes::default(),
                ),
            ],
            &["x"],
            &["n", "dims"],
            HashMap::new(),
            vec![("x", input)],
        );

        assert_rank0(&out["n"], expected, label);
        assert_empty_shape_vector(&out["dims"], &format!("Shape of {label}"));
    }
}

/// `Constant` with `value_float` / `value_int`: opset-21 documents these as "the
/// value for the sole element for the scalar ... output tensor", so both emit
/// rank 0.
///
/// A `Constant` node has no inputs, which makes it the graph's most foldable
/// node — so this runs at the default optimization level too, where the value
/// most plausibly comes from the optimizer's constant-folding pass rather than
/// from `ConstantOp::execute`. The two must not disagree about rank.
#[test]
fn constant_value_float_and_value_int_are_rank0() {
    for level in [OptLevel::None, OptLevel::All] {
        let mut float_attrs = Attributes::default();
        float_attrs.floats.insert("value_float".to_string(), 3.25);

        let out = run_at(
            level,
            vec![
                node(OpKind::Constant, "cf", &[], &["f"], float_attrs),
                node(
                    OpKind::Constant,
                    "ci",
                    &[],
                    &["i"],
                    int_attrs(&[("value_int", 42)]),
                ),
                node(
                    OpKind::Shape,
                    "shp",
                    &["f"],
                    &["fdims"],
                    Attributes::default(),
                ),
            ],
            &[],
            &["f", "i", "fdims"],
            HashMap::new(),
            vec![],
        );

        assert_rank0(
            &out["f"],
            3.25,
            &format!("Constant value_float at {level:?}"),
        );
        assert_rank0(&out["i"], 42.0, &format!("Constant value_int at {level:?}"));
        assert_empty_shape_vector(&out["fdims"], &format!("Shape of it at {level:?}"));
    }
}

/// A `Constant` carrying a full `value` **tensor** attribute is the arm that did
/// *not* change: it passes the parsed tensor's own shape through untouched, so a
/// `[3]` constant stays `[3]`. Pinning this alongside the rank-0 arms keeps the
/// migration from being over-applied to the one case that carries real shape
/// information.
#[test]
fn constant_with_a_tensor_value_keeps_its_own_shape() {
    let mut attrs = Attributes::default();
    attrs.tensors.insert(
        "value".to_string(),
        Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
    );

    let out = run_at(
        OptLevel::None,
        vec![node(OpKind::Constant, "c", &[], &["v"], attrs)],
        &[],
        &["v"],
        HashMap::new(),
        vec![],
    );

    assert_eq!(out["v"].shape, vec![3]);
    assert_eq!(out["v"].data, vec![1.0, 2.0, 3.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5 — the whole thing composed
// ═══════════════════════════════════════════════════════════════════════════

/// One graph that produces a rank-0 value three different ways and proves they
/// are interchangeable, by adding them together and getting a rank-0 result.
///
/// `Squeeze`-to-scalar, `ReduceSum(keepdims=0)` over all axes and `Size` each
/// reach rank 0 through different code (a shape-resolution helper, a reduction
/// kernel with a SIMD shortcut, and a direct constructor). If any one of them
/// still promoted to `[1]`, the `Add` chain below would broadcast back up to
/// `[1]` and the final `Shape` would report a length-1 vector instead of the
/// empty one — a single assertion that covers all three producers at once.
///
/// Expected: `5.0` (squeezed) + `276.0` (sum of `arange(24)`) + `24.0` (size)
/// = `305.0`, at shape `[]`.
#[test]
fn three_independent_rank0_producers_compose_into_one_rank0_output() {
    let out = run_at(
        OptLevel::None,
        vec![
            node(
                OpKind::Squeeze,
                "sq",
                &["one"],
                &["a"],
                Attributes::default(),
            ),
            node(
                OpKind::ReduceSum,
                "rs",
                &["big"],
                &["b"],
                int_attrs(&[("keepdims", 0)]),
            ),
            node(OpKind::Size, "sz", &["big"], &["c"], Attributes::default()),
            node(
                OpKind::Add,
                "add1",
                &["a", "b"],
                &["ab"],
                Attributes::default(),
            ),
            node(
                OpKind::Add,
                "add2",
                &["ab", "c"],
                &["total"],
                Attributes::default(),
            ),
            node(
                OpKind::Shape,
                "shp",
                &["total"],
                &["dims"],
                Attributes::default(),
            ),
        ],
        &["one", "big"],
        &["total", "dims"],
        HashMap::new(),
        vec![
            ("one", Tensor::new(vec![5.0], vec![1, 1])),
            (
                "big",
                Tensor::new((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]),
            ),
        ],
    );

    assert_rank0(&out["total"], 305.0, "Squeeze + ReduceSum + Size");
    assert_empty_shape_vector(&out["dims"], "Shape of the composed scalar");
}
