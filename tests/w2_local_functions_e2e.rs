//! Wave-2 end-to-end tests for ONNX **model-local functions**
//! (`ModelProto.functions`, wire field 25).
//!
//! `parse_model` used to drop field 25 through its catch-all `_ => {}`, so a
//! model that shipped a complete function body still failed with
//! `Unknown op: <FunctionName>`. PyTorch `dynamo` export and opset-18+ ONNX
//! emit these routinely, which made a growing class of real exports unloadable.
//!
//! Functions are now resolved by **inlining** every call site into the main
//! graph at model-build time, so the execution engine never sees one. What the
//! tests below pin is the substitution contract, in the order the corner cases
//! bite:
//!
//! 1. two call sites of the same function must not alias each other's
//!    intermediate names,
//! 2. an attribute referenced via `ref_attr_name` resolves *call site →
//!    function default → dropped* (dropped meaning the operator's own default
//!    applies, never a substituted zero),
//! 3. a formal input the call site omits maps to `""` (still omitted), not to
//!    a dangling prefixed name,
//! 4. nested calls expand recursively, and
//! 5. a self-recursive function — which the ONNX spec forbids — is a typed
//!    error, not a stack overflow.
//!
//! The models are hand-encoded protobuf (the encoder below mirrors
//! `tests/opset_plumbing_e2e.rs`) so the tests exercise the real
//! `Session::from_bytes` path, wire format included.

use std::collections::HashMap;

use oxionnx::{OptLevel, Session, SessionBuilder, Tensor};

// ── minimal protobuf encoder ────────────────────────────────────────────────

fn encode_varint(mut val: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;
        if val == 0 {
            buf.push(byte);
            break;
        }
        buf.push(byte | 0x80);
    }
    buf
}

fn varint_field(field: u32, val: u64) -> Vec<u8> {
    let mut buf = encode_varint((field << 3) as u64); // wire type 0
    buf.extend(encode_varint(val));
    buf
}

fn bytes_field(field: u32, data: &[u8]) -> Vec<u8> {
    let mut buf = encode_varint(((field << 3) | 2) as u64); // wire type 2
    buf.extend(encode_varint(data.len() as u64));
    buf.extend_from_slice(data);
    buf
}

fn float_field(field: u32, val: f32) -> Vec<u8> {
    let mut buf = encode_varint(((field << 3) | 5) as u64); // wire type 5
    buf.extend_from_slice(&val.to_le_bytes());
    buf
}

/// `AttributeProto` carrying a concrete float value.
fn float_attr(name: &str, value: f32) -> Vec<u8> {
    let mut attr = bytes_field(1, name.as_bytes()); // name
    attr.extend(float_field(2, value)); // f
    attr.extend(varint_field(20, 1)); // type = FLOAT
    attr
}

/// `AttributeProto` that takes its value from an enclosing function attribute
/// (`ref_attr_name`, wire field 21).
fn ref_attr(name: &str, ref_attr_name: &str) -> Vec<u8> {
    let mut attr = bytes_field(1, name.as_bytes()); // name
    attr.extend(varint_field(20, 1)); // type = FLOAT
    attr.extend(bytes_field(21, ref_attr_name.as_bytes())); // ref_attr_name
    attr
}

/// `NodeProto`.
fn node(
    op_type: &str,
    domain: &str,
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: &[Vec<u8>],
) -> Vec<u8> {
    let mut n = Vec::new();
    for input in inputs {
        n.extend(bytes_field(1, input.as_bytes()));
    }
    for output in outputs {
        n.extend(bytes_field(2, output.as_bytes()));
    }
    n.extend(bytes_field(3, name.as_bytes()));
    n.extend(bytes_field(4, op_type.as_bytes()));
    for attr in attrs {
        n.extend(bytes_field(5, attr));
    }
    if !domain.is_empty() {
        n.extend(bytes_field(7, domain.as_bytes()));
    }
    n
}

/// `ValueInfoProto` for a 1-D float32 tensor.
fn value_info(name: &str, len: u64) -> Vec<u8> {
    let dim = bytes_field(1, &varint_field(1, len)); // TensorShapeProto.dim{dim_value}
    let mut tensor_type = varint_field(1, 1); // elem_type = FLOAT
    tensor_type.extend(bytes_field(2, &dim)); // shape
    let type_proto = bytes_field(1, &tensor_type); // TypeProto.tensor_type

    let mut vi = bytes_field(1, name.as_bytes());
    vi.extend(bytes_field(2, &type_proto));
    vi
}

/// `GraphProto`.
fn graph(
    name: &str,
    nodes: &[Vec<u8>],
    inputs: &[(&str, u64)],
    outputs: &[(&str, u64)],
) -> Vec<u8> {
    let mut g = Vec::new();
    for n in nodes {
        g.extend(bytes_field(1, n));
    }
    g.extend(bytes_field(2, name.as_bytes()));
    for (n, len) in inputs {
        g.extend(bytes_field(11, &value_info(n, *len)));
    }
    for (n, len) in outputs {
        g.extend(bytes_field(12, &value_info(n, *len)));
    }
    g
}

fn opset_import(domain: &str, version: u64) -> Vec<u8> {
    let mut o = Vec::new();
    if !domain.is_empty() {
        o.extend(bytes_field(1, domain.as_bytes()));
    }
    o.extend(varint_field(2, version));
    o
}

/// One entry of the model's local function library.
struct Function<'a> {
    name: &'a str,
    domain: &'a str,
    inputs: &'a [&'a str],
    outputs: &'a [&'a str],
    /// Attribute names declared without a default (`FunctionProto.attribute`).
    attribute_names: &'a [&'a str],
    /// Attributes carrying a default (`FunctionProto.attribute_proto`).
    attribute_defaults: &'a [Vec<u8>],
    nodes: &'a [Vec<u8>],
}

fn function_proto(f: &Function<'_>) -> Vec<u8> {
    let mut p = bytes_field(1, f.name.as_bytes()); // name
    for i in f.inputs {
        p.extend(bytes_field(4, i.as_bytes()));
    }
    for o in f.outputs {
        p.extend(bytes_field(5, o.as_bytes()));
    }
    for a in f.attribute_names {
        p.extend(bytes_field(6, a.as_bytes()));
    }
    for n in f.nodes {
        p.extend(bytes_field(7, n));
    }
    p.extend(bytes_field(9, &opset_import("", 18))); // opset_import
    p.extend(bytes_field(10, f.domain.as_bytes())); // domain
    for a in f.attribute_defaults {
        p.extend(bytes_field(11, a)); // attribute_proto (with defaults)
    }
    p
}

/// `ModelProto` with an `ai.onnx` opset, a custom-domain opset and a function
/// library.
fn model(graph_bytes: &[u8], custom_domain: &str, functions: &[Vec<u8>]) -> Vec<u8> {
    let mut m = varint_field(1, 8); // ir_version
    m.extend(bytes_field(8, &opset_import("", 18)));
    m.extend(bytes_field(8, &opset_import(custom_domain, 1)));
    m.extend(bytes_field(7, graph_bytes));
    for f in functions {
        m.extend(bytes_field(25, f)); // ModelProto.functions
    }
    m
}

const DOMAIN: &str = "test.local";

// ── shared fixture ──────────────────────────────────────────────────────────

/// `ScaleShift(x)`, a three-node function whose every attribute arrives by
/// reference:
///
/// ```text
/// t = LeakyRelu(x, alpha = &scale)   // supplied at the call site
/// u = LeakyRelu(t, alpha = &shift)   // declared default 2.0
/// y = LeakyRelu(u, alpha = &unset)   // never supplied and has no default
/// ```
fn scale_shift_function() -> Vec<u8> {
    let nodes = vec![
        node(
            "LeakyRelu",
            "",
            "n0",
            &["x"],
            &["t"],
            &[ref_attr("alpha", "scale")],
        ),
        node(
            "LeakyRelu",
            "",
            "n1",
            &["t"],
            &["u"],
            &[ref_attr("alpha", "shift")],
        ),
        node(
            "LeakyRelu",
            "",
            "n2",
            &["u"],
            &["y"],
            &[ref_attr("alpha", "unset")],
        ),
    ];
    function_proto(&Function {
        name: "ScaleShift",
        domain: DOMAIN,
        inputs: &["x"],
        outputs: &["y"],
        attribute_names: &["scale", "unset"],
        attribute_defaults: &[float_attr("shift", 2.0)],
        nodes: &nodes,
    })
}

fn run(model_bytes: &[u8], x: Tensor) -> HashMap<String, Tensor> {
    let session = Session::from_bytes(model_bytes).expect("session build");
    let mut feed: HashMap<&str, Tensor> = HashMap::new();
    feed.insert("x", x);
    session.run(&feed).expect("run")
}

/// A session with **no** graph optimizations, for the tests that assert node
/// counts: CSE / dead-code elimination would otherwise let an unrelated change
/// move those numbers and turn a structural assertion into a flaky one.
fn unoptimized_session(model_bytes: &[u8]) -> Session {
    SessionBuilder::new()
        .with_optimization_level(OptLevel::None)
        .load_from_bytes(model_bytes)
        .expect("session build")
}

#[track_caller]
fn assert_close(actual: &Tensor, expected: &[f32], what: &str) {
    assert_eq!(actual.data.len(), expected.len(), "{what}: element count");
    for (i, (&a, &e)) in actual.data.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-6,
            "{what}: element {i}: got {a}, expected {e}"
        );
    }
}

// ── the function is called twice ────────────────────────────────────────────

/// The headline case: one function, two call sites, different attributes.
///
/// With `x = [-1, 2]` and `LeakyRelu(v, α) = v if v ≥ 0 else α·v`:
///
/// ```text
/// call 1 (scale = 0.5):   [-1, 2] → [-0.5,  2] → [-1.0, 2] → [-0.010, 2]
/// call 2 (scale = 0.25):  [-1, 2] → [-0.25, 2] → [-0.5, 2] → [-0.005, 2]
/// out = call1 + call2  =  [-0.015, 4]
/// ```
///
/// The second `LeakyRelu` uses the function's declared default `shift = 2.0`;
/// the third's `alpha` resolves to nothing and is therefore **dropped**, so
/// `LeakyRelu`'s own default `0.01` applies. A substituted `0.0` would give
/// `[0, 4]` and a leaked-through `scale` would give `[-0.3125, 4]` — both
/// distinguishable from the expected result.
#[test]
fn function_called_twice_inlines_independently() {
    let g = graph(
        "main",
        &[
            node(
                "ScaleShift",
                DOMAIN,
                "call1",
                &["x"],
                &["o1"],
                &[float_attr("scale", 0.5)],
            ),
            node(
                "ScaleShift",
                DOMAIN,
                "call2",
                &["x"],
                &["o2"],
                &[float_attr("scale", 0.25)],
            ),
            node("Add", "", "sum", &["o1", "o2"], &["out"], &[]),
        ],
        &[("x", 2)],
        &[("out", 2)],
    );
    let bytes = model(&g, DOMAIN, &[scale_shift_function()]);

    let out = run(&bytes, Tensor::new(vec![-1.0, 2.0], vec![2]));
    assert_close(&out["out"], &[-0.015, 4.0], "ScaleShift called twice");
}

/// The two expansions must not share intermediate names.
///
/// If both call sites wrote their `t` / `u` to the *same* graph names, the
/// second expansion would clobber the first and both branches would produce
/// call 2's answer — `2 * [-0.005, 2] = [-0.01, 4]`. Asserting the node count
/// as well pins that the calls really were expanded (6 body nodes + `Add`)
/// rather than left as two opaque nodes.
#[test]
fn two_call_sites_do_not_alias_intermediates() {
    let g = graph(
        "main",
        &[
            node(
                "ScaleShift",
                DOMAIN,
                "call1",
                &["x"],
                &["o1"],
                &[float_attr("scale", 0.5)],
            ),
            node(
                "ScaleShift",
                DOMAIN,
                "call2",
                &["x"],
                &["o2"],
                &[float_attr("scale", 0.25)],
            ),
            node("Add", "", "sum", &["o1", "o2"], &["out"], &[]),
        ],
        &[("x", 2)],
        &[("out", 2)],
    );
    let bytes = model(&g, DOMAIN, &[scale_shift_function()]);
    let session = unoptimized_session(&bytes);

    let nodes = session.nodes();
    assert_eq!(
        nodes.len(),
        7,
        "expected 3 + 3 inlined body nodes plus the Add, got {:?}",
        nodes.iter().map(|n| &n.op_type).collect::<Vec<_>>()
    );
    assert!(
        nodes.iter().all(|n| n.op_type != "ScaleShift"),
        "no function-call node may survive inlining"
    );
    let leaky = nodes.iter().filter(|n| n.op_type == "LeakyRelu").count();
    assert_eq!(
        leaky, 6,
        "both expansions must contribute 3 LeakyRelu nodes"
    );

    // Every intermediate produced by the inlined bodies must be unique.
    let mut produced: Vec<&str> = nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(|s| s.as_str()))
        .collect();
    let total = produced.len();
    produced.sort_unstable();
    produced.dedup();
    assert_eq!(
        produced.len(),
        total,
        "two expansions of the same function produced colliding tensor names"
    );
}

// ── nested calls ────────────────────────────────────────────────────────────

/// A function whose body calls another function: `Outer(v) = Neg(ScaleShift(v))`.
///
/// `ScaleShift(x, scale = 0.5) = [-0.01, 2]`, so `Outer(x) = [0.01, -2]`.
#[test]
fn nested_function_calls_expand_recursively() {
    let outer_nodes = vec![
        node(
            "ScaleShift",
            DOMAIN,
            "inner",
            &["v"],
            &["mid"],
            &[float_attr("scale", 0.5)],
        ),
        node("Neg", "", "neg", &["mid"], &["w"], &[]),
    ];
    let outer = function_proto(&Function {
        name: "Outer",
        domain: DOMAIN,
        inputs: &["v"],
        outputs: &["w"],
        attribute_names: &[],
        attribute_defaults: &[],
        nodes: &outer_nodes,
    });

    let g = graph(
        "main",
        &[node("Outer", DOMAIN, "call", &["x"], &["out"], &[])],
        &[("x", 2)],
        &[("out", 2)],
    );
    let bytes = model(&g, DOMAIN, &[scale_shift_function(), outer]);

    let session = unoptimized_session(&bytes);
    assert_eq!(
        session.nodes().len(),
        4,
        "3 ScaleShift body nodes + Neg after full expansion"
    );

    let out = run(&bytes, Tensor::new(vec![-1.0, 2.0], vec![2]));
    assert_close(&out["out"], &[0.01, -2.0], "Outer (nested)");
}

// ── omitted optional inputs ─────────────────────────────────────────────────

/// A formal input the call site does not pass maps to `""` — still *omitted* —
/// not to a prefixed name nothing produces.
///
/// `ClipLike(v, lo, hi) = Clip(v, lo, hi)`. Calling it with only `v` must leave
/// `Clip` with one input (no clipping at all); prefixing the omitted formals
/// instead would make `Clip` reference two tensors that no node produces.
#[test]
fn omitted_optional_inputs_stay_omitted() {
    let clip_nodes = vec![node("Clip", "", "clip", &["v", "lo", "hi"], &["y"], &[])];
    let clip_like = function_proto(&Function {
        name: "ClipLike",
        domain: DOMAIN,
        inputs: &["v", "lo", "hi"],
        outputs: &["y"],
        attribute_names: &[],
        attribute_defaults: &[],
        nodes: &clip_nodes,
    });

    let g = graph(
        "main",
        &[node("ClipLike", DOMAIN, "call", &["x"], &["out"], &[])],
        &[("x", 2)],
        &[("out", 2)],
    );
    let bytes = model(&g, DOMAIN, &[clip_like]);

    let session = unoptimized_session(&bytes);
    let nodes = session.nodes();
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].op_type, "Clip");
    for (idx, name) in nodes[0].inputs.iter().enumerate().skip(1) {
        assert!(
            name.is_empty(),
            "omitted formal input {idx} became '{name}' instead of staying empty"
        );
    }

    let out = run(&bytes, Tensor::new(vec![-1.0, 2.0], vec![2]));
    assert_close(&out["out"], &[-1.0, 2.0], "ClipLike with no bounds");
}

// ── malformed input ─────────────────────────────────────────────────────────

/// The ONNX spec forbids recursive function references. A self-recursive
/// function must be reported as a typed load error, never blow the stack.
#[test]
fn self_recursive_function_is_a_typed_error() {
    let recursive_nodes = vec![node("Loopy", DOMAIN, "self", &["v"], &["w"], &[])];
    let recursive = function_proto(&Function {
        name: "Loopy",
        domain: DOMAIN,
        inputs: &["v"],
        outputs: &["w"],
        attribute_names: &[],
        attribute_defaults: &[],
        nodes: &recursive_nodes,
    });

    let g = graph(
        "main",
        &[node("Loopy", DOMAIN, "call", &["x"], &["out"], &[])],
        &[("x", 2)],
        &[("out", 2)],
    );
    let bytes = model(&g, DOMAIN, &[recursive]);

    let err = match Session::from_bytes(&bytes) {
        Ok(_) => panic!("recursive function must fail to load"),
        Err(e) => e,
    };
    let text = format!("{err}");
    assert!(
        text.contains("recursive") || text.contains("nesting exceeds"),
        "unexpected error: {text}"
    );
}

/// Two functions with the same `(domain, name)` make which body runs depend on
/// map iteration order; reject the model instead.
#[test]
fn duplicate_function_names_are_rejected() {
    let body = vec![node("Neg", "", "n", &["v"], &["w"], &[])];
    let f = function_proto(&Function {
        name: "Dup",
        domain: DOMAIN,
        inputs: &["v"],
        outputs: &["w"],
        attribute_names: &[],
        attribute_defaults: &[],
        nodes: &body,
    });
    let g = graph(
        "main",
        &[node("Dup", DOMAIN, "call", &["x"], &["out"], &[])],
        &[("x", 2)],
        &[("out", 2)],
    );
    let bytes = model(&g, DOMAIN, &[f.clone(), f]);

    let err = match Session::from_bytes(&bytes) {
        Ok(_) => panic!("duplicate function must fail to load"),
        Err(e) => e,
    };
    assert!(
        format!("{err}").contains("two local functions"),
        "unexpected error: {err}"
    );
}

/// A model with **no** function library must be unaffected — the inliner is a
/// no-op and the graph keeps its original nodes.
#[test]
fn model_without_functions_is_untouched() {
    let g = graph(
        "main",
        &[node("Neg", "", "n", &["x"], &["out"], &[])],
        &[("x", 2)],
        &[("out", 2)],
    );
    let bytes = model(&g, DOMAIN, &[]);
    let out = run(&bytes, Tensor::new(vec![-1.0, 2.0], vec![2]));
    assert_close(&out["out"], &[1.0, -2.0], "plain model");
}

/// A function body containing **two** self-calls doubles the node count at
/// every level, so a depth cap alone is not a bound: reaching depth 64 would
/// mean emitting 2^64 nodes. The inliner therefore also carries a node budget,
/// and this ~200-byte model must be rejected in milliseconds rather than
/// hanging the loader.
#[test]
fn exponentially_recursive_function_is_rejected_quickly() {
    let body = vec![
        node("Boom", DOMAIN, "left", &["v"], &["a"], &[]),
        node("Boom", DOMAIN, "right", &["v"], &["b"], &[]),
        node("Add", "", "join", &["a", "b"], &["w"], &[]),
    ];
    let boom = function_proto(&Function {
        name: "Boom",
        domain: DOMAIN,
        inputs: &["v"],
        outputs: &["w"],
        attribute_names: &[],
        attribute_defaults: &[],
        nodes: &body,
    });
    let g = graph(
        "main",
        &[node("Boom", DOMAIN, "call", &["x"], &["out"], &[])],
        &[("x", 2)],
        &[("out", 2)],
    );
    let bytes = model(&g, DOMAIN, &[boom]);

    let started = std::time::Instant::now();
    let err = match Session::from_bytes(&bytes) {
        Ok(_) => panic!("exponentially recursive function must fail to load"),
        Err(e) => e,
    };
    let elapsed = started.elapsed();
    let text = format!("{err}");
    assert!(
        text.contains("budget") || text.contains("recursive") || text.contains("nesting exceeds"),
        "unexpected error: {text}"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(20),
        "rejection took {elapsed:?}; the bound must be on emitted work, not only on depth"
    );
}

/// `FunctionProto.overload` (IR 10+) disambiguates same-name functions, and a
/// call site selects one through `NodeProto.overload` (wire field 8). The
/// match is the exact triple `(domain, name, overload)` — a call site that
/// does not set field 8 asks for the *unnamed* overload (`""`), and that is
/// not the same request as "whichever overload the library happens to
/// declare". Here the library declares only `"variant_a"`, so this call site
/// (implicit `""`) does not match it: the call is left un-inlined rather than
/// silently run against a body it never named. Loading such a model still
/// succeeds — an unresolved call site is just an ordinary node the registry
/// does not implement, the same as any other unsupported operator — and the
/// mismatch surfaces loudly the moment the node would actually run.
///
/// See `oxionnx-proto/tests/w3_function_overloads.rs` for the matching-overload
/// case, where two overloads of one function name resolve independently.
#[test]
fn overload_mismatch_is_never_silently_matched_to_a_different_body() {
    let body = vec![node("Neg", "", "n", &["v"], &["w"], &[])];
    let mut f = function_proto(&Function {
        name: "Over",
        domain: DOMAIN,
        inputs: &["v"],
        outputs: &["w"],
        attribute_names: &[],
        attribute_defaults: &[],
        nodes: &body,
    });
    f.extend(bytes_field(13, b"variant_a")); // FunctionProto.overload

    let g = graph(
        "main",
        // No `NodeProto.overload` set here -> implicit "", which the
        // library above does NOT declare (it only has "variant_a").
        &[node("Over", DOMAIN, "call", &["x"], &["out"], &[])],
        &[("x", 2)],
        &[("out", 2)],
    );
    let bytes = model(&g, DOMAIN, &[f]);

    let session =
        Session::from_bytes(&bytes).expect("an unmatched overload is not itself a load error");
    assert!(
        session.nodes().iter().any(|n| n.op_type == "Over"),
        "the call site must NOT have been inlined against a different overload's body"
    );

    let mut feed: HashMap<&str, Tensor> = HashMap::new();
    feed.insert("x", Tensor::new(vec![-1.0, 2.0], vec![2]));
    let err = session
        .run(&feed)
        .expect_err("an unresolved function call must fail loudly, never run as a no-op");
    assert!(format!("{err}").contains("Over"), "unexpected error: {err}");
}
