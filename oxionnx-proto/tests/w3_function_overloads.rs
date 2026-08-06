//! Wave-3 tests for `FunctionProto.overload` / `NodeProto.overload`
//! (`ModelProto.functions` wire field 25's overload disambiguation, IR >= 10).
//!
//! Before this task, a model declaring two functions with the same
//! `(domain, name)` but different `overload` strings was refused outright at
//! load time (`FunctionProto` had no way to tell them apart from a call
//! site). The loader now keys its function library — and resolves every call
//! site — on the exact triple `(domain, name, overload)`:
//!
//! 1. two overloads of one function name resolve to distinct bodies, chosen
//!    per call site by `NodeProto.overload` (wire field 8),
//! 2. a call site whose overload does not match any declared overload is
//!    never silently run against a different body — it is left un-inlined
//!    and surfaces as an ordinary unsupported-operator node, and
//! 3. the streaming loader (`parse_streaming` / `parse_with_weight_filter`)
//!    resolves function calls — including overloaded ones — the same way the
//!    eager loader does, which earlier did not inline at all and left every
//!    call site as an unresolved node.
//!
//! Models are hand-encoded protobuf, mirroring the style already used by
//! `tests/w1_proto_parser.rs` and the root crate's `tests/w2_local_functions_e2e.rs`.

use std::io::Cursor;

use oxionnx_core::{Graph, OpKind};
use oxionnx_proto::model::load;
use oxionnx_proto::{build_graph, parse_streaming, parse_with_weight_filter};

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

/// `NodeProto`, with an optional `overload` (wire field 8) selecting which
/// overload of a model-local function this call site invokes.
fn node(
    op_type: &str,
    domain: &str,
    overload: &str,
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
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
    if !domain.is_empty() {
        n.extend(bytes_field(7, domain.as_bytes()));
    }
    if !overload.is_empty() {
        n.extend(bytes_field(8, overload.as_bytes()));
    }
    n
}

/// Appends `AttributeProto` entries (field 5) to already-built `NodeProto`
/// bytes (as returned by `node`), for the one test below that needs a node
/// carrying a subgraph attribute — every other call site has none.
fn with_attrs(mut node_bytes: Vec<u8>, attrs: &[Vec<u8>]) -> Vec<u8> {
    for a in attrs {
        node_bytes.extend(bytes_field(5, a));
    }
    node_bytes
}

/// `AttributeProto` carrying a single subgraph (field 6 = `g`, the shape
/// `If`'s `then_branch`/`else_branch` and `Loop`/`Scan`'s `body` all use).
fn graph_attr(name: &str, nested_graph_bytes: &[u8]) -> Vec<u8> {
    let mut attr = bytes_field(1, name.as_bytes()); // name
    attr.extend(bytes_field(6, nested_graph_bytes)); // g
    attr
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

/// `FunctionProto`, with an optional `overload` (wire field 13).
fn function_proto(
    name: &str,
    domain: &str,
    overload: &str,
    inputs: &[&str],
    outputs: &[&str],
    nodes: &[Vec<u8>],
) -> Vec<u8> {
    let mut p = bytes_field(1, name.as_bytes()); // name
    for i in inputs {
        p.extend(bytes_field(4, i.as_bytes()));
    }
    for o in outputs {
        p.extend(bytes_field(5, o.as_bytes()));
    }
    for n in nodes {
        p.extend(bytes_field(7, n));
    }
    p.extend(bytes_field(9, &opset_import("", 18))); // opset_import
    p.extend(bytes_field(10, domain.as_bytes())); // domain
    if !overload.is_empty() {
        p.extend(bytes_field(13, overload.as_bytes()));
    }
    p
}

/// `ModelProto` with an `ai.onnx` opset, a custom-domain opset and a function library.
fn model(graph_bytes: &[u8], custom_domain: &str, functions: &[Vec<u8>]) -> Vec<u8> {
    let mut m = varint_field(1, 10); // ir_version >= 10, so `overload` is meaningful
    m.extend(bytes_field(8, &opset_import("", 18)));
    m.extend(bytes_field(8, &opset_import(custom_domain, 1)));
    m.extend(bytes_field(7, graph_bytes));
    for f in functions {
        m.extend(bytes_field(25, f)); // ModelProto.functions
    }
    m
}

const DOMAIN: &str = "test.local";

/// Two `Poly` overloads (`"a"` -> `Neg`, `"b"` -> `Abs`) plus two call sites,
/// one per overload. Shared by the overload-resolution test and the
/// eager/streaming parity test below, so both exercise the same fixture.
fn two_overload_model_bytes() -> Vec<u8> {
    let body_a = vec![node("Neg", "", "", "n", &["v"], &["w"])];
    let overload_a = function_proto("Poly", DOMAIN, "a", &["v"], &["w"], &body_a);

    let body_b = vec![node("Abs", "", "", "n", &["v"], &["w"])];
    let overload_b = function_proto("Poly", DOMAIN, "b", &["v"], &["w"], &body_b);

    let g = graph(
        "main",
        &[
            node("Poly", DOMAIN, "a", "call_a", &["x"], &["out_a"]),
            node("Poly", DOMAIN, "b", "call_b", &["x"], &["out_b"]),
        ],
        &[("x", 2)],
        &[("out_a", 2), ("out_b", 2)],
    );
    model(&g, DOMAIN, &[overload_a, overload_b])
}

// ── two overloads resolve independently ─────────────────────────────────────

/// The headline case for this file: `Poly/"a"` and `Poly/"b"` are two
/// different functions that merely happen to share `(domain, name)`. Each
/// call site names its overload via `NodeProto.overload` and must inline the
/// matching body — never the other one, and never refuse to load as the
/// pre-overload-support loader did.
#[test]
fn two_overloads_of_one_function_name_resolve_to_distinct_bodies() {
    let bytes = two_overload_model_bytes();
    let (graph, _weights) = load(&bytes).expect("model with disambiguated overloads must load");

    assert_eq!(
        graph.nodes.len(),
        2,
        "both call sites must inline to exactly their (single-node) body, got {:?}",
        graph
            .nodes
            .iter()
            .map(|n| n.op.as_str())
            .collect::<Vec<_>>()
    );

    let neg = graph
        .nodes
        .iter()
        .find(|n| n.op == OpKind::Neg)
        .expect("overload 'a' must inline its Neg body");
    assert_eq!(
        neg.inputs,
        vec!["x".to_string()],
        "formal 'v' must resolve to the call site's actual input"
    );
    assert_eq!(neg.outputs, vec!["out_a".to_string()]);

    let abs = graph
        .nodes
        .iter()
        .find(|n| n.op == OpKind::Abs)
        .expect("overload 'b' must inline its Abs body");
    assert_eq!(abs.inputs, vec!["x".to_string()]);
    assert_eq!(abs.outputs, vec!["out_b".to_string()]);

    assert!(
        graph
            .nodes
            .iter()
            .all(|n| n.op != OpKind::Unknown("Poly".to_string())),
        "no call-site node may survive inlining when its overload is declared"
    );
}

/// A call site whose overload does not match any declared overload must never
/// be matched to a *different* one — that would silently run the wrong body.
/// It is left un-inlined instead, becoming an ordinary node this loader does
/// not recognize (the same outcome as calling any other undeclared operator).
#[test]
fn call_site_overload_with_no_match_is_never_matched_to_a_different_body() {
    let body = vec![node("Neg", "", "", "n", &["v"], &["w"])];
    let only_overload = function_proto("Solo", DOMAIN, "variant_a", &["v"], &["w"], &body);

    let g = graph(
        "main",
        // No `overload` set on the call site -> implicit "" -- which the
        // library does NOT declare (it only has "variant_a").
        &[node("Solo", DOMAIN, "", "call", &["x"], &["out"])],
        &[("x", 2)],
        &[("out", 2)],
    );
    let bytes = model(&g, DOMAIN, &[only_overload]);

    let (graph, _weights) = load(&bytes).expect("an unmatched overload is not itself a load error");
    assert_eq!(
        graph.nodes.len(),
        1,
        "the unmatched call site is not expanded"
    );
    assert_eq!(
        graph.nodes[0].op,
        OpKind::Unknown("Solo".to_string()),
        "an overload miss must surface as an unrecognized op, never as the wrong body's Neg"
    );
}

// ── streaming parity ────────────────────────────────────────────────────────

/// The streaming loader must resolve model-local function calls — including
/// overloaded ones — exactly as the eager loader does. Before this task,
/// `parse_streaming` never called the inliner at all, so a call site (of any
/// function, overloaded or not) survived into the returned `GraphProto`
/// unresolved; `build_graph` would then hand the session an `Unknown` op that
/// legitimately had a body available.
#[test]
fn streaming_load_inlines_overloaded_function_calls_the_same_as_eager_load() {
    let bytes = two_overload_model_bytes();

    let (eager_graph, eager_weights) = load(&bytes).expect("eager load must succeed");
    let (stream_proto, stream_weights) =
        parse_streaming(Cursor::new(bytes)).expect("streaming parse must succeed");
    let streamed_graph =
        build_graph(&stream_proto, &stream_weights).expect("streaming graph must build");

    assert_eq!(eager_weights.len(), stream_weights.len());
    assert_eq!(
        eager_graph.nodes.len(),
        streamed_graph.nodes.len(),
        "eager: {:?}\nstreamed: {:?}",
        eager_graph
            .nodes
            .iter()
            .map(|n| n.op.as_str())
            .collect::<Vec<_>>(),
        streamed_graph
            .nodes
            .iter()
            .map(|n| n.op.as_str())
            .collect::<Vec<_>>()
    );
    for (e, s) in eager_graph.nodes.iter().zip(streamed_graph.nodes.iter()) {
        assert_eq!(
            e.op, s.op,
            "op mismatch between eager and streaming inlining"
        );
        assert_eq!(e.inputs, s.inputs);
        assert_eq!(e.outputs, s.outputs);
    }
    assert_eq!(eager_graph.output_names, streamed_graph.output_names);
}

/// A function body's own node may carry a subgraph attribute — the same
/// shape `If`'s `then_branch`/`else_branch` and `Loop`/`Scan`'s `body` use —
/// and that subgraph may itself call another local function. Resolving that
/// nested call is `expand_node_subgraphs` recursing into the attribute, and
/// it must happen — on *both* loaders — before `build_graph` ever converts
/// the subgraph into a runtime `Graph`, so the conversion never sees an
/// unresolved call. This is the streaming path's least obvious case: the
/// function library only fully exists once the whole stream has been read
/// (`ModelProto.functions` sorts after `graph` on the wire), so nothing may
/// attempt to resolve a call — nested or not — before then.
#[test]
fn nested_function_call_inside_a_subgraph_attribute_resolves_on_both_loaders() {
    let leaf_body = vec![node("Neg", "", "", "n", &["v"], &["w"])];
    let leaf = function_proto("Leaf", DOMAIN, "", &["v"], &["w"], &leaf_body);

    let inner_graph = graph(
        "body_graph",
        &[node("Leaf", DOMAIN, "", "leaf_call", &["v"], &["w"])],
        &[("v", 2)],
        &[("w", 2)],
    );
    let holder = with_attrs(
        node("SubgraphHolder", "", "", "holder", &["v"], &["w"]),
        &[graph_attr("body", &inner_graph)],
    );
    let wrap = function_proto("Wrap", DOMAIN, "", &["v"], &["w"], &[holder]);

    let g = graph(
        "main",
        &[node("Wrap", DOMAIN, "", "call", &["x"], &["out"])],
        &[("x", 2)],
        &[("out", 2)],
    );
    let bytes = model(&g, DOMAIN, &[leaf, wrap]);

    // Eager.
    let (eager_graph, _weights) = load(&bytes).expect("eager load must succeed");
    assert_eq!(
        eager_graph.nodes.len(),
        1,
        "Wrap's body is a single holder node"
    );
    let eager_body = eager_graph.nodes[0]
        .attrs
        .graphs
        .get("body")
        .expect("the holder node must carry its 'body' subgraph attribute");
    assert_eq!(eager_body.nodes.len(), 1);
    assert_eq!(
        eager_body.nodes[0].op,
        OpKind::Neg,
        "the nested Leaf call inside the subgraph attribute must have been inlined (eager)"
    );

    // Streaming — the same bytes, through the loader this task's item (1) fixed.
    let (stream_proto, stream_weights) =
        parse_streaming(Cursor::new(bytes)).expect("streaming parse must succeed");
    let streamed_graph: Graph =
        build_graph(&stream_proto, &stream_weights).expect("streaming graph must build");
    assert_eq!(streamed_graph.nodes.len(), 1);
    let stream_body = streamed_graph.nodes[0]
        .attrs
        .graphs
        .get("body")
        .expect("the holder node must carry its 'body' subgraph attribute (streaming)");
    assert_eq!(stream_body.nodes.len(), 1);
    assert_eq!(
        stream_body.nodes[0].op,
        OpKind::Neg,
        "the nested Leaf call inside the subgraph attribute must have been inlined (streaming)"
    );
}

/// `parse_with_weight_filter` is the *second* `GraphProto` construction site
/// in the streaming convenience layer, independent of `parse_streaming`'s.
/// Both must inline local function calls, not just the first one.
#[test]
fn parse_with_weight_filter_also_inlines_local_function_calls() {
    let body = vec![node("Neg", "", "", "n", &["v"], &["w"])];
    let f = function_proto("Solo", DOMAIN, "", &["v"], &["w"], &body);
    let g = graph(
        "main",
        &[node("Solo", DOMAIN, "", "call", &["x"], &["out"])],
        &[("x", 2)],
        &[("out", 2)],
    );
    let bytes = model(&g, DOMAIN, &[f]);

    let (graph, _weights) = parse_with_weight_filter(Cursor::new(bytes), |_name, _shape| true)
        .and_then(|(g, w)| {
            build_graph(&g, &w)
                .map(|graph| (graph, w))
                .map_err(|e| e.to_string())
        })
        .expect("parse_with_weight_filter must inline the call site");

    assert_eq!(graph.nodes.len(), 1);
    assert_eq!(
        graph.nodes[0].op,
        OpKind::Neg,
        "the call site must have inlined its body"
    );
    assert_eq!(graph.nodes[0].inputs, vec!["x".to_string()]);
    assert_eq!(graph.nodes[0].outputs, vec!["out".to_string()]);
}

/// Sanity: a model with no function library at all must still round-trip
/// through both loaders identically (the common, function-free case must stay
/// cheap and unaffected).
#[test]
fn streaming_and_eager_agree_on_a_function_free_model_too() {
    let g = graph(
        "main",
        &[node("Neg", "", "", "n", &["x"], &["out"])],
        &[("x", 2)],
        &[("out", 2)],
    );
    let bytes = model(&g, DOMAIN, &[]);

    let (eager_graph, _) = load(&bytes).expect("eager load must succeed");
    let (stream_proto, stream_weights) =
        parse_streaming(Cursor::new(bytes)).expect("streaming parse must succeed");
    let streamed_graph: Graph =
        build_graph(&stream_proto, &stream_weights).expect("streaming graph must build");

    assert_eq!(eager_graph.nodes.len(), 1);
    assert_eq!(streamed_graph.nodes.len(), 1);
    assert_eq!(eager_graph.nodes[0].op, streamed_graph.nodes[0].op);
}
