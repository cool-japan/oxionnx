use crate::parser;
use crate::reader;
use crate::types::{AttributeProto, NodeProto, OpsetImport, TensorProto};
use oxionnx_core::Tensor;
use oxionnx_core::{Attributes, Graph, Node, OpKind};
use std::collections::HashMap;
use std::path::Path;

/// Raw metadata extracted from a `ModelProto`, before conversion to the session-layer type.
///
/// Returned alongside graph and weights by [`load_with_metadata`] and
/// [`load_with_metadata_and_path`].
pub struct RawModelMeta {
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub graph_name: String,
    pub ir_version: i64,
    pub opset_imports: Vec<(String, i64)>,
    pub metadata_props: Vec<(String, String)>,
}

/// Supported opset version range (inclusive).
pub const SUPPORTED_OPSET_RANGE: (i64, i64) = (7, 21);

/// Validate the opset version and emit a warning if out of supported range.
fn validate_opset(opset_imports: &[OpsetImport]) {
    for import in opset_imports {
        if import.domain.is_empty() {
            let (min, max) = SUPPORTED_OPSET_RANGE;
            if import.version < min || import.version > max {
                tracing::warn!(
                    opset = import.version,
                    min,
                    max,
                    "model uses opset outside supported range",
                );
            }
        }
    }
}

/// Load an ONNX model file and return (Graph, weight_tensors).
/// External data is NOT supported; use `load_with_path` for models with external data.
pub fn load(bytes: &[u8]) -> Result<(Graph, HashMap<String, Tensor>), String> {
    let (_, graph, weights) = load_with_metadata(bytes)?;
    Ok((graph, weights))
}

/// Load an ONNX model from bytes, resolving external data relative to `base_path`.
pub fn load_with_path(
    bytes: &[u8],
    base_path: &Path,
) -> Result<(Graph, HashMap<String, Tensor>), String> {
    let (_, graph, weights) = load_with_metadata_and_path(bytes, base_path)?;
    Ok((graph, weights))
}

/// Load an ONNX model file and return `(RawModelMeta, Graph, weight_tensors)`.
///
/// External data is NOT supported; use [`load_with_metadata_and_path`] for models
/// that store weights in separate external files.
pub fn load_with_metadata(
    bytes: &[u8],
) -> Result<(RawModelMeta, Graph, HashMap<String, Tensor>), String> {
    let model = parser::parse_model(bytes)?;
    validate_opset(&model.opset_imports);

    let meta = RawModelMeta {
        producer_name: model.producer_name.clone(),
        producer_version: model.producer_version.clone(),
        domain: model.domain.clone(),
        graph_name: model.graph.name.clone(),
        ir_version: model.ir_version,
        opset_imports: model
            .opset_imports
            .iter()
            .map(|o| (o.domain.clone(), o.version))
            .collect(),
        metadata_props: model.metadata_props.clone(),
    };

    // Resolve `ModelProto.functions` (field 25) before anything looks at the
    // node list: after this the graph contains only real operator nodes.
    // `parse_model` above already collected the library in the same pass, so
    // this reuses it instead of re-scanning `bytes` for field 25.
    let functions = model.functions;
    let mut graph_proto = model.graph;
    inline_local_functions(&mut graph_proto, &functions)?;

    // Collect initializer (weight) tensors first (needed for input_names filter and node attrs).
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    for init in &graph_proto.initializers {
        // Same external-data predicate as `reader::OnnxReader`, so both APIs
        // agree on a tensor that carries external_data without data_location.
        if reader::is_external(init) {
            return Err("External data requires load_with_path()".to_string());
        }
        weights.insert(init.name.clone(), decode_tensor_proto(init)?);
    }

    let (graph, weights_out) = build_graph_and_weights(graph_proto, weights, None)?;
    Ok((meta, graph, weights_out))
}

/// Internal helper: build Graph + return weights map from a GraphProto and pre-collected
/// weights. The caller hands ownership of `weights` in; this function borrows them for
/// attribute resolution and hands the same map straight back out together with the Graph.
///
/// `base_path`, when present, lets a subgraph's own (If/Loop/Scan body) initializer resolve
/// external data the same way a top-level initializer does; see [`build_graph_impl`].
fn build_graph_and_weights(
    graph_proto: crate::types::GraphProto,
    weights: HashMap<String, Tensor>,
    base_path: Option<&Path>,
) -> Result<(Graph, HashMap<String, Tensor>), String> {
    let graph = build_graph_impl(&graph_proto, &weights, base_path)?;
    Ok((graph, weights))
}

/// Load an ONNX model from bytes (resolving external data relative to `base_path`)
/// and return `(RawModelMeta, Graph, weight_tensors)`.
pub fn load_with_metadata_and_path(
    bytes: &[u8],
    base_path: &Path,
) -> Result<(RawModelMeta, Graph, HashMap<String, Tensor>), String> {
    let model = parser::parse_model(bytes)?;
    validate_opset(&model.opset_imports);

    let meta = RawModelMeta {
        producer_name: model.producer_name.clone(),
        producer_version: model.producer_version.clone(),
        domain: model.domain.clone(),
        graph_name: model.graph.name.clone(),
        ir_version: model.ir_version,
        opset_imports: model
            .opset_imports
            .iter()
            .map(|o| (o.domain.clone(), o.version))
            .collect(),
        metadata_props: model.metadata_props.clone(),
    };

    // Resolve `ModelProto.functions` (field 25) — see [`inline_local_functions`].
    let functions = model.functions;
    let mut graph_proto = model.graph;
    inline_local_functions(&mut graph_proto, &functions)?;

    // Collect initializer (weight) tensors
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    for init in &graph_proto.initializers {
        let tensor = if reader::is_external(init) {
            load_external_tensor(init, base_path)?
        } else {
            decode_tensor_proto(init)?
        };
        weights.insert(init.name.clone(), tensor);
    }

    let (graph, weights_out) = build_graph_and_weights(graph_proto, weights, Some(base_path))?;
    Ok((meta, graph, weights_out))
}

/// Decode an inline `TensorProto`, reporting a malformed one as a load error
/// instead of substituting silent zeros. `ReaderError` already names the
/// offending tensor, so no extra context prefix is added.
fn decode_tensor_proto(tp: &TensorProto) -> Result<Tensor, String> {
    tp.try_to_tensor().map_err(|e| e.to_string())
}

/// Decode a `TensorProto` that may carry external data, given an optional
/// base directory to resolve it against.
///
/// The top-level initializer loops in [`load_with_metadata`] /
/// [`load_with_metadata_and_path`] always know up front whether a base path
/// exists, so they branch on [`reader::is_external`] directly. This helper
/// exists for the paths where that isn't true at the call site — subgraph
/// (If/Loop/Scan body) initializers and attribute-embedded tensors — which
/// may or may not have a `base_path` depending on how the *model* was
/// loaded, not on where in the proto tree the tensor sits. Without a base
/// path, an external-data tensor is a named, typed error instead of the
/// misleading `MissingTensorData` that falls out of feeding an empty
/// `raw_data` straight into [`decode_tensor_proto`].
fn decode_tensor_proto_ext(tp: &TensorProto, base_path: Option<&Path>) -> Result<Tensor, String> {
    if reader::is_external(tp) {
        match base_path {
            Some(base) => load_external_tensor(tp, base),
            None => Err(format!(
                "tensor '{}' uses external data, but this load path has no base directory to \
                 resolve it against (use load_with_path or load_with_metadata_and_path, which \
                 accept one, instead of a bytes-only or streaming load)",
                tp.name
            )),
        }
    } else {
        decode_tensor_proto(tp)
    }
}

/// Load tensor data from an external file referenced by the TensorProto.
fn load_external_tensor(tensor_proto: &TensorProto, base_path: &Path) -> Result<Tensor, String> {
    let mut location = None;
    let mut offset: u64 = 0;
    let mut length: Option<u64> = None;

    for (key, value) in &tensor_proto.external_data {
        match key.as_str() {
            "location" => location = Some(value.clone()),
            "offset" => {
                offset = value
                    .parse::<u64>()
                    .map_err(|e| format!("Invalid offset '{}': {}", value, e))?;
            }
            "length" => {
                length = Some(
                    value
                        .parse::<u64>()
                        .map_err(|e| format!("Invalid length '{}': {}", value, e))?,
                );
            }
            _ => {} // ignore "checksum" and others
        }
    }

    let location = location.ok_or_else(|| {
        format!(
            "External tensor '{}' missing 'location' field",
            tensor_proto.name
        )
    })?;

    // `location` is attacker-controlled: sandbox it inside the model directory
    // before touching the filesystem.
    let file_path = reader::resolve_external_path(base_path, &location, &tensor_proto.name)
        .map_err(|e| e.to_string())?;
    let file_data = std::fs::read(&file_path).map_err(|e| {
        format!(
            "Cannot read external data file '{}': {}",
            file_path.display(),
            e
        )
    })?;

    let range_err = |detail: String| {
        format!(
            "External data for '{}': {detail} (file size {})",
            tensor_proto.name,
            file_data.len()
        )
    };
    let start = usize::try_from(offset)
        .map_err(|_| range_err(format!("offset {offset} is not addressable")))?;
    let end = match length {
        Some(len) => {
            let len = usize::try_from(len)
                .map_err(|_| range_err(format!("length {len} is not addressable")))?;
            start
                .checked_add(len)
                .ok_or_else(|| range_err(format!("offset {start} + length {len} overflows")))?
        }
        // No explicit length: the tensor owns the tail of the file.
        None => file_data.len(),
    };
    if start > end || end > file_data.len() {
        return Err(range_err(format!("byte range {start}..{end} is invalid")));
    }

    // One shared dtype table with the inline path, plus the dims/length check.
    tensor_proto
        .tensor_from_raw_bytes(&file_data[start..end])
        .map_err(|e| e.to_string())
}

/// Build an oxionnx-core `Graph` from a `GraphProto` and pre-extracted weights.
///
/// This is used by both the batch loader (`load`) and the streaming parser path
/// to convert parsed protobuf structures into the runtime graph representation.
///
/// External data cannot be resolved through this entry point — there is no base
/// directory to resolve it against here — whether it belongs to a nested
/// (If/Loop/Scan body) subgraph's own initializer or to a tensor embedded
/// directly in a node attribute (`AttributeProto.t`/`.tensors`). Either is
/// reported as a named error rather than silently read as zeros; use
/// [`load_with_path`] / [`load_with_metadata_and_path`] for models that need it
/// resolved.
pub fn build_graph(
    graph_proto: &crate::types::GraphProto,
    weights: &HashMap<String, Tensor>,
) -> Result<Graph, String> {
    build_graph_impl(graph_proto, weights, None)
}

/// Shared implementation behind the public [`build_graph`] and the two
/// path-aware `load_with_metadata*` entry points (via [`build_graph_and_weights`]).
///
/// `base_path`, when present, is threaded recursively into every nested
/// subgraph via [`convert_attributes`] / [`build_subgraph`] and passed to
/// every [`decode_tensor_proto_ext`] call, so an If/Loop/Scan body's own
/// initializer *and* a tensor embedded directly in a node attribute both
/// resolve external data exactly like a top-level initializer. `None` (the
/// only option reachable from the public `build_graph`, since its callers —
/// the streaming/bytes-only load paths — have no base directory of their
/// own) makes such a tensor a named error instead of a misleading
/// `MissingTensorData`.
fn build_graph_impl(
    graph_proto: &crate::types::GraphProto,
    weights: &HashMap<String, Tensor>,
    base_path: Option<&Path>,
) -> Result<Graph, String> {
    let mut nodes: Vec<Node> = Vec::with_capacity(graph_proto.nodes.len());
    for np in &graph_proto.nodes {
        let op = OpKind::parse(&np.op_type);
        if let OpKind::Unknown(ref name) = op {
            tracing::debug!(op = %name, "unsupported op, will be skipped");
        }
        let attrs = convert_attributes(&np.attributes, weights, base_path)?;
        nodes.push(Node {
            op,
            name: np.name.clone(),
            inputs: np.inputs.clone(),
            outputs: np.outputs.clone(),
            attrs,
        });
    }

    let input_names: Vec<String> = graph_proto
        .inputs
        .iter()
        .filter(|name| !weights.contains_key(name.as_str()))
        .cloned()
        .collect();

    let input_infos = graph_proto
        .input_value_infos
        .iter()
        .map(|vi| vi.to_tensor_info())
        .collect();
    let output_infos = graph_proto
        .output_value_infos
        .iter()
        .map(|vi| vi.to_tensor_info())
        .collect();

    Ok(Graph {
        name: graph_proto.name.clone(),
        nodes,
        input_names,
        output_names: graph_proto.outputs.clone(),
        input_infos,
        output_infos,
    })
}

/// Extract training information from model bytes.
///
/// Returns an empty vector if the model contains no training info.
pub fn extract_training_info(bytes: &[u8]) -> Result<Vec<crate::types::TrainingInfo>, String> {
    let model = parser::parse_model(bytes)?;
    Ok(model.training_info)
}

/// Convert a `GraphProto` subgraph into a runtime `Graph`, prepending synthesized
/// `Constant` nodes for the subgraph's own initializers and removing those names
/// from `input_names` (they are not real external inputs).
///
/// `base_path` is forwarded from the enclosing [`convert_attributes`] call so a
/// subgraph-local initializer that carries external data resolves it the same
/// way a top-level one does (see [`decode_tensor_proto_ext`]).
fn build_subgraph(
    gp: &crate::types::GraphProto,
    weights: &HashMap<String, Tensor>,
    base_path: Option<&Path>,
) -> Result<Graph, String> {
    // Convert the nested graph (nodes, input_names, output_names, etc.)
    let mut graph = build_graph_impl(gp, weights, base_path)?;

    // Convert local initializers into synthesized Constant nodes prepended to the graph.
    // These initializers are "constants" within the subgraph — they must not appear in
    // input_names, and must be resolved before any node that references them.
    let local_init_names: std::collections::HashSet<String> =
        gp.initializers.iter().map(|i| i.name.clone()).collect();

    // Remove local initializer names from input_names (they are not real graph inputs)
    graph.input_names.retain(|n| !local_init_names.contains(n));
    graph
        .input_infos
        .retain(|vi| !local_init_names.contains(&vi.name));

    // Prepend one Constant node per local initializer
    let mut const_nodes: Vec<Node> = Vec::with_capacity(gp.initializers.len());
    for init in &gp.initializers {
        let mut constant_attrs = Attributes::default();
        constant_attrs.tensors.insert(
            "value".to_string(),
            decode_tensor_proto_ext(init, base_path)?,
        );
        const_nodes.push(Node {
            op: OpKind::Constant,
            name: format!("__const_{}", init.name),
            inputs: vec![],
            outputs: vec![init.name.clone()],
            attrs: constant_attrs,
        });
    }
    const_nodes.append(&mut graph.nodes);
    graph.nodes = const_nodes;

    Ok(graph)
}

fn convert_attributes(
    attrs: &[AttributeProto],
    weights: &HashMap<String, Tensor>,
    base_path: Option<&Path>,
) -> Result<Attributes, String> {
    let mut a = Attributes::default();
    for attr in attrs {
        let name = attr.name.clone();
        let v = &attr.value;

        // Handle graph-valued attributes (If then_branch/else_branch, Loop/Scan body).
        // These are parsed into AttributeValue.g and do not use attr_type dispatch.
        if let Some(ref gp) = v.g {
            a.graphs
                .insert(name.clone(), build_subgraph(gp, weights, base_path)?);
            continue; // skip the attr_type match for this attribute
        }

        // GRAPHS list (AttributeType::GRAPHS, wire field 11 — `AttributeValue.graphs`).
        // Checked by field population rather than `v.attr_type`, mirroring `v.g`
        // above: the parser pushes onto `graphs` whenever field 11 occurs on the
        // wire regardless of what the (separate) attr_type field claims, so a
        // dispatch keyed on attr_type could silently miss a mistyped-but-populated
        // attribute. `Attributes.graphs` has no repeated-GRAPH slot (one `Graph`
        // per name), so only the representable cases are handled here: an empty
        // list contributes nothing, and a single-element list is exactly as
        // representable as the `g` (singular GRAPH) case above and is stored the
        // same way, under the attribute's own name — reachable by any consumer
        // that already does `attrs.graph(name)` / `attrs.graphs.get(name)`. A
        // list of more than one graph has no slot to go in; per the "unrepresentable
        // input is a typed error, never a silent drop" rule, that is reported
        // rather than truncated or key-mangled into an entry nothing will ever look
        // up (no consumer anywhere in this workspace probes an indexed key).
        if !v.graphs.is_empty() {
            if v.graphs.len() > 1 {
                return Err(format!(
                    "attribute '{name}': {} subgraphs given, but this runtime has no \
                     repeated-GRAPH slot (only one subgraph per attribute name is \
                     representable)",
                    v.graphs.len()
                ));
            }
            let sub = build_subgraph(&v.graphs[0], weights, base_path)?;
            a.graphs.insert(name.clone(), sub);
            continue;
        }

        // TENSORS list (AttributeType::TENSORS, wire field 10 — `AttributeValue.tensors`).
        // Same reasoning and the same representable/error split as the GRAPHS list
        // above, against `Attributes.tensors` (one `Tensor` per name).
        if !v.tensors.is_empty() {
            if v.tensors.len() > 1 {
                return Err(format!(
                    "attribute '{name}': {} tensors given, but this runtime has no \
                     repeated-TENSOR slot (only one tensor per attribute name is \
                     representable)",
                    v.tensors.len()
                ));
            }
            let tensor = decode_tensor_proto_ext(&v.tensors[0], base_path)?;
            a.tensors.insert(name.clone(), tensor);
            continue;
        }

        // attr_type: 1=f, 2=i, 3=s, 4=t, 6=floats, 7=ints, 8=strings
        // (5=g/GRAPH and 9=TENSORS/10=GRAPHS are handled above via field
        // population, not this dispatch — see the comments there.)
        match v.attr_type {
            1 => {
                a.floats.insert(name, v.f);
            }
            2 => {
                a.ints.insert(name.clone(), v.i);
            }
            3 => {
                a.strings.insert(name, v.s.clone());
            }
            4 => {
                if let Some(ref tp) = v.t {
                    a.tensors
                        .insert(name, decode_tensor_proto_ext(tp, base_path)?);
                }
            }
            6 => {
                a.float_lists.insert(name, v.floats.clone());
            }
            7 => {
                a.int_lists.insert(name, v.ints.clone());
            }
            8 => {
                // STRINGS: e.g. TreeEnsemble* `nodes_modes`, LSTM/GRU
                // `activations`, StringNormalizer `stopwords`.
                a.string_lists.insert(name, v.strings.clone());
            }
            0 => {
                // attr_type=0 means unset; infer from which field is populated
                if v.f != 0.0 {
                    a.floats.insert(name.clone(), v.f);
                }
                if v.i != 0 {
                    a.ints.insert(name.clone(), v.i);
                }
                if !v.s.is_empty() {
                    a.strings.insert(name.clone(), v.s.clone());
                }
                if !v.floats.is_empty() {
                    a.float_lists.insert(name.clone(), v.floats.clone());
                }
                if !v.ints.is_empty() {
                    a.int_lists.insert(name.clone(), v.ints.clone());
                }
                if !v.strings.is_empty() {
                    a.string_lists.insert(name.clone(), v.strings.clone());
                }
                if let Some(ref tp) = v.t {
                    a.tensors
                        .insert(name, decode_tensor_proto_ext(tp, base_path)?);
                }
            }
            _ => {}
        }
    }
    Ok(a)
}

// ─── Model-local function inlining (ModelProto field 25) ──────────────────
//
// A model-local function is a named, reusable subgraph carried in
// `ModelProto.functions`; nodes call it by `(domain, op_type)`, disambiguated
// further by `overload` when the library declares more than one function
// under the same `(domain, op_type)` (see [`FunctionKey`]). Nothing in the
// execution engine knows about functions, so they are resolved here, at
// model-build time, by *inlining* every call site into the main graph. After
// this pass the graph contains only ordinary operator nodes.
//
// Priority: a model-local function always wins over a built-in operator of the
// same `(domain, name)`. The ONNX spec leaves the conflict resolution to the
// runtime; inlining what the model actually shipped is the choice that cannot
// silently change a model's meaning.

/// How deeply one function body may call other function bodies.
///
/// Reuses the parser's [`parser::MAX_NESTING_DEPTH`] so a hostile or malformed
/// model cannot drive this recursion past the thread stack. The ONNX spec
/// forbids recursive function references outright, so exceeding the cap is
/// reported as (probable) recursion rather than as a size limit.
const MAX_FUNCTION_DEPTH: u32 = parser::MAX_NESTING_DEPTH;

/// Maps a function body's local names onto the call site's actual names.
struct Renamer {
    /// Formal input/output name -> actual name at the call site. An actual of
    /// `""` means the call site omitted that optional input.
    formals: HashMap<String, String>,
    /// Prefix applied to every *other* (purely internal) name in the body, so
    /// two call sites of the same function cannot collide.
    prefix: String,
}

impl Renamer {
    /// Rewrite one name from the function body into main-graph space.
    ///
    /// An empty name stays empty: ONNX uses `""` for an omitted optional input,
    /// and prefixing it would turn "not supplied" into a dangling tensor
    /// reference that no node produces.
    fn apply(&self, name: &str) -> String {
        if name.is_empty() {
            return String::new();
        }
        match self.formals.get(name) {
            Some(actual) => actual.clone(),
            None => format!("{}{}", self.prefix, name),
        }
    }
}

/// The attributes visible to one function expansion: what the call site passed,
/// and what the `FunctionProto` declares as defaults.
struct AttrScope<'a> {
    call_site: HashMap<&'a str, &'a AttributeProto>,
    defaults: HashMap<&'a str, &'a AttributeProto>,
}

impl AttrScope<'_> {
    /// Resolve a body attribute that references a function attribute by name
    /// (`AttributeProto.ref_attr_name`, wire field 21).
    ///
    /// Order: the call site's value, then the function's declared default,
    /// then `None` — which means the attribute is **dropped entirely** so the
    /// operator's own default applies. Substituting a zero instead would
    /// silently pick a different (and usually wrong) operator behaviour.
    ///
    /// The flag says whether the value came from the *call site*, which
    /// matters for a GRAPH-typed attribute: a subgraph handed in by the caller
    /// already names tensors in the caller's namespace and must **not** be put
    /// through the callee's renamer, whereas one declared inside the function
    /// (literally, or as a default) must.
    fn resolve(&self, ref_attr_name: &str) -> Option<(&AttributeProto, bool)> {
        if let Some(found) = self.call_site.get(ref_attr_name) {
            return Some((found, true));
        }
        self.defaults
            .get(ref_attr_name)
            .map(|found| (*found, false))
    }
}

/// Key a function is looked up by: `(domain, name, overload)`.
///
/// `overload` (`FunctionProto.overload` / `NodeProto.overload`, wire field 13
/// / field 8, IR ≥ 10) disambiguates same-`(domain, name)` functions. A call
/// site that does not set field 8 asks for the *unnamed* overload
/// (`overload == ""`) — never "whichever one happens to be in the library" —
/// so this key is an exact triple match, not a `(domain, name)` match with
/// `overload` as a tiebreaker.
type FunctionKey = (String, String, String);

/// Rewrite the names inside a function body's subgraph attribute (an `If`
/// branch, a `Loop`/`Scan` body) so it stays consistent with the inlined body
/// around it.
///
/// Everything a subgraph names is rewritten with the *same* [`Renamer`]: its
/// own formal inputs and outputs, its initializers, its value-info entries and
/// every node inside it. That keeps two things correct at once — the
/// subgraph's internal wiring, and its outer-scope captures, which reference
/// names in the enclosing function body that this same renamer prefixes.
fn rename_subgraph(
    graph: &crate::types::GraphProto,
    renamer: &Renamer,
    scope: &AttrScope<'_>,
    depth: u32,
) -> Result<crate::types::GraphProto, String> {
    if depth > MAX_FUNCTION_DEPTH {
        return Err(format!(
            "function body subgraph nesting exceeds maximum depth {MAX_FUNCTION_DEPTH}"
        ));
    }
    let mut out = graph.clone();
    out.name = format!("{}{}", renamer.prefix, graph.name);
    out.inputs = graph.inputs.iter().map(|n| renamer.apply(n)).collect();
    out.outputs = graph.outputs.iter().map(|n| renamer.apply(n)).collect();
    for vi in out
        .input_value_infos
        .iter_mut()
        .chain(out.output_value_infos.iter_mut())
        .chain(out.value_infos.iter_mut())
    {
        vi.name = renamer.apply(&vi.name);
    }
    for init in out.initializers.iter_mut() {
        init.name = renamer.apply(&init.name);
    }
    out.nodes = graph
        .nodes
        .iter()
        .map(|node| rename_node(node, renamer, scope, depth + 1))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(out)
}

/// Rewrite one body node into main-graph space: names through the [`Renamer`],
/// attributes through the [`AttrScope`], subgraphs recursively.
fn rename_node(
    node: &NodeProto,
    renamer: &Renamer,
    scope: &AttrScope<'_>,
    depth: u32,
) -> Result<NodeProto, String> {
    let mut out = NodeProto {
        inputs: node.inputs.iter().map(|n| renamer.apply(n)).collect(),
        outputs: node.outputs.iter().map(|n| renamer.apply(n)).collect(),
        name: if node.name.is_empty() {
            String::new()
        } else {
            format!("{}{}", renamer.prefix, node.name)
        },
        op_type: node.op_type.clone(),
        attributes: Vec::with_capacity(node.attributes.len()),
        domain: node.domain.clone(),
        // A renamed body node may itself be a call site for another
        // (possibly overloaded) local function; dropping this would silently
        // demote such a nested call to the unnamed overload.
        overload: node.overload.clone(),
    };

    for attr in &node.attributes {
        // A body attribute may take its value from one of the *function's*
        // attributes instead of carrying one itself.
        let (source, from_call_site): (&AttributeProto, bool) =
            if attr.value.ref_attr_name.is_empty() {
                (attr, false)
            } else {
                match scope.resolve(&attr.value.ref_attr_name) {
                    Some(found) => found,
                    // Neither supplied nor defaulted: leave the attribute unset.
                    None => continue,
                }
            };

        let mut resolved = AttributeProto {
            name: attr.name.clone(),
            value: source.value.clone(),
        };
        // The reference is consumed by this substitution; leaving it in place
        // would make a nested expansion try to resolve it a second time.
        resolved.value.ref_attr_name = String::new();

        // Only a subgraph that belongs to the function body (or to one of its
        // declared defaults) lives in the callee's namespace; one substituted
        // in from the call site is already in the caller's and is left alone.
        if !from_call_site {
            if let Some(sub) = resolved.value.g.take() {
                resolved.value.g =
                    Some(Box::new(rename_subgraph(&sub, renamer, scope, depth + 1)?));
            }
            if !resolved.value.graphs.is_empty() {
                resolved.value.graphs = resolved
                    .value
                    .graphs
                    .iter()
                    .map(|g| rename_subgraph(g, renamer, scope, depth + 1))
                    .collect::<Result<Vec<_>, _>>()?;
            }
        }
        out.attributes.push(resolved);
    }

    Ok(out)
}

/// Mutable state carried through one whole inlining pass.
struct ExpandState {
    /// Hands each expansion a unique prefix so the same function called twice
    /// cannot alias its own intermediates.
    counter: usize,
    /// Nodes this pass may still emit.
    ///
    /// The depth cap alone does **not** bound the work: a body containing two
    /// self-calls doubles the node count per level, so reaching depth 64 would
    /// mean emitting 2^64 nodes — a ~200-byte hostile model that hangs the
    /// loader long before any depth check fires. The budget makes the *output
    /// size* the bound, which is the quantity that actually matters.
    budget: usize,
}

impl ExpandState {
    /// Account for one node about to be emitted.
    fn spend(&mut self) -> Result<(), String> {
        self.budget = self.budget.checked_sub(1).ok_or_else(|| {
            "model-local function inlining exceeded its node budget; the ONNX spec forbids \
             recursive function references, so this model most likely contains a function \
             that (directly or mutually) calls itself"
                .to_string()
        })?;
        Ok(())
    }
}

/// Expand a node list, replacing every call to a model-local function with
/// that function's (renamed, attribute-substituted) body.
fn expand_nodes(
    nodes: &[NodeProto],
    library: &HashMap<FunctionKey, &crate::parser::FunctionProto>,
    state: &mut ExpandState,
    depth: u32,
) -> Result<Vec<NodeProto>, String> {
    if depth > MAX_FUNCTION_DEPTH {
        return Err(format!(
            "model-local function expansion exceeds maximum nesting depth {MAX_FUNCTION_DEPTH}"
        ));
    }
    let mut out: Vec<NodeProto> = Vec::with_capacity(nodes.len());
    for node in nodes {
        // Charge every node the pass *visits*, call site or not. Charging only
        // emitted nodes would leave a body made purely of self-calls (no leaf
        // operator at all) free to double per level until the depth cap, which
        // is 2^64 visits.
        state.spend()?;
        let key = (
            node.domain.clone(),
            node.op_type.clone(),
            node.overload.clone(),
        );
        let Some(func) = library.get(&key).copied() else {
            // Not a function call — either a real operator, or a call site
            // naming an overload the library does not declare (never
            // silently matched against a different one — see `FunctionKey`).
            // Either way an `If`/`Loop`/`Scan` body hanging off it may still
            // contain a function call of its own.
            out.push(expand_node_subgraphs(node, library, state, depth + 1)?);
            continue;
        };
        if depth >= MAX_FUNCTION_DEPTH {
            return Err(format!(
                "local function '{}' (domain '{}'): call nesting exceeds maximum depth \
                 {MAX_FUNCTION_DEPTH}; the ONNX spec forbids recursive function references, \
                 so this model most likely contains a function that calls itself",
                func.name, func.domain
            ));
        }
        if node.inputs.len() > func.inputs.len() {
            return Err(format!(
                "local function '{}' (domain '{}'): call site passes {} inputs but the \
                 function declares {}",
                func.name,
                func.domain,
                node.inputs.len(),
                func.inputs.len()
            ));
        }
        if node.outputs.len() > func.outputs.len() {
            return Err(format!(
                "local function '{}' (domain '{}'): call site expects {} outputs but the \
                 function declares {}",
                func.name,
                func.domain,
                node.outputs.len(),
                func.outputs.len()
            ));
        }

        state.counter += 1;
        let prefix = format!("__fn{}_{}__", state.counter, func.name);

        let mut formals: HashMap<String, String> = HashMap::new();
        for (idx, formal) in func.inputs.iter().enumerate() {
            // A formal past the end of the call site's input list is a trailing
            // optional the caller omitted: it maps to `""`, never to a
            // prefixed name nothing produces.
            let actual = node.inputs.get(idx).cloned().unwrap_or_default();
            formals.insert(formal.clone(), actual);
        }
        for (idx, formal) in func.outputs.iter().enumerate() {
            // An omitted *output*, by contrast, is still computed by the body;
            // it just goes to a private name instead of a graph value.
            let actual = match node.outputs.get(idx) {
                Some(name) if !name.is_empty() => name.clone(),
                _ => format!("{prefix}{formal}"),
            };
            formals.insert(formal.clone(), actual);
        }
        let renamer = Renamer { formals, prefix };

        let scope = AttrScope {
            call_site: node
                .attributes
                .iter()
                .map(|a| (a.name.as_str(), a))
                .collect(),
            defaults: func
                .attribute_defaults
                .iter()
                .map(|a| (a.name.as_str(), a))
                .collect(),
        };

        let body: Vec<NodeProto> = func
            .nodes
            .iter()
            .map(|body_node| rename_node(body_node, &renamer, &scope, 0))
            .collect::<Result<Vec<_>, _>>()?;

        // The substituted body may itself call other local functions.
        out.extend(expand_nodes(&body, library, state, depth + 1)?);
    }
    Ok(out)
}

/// Expand function calls that live inside a node's subgraph attributes.
///
/// Returns the node unchanged when it carries no subgraph, so the common case
/// costs one clone and nothing else.
fn expand_node_subgraphs(
    node: &NodeProto,
    library: &HashMap<FunctionKey, &crate::parser::FunctionProto>,
    state: &mut ExpandState,
    depth: u32,
) -> Result<NodeProto, String> {
    let has_subgraph = node
        .attributes
        .iter()
        .any(|a| a.value.g.is_some() || !a.value.graphs.is_empty());
    let mut out = node.clone();
    if !has_subgraph {
        return Ok(out);
    }
    for attr in out.attributes.iter_mut() {
        if let Some(sub) = attr.value.g.as_mut() {
            sub.nodes = expand_nodes(&sub.nodes, library, state, depth)?;
        }
        for sub in attr.value.graphs.iter_mut() {
            sub.nodes = expand_nodes(&sub.nodes, library, state, depth)?;
        }
    }
    Ok(out)
}

/// Inline every model-local function call in `graph_proto`.
///
/// A no-op when the model declares no functions. After this returns, no node
/// in the graph (or in any of its subgraphs) refers to a function body.
pub(crate) fn inline_local_functions(
    graph_proto: &mut crate::types::GraphProto,
    functions: &[crate::parser::FunctionProto],
) -> Result<(), String> {
    if functions.is_empty() {
        return Ok(());
    }
    let mut library: HashMap<FunctionKey, &crate::parser::FunctionProto> = HashMap::new();
    for func in functions {
        if func.name.is_empty() {
            return Err("model-local function has an empty name".to_string());
        }
        // Uniqueness — and call-site resolution below — is the exact triple
        // `(domain, name, overload)` per the spec. `NodeProto.overload` (wire
        // field 8, IR >= 10) carries the call site's choice; `""` means the
        // unnamed overload, not "whichever body happens to be in the
        // library". Keying on `(domain, name)` alone and ignoring `overload`
        // could silently run the wrong body for an overloaded function, which
        // an exact-triple miss cannot do: it falls through to "not a function
        // call" and surfaces as a loud unknown-operator error instead.
        let key = (
            func.domain.clone(),
            func.name.clone(),
            func.overload.clone(),
        );
        if library.insert(key, func).is_some() {
            return Err(format!(
                "model declares two local functions named '{}' in domain '{}' \
                 with the same overload '{}'",
                func.name, func.domain, func.overload
            ));
        }
    }

    // Inlining is a *substitution*, so its output can be much larger than its
    // input; the budget below is what stops a malformed (recursive) library
    // from expanding without bound. It is generous enough that no legitimate
    // model reaches it: real function bodies are a handful of nodes, so a
    // graph of N call sites expands to a small multiple of N.
    let node_count = graph_proto.nodes.len();
    // Generous but finite: at least 100k nodes, ~4096x the call-site count,
    // hard-capped at 16M, and — applied last, so the cap can never reject a
    // graph purely for being large — never below 4x the input graph.
    let budget = node_count
        .saturating_mul(4096)
        .clamp(100_000, 16_000_000)
        .max(node_count.saturating_mul(4));
    let mut state = ExpandState { counter: 0, budget };
    graph_proto.nodes = expand_nodes(&graph_proto.nodes, &library, &mut state, 0)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: encode a varint into bytes.
    fn encode_varint(mut val: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        loop {
            let byte = (val & 0x7F) as u8;
            val >>= 7;
            if val == 0 {
                buf.push(byte);
                break;
            } else {
                buf.push(byte | 0x80);
            }
        }
        buf
    }

    fn encode_varint_field(field: u32, val: u64) -> Vec<u8> {
        let tag = field << 3;
        let mut buf = encode_varint(tag as u64);
        buf.extend(encode_varint(val));
        buf
    }

    fn encode_bytes_field(field: u32, data: &[u8]) -> Vec<u8> {
        let tag = (field << 3) | 2;
        let mut buf = encode_varint(tag as u64);
        buf.extend(encode_varint(data.len() as u64));
        buf.extend(data);
        buf
    }

    /// Build a minimal ONNX model binary with one initializer tensor.
    fn build_model_with_initializer(tensor_proto_bytes: &[u8]) -> Vec<u8> {
        // GraphProto: field 5 = initializer (TensorProto)
        let graph_bytes = encode_bytes_field(5, tensor_proto_bytes);
        // ModelProto: field 1 = ir_version, field 7 = graph, field 8 = opset
        let opset = encode_varint_field(2, 13);
        let mut model_bytes = encode_varint_field(1, 7);
        model_bytes.extend(encode_bytes_field(8, &opset));
        model_bytes.extend(encode_bytes_field(7, &graph_bytes));
        model_bytes
    }

    #[test]
    fn test_load_with_external_data() {
        // Create a temp directory with an external data file
        let tmp_dir = std::env::temp_dir().join("oxionnx_test_ext_data");
        let _ = std::fs::create_dir_all(&tmp_dir);

        // Write 8 floats (2x4 tensor) as raw f32 LE bytes
        let floats: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let raw_bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Put 16 bytes of padding then our data
        let offset = 16u64;
        let mut file_data = vec![0u8; offset as usize];
        file_data.extend(&raw_bytes);
        let data_file = tmp_dir.join("weights.bin");
        std::fs::write(&data_file, &file_data).expect("write external data");

        // Build TensorProto with external data
        let mut tensor_bytes = Vec::new();
        // dims packed: [2, 4]
        let mut dims_packed = encode_varint(2);
        dims_packed.extend(encode_varint(4));
        tensor_bytes.extend(encode_bytes_field(1, &dims_packed));
        // data_type = 1 (float32)
        tensor_bytes.extend(encode_varint_field(2, 1));
        // name = "my_weight"
        tensor_bytes.extend(encode_bytes_field(8, b"my_weight"));
        // external_data entries (field 13, repeated StringStringEntryProto)
        let mut entry_loc = encode_bytes_field(1, b"location");
        entry_loc.extend(encode_bytes_field(2, b"weights.bin"));
        tensor_bytes.extend(encode_bytes_field(13, &entry_loc));

        let mut entry_off = encode_bytes_field(1, b"offset");
        entry_off.extend(encode_bytes_field(2, b"16"));
        tensor_bytes.extend(encode_bytes_field(13, &entry_off));

        let mut entry_len = encode_bytes_field(1, b"length");
        entry_len.extend(encode_bytes_field(2, b"32")); // 8 * 4 bytes
        tensor_bytes.extend(encode_bytes_field(13, &entry_len));
        // data_location = 1 (field 14, enum)
        tensor_bytes.extend(encode_varint_field(14, 1));

        let model_bytes = build_model_with_initializer(&tensor_bytes);

        let (_graph, weights) =
            load_with_path(&model_bytes, &tmp_dir).expect("load_with_path should succeed");

        let tensor = weights.get("my_weight").expect("weight should exist");
        assert_eq!(tensor.shape, vec![2, 4]);
        assert_eq!(tensor.data, floats);

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_load_rejects_external_data_without_path() {
        // Build a TensorProto with data_location=1
        let mut tensor_bytes = Vec::new();
        let mut dims_packed = encode_varint(2);
        dims_packed.extend(encode_varint(2));
        tensor_bytes.extend(encode_bytes_field(1, &dims_packed));
        tensor_bytes.extend(encode_varint_field(2, 1));
        tensor_bytes.extend(encode_bytes_field(8, b"ext_weight"));
        tensor_bytes.extend(encode_varint_field(14, 1));

        let model_bytes = build_model_with_initializer(&tensor_bytes);

        let result = load(&model_bytes);
        assert!(result.is_err());
        let err = result.expect_err("should be error");
        assert!(
            err.contains("External data requires load_with_path()"),
            "got: {err}"
        );
    }

    #[test]
    fn test_opset_validation_in_range() {
        // Opset 13 is in supported range, should not error
        let opset = encode_varint_field(2, 13);
        let mut model_bytes = encode_varint_field(1, 7);
        model_bytes.extend(encode_bytes_field(8, &opset));

        let result = load(&model_bytes);
        assert!(result.is_ok());
    }

    #[test]
    fn test_subgraph_attribute_wired_into_graph() {
        // Build a minimal ONNX model containing an If node whose then_branch carries
        // a subgraph (a single Relu node). Verify that after load(), the If node's
        // Attributes.graphs map contains "then_branch" with one Relu node.

        // ── encode a Relu NodeProto ───────────────────────────────────────────
        let mut relu_node = Vec::new();
        relu_node.extend(encode_bytes_field(1, b"X")); // input
        relu_node.extend(encode_bytes_field(2, b"Y")); // output
        relu_node.extend(encode_bytes_field(3, b"relu_sub")); // name
        relu_node.extend(encode_bytes_field(4, b"Relu")); // op_type

        // ── encode a GraphProto for then_branch ───────────────────────────────
        let mut then_graph = Vec::new();
        then_graph.extend(encode_bytes_field(1, &relu_node)); // node
        then_graph.extend(encode_bytes_field(2, b"then_branch")); // name

        // ── encode AttributeProto: name="then_branch", g=then_graph, attr_type=5 ──
        let mut then_attr = Vec::new();
        then_attr.extend(encode_bytes_field(1, b"then_branch")); // name
        then_attr.extend(encode_bytes_field(6, &then_graph)); // g: GraphProto (field 6)
        then_attr.extend(encode_varint_field(20, 5)); // attr_type = GRAPH

        // ── encode an If NodeProto ────────────────────────────────────────────
        let mut if_node = Vec::new();
        if_node.extend(encode_bytes_field(1, b"cond")); // input
        if_node.extend(encode_bytes_field(2, b"result")); // output
        if_node.extend(encode_bytes_field(3, b"if_op")); // name
        if_node.extend(encode_bytes_field(4, b"If")); // op_type
        if_node.extend(encode_bytes_field(5, &then_attr)); // attribute

        // ── encode a GraphProto (outer model graph) ────────────────────────────
        let mut graph_bytes = Vec::new();
        graph_bytes.extend(encode_bytes_field(1, &if_node)); // node

        // ── encode a full ModelProto ──────────────────────────────────────────
        let opset = encode_varint_field(2, 13); // OperatorSetIdProto: version=13
        let mut model_bytes = encode_varint_field(1, 7); // ir_version=7
        model_bytes.extend(encode_bytes_field(8, &opset)); // opset_import
        model_bytes.extend(encode_bytes_field(7, &graph_bytes)); // graph

        let (graph, _weights) = load(&model_bytes).expect("load should succeed");

        // The outer graph has one node: the If node.
        assert_eq!(graph.nodes.len(), 1);
        let if_graph_node = &graph.nodes[0];

        // The If node's attrs must contain the then_branch subgraph.
        assert!(
            if_graph_node.attrs.graphs.contains_key("then_branch"),
            "If node must have then_branch in attrs.graphs"
        );

        let then_branch = if_graph_node
            .attrs
            .graphs
            .get("then_branch")
            .expect("then_branch must exist");

        // The then_branch subgraph must contain exactly one node: Relu.
        assert_eq!(
            then_branch.nodes.len(),
            1,
            "then_branch subgraph should have 1 node"
        );
        assert_eq!(
            then_branch.nodes[0].op,
            oxionnx_core::OpKind::Relu,
            "then_branch subgraph node must be Relu"
        );
    }

    // ── [stitch S1-1] TENSORS (field 10) / GRAPHS (field 11) list attributes ──

    #[test]
    fn test_tensors_list_attribute_single_element_is_reachable_by_name() {
        // AttributeType::TENSORS (attr_type=9) with exactly one tensor is fully
        // representable in Attributes.tensors (one Tensor per name): it must land
        // under the attribute's own name, not a mangled/indexed one nothing reads.
        let mut tensor_bytes = Vec::new();
        let dims_packed = encode_varint(3);
        tensor_bytes.extend(encode_bytes_field(1, &dims_packed)); // dims = [3]
        tensor_bytes.extend(encode_varint_field(2, 1)); // data_type = float32
        tensor_bytes.extend(encode_bytes_field(8, b"t0")); // name
        let raw: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        tensor_bytes.extend(encode_bytes_field(9, &raw)); // raw_data

        let mut attr_bytes = Vec::new();
        attr_bytes.extend(encode_bytes_field(1, b"consts")); // name
        attr_bytes.extend(encode_bytes_field(10, &tensor_bytes)); // tensors[0] (field 10)
        attr_bytes.extend(encode_varint_field(20, 9)); // attr_type = TENSORS

        let mut node_bytes = Vec::new();
        node_bytes.extend(encode_bytes_field(2, b"y")); // output
        node_bytes.extend(encode_bytes_field(4, b"CustomOp")); // op_type
        node_bytes.extend(encode_bytes_field(5, &attr_bytes)); // attribute

        let mut graph_bytes = Vec::new();
        graph_bytes.extend(encode_bytes_field(1, &node_bytes));

        let opset = encode_varint_field(2, 13);
        let mut model_bytes = encode_varint_field(1, 7);
        model_bytes.extend(encode_bytes_field(8, &opset));
        model_bytes.extend(encode_bytes_field(7, &graph_bytes));

        let (graph, _weights) = load(&model_bytes).expect("load should succeed");
        let tensor = graph.nodes[0]
            .attrs
            .tensors
            .get("consts")
            .expect("single-element TENSORS list must be stored under its own name");
        assert_eq!(tensor.shape, vec![3]);
        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tensors_list_attribute_multi_element_is_a_typed_error() {
        // Two tensors under one TENSORS attribute cannot be represented
        // (Attributes has exactly one Tensor slot per name) — this must fail
        // the load loudly, not truncate to the first entry or drop silently.
        fn one_tensor(name: &str, val: f32) -> Vec<u8> {
            let mut tensor_bytes = Vec::new();
            let dims_packed = encode_varint(1);
            tensor_bytes.extend(encode_bytes_field(1, &dims_packed));
            tensor_bytes.extend(encode_varint_field(2, 1));
            tensor_bytes.extend(encode_bytes_field(8, name.as_bytes()));
            tensor_bytes.extend(encode_bytes_field(9, &val.to_le_bytes()));
            tensor_bytes
        }

        let mut attr_bytes = Vec::new();
        attr_bytes.extend(encode_bytes_field(1, b"consts"));
        attr_bytes.extend(encode_bytes_field(10, &one_tensor("t0", 1.0)));
        attr_bytes.extend(encode_bytes_field(10, &one_tensor("t1", 2.0)));
        attr_bytes.extend(encode_varint_field(20, 9));

        let mut node_bytes = Vec::new();
        node_bytes.extend(encode_bytes_field(2, b"y"));
        node_bytes.extend(encode_bytes_field(4, b"CustomOp"));
        node_bytes.extend(encode_bytes_field(5, &attr_bytes));

        let mut graph_bytes = Vec::new();
        graph_bytes.extend(encode_bytes_field(1, &node_bytes));

        let opset = encode_varint_field(2, 13);
        let mut model_bytes = encode_varint_field(1, 7);
        model_bytes.extend(encode_bytes_field(8, &opset));
        model_bytes.extend(encode_bytes_field(7, &graph_bytes));

        let err = load(&model_bytes).expect_err("2-element TENSORS list must be a typed error");
        assert!(
            err.contains("consts") && err.contains("2 tensors"),
            "error should name the attribute and the count: {err}"
        );
    }

    #[test]
    fn test_graphs_list_attribute_single_element_is_reachable_by_name() {
        // AttributeType::GRAPHS (attr_type=10) with exactly one subgraph goes
        // through build_subgraph exactly like the singular `g` case, and lands
        // under the attribute's own name.
        let mut relu_node = Vec::new();
        relu_node.extend(encode_bytes_field(1, b"X"));
        relu_node.extend(encode_bytes_field(2, b"Y"));
        relu_node.extend(encode_bytes_field(4, b"Relu"));

        let mut sub_graph = Vec::new();
        sub_graph.extend(encode_bytes_field(1, &relu_node));
        sub_graph.extend(encode_bytes_field(2, b"branch0"));

        let mut attr_bytes = Vec::new();
        attr_bytes.extend(encode_bytes_field(1, b"branches"));
        attr_bytes.extend(encode_bytes_field(11, &sub_graph)); // graphs[0] (field 11)
        attr_bytes.extend(encode_varint_field(20, 10)); // attr_type = GRAPHS

        let mut node_bytes = Vec::new();
        node_bytes.extend(encode_bytes_field(2, b"y"));
        node_bytes.extend(encode_bytes_field(4, b"CustomOp"));
        node_bytes.extend(encode_bytes_field(5, &attr_bytes));

        let mut graph_bytes = Vec::new();
        graph_bytes.extend(encode_bytes_field(1, &node_bytes));

        let opset = encode_varint_field(2, 13);
        let mut model_bytes = encode_varint_field(1, 7);
        model_bytes.extend(encode_bytes_field(8, &opset));
        model_bytes.extend(encode_bytes_field(7, &graph_bytes));

        let (graph, _weights) = load(&model_bytes).expect("load should succeed");
        let branch = graph.nodes[0]
            .attrs
            .graphs
            .get("branches")
            .expect("single-element GRAPHS list must be stored under its own name");
        assert_eq!(branch.nodes.len(), 1);
        assert_eq!(branch.nodes[0].op, OpKind::Relu);
    }

    #[test]
    fn test_graphs_list_attribute_multi_element_is_a_typed_error() {
        fn one_graph(name: &str) -> Vec<u8> {
            let mut relu_node = Vec::new();
            relu_node.extend(encode_bytes_field(1, b"X"));
            relu_node.extend(encode_bytes_field(2, b"Y"));
            relu_node.extend(encode_bytes_field(4, b"Relu"));
            let mut g = Vec::new();
            g.extend(encode_bytes_field(1, &relu_node));
            g.extend(encode_bytes_field(2, name.as_bytes()));
            g
        }

        let mut attr_bytes = Vec::new();
        attr_bytes.extend(encode_bytes_field(1, b"branches"));
        attr_bytes.extend(encode_bytes_field(11, &one_graph("branch0")));
        attr_bytes.extend(encode_bytes_field(11, &one_graph("branch1")));
        attr_bytes.extend(encode_varint_field(20, 10));

        let mut node_bytes = Vec::new();
        node_bytes.extend(encode_bytes_field(2, b"y"));
        node_bytes.extend(encode_bytes_field(4, b"CustomOp"));
        node_bytes.extend(encode_bytes_field(5, &attr_bytes));

        let mut graph_bytes = Vec::new();
        graph_bytes.extend(encode_bytes_field(1, &node_bytes));

        let opset = encode_varint_field(2, 13);
        let mut model_bytes = encode_varint_field(1, 7);
        model_bytes.extend(encode_bytes_field(8, &opset));
        model_bytes.extend(encode_bytes_field(7, &graph_bytes));

        let err = load(&model_bytes).expect_err("2-element GRAPHS list must be a typed error");
        assert!(
            err.contains("branches") && err.contains("2 subgraphs"),
            "error should name the attribute and the count: {err}"
        );
    }

    // ── [stitch S1-5] base_path threaded through subgraph initializers ──

    #[test]
    fn test_subgraph_external_initializer_resolves_with_base_path() {
        let tmp_dir = std::env::temp_dir().join("oxionnx_test_subgraph_ext_data");
        let _ = std::fs::create_dir_all(&tmp_dir);

        let floats: Vec<f32> = vec![9.0, 8.0];
        let raw_bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let data_file = tmp_dir.join("sub_weights.bin");
        std::fs::write(&data_file, &raw_bytes).expect("write external data");

        let mut tensor_bytes = Vec::new();
        let dims_packed = encode_varint(2);
        tensor_bytes.extend(encode_bytes_field(1, &dims_packed)); // dims = [2]
        tensor_bytes.extend(encode_varint_field(2, 1)); // data_type = float32
        tensor_bytes.extend(encode_bytes_field(8, b"sub_const")); // name
        let mut entry_loc = encode_bytes_field(1, b"location");
        entry_loc.extend(encode_bytes_field(2, b"sub_weights.bin"));
        tensor_bytes.extend(encode_bytes_field(13, &entry_loc)); // external_data
        tensor_bytes.extend(encode_varint_field(14, 1)); // data_location = EXTERNAL

        // then_branch GraphProto: just the one (external) initializer, no nodes.
        let mut then_graph = Vec::new();
        then_graph.extend(encode_bytes_field(5, &tensor_bytes)); // initializer
        then_graph.extend(encode_bytes_field(2, b"then_branch")); // name

        let mut then_attr = Vec::new();
        then_attr.extend(encode_bytes_field(1, b"then_branch"));
        then_attr.extend(encode_bytes_field(6, &then_graph)); // g (field 6)
        then_attr.extend(encode_varint_field(20, 5)); // attr_type = GRAPH

        let mut if_node = Vec::new();
        if_node.extend(encode_bytes_field(1, b"cond"));
        if_node.extend(encode_bytes_field(2, b"sub_const"));
        if_node.extend(encode_bytes_field(4, b"If"));
        if_node.extend(encode_bytes_field(5, &then_attr));

        let mut graph_bytes = Vec::new();
        graph_bytes.extend(encode_bytes_field(1, &if_node));

        let opset = encode_varint_field(2, 13);
        let mut model_bytes = encode_varint_field(1, 7);
        model_bytes.extend(encode_bytes_field(8, &opset));
        model_bytes.extend(encode_bytes_field(7, &graph_bytes));

        // Without a base path: a clear, specific error naming external data as
        // the cause — not the generic "missing tensor data" that falls out of
        // feeding an empty raw_data straight into the decoder.
        let no_path_err =
            load(&model_bytes).expect_err("external data with no base path must fail");
        assert!(
            no_path_err.contains("external data") && no_path_err.contains("sub_const"),
            "error should name external data as the cause: {no_path_err}"
        );

        // With a base path: the subgraph's own initializer resolves correctly,
        // the same way a top-level one does.
        let (graph, _weights) = load_with_path(&model_bytes, &tmp_dir)
            .expect("load_with_path should resolve subgraph external data");
        let then_branch = graph.nodes[0]
            .attrs
            .graphs
            .get("then_branch")
            .expect("then_branch present");
        assert_eq!(then_branch.nodes.len(), 1);
        assert_eq!(then_branch.nodes[0].op, OpKind::Constant);
        let const_tensor = then_branch.nodes[0]
            .attrs
            .tensors
            .get("value")
            .expect("value tensor");
        assert_eq!(const_tensor.shape, vec![2]);
        assert_eq!(const_tensor.data, floats);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_nested_subgraph_external_initializer_resolves_two_levels_deep() {
        // If (outer) -> then_branch -> If (inner) -> then_branch -> external
        // initializer. Proves base_path threads recursively through
        // convert_attributes -> build_subgraph -> build_graph_impl at every
        // nesting depth, not just the first.
        let tmp_dir = std::env::temp_dir().join("oxionnx_test_nested_subgraph_ext_data");
        let _ = std::fs::create_dir_all(&tmp_dir);

        let floats: Vec<f32> = vec![4.0, 5.0, 6.0];
        let raw_bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let data_file = tmp_dir.join("leaf.bin");
        std::fs::write(&data_file, &raw_bytes).expect("write external data");

        // ── innermost initializer (external) ──
        let mut leaf_tensor = Vec::new();
        let dims_packed = encode_varint(3);
        leaf_tensor.extend(encode_bytes_field(1, &dims_packed)); // dims = [3]
        leaf_tensor.extend(encode_varint_field(2, 1));
        leaf_tensor.extend(encode_bytes_field(8, b"leaf_const"));
        let mut entry_loc = encode_bytes_field(1, b"location");
        entry_loc.extend(encode_bytes_field(2, b"leaf.bin"));
        leaf_tensor.extend(encode_bytes_field(13, &entry_loc));
        leaf_tensor.extend(encode_varint_field(14, 1));

        // ── innermost then_branch: just the one initializer ──
        let mut innermost_graph = Vec::new();
        innermost_graph.extend(encode_bytes_field(5, &leaf_tensor));
        innermost_graph.extend(encode_bytes_field(2, b"innermost_then"));

        let mut inner_then_attr = Vec::new();
        inner_then_attr.extend(encode_bytes_field(1, b"then_branch"));
        inner_then_attr.extend(encode_bytes_field(6, &innermost_graph));
        inner_then_attr.extend(encode_varint_field(20, 5));

        let mut inner_if_node = Vec::new();
        inner_if_node.extend(encode_bytes_field(1, b"cond2"));
        inner_if_node.extend(encode_bytes_field(2, b"leaf_const"));
        inner_if_node.extend(encode_bytes_field(4, b"If"));
        inner_if_node.extend(encode_bytes_field(5, &inner_then_attr));

        // ── middle graph (outer then_branch): contains only the inner If ──
        let mut middle_graph = Vec::new();
        middle_graph.extend(encode_bytes_field(1, &inner_if_node));
        middle_graph.extend(encode_bytes_field(2, b"middle_then"));

        let mut outer_then_attr = Vec::new();
        outer_then_attr.extend(encode_bytes_field(1, b"then_branch"));
        outer_then_attr.extend(encode_bytes_field(6, &middle_graph));
        outer_then_attr.extend(encode_varint_field(20, 5));

        let mut outer_if_node = Vec::new();
        outer_if_node.extend(encode_bytes_field(1, b"cond1"));
        outer_if_node.extend(encode_bytes_field(2, b"leaf_const"));
        outer_if_node.extend(encode_bytes_field(4, b"If"));
        outer_if_node.extend(encode_bytes_field(5, &outer_then_attr));

        let mut graph_bytes = Vec::new();
        graph_bytes.extend(encode_bytes_field(1, &outer_if_node));

        let opset = encode_varint_field(2, 13);
        let mut model_bytes = encode_varint_field(1, 7);
        model_bytes.extend(encode_bytes_field(8, &opset));
        model_bytes.extend(encode_bytes_field(7, &graph_bytes));

        let (graph, _weights) = load_with_path(&model_bytes, &tmp_dir)
            .expect("load_with_path should resolve a two-level-nested subgraph's external data");

        let middle = graph.nodes[0]
            .attrs
            .graphs
            .get("then_branch")
            .expect("outer then_branch present");
        // middle has no local initializers of its own, so its one node is the
        // inner If directly (no synthesized Constant prepended).
        assert_eq!(middle.nodes.len(), 1);
        let innermost = middle.nodes[0]
            .attrs
            .graphs
            .get("then_branch")
            .expect("inner then_branch present");
        assert_eq!(innermost.nodes.len(), 1);
        assert_eq!(innermost.nodes[0].op, OpKind::Constant);
        let tensor = innermost.nodes[0]
            .attrs
            .tensors
            .get("value")
            .expect("leaf tensor decoded");
        assert_eq!(tensor.shape, vec![3]);
        assert_eq!(tensor.data, floats);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_attribute_embedded_external_tensor_resolves_with_base_path() {
        // Same external-data resolution, but through the OTHER path that was
        // widened to use `decode_tensor_proto_ext`: a tensor embedded directly
        // as a node attribute value (AttributeProto.t, attr_type=4 / TENSOR),
        // not a graph initializer at all. A Constant node's `value` attribute
        // is the natural real-world example of this.
        let tmp_dir = std::env::temp_dir().join("oxionnx_test_attr_embedded_ext_data");
        let _ = std::fs::create_dir_all(&tmp_dir);

        let floats: Vec<f32> = vec![7.0, 6.0, 5.0, 4.0];
        let raw_bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let data_file = tmp_dir.join("attr_weights.bin");
        std::fs::write(&data_file, &raw_bytes).expect("write external data");

        let mut tensor_bytes = Vec::new();
        let dims_packed = encode_varint(4);
        tensor_bytes.extend(encode_bytes_field(1, &dims_packed)); // dims = [4]
        tensor_bytes.extend(encode_varint_field(2, 1)); // data_type = float32
        tensor_bytes.extend(encode_bytes_field(8, b"embedded_const")); // name
        let mut entry_loc = encode_bytes_field(1, b"location");
        entry_loc.extend(encode_bytes_field(2, b"attr_weights.bin"));
        tensor_bytes.extend(encode_bytes_field(13, &entry_loc)); // external_data
        tensor_bytes.extend(encode_varint_field(14, 1)); // data_location = EXTERNAL

        // AttributeProto: name="value", t=tensor_bytes (field 5), attr_type=4 (TENSOR).
        let mut value_attr = Vec::new();
        value_attr.extend(encode_bytes_field(1, b"value"));
        value_attr.extend(encode_bytes_field(5, &tensor_bytes)); // t (field 5)
        value_attr.extend(encode_varint_field(20, 4)); // attr_type = TENSOR

        let mut const_node = Vec::new();
        const_node.extend(encode_bytes_field(2, b"y")); // output
        const_node.extend(encode_bytes_field(4, b"Constant")); // op_type
        const_node.extend(encode_bytes_field(5, &value_attr)); // attribute

        let mut graph_bytes = Vec::new();
        graph_bytes.extend(encode_bytes_field(1, &const_node));

        let opset = encode_varint_field(2, 13);
        let mut model_bytes = encode_varint_field(1, 7);
        model_bytes.extend(encode_bytes_field(8, &opset));
        model_bytes.extend(encode_bytes_field(7, &graph_bytes));

        // Without a base path: named error, not a silent MissingTensorData.
        let no_path_err =
            load(&model_bytes).expect_err("attribute-embedded external data needs a base path");
        assert!(
            no_path_err.contains("external data") && no_path_err.contains("embedded_const"),
            "error should name external data as the cause: {no_path_err}"
        );

        // With a base path: resolves exactly like a graph initializer would.
        let (graph, _weights) = load_with_path(&model_bytes, &tmp_dir)
            .expect("load_with_path should resolve an attribute-embedded external tensor");
        let tensor = graph.nodes[0]
            .attrs
            .tensors
            .get("value")
            .expect("value tensor decoded");
        assert_eq!(tensor.shape, vec![4]);
        assert_eq!(tensor.data, floats);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}
