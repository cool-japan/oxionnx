//! Session caching: save the **optimized** graph + weights to disk and reload
//! it without paying for the optimization pipeline again.
//!
//! # What is cached
//!
//! Everything the engine needs to execute, in the shape it has *after*
//! session construction has run constant folding, CSE, fusion, dead-node
//! elimination and the topological sort:
//!
//! * the optimized, topologically sorted node list (including nested subgraph
//!   attributes for `If` / `Loop` / `Scan`),
//! * the weight table — which the optimizer *rewrites*: constant folding adds
//!   folded initializers and fusions synthesise new ones, so re-deriving it from
//!   the original `.onnx` would not reproduce this graph,
//! * graph input/output names and their `ValueInfo` metadata,
//! * model metadata (producer, IR version, opset imports, custom props) — the
//!   opset in particular is load-bearing, because version-sensitive operators
//!   read it off `OpContext::opset`.
//!
//! What is **not** cached is anything that is a property of the *machine* rather
//! than the model: thread counts, provider lists, profiling, the memory pool.
//! Those come from the [`crate::SessionBuilder`] that loads the cache, which is
//! the point — the same cache file is meant to be usable by a process configured
//! differently from the one that wrote it.
//!
//! # Format
//!
//! A hand-rolled, length-prefixed, little-endian binary format, versioned by
//! [`SESSION_CACHE_FORMAT_VERSION`] behind the [`SESSION_CACHE_MAGIC`] tag:
//!
//! ```text
//! magic        : 8 bytes  b"OXIONNXS"
//! version      : u32
//! node_count   : u64          ─┐ header, readable without decoding the body
//! weight_count : u64          ─┘ (see `Session::peek_optimized_header`)
//! metadata     : ModelMetadata
//! input_names  : [String]
//! output_names : [String]
//! input_infos  : [TensorInfo]
//! output_infos : [TensorInfo]
//! nodes        : [Node]
//! weights      : [(String, Tensor)]     sorted by name
//! ```
//!
//! Every variable-length item is preceded by a `u64` count, and every map is
//! written in sorted key order, so **the same session always serialises to the
//! same bytes** (a hash of the file is a usable cache key).
//!
//! # Why hand-rolled rather than a derive
//!
//! The types that have to cross the boundary — `Node`, `Attributes`, `Graph`,
//! `TensorInfo`, `DType`, `OpKind` — live in `oxionnx-core`, where a derive
//! would mean a serialization dependency in the crate that is deliberately the
//! most dependency-free one in the workspace (it is `no_std`-capable). Mirror
//! structs plus conversions would cost more code than the writer does, and
//! would still leave the *reader* — the part that must survive a truncated or
//! hostile file without panicking — to be audited by hand. Writing both ends
//! here keeps that audit in one file: **every** read is bounds-checked against
//! the remaining input before a single byte is consumed or a single element
//! reserved, so a corrupt file is always [`OnnxError::Parse`] and never a panic
//! or a multi-gigabyte allocation.

use crate::execution_providers::OpPlacement;
use crate::graph::{Attributes, Graph, Node, OpKind};
use crate::tensor::Tensor;
use crate::OnnxError;
use oxionnx_core::{DType, OperatorRegistry, TensorInfo};
use std::collections::HashMap;
use std::path::Path;

use super::types::{ModelMetadata, OptLevel};
use super::Session;

/// File tag identifying an oxionnx session cache.
pub const SESSION_CACHE_MAGIC: &[u8; 8] = b"OXIONNXS";

/// Version of the session-cache binary layout written by this build.
///
/// A file whose version does not match is rejected outright rather than
/// best-effort decoded: the cache is a pure performance artefact that can always
/// be regenerated from the `.onnx`, so a mismatch is a cache miss, not an error
/// worth compatibility shims.
pub const SESSION_CACHE_FORMAT_VERSION: u32 = 1;

/// Fixed-size prefix of a session cache, readable without decoding the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionCacheHeader {
    /// Layout version the file was written with.
    pub format_version: u32,
    /// Number of nodes in the cached (optimized) graph.
    pub node_count: u64,
    /// Number of entries in the cached weight table.
    pub weight_count: u64,
}

/// Length of the fixed header: magic + version + two counts.
const HEADER_LEN: usize = 8 + 4 + 8 + 8;

/// Hard cap on nested subgraph depth, applied when writing *and* reading.
///
/// A cache file is untrusted input like any other file on disk. Without this,
/// a file claiming a million nested `Loop` bodies would recurse the decoder into
/// a stack overflow — which is a crash, not an `Err`.
const MAX_GRAPH_DEPTH: u32 = 32;

// ────────────────────────────────────────────────────────────────────────────
// Writing
// ────────────────────────────────────────────────────────────────────────────

/// Append `bytes.len()` as a `u64` followed by the bytes themselves.
fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_i64(out: &mut Vec<u8>, v: i64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    put_bytes(out, s.as_bytes());
}

fn put_count(out: &mut Vec<u8>, n: usize) {
    put_u64(out, n as u64);
}

fn put_str_list(out: &mut Vec<u8>, items: &[String]) {
    put_count(out, items.len());
    for s in items {
        put_str(out, s);
    }
}

fn put_f32_slice(out: &mut Vec<u8>, data: &[f32]) {
    put_count(out, data.len());
    #[cfg(target_endian = "little")]
    out.extend_from_slice(bytemuck::cast_slice::<f32, u8>(data));
    #[cfg(not(target_endian = "little"))]
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn put_usize_slice(out: &mut Vec<u8>, dims: &[usize]) {
    put_count(out, dims.len());
    for &d in dims {
        put_u64(out, d as u64);
    }
}

fn put_tensor(out: &mut Vec<u8>, tensor: &Tensor) {
    put_usize_slice(out, &tensor.shape);
    put_f32_slice(out, &tensor.data);
}

fn put_optional_usize_slice(out: &mut Vec<u8>, dims: &[Option<usize>]) {
    put_count(out, dims.len());
    for dim in dims {
        match dim {
            Some(d) => {
                out.push(1);
                put_u64(out, *d as u64);
            }
            None => out.push(0),
        }
    }
}

fn put_optional_str_list(out: &mut Vec<u8>, items: &[Option<String>]) {
    put_count(out, items.len());
    for item in items {
        match item {
            Some(s) => {
                out.push(1);
                put_str(out, s);
            }
            None => out.push(0),
        }
    }
}

fn put_tensor_info(out: &mut Vec<u8>, info: &TensorInfo) {
    put_str(out, &info.name);
    put_i32(out, info.dtype.to_onnx());
    put_optional_usize_slice(out, &info.shape);
    put_optional_str_list(out, &info.dim_params);
}

fn put_tensor_info_list(out: &mut Vec<u8>, infos: &[TensorInfo]) {
    put_count(out, infos.len());
    for info in infos {
        put_tensor_info(out, info);
    }
}

/// Map entries in sorted key order, so the encoding is deterministic.
fn sorted<V>(map: &HashMap<String, V>) -> Vec<(&str, &V)> {
    let mut entries: Vec<(&str, &V)> = map.iter().map(|(k, v)| (k.as_str(), v)).collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries
}

fn put_attributes(out: &mut Vec<u8>, attrs: &Attributes, depth: u32) -> Result<(), OnnxError> {
    let floats = sorted(&attrs.floats);
    put_count(out, floats.len());
    for (name, value) in floats {
        put_str(out, name);
        out.extend_from_slice(&value.to_le_bytes());
    }

    let ints = sorted(&attrs.ints);
    put_count(out, ints.len());
    for (name, value) in ints {
        put_str(out, name);
        put_i64(out, *value);
    }

    let strings = sorted(&attrs.strings);
    put_count(out, strings.len());
    for (name, value) in strings {
        put_str(out, name);
        put_str(out, value);
    }

    let tensors = sorted(&attrs.tensors);
    put_count(out, tensors.len());
    for (name, value) in tensors {
        put_str(out, name);
        put_tensor(out, value);
    }

    let float_lists = sorted(&attrs.float_lists);
    put_count(out, float_lists.len());
    for (name, value) in float_lists {
        put_str(out, name);
        put_f32_slice(out, value);
    }

    let int_lists = sorted(&attrs.int_lists);
    put_count(out, int_lists.len());
    for (name, value) in int_lists {
        put_str(out, name);
        put_count(out, value.len());
        for v in value {
            put_i64(out, *v);
        }
    }

    let string_lists = sorted(&attrs.string_lists);
    put_count(out, string_lists.len());
    for (name, value) in string_lists {
        put_str(out, name);
        put_str_list(out, value);
    }

    let graphs = sorted(&attrs.graphs);
    put_count(out, graphs.len());
    for (name, value) in graphs {
        put_str(out, name);
        put_graph(out, value, depth + 1)?;
    }

    Ok(())
}

fn put_node(out: &mut Vec<u8>, node: &Node, depth: u32) -> Result<(), OnnxError> {
    // `OpKind::Unknown(name).as_str()` returns the original name, and
    // `OpKind::parse` maps any unrecognised name back to `Unknown(name)`, so the
    // op string round-trips exactly — including custom operators registered
    // outside the `OpKind` enum.
    put_str(out, node.op.as_str());
    put_str(out, &node.name);
    put_str_list(out, &node.inputs);
    put_str_list(out, &node.outputs);
    put_attributes(out, &node.attrs, depth)
}

fn put_graph(out: &mut Vec<u8>, graph: &Graph, depth: u32) -> Result<(), OnnxError> {
    if depth > MAX_GRAPH_DEPTH {
        return Err(OnnxError::InvalidModel(format!(
            "session cache: subgraph nesting deeper than {MAX_GRAPH_DEPTH} cannot be cached"
        )));
    }
    put_str(out, &graph.name);
    put_count(out, graph.nodes.len());
    for node in &graph.nodes {
        put_node(out, node, depth)?;
    }
    put_str_list(out, &graph.input_names);
    put_str_list(out, &graph.output_names);
    put_tensor_info_list(out, &graph.input_infos);
    put_tensor_info_list(out, &graph.output_infos);
    Ok(())
}

fn put_metadata(out: &mut Vec<u8>, meta: &ModelMetadata) {
    put_str(out, &meta.producer_name);
    put_str(out, &meta.producer_version);
    put_str(out, &meta.domain);
    put_str(out, &meta.graph_name);
    put_i64(out, meta.ir_version);
    put_count(out, meta.opset_imports.len());
    for (domain, version) in &meta.opset_imports {
        put_str(out, domain);
        put_i64(out, *version);
    }
    let custom = sorted(&meta.custom_metadata);
    put_count(out, custom.len());
    for (key, value) in custom {
        put_str(out, key);
        put_str(out, value);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Reading
// ────────────────────────────────────────────────────────────────────────────

/// A bounds-checked cursor over the cache bytes.
///
/// The only way to consume input. Every method validates against the remaining
/// length *before* indexing, and every count is validated against the remaining
/// length *before* it is used to reserve capacity — so a file claiming
/// `u64::MAX` elements produces a `Parse` error rather than an allocation
/// failure.
struct Reader<'b> {
    bytes: &'b [u8],
    pos: usize,
}

impl<'b> Reader<'b> {
    fn new(bytes: &'b [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn truncated(&self, what: &str, need: usize) -> OnnxError {
        OnnxError::Parse(format!(
            "session cache truncated at byte {}: needed {need} more bytes for {what}, {} remain",
            self.pos,
            self.remaining()
        ))
    }

    fn take(&mut self, n: usize, what: &str) -> Result<&'b [u8], OnnxError> {
        if self.remaining() < n {
            return Err(self.truncated(what, n));
        }
        let start = self.pos;
        self.pos += n;
        Ok(&self.bytes[start..self.pos])
    }

    fn u8(&mut self, what: &str) -> Result<u8, OnnxError> {
        Ok(self.take(1, what)?[0])
    }

    fn u32(&mut self, what: &str) -> Result<u32, OnnxError> {
        let b = self.take(4, what)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&mut self, what: &str) -> Result<i32, OnnxError> {
        Ok(self.u32(what)? as i32)
    }

    fn u64(&mut self, what: &str) -> Result<u64, OnnxError> {
        let b = self.take(8, what)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn i64(&mut self, what: &str) -> Result<i64, OnnxError> {
        Ok(self.u64(what)? as i64)
    }

    fn f32(&mut self, what: &str) -> Result<f32, OnnxError> {
        let b = self.take(4, what)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// A `u64` element count, rejected unless the input could actually hold
    /// that many elements at `min_bytes_each` bytes apiece.
    ///
    /// This is the allocation guard: it runs *before* any `with_capacity`, so no
    /// count read out of the file can ask for memory the file could not possibly
    /// describe.
    fn count(&mut self, what: &str, min_bytes_each: usize) -> Result<usize, OnnxError> {
        let raw = self.u64(what)?;
        let n = usize::try_from(raw).map_err(|_| {
            OnnxError::Parse(format!(
                "session cache: {what} count {raw} exceeds this platform's address space"
            ))
        })?;
        let affordable = self.remaining() / min_bytes_each.max(1);
        if n > affordable {
            return Err(OnnxError::Parse(format!(
                "session cache: {what} claims {n} elements but only {} bytes remain \
                 (at least {min_bytes_each} bytes each are required)",
                self.remaining()
            )));
        }
        Ok(n)
    }

    fn string(&mut self, what: &str) -> Result<String, OnnxError> {
        let len = self.count(what, 1)?;
        let bytes = self.take(len, what)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| OnnxError::Parse(format!("session cache: {what} is not valid UTF-8: {e}")))
    }

    fn string_list(&mut self, what: &str) -> Result<Vec<String>, OnnxError> {
        let n = self.count(what, 8)?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.string(what)?);
        }
        Ok(out)
    }

    fn f32_vec(&mut self, what: &str) -> Result<Vec<f32>, OnnxError> {
        let n = self.count(what, 4)?;
        let bytes = self.take(n * 4, what)?;
        let mut out = Vec::with_capacity(n);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    }

    fn i64_vec(&mut self, what: &str) -> Result<Vec<i64>, OnnxError> {
        let n = self.count(what, 8)?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.i64(what)?);
        }
        Ok(out)
    }

    fn usize_vec(&mut self, what: &str) -> Result<Vec<usize>, OnnxError> {
        let n = self.count(what, 8)?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let raw = self.u64(what)?;
            out.push(usize::try_from(raw).map_err(|_| {
                OnnxError::Parse(format!(
                    "session cache: {what} dimension {raw} is out of range"
                ))
            })?);
        }
        Ok(out)
    }

    fn optional_usize_vec(&mut self, what: &str) -> Result<Vec<Option<usize>>, OnnxError> {
        let n = self.count(what, 1)?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            match self.u8(what)? {
                0 => out.push(None),
                1 => {
                    let raw = self.u64(what)?;
                    out.push(Some(usize::try_from(raw).map_err(|_| {
                        OnnxError::Parse(format!(
                            "session cache: {what} dimension {raw} is out of range"
                        ))
                    })?));
                }
                tag => {
                    return Err(OnnxError::Parse(format!(
                        "session cache: {what} has invalid optional tag {tag} (expected 0 or 1)"
                    )))
                }
            }
        }
        Ok(out)
    }

    fn optional_string_vec(&mut self, what: &str) -> Result<Vec<Option<String>>, OnnxError> {
        let n = self.count(what, 1)?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            match self.u8(what)? {
                0 => out.push(None),
                1 => out.push(Some(self.string(what)?)),
                tag => {
                    return Err(OnnxError::Parse(format!(
                        "session cache: {what} has invalid optional tag {tag} (expected 0 or 1)"
                    )))
                }
            }
        }
        Ok(out)
    }

    fn tensor(&mut self, what: &str) -> Result<Tensor, OnnxError> {
        let shape = self.usize_vec(what)?;
        let data = self.f32_vec(what)?;
        // `try_new` (not `new`) on purpose: the shape and the element count come
        // from the file independently, so a corrupt or hand-edited cache can
        // disagree about them. `Tensor::new` only debug-asserts, which in a
        // release build would build an inconsistent tensor that explodes much
        // later inside an operator.
        Tensor::try_new(data, shape)
    }

    fn tensor_info(&mut self, what: &str) -> Result<TensorInfo, OnnxError> {
        let name = self.string(what)?;
        let raw_dtype = self.i32(what)?;
        let dtype = DType::from_onnx(raw_dtype).ok_or_else(|| {
            OnnxError::Parse(format!(
                "session cache: {what} '{name}' has unknown ONNX dtype code {raw_dtype}"
            ))
        })?;
        let shape = self.optional_usize_vec(what)?;
        let dim_params = self.optional_string_vec(what)?;
        Ok(TensorInfo {
            name,
            dtype,
            shape,
            dim_params,
        })
    }

    fn tensor_info_list(&mut self, what: &str) -> Result<Vec<TensorInfo>, OnnxError> {
        let n = self.count(what, 8)?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.tensor_info(what)?);
        }
        Ok(out)
    }

    fn attributes(&mut self, depth: u32) -> Result<Attributes, OnnxError> {
        let mut attrs = Attributes::default();

        let n = self.count("attr floats", 8)?;
        for _ in 0..n {
            let key = self.string("attr float name")?;
            attrs.floats.insert(key, self.f32("attr float")?);
        }

        let n = self.count("attr ints", 8)?;
        for _ in 0..n {
            let key = self.string("attr int name")?;
            attrs.ints.insert(key, self.i64("attr int")?);
        }

        let n = self.count("attr strings", 8)?;
        for _ in 0..n {
            let key = self.string("attr string name")?;
            attrs.strings.insert(key, self.string("attr string")?);
        }

        let n = self.count("attr tensors", 8)?;
        for _ in 0..n {
            let key = self.string("attr tensor name")?;
            attrs.tensors.insert(key, self.tensor("attr tensor")?);
        }

        let n = self.count("attr float lists", 8)?;
        for _ in 0..n {
            let key = self.string("attr float list name")?;
            attrs.float_lists.insert(key, self.f32_vec("attr floats")?);
        }

        let n = self.count("attr int lists", 8)?;
        for _ in 0..n {
            let key = self.string("attr int list name")?;
            attrs.int_lists.insert(key, self.i64_vec("attr ints")?);
        }

        let n = self.count("attr string lists", 8)?;
        for _ in 0..n {
            let key = self.string("attr string list name")?;
            attrs
                .string_lists
                .insert(key, self.string_list("attr strings")?);
        }

        let n = self.count("attr graphs", 8)?;
        for _ in 0..n {
            let key = self.string("attr graph name")?;
            let graph = self.graph(depth + 1)?;
            attrs.graphs.insert(key, graph);
        }

        Ok(attrs)
    }

    fn node(&mut self, depth: u32) -> Result<Node, OnnxError> {
        let op = OpKind::parse(&self.string("node op type")?);
        let name = self.string("node name")?;
        let inputs = self.string_list("node inputs")?;
        let outputs = self.string_list("node outputs")?;
        let attrs = self.attributes(depth)?;
        Ok(Node {
            op,
            name,
            inputs,
            outputs,
            attrs,
        })
    }

    fn graph(&mut self, depth: u32) -> Result<Graph, OnnxError> {
        if depth > MAX_GRAPH_DEPTH {
            return Err(OnnxError::Parse(format!(
                "session cache: subgraph nesting exceeds the {MAX_GRAPH_DEPTH}-level limit"
            )));
        }
        let name = self.string("subgraph name")?;
        let node_count = self.count("subgraph nodes", 8)?;
        let mut nodes = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            nodes.push(self.node(depth)?);
        }
        Ok(Graph {
            name,
            nodes,
            input_names: self.string_list("subgraph inputs")?,
            output_names: self.string_list("subgraph outputs")?,
            input_infos: self.tensor_info_list("subgraph input info")?,
            output_infos: self.tensor_info_list("subgraph output info")?,
        })
    }

    fn metadata(&mut self) -> Result<ModelMetadata, OnnxError> {
        let producer_name = self.string("producer name")?;
        let producer_version = self.string("producer version")?;
        let domain = self.string("domain")?;
        let graph_name = self.string("graph name")?;
        let ir_version = self.i64("ir version")?;

        let n = self.count("opset imports", 16)?;
        let mut opset_imports = Vec::with_capacity(n);
        for _ in 0..n {
            let dom = self.string("opset domain")?;
            opset_imports.push((dom, self.i64("opset version")?));
        }

        let n = self.count("custom metadata", 16)?;
        let mut custom_metadata = HashMap::with_capacity(n);
        for _ in 0..n {
            let key = self.string("metadata key")?;
            custom_metadata.insert(key, self.string("metadata value")?);
        }

        Ok(ModelMetadata {
            producer_name,
            producer_version,
            domain,
            graph_name,
            ir_version,
            opset_imports,
            custom_metadata,
        })
    }
}

/// Decode the fixed header, validating magic and version.
fn read_header(bytes: &[u8]) -> Result<(SessionCacheHeader, Reader<'_>), OnnxError> {
    if bytes.len() < HEADER_LEN {
        return Err(OnnxError::Parse(format!(
            "session cache is {} bytes, shorter than the {HEADER_LEN}-byte header",
            bytes.len()
        )));
    }
    let mut reader = Reader::new(bytes);
    let magic = reader.take(8, "magic")?;
    if magic != SESSION_CACHE_MAGIC.as_slice() {
        return Err(OnnxError::Parse(
            "not an oxionnx session cache: magic bytes do not match".to_string(),
        ));
    }
    let format_version = reader.u32("format version")?;
    if format_version != SESSION_CACHE_FORMAT_VERSION {
        return Err(OnnxError::Parse(format!(
            "session cache format version {format_version} is not supported by this build \
             (expected {SESSION_CACHE_FORMAT_VERSION}); delete the cache and re-save it"
        )));
    }
    let node_count = reader.u64("node count")?;
    let weight_count = reader.u64("weight count")?;
    Ok((
        SessionCacheHeader {
            format_version,
            node_count,
            weight_count,
        },
        reader,
    ))
}

/// The decoded body of a cache file, before it becomes a `Session`.
pub(crate) struct CachedGraph {
    pub(crate) graph: Graph,
    pub(crate) weights: HashMap<String, Tensor>,
    pub(crate) metadata: ModelMetadata,
}

/// Tripwire: every node the cache described must survive into the rebuilt
/// session.
///
/// Loading re-runs the topological sort — the cheap part; the *optimization*
/// passes are what the cache exists to skip — and `Graph::topological_sort`
/// today keeps nodes it cannot schedule, appending them in their original order
/// rather than discarding them.  So this **cannot fire** as the code stands, and
/// that is the point: it is one integer comparison guarding the assumption.  If
/// the sort ever starts dropping unschedulable nodes, a corrupt or hand-edited
/// cache would otherwise load into a session that silently returns fewer
/// outputs instead of failing, which is the worst possible failure mode for a
/// cache.  See `a_cache_whose_graph_is_unschedulable_loads_whole_and_fails_at_run`.
pub(crate) fn check_no_nodes_were_dropped(
    session: &Session,
    expected: usize,
) -> Result<(), OnnxError> {
    if session.sorted_nodes.len() == expected {
        return Ok(());
    }
    Err(OnnxError::Parse(format!(
        "session cache describes {expected} nodes but only {} could be scheduled; \
         the cached graph references a value nothing produces",
        session.sorted_nodes.len()
    )))
}

/// Decode a whole cache file into the parts a session is assembled from.
pub(crate) fn decode(bytes: &[u8]) -> Result<CachedGraph, OnnxError> {
    let (header, mut reader) = read_header(bytes)?;

    let metadata = reader.metadata()?;
    let input_names = reader.string_list("graph inputs")?;
    let output_names = reader.string_list("graph outputs")?;
    let input_infos = reader.tensor_info_list("graph input info")?;
    let output_infos = reader.tensor_info_list("graph output info")?;

    let node_count = reader.count("nodes", 8)?;
    if node_count as u64 != header.node_count {
        return Err(OnnxError::Parse(format!(
            "session cache header claims {} nodes but the body holds {node_count}",
            header.node_count
        )));
    }
    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        nodes.push(reader.node(0)?);
    }

    let weight_count = reader.count("weights", 8)?;
    if weight_count as u64 != header.weight_count {
        return Err(OnnxError::Parse(format!(
            "session cache header claims {} weights but the body holds {weight_count}",
            header.weight_count
        )));
    }
    let mut weights = HashMap::with_capacity(weight_count);
    for _ in 0..weight_count {
        let name = reader.string("weight name")?;
        weights.insert(name, reader.tensor("weight")?);
    }

    if reader.remaining() != 0 {
        return Err(OnnxError::Parse(format!(
            "session cache has {} trailing bytes after the weight table",
            reader.remaining()
        )));
    }

    Ok(CachedGraph {
        graph: Graph {
            name: metadata.graph_name.clone(),
            nodes,
            input_names,
            output_names,
            input_infos,
            output_infos,
        },
        weights,
        metadata,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────────────

impl Session {
    /// Serialize this session's **optimized** graph and weights to bytes.
    ///
    /// The result is deterministic: two sessions with the same optimized graph
    /// produce byte-identical output, so the bytes can be hashed for use as a
    /// cache key.
    ///
    /// # Errors
    ///
    /// [`OnnxError::InvalidModel`] if the graph nests subgraphs deeper than the
    /// format's 32-level limit.
    pub fn to_optimized_bytes(&self) -> Result<Vec<u8>, OnnxError> {
        let mut out = Vec::new();
        out.extend_from_slice(SESSION_CACHE_MAGIC);
        put_u32(&mut out, SESSION_CACHE_FORMAT_VERSION);
        put_count(&mut out, self.sorted_nodes.len());
        put_count(&mut out, self.weights.len());

        put_metadata(&mut out, &self.metadata);
        put_str_list(&mut out, &self.input_names);
        put_str_list(&mut out, &self.output_names);
        put_tensor_info_list(&mut out, &self.input_infos);
        put_tensor_info_list(&mut out, &self.output_infos);

        put_count(&mut out, self.sorted_nodes.len());
        for node in &self.sorted_nodes {
            put_node(&mut out, node, 0)?;
        }

        let weights = sorted(&self.weights);
        put_count(&mut out, weights.len());
        for (name, tensor) in weights {
            put_str(&mut out, name);
            put_tensor(&mut out, tensor);
        }
        Ok(out)
    }

    /// Write this session's optimized graph and weights to `path`.
    ///
    /// # Errors
    ///
    /// [`OnnxError::Internal`] if the file cannot be written; otherwise as
    /// [`Session::to_optimized_bytes`].
    pub fn save_optimized(&self, path: &Path) -> Result<(), OnnxError> {
        let bytes = self.to_optimized_bytes()?;
        std::fs::write(path, bytes).map_err(|e| {
            OnnxError::Internal(format!(
                "cannot write session cache {}: {e}",
                path.display()
            ))
        })
    }

    /// Rebuild a session from bytes written by [`Session::to_optimized_bytes`],
    /// **without running any optimization pass**.
    ///
    /// The cached node list is already optimized and already topologically
    /// sorted, so it is loaded at [`OptLevel::None`]: no constant folding, no
    /// fusion, no CSE, no dead-node elimination. That is the whole point of the
    /// cache, and it is what the `session_cache` tests assert by counting
    /// operator executions during load.
    ///
    /// # Errors
    ///
    /// [`OnnxError::Parse`] for any malformed, truncated, wrong-version or
    /// foreign file — a cache file is untrusted input and never panics the
    /// decoder.
    pub fn from_optimized_bytes(bytes: &[u8]) -> Result<Self, OnnxError> {
        Self::from_optimized_bytes_with_registry(bytes, oxionnx_ops::default_registry())
    }

    /// [`Session::from_optimized_bytes`] with a custom operator registry.
    pub fn from_optimized_bytes_with_registry(
        bytes: &[u8],
        registry: OperatorRegistry,
    ) -> Result<Self, OnnxError> {
        let cached = decode(bytes)?;
        let expected_nodes = cached.graph.nodes.len();
        let session = Self::build_from_graph(
            cached.graph,
            cached.weights,
            cached.metadata,
            registry,
            OptLevel::None,
            // Runtime knobs deliberately match `Session::from_file`: profiling
            // off, memory pool off, sequential, f32.  A caller who wants
            // different ones loads the cache through
            // [`crate::SessionBuilder::load_optimized`], which applies its own.
            false,
            false,
            false,
            false,
            None,
            OpPlacement::default(),
            Vec::new(),
        )?;
        check_no_nodes_were_dropped(&session, expected_nodes)?;
        Ok(session)
    }

    /// Load a session cache written by [`Session::save_optimized`].
    pub fn load_optimized(path: &Path) -> Result<Self, OnnxError> {
        Self::load_optimized_with_registry(path, oxionnx_ops::default_registry())
    }

    /// [`Session::load_optimized`] with a custom operator registry.
    pub fn load_optimized_with_registry(
        path: &Path,
        registry: OperatorRegistry,
    ) -> Result<Self, OnnxError> {
        let bytes = std::fs::read(path).map_err(|e| {
            OnnxError::Parse(format!("cannot read session cache {}: {e}", path.display()))
        })?;
        Self::from_optimized_bytes_with_registry(&bytes, registry)
    }

    /// Read only the fixed header of a session cache.
    ///
    /// Cheap enough to call on every candidate file in a cache directory: it
    /// touches the first 28 bytes and decodes nothing else.
    ///
    /// # Errors
    ///
    /// [`OnnxError::Parse`] if the bytes are too short, do not carry the
    /// [`SESSION_CACHE_MAGIC`] tag, or were written by an incompatible format
    /// version.
    pub fn peek_optimized_header(bytes: &[u8]) -> Result<SessionCacheHeader, OnnxError> {
        read_header(bytes).map(|(header, _)| header)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every prefix of a valid cache must be rejected with a typed error — no
    /// panic, no hang, no silent success.
    #[test]
    fn every_truncation_of_a_valid_cache_is_a_typed_error() {
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), Tensor::new(vec![1.0, 2.0], vec![2]));
        let graph = Graph {
            name: "g".to_string(),
            nodes: vec![Node {
                op: OpKind::Add,
                name: "add".to_string(),
                inputs: vec!["x".to_string(), "w".to_string()],
                outputs: vec!["y".to_string()],
                attrs: Attributes::default(),
            }],
            input_names: vec!["x".to_string()],
            output_names: vec!["y".to_string()],
            input_infos: Vec::new(),
            output_infos: Vec::new(),
        };
        let session = Session::from_graph(graph, weights).expect("session builds");
        let bytes = session.to_optimized_bytes().expect("serialises");
        assert!(bytes.len() > HEADER_LEN);

        for cut in 0..bytes.len() {
            let err = Session::from_optimized_bytes(&bytes[..cut]);
            assert!(
                err.is_err(),
                "a {cut}-byte prefix of a {}-byte cache must not decode",
                bytes.len()
            );
        }
    }

    #[test]
    fn a_hostile_element_count_is_rejected_before_it_allocates() {
        // A well-formed header followed by a weight-name length of u64::MAX.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SESSION_CACHE_MAGIC);
        put_u32(&mut bytes, SESSION_CACHE_FORMAT_VERSION);
        put_u64(&mut bytes, 0);
        put_u64(&mut bytes, 0);
        // metadata: four empty strings, ir_version, two empty maps
        for _ in 0..4 {
            put_str(&mut bytes, "");
        }
        put_i64(&mut bytes, 7);
        put_u64(&mut bytes, 0);
        put_u64(&mut bytes, 0);
        // input_names claims u64::MAX entries
        put_u64(&mut bytes, u64::MAX);

        match Session::from_optimized_bytes(&bytes).map(|_| ()) {
            Err(OnnxError::Parse(msg)) => {
                assert!(msg.contains("elements"), "unexpected message: {msg}");
            }
            other => panic!("expected a Parse error, got {other:?}"),
        }
    }

    #[test]
    fn a_foreign_file_is_rejected_by_its_magic() {
        let bytes = vec![0u8; 128];
        match Session::from_optimized_bytes(&bytes).map(|_| ()) {
            Err(OnnxError::Parse(msg)) => assert!(msg.contains("magic"), "got: {msg}"),
            other => panic!("expected a Parse error, got {other:?}"),
        }
    }

    #[test]
    fn a_future_format_version_is_rejected_rather_than_guessed_at() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SESSION_CACHE_MAGIC);
        put_u32(&mut bytes, SESSION_CACHE_FORMAT_VERSION + 1);
        put_u64(&mut bytes, 0);
        put_u64(&mut bytes, 0);
        match Session::peek_optimized_header(&bytes) {
            Err(OnnxError::Parse(msg)) => assert!(msg.contains("format version"), "got: {msg}"),
            other => panic!("expected a Parse error, got {other:?}"),
        }
    }

    #[test]
    fn a_tensor_whose_shape_contradicts_its_data_is_rejected() {
        let mut bytes = Vec::new();
        // shape [4] but only two elements follow
        put_usize_slice(&mut bytes, &[4]);
        put_f32_slice(&mut bytes, &[1.0, 2.0]);
        let mut reader = Reader::new(&bytes);
        match reader.tensor("weight") {
            Err(OnnxError::ShapeMismatch(_)) => {}
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }
}
