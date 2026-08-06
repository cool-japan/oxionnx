use crate::execution_providers::{OpPlacement, ProviderKind};
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::OnnxError;
use std::collections::HashMap;
use std::path::Path;

use super::cancellation::CancellationToken;
use super::types::{raw_meta_to_model_metadata, ModelMetadata, OptLevel};
use super::Session;
use oxionnx_core::OperatorRegistry;

/// Builder for configuring and creating a Session.
pub struct SessionBuilder {
    pub(crate) opt_level: OptLevel,
    pub(crate) registry: Option<OperatorRegistry>,
    pub(crate) enable_profiling: bool,
    pub(crate) enable_memory_pool: bool,
    pub(crate) parallel: bool,
    pub(crate) mixed_precision: bool,
    pub(crate) num_threads: Option<usize>,
    pub(crate) op_placement: OpPlacement,
    /// Ordered list of execution provider backends to attempt, in priority order.
    ///
    /// When non-empty, the dispatch loop tries each provider in turn and uses the
    /// first that returns `Some(result)`. CPU is always the implicit terminal
    /// fallback — it is tried even if absent from this list.
    ///
    /// When empty (the default), the session falls back to the legacy
    /// heuristic / compile-time feature-flag dispatch, preserving backward
    /// compatibility with callers that never call `with_execution_providers`.
    pub(crate) providers: Vec<ProviderKind>,
    /// Session-scoped cooperative cancellation token, when the caller supplied
    /// one.  Applied after the graph is optimized, because the guard registry is
    /// built from the *final* node list.  See [`crate::session::cancellation`].
    pub(crate) cancellation: Option<CancellationToken>,
}

impl SessionBuilder {
    /// Apply the settings that can only be applied to a finished session.
    ///
    /// Today that is exactly one: cancellation, whose operator guards are built
    /// from the optimized node list and therefore cannot be installed until
    /// [`Session::build_from_graph`] has produced it.
    fn finish(
        session: Result<Session, OnnxError>,
        cancellation: Option<CancellationToken>,
    ) -> Result<Session, OnnxError> {
        let mut session = session?;
        if let Some(token) = cancellation {
            session.set_session_cancellation(token);
        }
        Ok(session)
    }
    /// Create a new builder with default settings (all optimizations, no profiling, no pool,
    /// sequential execution).
    pub fn new() -> Self {
        Self {
            opt_level: OptLevel::All,
            registry: None,
            enable_profiling: false,
            enable_memory_pool: true,
            parallel: false,
            mixed_precision: false,
            num_threads: None,
            op_placement: OpPlacement::default(),
            providers: Vec::new(),
            cancellation: None,
        }
    }

    /// Bind a [`CancellationToken`] to the session this builder produces.
    ///
    /// Every operator the model uses will consult the token before it executes,
    /// so `run()` stops at the first node boundary after
    /// [`CancellationToken::cancel`] and returns [`OnnxError::Cancelled`].
    ///
    /// # The token is session-scoped, not run-scoped
    ///
    /// The name says so on purpose. One token covers the whole session:
    /// cancelling it aborts **every** run in flight on that session, not one
    /// request. Give each concurrently-cancellable workload its own session, or
    /// cancel at the granularity of [`crate::streaming::TokenStream`], which
    /// checks between decode steps and is per-generation by construction.
    ///
    /// ```
    /// use oxionnx::{CancellationToken, Session, Tensor};
    /// use std::collections::HashMap;
    /// # use oxionnx::{Attributes, Graph, Node, OpKind};
    /// # let graph = Graph {
    /// #     name: "g".into(),
    /// #     nodes: vec![Node { op: OpKind::Relu, name: "r".into(),
    /// #         inputs: vec!["x".into()], outputs: vec!["y".into()],
    /// #         attrs: Attributes::default() }],
    /// #     input_names: vec!["x".into()], output_names: vec!["y".into()],
    /// #     input_infos: Vec::new(), output_infos: Vec::new(),
    /// # };
    /// let token = CancellationToken::new();
    /// let session = Session::builder()
    ///     .with_session_cancellation(token.clone())
    ///     .build_from_graph(graph, HashMap::new())?;
    ///
    /// token.cancel();
    /// let mut inputs = HashMap::new();
    /// inputs.insert("x", Tensor::new(vec![-1.0, 1.0], vec![2]));
    /// assert!(matches!(session.run(&inputs), Err(oxionnx::OnnxError::Cancelled(_))));
    /// # Ok::<(), oxionnx::OnnxError>(())
    /// ```
    #[must_use]
    pub fn with_session_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = Some(token);
        self
    }

    /// Set the optimization level for graph optimization passes.
    pub fn with_optimization_level(mut self, level: OptLevel) -> Self {
        self.opt_level = level;
        self
    }

    /// Set a custom operator registry.
    pub fn with_registry(mut self, registry: OperatorRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Enable per-node profiling during `run()`.
    pub fn with_profiling(mut self) -> Self {
        self.enable_profiling = true;
        self
    }

    /// Enable the activation memory pool for buffer reuse during inference.
    pub fn with_memory_pool(mut self, enabled: bool) -> Self {
        self.enable_memory_pool = enabled;
        self
    }

    /// Enable or disable multi-threaded parallel execution of independent nodes.
    /// When enabled, nodes at the same topological depth are executed concurrently
    /// using rayon. On `wasm32` targets, this flag is ignored and execution is
    /// always sequential. Default: `false`.
    pub fn with_parallel_execution(mut self, enabled: bool) -> Self {
        self.parallel = enabled;
        self
    }

    /// Enable mixed-precision inference (f16 activations, f32 accumulation).
    pub fn with_mixed_precision(mut self, enabled: bool) -> Self {
        self.mixed_precision = enabled;
        self
    }

    /// Set operator placement strategy for routing ops to CPU/GPU.
    pub fn with_op_placement(mut self, placement: OpPlacement) -> Self {
        self.op_placement = placement;
        self
    }

    /// Load an ONNX model from a file path.
    /// Supports models with external data by resolving paths relative to the file's directory.
    pub fn load(self, path: &Path) -> Result<Session, OnnxError> {
        let bytes = std::fs::read(path).map_err(|e| {
            OnnxError::Parse(format!("Cannot read ONNX file {}: {e}", path.display()))
        })?;
        let base_path = path.parent().unwrap_or_else(|| Path::new("."));
        let registry = self.registry.unwrap_or_else(oxionnx_ops::default_registry);
        let (raw_meta, graph, weights) = super::loading::parse_stage("file", bytes.len(), || {
            crate::model::load_with_metadata_and_path(&bytes, base_path)
        })
        .map_err(OnnxError::Parse)?;
        let metadata = raw_meta_to_model_metadata(raw_meta);
        let built = Session::build_from_graph(
            graph,
            weights,
            metadata,
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
            self.mixed_precision,
            self.num_threads,
            self.op_placement,
            self.providers,
        );
        Self::finish(built, self.cancellation)
    }

    /// Load an ONNX model from a file using memory mapping.
    ///
    /// The file is memory-mapped instead of being read entirely into a `Vec<u8>`.
    /// This lets the OS virtual-memory subsystem page out weight data that is not
    /// actively used, reducing resident memory for large models.
    ///
    /// # Why this does not call `oxionnx_proto::mmap_loader::MmapModel::open`
    ///
    /// `MmapModel::open` parses via [`oxionnx_proto::model::load_with_path`], which
    /// discards the model's `RawModelMeta` (producer name, IR version, opset
    /// imports, custom metadata props) — the same information [`Self::load`] and
    /// [`Self::load_from_bytes`] preserve via `load_with_metadata_and_path` /
    /// `load_with_metadata`. Mapping the file here and calling
    /// [`crate::model::load_with_metadata_and_path`] directly does the *same*
    /// single parse pass over the *same* mapped bytes — `load_with_path` is just
    /// `load_with_metadata_and_path` with the metadata half of its return value
    /// dropped — while keeping it, so `session.metadata()` after `load_mmap`
    /// agrees with what `load`/`load_from_bytes` report for identical model
    /// bytes, instead of silently reporting [`ModelMetadata::default`].
    #[cfg(feature = "mmap")]
    pub fn load_mmap(self, path: &Path) -> Result<Session, OnnxError> {
        let (raw_meta, graph, weights) = super::loading::parse_stage("mmap", 0, || {
            let file = std::fs::File::open(path)
                .map_err(|e| format!("Cannot open '{}': {}", path.display(), e))?;
            // SAFETY: matches the contract `oxionnx_proto::mmap_loader::MmapModel::open`
            // documents for its own (otherwise-equivalent) mapping — the file is held
            // open for exactly the duration of this mapping, and the mapped bytes are
            // fully copied out into owned `Tensor`/`String` storage by the parser below
            // before this closure returns, so nothing borrows the mapping afterwards.
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| format!("mmap failed for '{}': {}", path.display(), e))?;
            let base_path = path.parent().unwrap_or_else(|| Path::new("."));
            crate::model::load_with_metadata_and_path(&mmap, base_path)
        })
        .map_err(OnnxError::Parse)?;
        let metadata = raw_meta_to_model_metadata(raw_meta);
        let registry = self.registry.unwrap_or_else(oxionnx_ops::default_registry);
        let built = Session::build_from_graph(
            graph,
            weights,
            metadata,
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
            self.mixed_precision,
            self.num_threads,
            self.op_placement,
            self.providers,
        );
        Self::finish(built, self.cancellation)
    }

    /// Load an ONNX model from raw bytes.
    pub fn load_from_bytes(self, bytes: &[u8]) -> Result<Session, OnnxError> {
        let registry = self.registry.unwrap_or_else(oxionnx_ops::default_registry);
        let (raw_meta, graph, weights) = super::loading::parse_stage("bytes", bytes.len(), || {
            crate::model::load_with_metadata(bytes)
        })
        .map_err(OnnxError::Parse)?;
        let metadata = raw_meta_to_model_metadata(raw_meta);
        let built = Session::build_from_graph(
            graph,
            weights,
            metadata,
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
            self.mixed_precision,
            self.num_threads,
            self.op_placement,
            self.providers,
        );
        Self::finish(built, self.cancellation)
    }

    /// Load an ONNX model from a `Read` source (streaming).
    ///
    /// Parses the model incrementally from the reader without loading the entire
    /// file into memory at once. Useful for multi-GB models.
    pub fn load_from_reader<R: std::io::Read>(self, reader: R) -> Result<Session, OnnxError> {
        let registry = self.registry.unwrap_or_else(oxionnx_ops::default_registry);
        let (graph, weights) = super::loading::parse_stage("reader", 0, || {
            let (graph_proto, weights) = oxionnx_proto::parse_streaming(reader)?;
            let graph = oxionnx_proto::build_graph(&graph_proto, &weights)?;
            Ok::<_, String>((graph, weights))
        })
        .map_err(OnnxError::Parse)?;
        let built = Session::build_from_graph(
            graph,
            weights,
            ModelMetadata::default(),
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
            self.mixed_precision,
            self.num_threads,
            self.op_placement,
            self.providers,
        );
        Self::finish(built, self.cancellation)
    }

    /// Load an ONNX model with selective weight loading.
    ///
    /// The `weight_filter` closure receives each weight's name and shape.
    /// If it returns `true`, the weight is loaded; if `false`, it is skipped.
    /// This is useful for loading only needed layers from a large model.
    pub fn load_filtered<F>(self, path: &Path, weight_filter: F) -> Result<Session, OnnxError>
    where
        F: FnMut(&str, &[usize]) -> bool,
    {
        let file = std::fs::File::open(path).map_err(|e| {
            OnnxError::Parse(format!("Cannot read ONNX file {}: {e}", path.display()))
        })?;
        let registry = self.registry.unwrap_or_else(oxionnx_ops::default_registry);
        let (graph, weights) = super::loading::parse_stage("filtered", 0, || {
            let (graph_proto, weights) =
                oxionnx_proto::parse_with_weight_filter(file, weight_filter)?;
            let graph = oxionnx_proto::build_graph(&graph_proto, &weights)?;
            Ok::<_, String>((graph, weights))
        })
        .map_err(OnnxError::Parse)?;
        let built = Session::build_from_graph(
            graph,
            weights,
            ModelMetadata::default(),
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
            self.mixed_precision,
            self.num_threads,
            self.op_placement,
            self.providers,
        );
        Self::finish(built, self.cancellation)
    }

    /// Build a Session from a pre-parsed Graph and weights.
    pub fn build_from_graph(
        self,
        graph: Graph,
        weights: HashMap<String, Tensor>,
    ) -> Result<Session, OnnxError> {
        let registry = self.registry.unwrap_or_else(oxionnx_ops::default_registry);
        let built = Session::build_from_graph(
            graph,
            weights,
            ModelMetadata::default(),
            registry,
            self.opt_level,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
            self.mixed_precision,
            self.num_threads,
            self.op_placement,
            self.providers,
        );
        Self::finish(built, self.cancellation)
    }

    // ── Session cache ────────────────────────────────────────────────────────

    /// Load a session cache written by [`Session::save_optimized`], applying
    /// this builder's *runtime* settings.
    ///
    /// The cached graph is already optimized, so
    /// [`SessionBuilder::with_optimization_level`] is deliberately ignored here
    /// — re-optimising a cache would defeat its purpose, and the level is
    /// pinned to [`OptLevel::None`]. Everything that describes the *machine*
    /// rather than the model (threads, providers, profiling, memory pool,
    /// placement, cancellation) is applied exactly as for a `.onnx` load, which
    /// is why one cache file is usable by differently-configured processes.
    ///
    /// # Errors
    ///
    /// [`OnnxError::Parse`] if the file cannot be read, is not a session cache,
    /// or was written by an incompatible format version.
    pub fn load_optimized(self, path: &Path) -> Result<Session, OnnxError> {
        let bytes = std::fs::read(path).map_err(|e| {
            OnnxError::Parse(format!("cannot read session cache {}: {e}", path.display()))
        })?;
        self.load_optimized_from_bytes(&bytes)
    }

    /// [`SessionBuilder::load_optimized`] from bytes already in memory.
    pub fn load_optimized_from_bytes(self, bytes: &[u8]) -> Result<Session, OnnxError> {
        let cached =
            super::loading::parse_stage("cache", bytes.len(), || super::serialize::decode(bytes))?;
        let expected_nodes = cached.graph.nodes.len();
        let registry = self.registry.unwrap_or_else(oxionnx_ops::default_registry);
        let built = Session::build_from_graph(
            cached.graph,
            cached.weights,
            cached.metadata,
            registry,
            // Pinned, not `self.opt_level`: the cache *is* the optimized graph.
            OptLevel::None,
            self.enable_profiling,
            self.enable_memory_pool,
            self.parallel,
            self.mixed_precision,
            self.num_threads,
            self.op_placement,
            self.providers,
        );
        let session = Self::finish(built, self.cancellation)?;
        super::serialize::check_no_nodes_were_dropped(&session, expected_nodes)?;
        Ok(session)
    }

    // ── ort-compatibility aliases ────────────────────────────────────────────

    /// `ort`-compatible alias for [`SessionBuilder::load`].
    ///
    /// Allows callers migrating from `ort` to use `commit_from_file` without
    /// changing call-sites.
    pub fn commit_from_file(self, path: impl AsRef<std::path::Path>) -> Result<Session, OnnxError> {
        self.load(path.as_ref())
    }

    /// `ort`-compatible alias for [`SessionBuilder::load_from_bytes`].
    pub fn commit_from_memory(self, bytes: &[u8]) -> Result<Session, OnnxError> {
        self.load_from_bytes(bytes)
    }

    /// Set the number of threads for intra-op parallelism.
    ///
    /// When set, a per-session rayon thread pool is created with this many threads.
    /// If not set (or on `wasm32`), the global rayon pool is used.
    /// Also enables parallel execution automatically.
    pub fn with_intra_threads(mut self, n: usize) -> Self {
        self.num_threads = Some(n);
        self.parallel = true;
        self
    }

    /// Set the number of threads for inter-op parallelism.
    ///
    /// Currently an alias for [`SessionBuilder::with_intra_threads`].
    pub fn with_inter_threads(mut self, n: usize) -> Self {
        self.num_threads = Some(n);
        self.parallel = true;
        self
    }

    /// Set the ordered list of execution provider backends to try, in priority order.
    ///
    /// Each `ProviderKind` in the iterator is attempted for every ONNX graph node
    /// during inference; the first provider that returns `Some(result)` wins.
    /// CPU is always the implicit terminal fallback — it is tried even if
    /// absent from this list, guaranteeing that no provider selection can
    /// silently break CPU-only inference.
    ///
    /// Passing an empty iterator restores the legacy heuristic / compile-time
    /// feature-flag dispatch (backward-compatible default).
    ///
    /// ## `ort` compatibility
    ///
    /// The `ort` 2.x API accepts [`crate::execution_providers::ExecutionProviderDispatch`]
    /// tokens.  To support callers migrating from `ort`, this method also accepts
    /// those tokens — they are silently discarded so that existing call sites
    /// compile without change.  Use [`SessionBuilder::with_provider_kinds`] to
    /// pass typed [`ProviderKind`] values that actually affect dispatch.
    pub fn with_execution_providers<I>(self, _providers: I) -> Self
    where
        I: IntoIterator<Item = crate::execution_providers::ExecutionProviderDispatch>,
    {
        // `ExecutionProviderDispatch` is an opaque ort-compat token; discarding
        // it preserves backward compatibility (callers migrating from ort).
        self
    }

    /// Set the ordered list of [`ProviderKind`] backends to attempt, in priority order.
    ///
    /// Unlike [`SessionBuilder::with_execution_providers`], which accepts the
    /// `ort`-compatible opaque token, this method accepts typed [`ProviderKind`]
    /// values that **actually route dispatch** at runtime.
    ///
    /// # CPU fallback guarantee
    ///
    /// CPU is always tried last even if not present in `providers`.
    /// An empty list is equivalent to CPU-only execution.
    ///
    /// # Feature gating
    ///
    /// Provider variants are only present when the corresponding Cargo feature
    /// is enabled:
    /// - `ProviderKind::Gpu` requires feature `gpu`
    /// - `ProviderKind::Cuda` requires feature `cuda`
    /// - `ProviderKind::DirectMl` requires feature `directml`
    ///
    /// Passing a provider whose feature is not enabled is a compile error.
    pub fn with_provider_kinds(
        mut self,
        providers: impl IntoIterator<Item = ProviderKind>,
    ) -> Self {
        self.providers = providers.into_iter().collect();
        self
    }

    /// Return the currently configured provider kind list.
    ///
    /// Useful for introspection in tests and diagnostic tooling.
    /// Returns an empty slice when no explicit list has been set (legacy
    /// heuristic dispatch will be used in that case).
    #[must_use]
    pub fn provider_kinds(&self) -> &[ProviderKind] {
        &self.providers
    }
}

impl Default for SessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, feature = "mmap"))]
mod tests {
    use super::*;

    // ── minimal ONNX protobuf encoder ───────────────────────────────────────
    //
    // Mirrors the wire-format helpers `oxionnx-proto/src/mmap_loader.rs`'s own
    // `#[cfg(test)]` module and `tests/w3_mmap_session_load_e2e.rs` already use.

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

    /// `TensorProto` with a flat `[floats.len()]` shape.
    fn tensor_proto(name: &str, floats: &[f32]) -> Vec<u8> {
        let mut t = Vec::new();
        let dims_packed = encode_varint(floats.len() as u64);
        t.extend(encode_bytes_field(1, &dims_packed)); // dims (packed repeated int64)
        t.extend(encode_varint_field(2, 1)); // data_type = 1 (FLOAT)
        t.extend(encode_bytes_field(8, name.as_bytes())); // name
        let raw: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        t.extend(encode_bytes_field(9, &raw)); // raw_data
        t
    }

    /// A minimal-but-real ONNX model (one f32 initializer, no nodes/inputs/
    /// outputs — this test only exercises metadata, not `Session::run`)
    /// carrying every field `raw_meta_to_model_metadata` maps onto
    /// `ModelMetadata`: producer name/version, domain, graph name, ir_version,
    /// one opset import, and one custom `metadata_props` entry.
    fn build_model_with_metadata(w: &[f32]) -> Vec<u8> {
        let mut graph = Vec::new();
        graph.extend(encode_bytes_field(2, b"mmap_meta_test_graph")); // GraphProto.name
        graph.extend(encode_bytes_field(5, &tensor_proto("w", w))); // initializer

        let opset = encode_varint_field(2, 13); // OperatorSetIdProto.version = 13
        let mut metadata_entry = encode_bytes_field(1, b"custom_key"); // StringStringEntryProto.key
        metadata_entry.extend(encode_bytes_field(2, b"custom_value")); // .value

        let mut model = encode_varint_field(1, 9); // ModelProto.ir_version = 9
        model.extend(encode_bytes_field(2, b"oxionnx-builder-test")); // producer_name
        model.extend(encode_bytes_field(3, b"1.2.3")); // producer_version
        model.extend(encode_bytes_field(4, b"ai.oxionnx.test")); // domain
        model.extend(encode_bytes_field(8, &opset)); // opset_import
        model.extend(encode_bytes_field(14, &metadata_entry)); // metadata_props
        model.extend(encode_bytes_field(7, &graph)); // graph
        model
    }

    /// `load_mmap` must populate the model's real parsed metadata — the same
    /// `RawModelMeta` `load()`/`load_from_bytes()` already surface — instead of
    /// silently reporting `ModelMetadata::default()`.
    ///
    /// Checked against **two** independent baselines, matching
    /// `tests/w3_mmap_session_load_e2e.rs`'s convention: hand-specified
    /// expected values (so `load_mmap` and `load_from_bytes` cannot both be
    /// wrong in the same way and still pass), and `load_from_bytes` on the
    /// identical bytes (so the two loading paths cannot silently diverge).
    #[test]
    fn load_mmap_populates_real_metadata_matching_load_from_bytes() {
        let model_bytes = build_model_with_metadata(&[1.0, 2.0, 3.0]);

        let dir = std::env::temp_dir().join("oxionnx_builder_mmap_meta_tests");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("mmap_meta_test.onnx");
        std::fs::write(&path, &model_bytes).expect("write temp model");

        let session_mmap = SessionBuilder::new()
            .load_mmap(&path)
            .expect("load_mmap should succeed on a well-formed model");
        let session_bytes = SessionBuilder::new()
            .load_from_bytes(&model_bytes)
            .expect("load_from_bytes should succeed on the same bytes");

        let _ = std::fs::remove_file(&path);

        let mmap_meta = session_mmap.metadata();
        let bytes_meta = session_bytes.metadata();

        // Hand-specified expected values — not just "the two paths agree",
        // since two identically-broken loaders (e.g. both reporting the
        // default) could agree with each other and still be wrong.
        assert_eq!(mmap_meta.producer_name, "oxionnx-builder-test");
        assert_eq!(mmap_meta.producer_version, "1.2.3");
        assert_eq!(mmap_meta.domain, "ai.oxionnx.test");
        assert_eq!(mmap_meta.graph_name, "mmap_meta_test_graph");
        assert_eq!(mmap_meta.ir_version, 9);
        assert_eq!(mmap_meta.opset_imports, vec![(String::new(), 13)]);
        assert_eq!(
            mmap_meta
                .custom_metadata
                .get("custom_key")
                .map(String::as_str),
            Some("custom_value")
        );

        // And the two loading paths must agree on every field for identical
        // model bytes.
        assert_eq!(mmap_meta.producer_name, bytes_meta.producer_name);
        assert_eq!(mmap_meta.producer_version, bytes_meta.producer_version);
        assert_eq!(mmap_meta.domain, bytes_meta.domain);
        assert_eq!(mmap_meta.graph_name, bytes_meta.graph_name);
        assert_eq!(mmap_meta.ir_version, bytes_meta.ir_version);
        assert_eq!(mmap_meta.opset_imports, bytes_meta.opset_imports);
        assert_eq!(mmap_meta.custom_metadata, bytes_meta.custom_metadata);
    }

    /// `load_mmap` on a path that does not exist must still return a typed
    /// `OnnxError`, never panic, now that the parse closure does its own
    /// `File::open`/`Mmap::map` instead of delegating to `MmapModel::open`.
    #[test]
    fn load_mmap_reports_a_typed_error_for_a_missing_file() {
        let missing =
            std::env::temp_dir().join("oxionnx_builder_mmap_meta_tests_definitely_absent.onnx");
        let _ = std::fs::remove_file(&missing);

        let result = SessionBuilder::new().load_mmap(&missing);
        assert!(
            result.is_err(),
            "load_mmap on a missing path must return Err"
        );
    }
}
