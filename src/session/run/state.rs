use crate::memory::SizeClassPool;
use crate::tensor::Tensor;
use crate::OnnxError;
use std::collections::HashMap;
use std::sync::Mutex;

// ── SessionRunState ──────────────────────────────────────────────────────────

/// Intermediate tensor storage for a single inference run.
///
/// Wraps the tensor map with pool-backed buffer release on completion.
/// Replaces the bare `HashMap<String, Tensor>` used in previous versions.
/// IoBinding integration (bound_outputs wiring) is deferred to a later item.
pub(crate) struct SessionRunState {
    /// Active intermediate tensors keyed by name.
    tensors: HashMap<String, Tensor>,
}

impl SessionRunState {
    /// Create a new run state with a pre-allocated tensor map capacity.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            tensors: HashMap::with_capacity(capacity),
        }
    }

    /// Look up a tensor by name (immutable).
    #[inline]
    pub(crate) fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Insert or replace a tensor. If a tensor already exists at this name,
    /// its data buffer is released to the pool.
    ///
    /// One hash lookup, not two: this was `remove` followed by `insert`, which
    /// hashed the key twice on the hottest write path in the engine (once per
    /// node output, per run).  `HashMap::insert` already returns the displaced
    /// value, and releasing it after the store rather than before is
    /// unobservable — a pool release touches only the pool.
    pub(crate) fn insert(
        &mut self,
        name: String,
        tensor: Tensor,
        pool: Option<&Mutex<SizeClassPool>>,
    ) {
        if let Some(old) = self.tensors.insert(name, tensor) {
            release_to_pool(old, pool);
        }
    }

    /// Remove a tensor from state, returning ownership (no pool release).
    /// Used for in-place execution where the caller takes ownership of the buffer.
    pub(crate) fn take(&mut self, name: &str) -> Option<Tensor> {
        self.tensors.remove(name)
    }

    /// Expose the tensor map as an immutable reference (for GPU dispatch functions
    /// that accept `&HashMap<String, Tensor>`).
    #[inline]
    pub(crate) fn as_map(&self) -> &HashMap<String, Tensor> {
        &self.tensors
    }

    /// Extract the named output tensors and release all remaining intermediates
    /// back to the pool.
    ///
    /// # A declared output that was never produced is an error
    ///
    /// This used to `filter_map` the misses away, so `Session::run` returned
    /// `Ok(map)` with a graph output silently absent — the caller's
    /// `outputs.get("y")` was `None` and nothing anywhere said why.  Combined
    /// with the run loops' old "skip nodes whose op is unknown" behaviour, an
    /// entire branch of the graph could vanish without a single diagnostic.
    ///
    /// `weights` is consulted before declaring an output missing: constant
    /// folding can promote a graph output to an initializer, and a model may
    /// legitimately name an initializer as an output.  Such an output is never
    /// written by any node, yet it is perfectly well defined.
    ///
    /// # Errors
    ///
    /// [`OnnxError::TensorNotFound`], naming every missing output, when a
    /// declared output is neither in the run state nor an initializer.
    pub(crate) fn take_outputs(
        mut self,
        output_names: &[String],
        weights: &HashMap<String, Tensor>,
        pool: Option<&Mutex<SizeClassPool>>,
    ) -> Result<HashMap<String, Tensor>, OnnxError> {
        // Remove outputs first (these are returned to the caller, not pooled)
        let mut result: HashMap<String, Tensor> = HashMap::with_capacity(output_names.len());
        let mut missing: Vec<&str> = Vec::new();
        for name in output_names {
            if let Some(t) = self.tensors.remove(name) {
                result.insert(name.clone(), t);
            } else if let Some(w) = weights.get(name) {
                // A graph output that is an initializer (or was constant-folded
                // into one): no node writes it, but it is well defined.
                result.insert(name.clone(), w.clone());
            } else {
                missing.push(name.as_str());
            }
        }
        // Release all remaining intermediates back to the pool, error or not:
        // the buffers are ours to recycle either way.
        for (_name, tensor) in self.tensors.drain() {
            release_to_pool(tensor, pool);
        }
        if !missing.is_empty() {
            return Err(OnnxError::TensorNotFound(format!(
                "graph output(s) {missing:?} were never produced by any node — the model \
                 declares them as outputs but nothing in the executed graph writes them",
            )));
        }
        Ok(result)
    }
}

/// Release a tensor's data buffer back into the pool, if a pool is available.
#[inline]
pub(super) fn release_to_pool(mut tensor: Tensor, pool: Option<&Mutex<SizeClassPool>>) {
    if let Some(pool_mutex) = pool {
        if let Ok(mut guard) = pool_mutex.lock() {
            let buf = std::mem::take(&mut tensor.data);
            if !buf.is_empty() {
                guard.release(buf);
            }
        }
    }
}

// ── TypedSessionRunState ─────────────────────────────────────────────────────

/// Intermediate typed-tensor storage for a single `run_typed` inference run.
///
/// Parallel to `SessionRunState` but carries `TypedTensor` intermediates so that
/// integer and half-precision dtypes are preserved per-node without an f32 round-trip.
/// Pool integration is intentionally absent: `TypedTensor` heap buffers are owned by
/// the enum variants and freed by Rust's ordinary drop machinery.
pub(super) struct TypedSessionRunState {
    pub(super) slots: HashMap<String, oxionnx_core::TypedTensor>,
}

impl TypedSessionRunState {
    pub(super) fn new() -> Self {
        Self {
            slots: HashMap::new(),
        }
    }

    #[inline]
    pub(super) fn get(&self, name: &str) -> Option<&oxionnx_core::TypedTensor> {
        self.slots.get(name)
    }

    /// The live slot map, for use as a subgraph operator's outer scope.
    #[inline]
    pub(super) fn slots(&self) -> &HashMap<String, oxionnx_core::TypedTensor> {
        &self.slots
    }

    #[inline]
    pub(super) fn insert(&mut self, name: String, tensor: oxionnx_core::TypedTensor) {
        self.slots.insert(name, tensor);
    }

    /// Remove and return the requested output tensors; intermediate slots are dropped.
    ///
    /// Mirrors [`SessionRunState::take_outputs`]: a declared output that no node
    /// produced is an error rather than a silently absent map entry, and
    /// `weights` is consulted first because initializers are not seeded into the
    /// typed run state (seeding them deep-copied every model parameter on every
    /// `run_typed` call).
    ///
    /// # Errors
    ///
    /// [`OnnxError::TensorNotFound`], naming every missing output.
    pub(super) fn take_outputs(
        &mut self,
        output_names: &[String],
        weights: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, oxionnx_core::TypedTensor>, OnnxError> {
        let mut result = HashMap::with_capacity(output_names.len());
        let mut missing: Vec<&str> = Vec::new();
        for name in output_names {
            if let Some(t) = self.slots.remove(name) {
                result.insert(name.clone(), t);
            } else if let Some(w) = weights.get(name) {
                result.insert(
                    name.clone(),
                    oxionnx_core::TypedTensor::new(
                        oxionnx_core::TensorStorage::F32(w.data.clone()),
                        w.shape.clone(),
                    ),
                );
            } else {
                missing.push(name.as_str());
            }
        }
        if !missing.is_empty() {
            return Err(OnnxError::TensorNotFound(format!(
                "graph output(s) {missing:?} were never produced by any node — the model \
                 declares them as outputs but nothing in the executed graph writes them",
            )));
        }
        Ok(result)
    }
}
