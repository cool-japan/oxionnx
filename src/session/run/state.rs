use crate::memory::SizeClassPool;
use crate::tensor::Tensor;
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
    /// its data buffer is released to the pool before the new tensor is stored.
    pub(crate) fn insert(
        &mut self,
        name: String,
        tensor: Tensor,
        pool: Option<&Mutex<SizeClassPool>>,
    ) {
        if let Some(old) = self.tensors.remove(&name) {
            release_to_pool(old, pool);
        }
        self.tensors.insert(name, tensor);
    }

    /// Remove a tensor from state, returning ownership (no pool release).
    /// Used for in-place execution where the caller takes ownership of the buffer.
    pub(crate) fn take(&mut self, name: &str) -> Option<Tensor> {
        self.tensors.remove(name)
    }

    /// Expose the tensor map as an immutable reference (for GPU dispatch functions
    /// that accept `&HashMap<String, Tensor>`).
    #[inline]
    #[cfg_attr(
        not(any(feature = "gpu", feature = "cuda", feature = "directml")),
        allow(dead_code)
    )]
    pub(crate) fn as_map(&self) -> &HashMap<String, Tensor> {
        &self.tensors
    }

    /// Extract the named output tensors and release all remaining intermediates
    /// back to the pool. Returns `HashMap<String, Tensor>` of the requested outputs.
    pub(crate) fn take_outputs(
        mut self,
        output_names: &[String],
        pool: Option<&Mutex<SizeClassPool>>,
    ) -> HashMap<String, Tensor> {
        // Remove outputs first (these are returned to the caller, not pooled)
        let mut result: HashMap<String, Tensor> = HashMap::with_capacity(output_names.len());
        for name in output_names {
            if let Some(t) = self.tensors.remove(name) {
                result.insert(name.clone(), t);
            }
        }
        // Release all remaining intermediates back to the pool
        for (_name, tensor) in self.tensors.drain() {
            release_to_pool(tensor, pool);
        }
        result
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

    #[inline]
    pub(super) fn insert(&mut self, name: String, tensor: oxionnx_core::TypedTensor) {
        self.slots.insert(name, tensor);
    }

    /// Remove and return the requested output tensors; intermediate slots are dropped.
    pub(super) fn take_outputs(
        &mut self,
        output_names: &[String],
    ) -> HashMap<String, oxionnx_core::TypedTensor> {
        output_names
            .iter()
            .filter_map(|n| self.slots.remove(n).map(|t| (n.clone(), t)))
            .collect()
    }
}
