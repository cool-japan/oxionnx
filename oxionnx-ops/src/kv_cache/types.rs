//! KvCache struct and implementation.

use oxionnx_core::Tensor;

/// Smallest sequence capacity a layer slot is grown to on first use.
const MIN_SEQ_CAPACITY: usize = 8;

/// Storage for one layer's key (or value) cache.
///
/// Laid out as `[batch, heads, cap, head_dim]` with a logical window
/// `[start, start + len)` inside every `(batch, head)` block. Appending writes
/// only the *new* tokens into the spare capacity, so a decode step costs
/// `O(new_seq)` instead of re-copying the whole cache, and the ring-buffer
/// eviction is a pure index update (`start += drop`) instead of a second copy.
///
/// The capacity grows geometrically, so the amortised cost of a decode step
/// stays `O(new_seq)` even though the buffer is occasionally re-laid-out.
struct LayerCache {
    data: Vec<f32>,
    batch: usize,
    heads: usize,
    head_dim: usize,
    /// Sequence capacity per `(batch, head)` block.
    cap: usize,
    /// First live sequence position inside each block.
    start: usize,
    /// Live sequence length.
    len: usize,
}

impl LayerCache {
    /// Elements per `(batch, head)` block, including spare capacity.
    #[inline]
    fn block_stride(&self) -> usize {
        self.cap * self.head_dim
    }

    /// Number of `(batch, head)` blocks.
    #[inline]
    fn blocks(&self) -> usize {
        self.batch * self.heads
    }

    /// Create a slot sized for `new_tensor`, with room to grow.
    fn with_first(new_tensor: &Tensor) -> Self {
        let batch = new_tensor.shape[0];
        let heads = new_tensor.shape[1];
        let new_seq = new_tensor.shape[2];
        let head_dim = new_tensor.shape[3];
        let cap = new_seq.max(MIN_SEQ_CAPACITY);
        let mut slot = Self {
            data: vec![0.0f32; batch * heads * cap * head_dim],
            batch,
            heads,
            head_dim,
            cap,
            start: 0,
            len: 0,
        };
        slot.write_new(new_tensor, new_seq);
        slot.len = new_seq;
        slot
    }

    /// Make room for `need` live positions starting at `start`, compacting or
    /// reallocating as required.
    fn reserve(&mut self, need: usize) {
        if self.start + need <= self.cap {
            return;
        }
        let live = self.len * self.head_dim;
        if need <= self.cap {
            // Enough capacity overall — slide every block's window back to 0.
            if self.start > 0 && live > 0 {
                let stride = self.block_stride();
                for bh in 0..self.blocks() {
                    let src = bh * stride + self.start * self.head_dim;
                    self.data.copy_within(src..src + live, bh * stride);
                }
            }
            self.start = 0;
            return;
        }
        // Geometric growth keeps append amortised O(new_seq).
        let new_cap = need.max(self.cap.saturating_mul(2)).max(MIN_SEQ_CAPACITY);
        let new_stride = new_cap * self.head_dim;
        let mut fresh = vec![0.0f32; self.blocks() * new_stride];
        if live > 0 {
            let stride = self.block_stride();
            for bh in 0..self.blocks() {
                let src = bh * stride + self.start * self.head_dim;
                fresh[bh * new_stride..bh * new_stride + live]
                    .copy_from_slice(&self.data[src..src + live]);
            }
        }
        self.data = fresh;
        self.cap = new_cap;
        self.start = 0;
    }

    /// Copy `new_tensor`'s rows into the spare capacity after the live window.
    fn write_new(&mut self, new_tensor: &Tensor, new_seq: usize) {
        let stride = self.block_stride();
        let new_block = new_seq * self.head_dim;
        if new_block == 0 {
            return;
        }
        let offset = (self.start + self.len) * self.head_dim;
        for bh in 0..self.blocks() {
            let dst = bh * stride + offset;
            let src = bh * new_block;
            self.data[dst..dst + new_block].copy_from_slice(&new_tensor.data[src..src + new_block]);
        }
    }

    /// Append `new_tensor` in place (reserve + write), growing if needed.
    fn append(&mut self, new_tensor: &Tensor, new_seq: usize) {
        self.reserve(self.len + new_seq);
        self.write_new(new_tensor, new_seq);
        self.len += new_seq;
    }

    /// Materialise the live window as a dense `[batch, heads, len, head_dim]`
    /// tensor.
    fn gather(&self) -> Tensor {
        let shape = vec![self.batch, self.heads, self.len, self.head_dim];
        if self.start == 0 && self.len == self.cap {
            // The buffer already *is* the dense tensor — one memcpy.
            return Tensor::new(self.data.clone(), shape);
        }
        let out_block = self.len * self.head_dim;
        let mut out = vec![0.0f32; self.blocks() * out_block];
        if out_block > 0 {
            let stride = self.block_stride();
            for bh in 0..self.blocks() {
                let src = bh * stride + self.start * self.head_dim;
                out[bh * out_block..(bh + 1) * out_block]
                    .copy_from_slice(&self.data[src..src + out_block]);
            }
        }
        Tensor::new(out, shape)
    }

    /// Drop the oldest positions so at most `max_len` remain — `O(1)`.
    fn trim_front(&mut self, max_len: usize) {
        if self.len > max_len {
            self.start += self.len - max_len;
            self.len = max_len;
        }
    }

    /// The logical shape of the currently cached tensor (for error messages).
    fn logical_shape(&self) -> Vec<usize> {
        vec![self.batch, self.heads, self.len, self.head_dim]
    }
}

/// Key-Value cache for autoregressive (incremental) inference.
///
/// Stores past key and value tensors per layer so that autoregressive
/// decoding only needs to compute attention for the new token(s) against
/// the full cached sequence.
///
/// Supports two modes:
/// - **Unbounded**: the cache grows without limit (suitable for short sequences).
/// - **Ring buffer**: once `max_seq_len` is reached, the oldest entries are
///   evicted to keep memory bounded (suitable for long-running generation).
pub struct KvCache {
    /// Past key storage per layer (logical shape `[batch, heads, past_seq_len, head_dim]`).
    past_keys: Vec<Option<LayerCache>>,
    /// Past value storage per layer (logical shape `[batch, heads, past_seq_len, head_dim]`).
    past_values: Vec<Option<LayerCache>>,
    /// Maximum sequence length (enables ring buffer mode when `Some`).
    max_seq_len: Option<usize>,
    /// Number of layers this cache was created for.
    num_layers: usize,
}
impl KvCache {
    /// Create an empty cache for `num_layers` layers (unbounded mode).
    pub fn new(num_layers: usize) -> Self {
        Self {
            past_keys: (0..num_layers).map(|_| None).collect(),
            past_values: (0..num_layers).map(|_| None).collect(),
            max_seq_len: None,
            num_layers,
        }
    }
    /// Create an empty cache with a maximum sequence length (ring buffer mode).
    ///
    /// Once the cached sequence exceeds `max_seq_len`, the oldest entries are
    /// dropped to stay within bounds.
    pub fn with_max_seq_len(num_layers: usize, max_seq_len: usize) -> Self {
        Self {
            past_keys: (0..num_layers).map(|_| None).collect(),
            past_values: (0..num_layers).map(|_| None).collect(),
            max_seq_len: Some(max_seq_len),
            num_layers,
        }
    }
    /// Number of layers this cache manages.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
    /// Current cached sequence length for a given layer.
    ///
    /// Returns `0` if no entries have been added yet for that layer.
    pub fn seq_len(&self, layer: usize) -> usize {
        if layer >= self.num_layers {
            return 0;
        }
        match &self.past_keys[layer] {
            Some(slot) => slot.len,
            None => 0,
        }
    }
    /// Reset all layers, discarding all cached key/value tensors.
    pub fn clear(&mut self) {
        for slot in self.past_keys.iter_mut() {
            *slot = None;
        }
        for slot in self.past_values.iter_mut() {
            *slot = None;
        }
    }
    /// Update the cache for `layer` by appending `new_key` and `new_value`.
    ///
    /// # Arguments
    /// * `layer` — layer index (must be < `num_layers`)
    /// * `new_key` — new key tensor `[batch, num_heads, new_seq, head_dim]`
    /// * `new_value` — new value tensor `[batch, num_heads, new_seq, head_dim]`
    ///
    /// # Returns
    /// A tuple `(full_key, full_value)` containing the concatenated
    /// past + new tensors along dimension 2 (the sequence dimension).
    /// If `max_seq_len` is set and the total would exceed it, the oldest
    /// entries are dropped from the front.
    ///
    /// The new tokens are written into spare capacity of the existing buffer
    /// (`O(new_seq)`); only the returned dense view costs `O(past + new)`.
    /// Every shape is validated *before* anything is mutated, so a rejected
    /// update leaves the cache exactly as it was.
    pub fn update(
        &mut self,
        layer: usize,
        new_key: &Tensor,
        new_value: &Tensor,
    ) -> Result<(Tensor, Tensor), String> {
        if layer >= self.num_layers {
            return Err(format!(
                "KvCache::update: layer {} out of range (num_layers={})",
                layer, self.num_layers
            ));
        }
        if new_key.ndim() != 4 {
            return Err(format!(
                "KvCache::update: new_key must be 4D [batch, heads, seq, dim], got {}D",
                new_key.ndim()
            ));
        }
        if new_value.ndim() != 4 {
            return Err(format!(
                "KvCache::update: new_value must be 4D, got {}D",
                new_value.ndim()
            ));
        }
        if new_key.shape[2] == 0 || new_value.shape[2] == 0 {
            return Err(
                format!(
                    "KvCache::update: empty sequence not allowed: new_key shape={:?}, new_value shape={:?}",
                    new_key.shape, new_value.shape
                ),
            );
        }
        // Validate both tensors first: a rejected update must leave the cache
        // usable, which append-then-validate would not.
        Self::check_appendable(self.past_keys[layer].as_ref(), new_key)?;
        Self::check_appendable(self.past_values[layer].as_ref(), new_value)?;

        let key_seq = new_key.shape[2];
        let value_seq = new_value.shape[2];
        match self.past_keys[layer].as_mut() {
            Some(slot) => slot.append(new_key, key_seq),
            None => self.past_keys[layer] = Some(LayerCache::with_first(new_key)),
        }
        match self.past_values[layer].as_mut() {
            Some(slot) => slot.append(new_value, value_seq),
            None => self.past_values[layer] = Some(LayerCache::with_first(new_value)),
        }

        // Gather the *untrimmed* window first: `update` returns past + new even
        // when that exceeds `max_seq_len`; only the stored cache is trimmed.
        let (full_key, full_value) = match (&self.past_keys[layer], &self.past_values[layer]) {
            (Some(k), Some(v)) => (k.gather(), v.gather()),
            // Unreachable: both slots were just populated above.
            _ => {
                return Err(format!(
                    "KvCache::update: layer {layer} storage missing after append"
                ))
            }
        };
        if let Some(max_len) = self.max_seq_len {
            if let Some(slot) = self.past_keys[layer].as_mut() {
                slot.trim_front(max_len);
            }
            if let Some(slot) = self.past_values[layer].as_mut() {
                slot.trim_front(max_len);
            }
        }
        Ok((full_key, full_value))
    }

    /// Verify `new_tensor` can be appended to `past` without mutating anything.
    ///
    /// Error messages are the historical `concat_along_seq` ones — callers and
    /// tests match on the `batch` / `num_heads` / `head_dim` wording.
    fn check_appendable(past: Option<&LayerCache>, new_tensor: &Tensor) -> Result<(), String> {
        let expected = new_tensor.shape[0]
            .saturating_mul(new_tensor.shape[1])
            .saturating_mul(new_tensor.shape[2])
            .saturating_mul(new_tensor.shape[3]);
        if new_tensor.data.len() < expected {
            return Err(format!(
                "KvCache: tensor holds {} element(s) but shape {:?} needs {expected}",
                new_tensor.data.len(),
                new_tensor.shape
            ));
        }
        let Some(slot) = past else {
            return Ok(());
        };
        if slot.batch != new_tensor.shape[0] {
            return Err(format!(
                "KvCache: batch mismatch (past={}, new={}) past_shape={:?} new_shape={:?}",
                slot.batch,
                new_tensor.shape[0],
                slot.logical_shape(),
                new_tensor.shape
            ));
        }
        if slot.heads != new_tensor.shape[1] {
            return Err(format!(
                "KvCache: num_heads mismatch (past={}, new={}) past_shape={:?} new_shape={:?}",
                slot.heads,
                new_tensor.shape[1],
                slot.logical_shape(),
                new_tensor.shape
            ));
        }
        if slot.head_dim != new_tensor.shape[3] {
            return Err(format!(
                "KvCache: head_dim mismatch (past={}, new={}) past_shape={:?} new_shape={:?}",
                slot.head_dim,
                new_tensor.shape[3],
                slot.logical_shape(),
                new_tensor.shape
            ));
        }
        Ok(())
    }
}
