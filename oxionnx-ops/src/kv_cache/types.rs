//! KvCache struct and implementation.

use oxionnx_core::Tensor;

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
    /// Past key tensors per layer: shape `[batch, num_heads, past_seq_len, head_dim]`
    past_keys: Vec<Option<Tensor>>,
    /// Past value tensors per layer: shape `[batch, num_heads, past_seq_len, head_dim]`
    past_values: Vec<Option<Tensor>>,
    /// Maximum sequence length (enables ring buffer mode when `Some`).
    max_seq_len: Option<usize>,
    /// Number of layers this cache was created for.
    num_layers: usize,
}
impl KvCache {
    /// Create an empty cache for `num_layers` layers (unbounded mode).
    pub fn new(num_layers: usize) -> Self {
        Self {
            past_keys: vec![None; num_layers],
            past_values: vec![None; num_layers],
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
            past_keys: vec![None; num_layers],
            past_values: vec![None; num_layers],
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
            Some(t) if t.ndim() >= 3 => t.shape[2],
            _ => 0,
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
        let full_key = self.concat_along_seq(&self.past_keys[layer], new_key)?;
        let full_value = self.concat_along_seq(&self.past_values[layer], new_value)?;
        let (trimmed_key, trimmed_value) = match self.max_seq_len {
            Some(max_len) if full_key.shape[2] > max_len => {
                let k = Self::trim_front_seq(&full_key, max_len);
                let v = Self::trim_front_seq(&full_value, max_len);
                (k, v)
            }
            _ => (full_key.clone(), full_value.clone()),
        };
        self.past_keys[layer] = Some(trimmed_key);
        self.past_values[layer] = Some(trimmed_value);
        Ok((full_key, full_value))
    }
    /// Concatenate `past` (if present) with `new_tensor` along dimension 2.
    ///
    /// Both tensors are 4D: `[batch, heads, seq, dim]`.
    fn concat_along_seq(
        &self,
        past: &Option<Tensor>,
        new_tensor: &Tensor,
    ) -> Result<Tensor, String> {
        let past_ref = match past {
            Some(p) => p,
            None => return Ok(new_tensor.clone()),
        };
        if past_ref.shape[0] != new_tensor.shape[0] {
            return Err(format!(
                "KvCache: batch mismatch (past={}, new={}) past_shape={:?} new_shape={:?}",
                past_ref.shape[0], new_tensor.shape[0], past_ref.shape, new_tensor.shape
            ));
        }
        if past_ref.shape[1] != new_tensor.shape[1] {
            return Err(format!(
                "KvCache: num_heads mismatch (past={}, new={}) past_shape={:?} new_shape={:?}",
                past_ref.shape[1], new_tensor.shape[1], past_ref.shape, new_tensor.shape
            ));
        }
        if past_ref.shape[3] != new_tensor.shape[3] {
            return Err(format!(
                "KvCache: head_dim mismatch (past={}, new={}) past_shape={:?} new_shape={:?}",
                past_ref.shape[3], new_tensor.shape[3], past_ref.shape, new_tensor.shape
            ));
        }
        let batch = past_ref.shape[0];
        let heads = past_ref.shape[1];
        let past_seq = past_ref.shape[2];
        let new_seq = new_tensor.shape[2];
        let head_dim = past_ref.shape[3];
        let total_seq = past_seq + new_seq;
        let mut data = vec![0.0f32; batch * heads * total_seq * head_dim];
        let past_bh_stride = past_seq * head_dim;
        let new_bh_stride = new_seq * head_dim;
        let out_bh_stride = total_seq * head_dim;
        for b in 0..batch {
            for h in 0..heads {
                let bh = b * heads + h;
                let past_off = bh * past_bh_stride;
                let new_off = bh * new_bh_stride;
                let out_off = bh * out_bh_stride;
                data[out_off..out_off + past_bh_stride]
                    .copy_from_slice(&past_ref.data[past_off..past_off + past_bh_stride]);
                data[out_off + past_bh_stride..out_off + past_bh_stride + new_bh_stride]
                    .copy_from_slice(&new_tensor.data[new_off..new_off + new_bh_stride]);
            }
        }
        Ok(Tensor::new(data, vec![batch, heads, total_seq, head_dim]))
    }
    /// Trim a 4D tensor along dimension 2, keeping only the last `max_len` entries.
    fn trim_front_seq(tensor: &Tensor, max_len: usize) -> Tensor {
        let batch = tensor.shape[0];
        let heads = tensor.shape[1];
        let total_seq = tensor.shape[2];
        let head_dim = tensor.shape[3];
        if total_seq <= max_len {
            return tensor.clone();
        }
        let drop = total_seq - max_len;
        let mut data = vec![0.0f32; batch * heads * max_len * head_dim];
        let in_bh_stride = total_seq * head_dim;
        let out_bh_stride = max_len * head_dim;
        for b in 0..batch {
            for h in 0..heads {
                let bh = b * heads + h;
                let in_off = bh * in_bh_stride + drop * head_dim;
                let out_off = bh * out_bh_stride;
                data[out_off..out_off + out_bh_stride]
                    .copy_from_slice(&tensor.data[in_off..in_off + out_bh_stride]);
            }
        }
        Tensor::new(data, vec![batch, heads, max_len, head_dim])
    }
}
