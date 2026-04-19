//! Flash Attention v2 (Pure Rust) — tiled attention with online softmax.
//!
//! Processes Q, K, V in blocks to achieve O(Br × Bc) extra memory
//! instead of O(N²) for the full attention matrix.
//!
//! ## Module layout
//! - `kernel`     — re-exports [`flash_attention`] and [`flash_attention_with_block_size`]
//! - `multi_head` — re-exports [`multi_head_flash_attention`]
//! - `cached`     — re-exports [`cached_flash_attention`]

use oxionnx_core::Tensor;

// ── Sub-modules ──────────────────────────────────────────────────────────────

pub(crate) mod cached;
pub(crate) mod kernel;
pub(crate) mod multi_head;

#[cfg(test)]
mod tests;

// ── Re-exports (public API) ──────────────────────────────────────────────────

pub use cached::cached_flash_attention;
pub use kernel::{flash_attention, flash_attention_with_block_size};
pub use multi_head::multi_head_flash_attention;

// ── Constants ────────────────────────────────────────────────────────────────

/// Default block size for Flash Attention query rows.
pub(crate) const FLASH_DEFAULT_BLOCK_R: usize = 64;
/// Default block size for Flash Attention key/value columns.
pub(crate) const FLASH_DEFAULT_BLOCK_C: usize = 64;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Retrieve an additive mask value with broadcasting support.
///
/// Supports 2D `[seq_q, seq_k]`, 3D `[batch, seq_q, seq_k]`, and
/// 4D `[batch, heads, seq_q, seq_k]` masks (dims of size 1 are broadcast).
pub(crate) fn flash_mask_value(mask: &Tensor, b: usize, h: usize, i: usize, j: usize) -> f32 {
    match mask.ndim() {
        2 => mask.data[i * mask.shape[1] + j],
        3 => {
            let mb = if mask.shape[0] == 1 { 0 } else { b };
            let (s1, s2) = (mask.shape[1], mask.shape[2]);
            mask.data[mb * s1 * s2 + i * s2 + j]
        }
        4 => {
            let mb = if mask.shape[0] == 1 { 0 } else { b };
            let mh = if mask.shape[1] == 1 { 0 } else { h };
            let (s1, s2, s3) = (mask.shape[1], mask.shape[2], mask.shape[3]);
            mask.data[mb * s1 * s2 * s3 + mh * s2 * s3 + i * s3 + j]
        }
        _ => 0.0,
    }
}
