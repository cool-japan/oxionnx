//! Attention mechanism kernels: scaled dot-product, multi-head, rotary embedding,
//! grouped/multi-query, ALiBi, and cached (KV-cache) attention variants.

pub mod cached;
pub mod core;
#[cfg(feature = "simd")]
pub(crate) mod simd_sdpa;
pub(crate) mod typed;
pub mod variants;

// Re-export public API
pub use cached::{cached_attention, cached_multi_head_attention};
pub use core::{multi_head_attention, rotary_embedding, scaled_dot_product_attention};
pub(crate) use core::{reshape_from_heads, reshape_to_heads};
pub use variants::{alibi_attention, grouped_query_attention, multi_query_attention};

// Re-export typed kernel types
pub(crate) use typed::{
    multi_head_attention_bf16, multi_head_attention_f16, scaled_dot_product_attention_bf16,
    scaled_dot_product_attention_f16, MhaDims, SdpaDims,
};

#[cfg(all(test, feature = "simd"))]
mod simd_tests;

#[cfg(test)]
mod tests;
