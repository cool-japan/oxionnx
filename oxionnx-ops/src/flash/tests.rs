//! Flash Attention unit tests — numerical parity against the scalar SDPA
//! reference implementation and edge-case coverage.

use oxionnx_core::Tensor;

use crate::attention::{multi_head_attention, scaled_dot_product_attention};

use super::{flash_attention, flash_attention_with_block_size, multi_head_flash_attention};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compare two tensors element-wise within `tol`.
fn assert_tensors_close(a: &Tensor, b: &Tensor, tol: f32, label: &str) {
    assert_eq!(a.shape, b.shape, "{label}: shape mismatch");
    for (i, (av, bv)) in a.data.iter().zip(b.data.iter()).enumerate() {
        assert!(
            (av - bv).abs() < tol,
            "{label}: mismatch at index {i}: {av} vs {bv} (diff={})",
            (av - bv).abs()
        );
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn test_flash_attention_matches_sdpa_small() {
    // 4×4 attention with block_size=2 to exercise the tiled algorithm
    let (batch, heads, seq, dim) = (1, 1, 4, 8);
    let n = batch * heads * seq * dim;
    let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin()).collect();
    let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.2).cos()).collect();
    let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.15 + 1.0).sin()).collect();

    let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
    let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
    let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

    let flash = flash_attention_with_block_size(&q, &k, &v, None, false, 2, 2)
        .expect("flash should succeed");
    let sdpa = scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA should succeed");

    assert_tensors_close(&flash, &sdpa, 1e-5, "flash_vs_sdpa_4x4");
}

#[test]
fn test_flash_attention_matches_sdpa_block3() {
    // block_size=3 with seq=4 (not divisible) to exercise boundary handling
    let (batch, heads, seq, dim) = (1, 1, 4, 8);
    let n = batch * heads * seq * dim;
    let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.3).sin()).collect();
    let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.17).cos()).collect();
    let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05 + 2.0).sin()).collect();

    let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
    let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
    let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

    let flash = flash_attention_with_block_size(&q, &k, &v, None, false, 3, 3)
        .expect("flash should succeed");
    let sdpa = scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA should succeed");

    assert_tensors_close(&flash, &sdpa, 1e-5, "flash_vs_sdpa_block3");
}

#[test]
fn test_flash_attention_causal() {
    // Verify causal masking matches SDPA with an explicit causal mask
    let (batch, heads, seq, dim) = (1, 2, 6, 4);
    let n = batch * heads * seq * dim;
    let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).sin()).collect();
    let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.13).cos()).collect();
    let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.11 + 0.5).sin()).collect();

    let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
    let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
    let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

    // Flash attention with causal=true, block_size=2
    let flash = flash_attention_with_block_size(&q, &k, &v, None, true, 2, 2)
        .expect("flash causal should succeed");

    // Reference: SDPA with explicit causal mask
    let mut mask_data = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            if j > i {
                mask_data[i * seq + j] = f32::NEG_INFINITY;
            }
        }
    }
    let causal_mask = Tensor::new(mask_data, vec![seq, seq]);
    let sdpa = scaled_dot_product_attention(&q, &k, &v, Some(&causal_mask), None)
        .expect("SDPA with causal mask should succeed");

    assert_tensors_close(&flash, &sdpa, 1e-5, "flash_causal");
}

#[test]
fn test_flash_attention_causal_with_additive_mask() {
    // Causal + additive mask combined
    let (batch, heads, seq, dim) = (1, 1, 5, 4);
    let n = batch * heads * seq * dim;
    let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.09).sin()).collect();
    let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.12).cos()).collect();
    let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.08 + 1.5).sin()).collect();

    let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
    let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
    let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

    // Additive mask: penalize position 0 key by -5
    let mut add_mask = vec![0.0f32; seq * seq];
    for i in 0..seq {
        add_mask[i * seq] = -5.0;
    }
    let add_tensor = Tensor::new(add_mask.clone(), vec![seq, seq]);

    let flash = flash_attention_with_block_size(&q, &k, &v, Some(&add_tensor), true, 2, 2)
        .expect("flash causal+mask should succeed");

    // Reference: SDPA with combined causal + additive mask
    let mut combined = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            if j > i {
                combined[i * seq + j] = f32::NEG_INFINITY;
            }
            combined[i * seq + j] += add_mask[i * seq + j];
        }
    }
    let combined_mask = Tensor::new(combined, vec![seq, seq]);
    let sdpa = scaled_dot_product_attention(&q, &k, &v, Some(&combined_mask), None)
        .expect("SDPA with combined mask should succeed");

    assert_tensors_close(&flash, &sdpa, 1e-5, "flash_causal_additive");
}

#[test]
fn test_flash_attention_batch_multihead() {
    // batch=2, heads=4
    let (batch, heads, seq, dim) = (2, 4, 8, 8);
    let n = batch * heads * seq * dim;
    let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.03).sin()).collect();
    let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).cos()).collect();
    let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.04 + 0.7).sin()).collect();

    let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
    let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
    let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

    // Use block_size=3 (not dividing seq=8 evenly)
    let flash = flash_attention_with_block_size(&q, &k, &v, None, false, 3, 3)
        .expect("flash batch+heads should succeed");
    let sdpa = scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA should succeed");

    assert_eq!(flash.shape, vec![batch, heads, seq, dim]);
    assert_tensors_close(&flash, &sdpa, 1e-5, "flash_batch_multihead");
}

#[test]
fn test_flash_attention_large_seq_stability() {
    // 256 tokens, verify no NaN/Inf
    let (batch, heads, seq, dim) = (1, 2, 256, 16);
    let n = batch * heads * seq * dim;
    let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.02).sin()).collect();
    let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.025).cos()).collect();
    let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.018 + 0.3).sin()).collect();

    let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
    let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
    let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

    // block_size=32, so we get 256/32 = 8 blocks
    let flash = flash_attention_with_block_size(&q, &k, &v, None, false, 32, 32)
        .expect("flash large seq should succeed");

    assert_eq!(flash.shape, vec![batch, heads, seq, dim]);
    for (i, &val) in flash.data.iter().enumerate() {
        assert!(val.is_finite(), "NaN/Inf at index {i} (val={val})",);
    }

    // Also verify against SDPA reference
    let sdpa = scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA should succeed");
    assert_tensors_close(&flash, &sdpa, 1e-4, "flash_large_seq");
}

#[test]
fn test_flash_attention_large_seq_causal_stability() {
    // 256 tokens, causal, verify numerical stability
    let (batch, heads, seq, dim) = (1, 1, 256, 16);
    let n = batch * heads * seq * dim;
    let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.04).sin()).collect();
    let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.06).cos()).collect();
    let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.035 + 1.2).sin()).collect();

    let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
    let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
    let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

    let flash = flash_attention_with_block_size(&q, &k, &v, None, true, 32, 32)
        .expect("flash large causal should succeed");

    for (i, &val) in flash.data.iter().enumerate() {
        assert!(val.is_finite(), "NaN/Inf at index {i}");
    }

    // Reference with explicit causal mask
    let mut mask_data = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            if j > i {
                mask_data[i * seq + j] = f32::NEG_INFINITY;
            }
        }
    }
    let causal_mask = Tensor::new(mask_data, vec![seq, seq]);
    let sdpa = scaled_dot_product_attention(&q, &k, &v, Some(&causal_mask), None)
        .expect("SDPA should succeed");
    assert_tensors_close(&flash, &sdpa, 1e-4, "flash_large_causal");
}

#[test]
fn test_flash_attention_block_boundary_edge() {
    // seq=7, block_size=3 → blocks of [3,3,1]
    let (batch, heads, seq, dim) = (1, 1, 7, 4);
    let n = batch * heads * seq * dim;
    let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.23).sin()).collect();
    let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.19).cos()).collect();
    let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.27 + 0.4).sin()).collect();

    let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
    let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
    let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

    let flash = flash_attention_with_block_size(&q, &k, &v, None, false, 3, 3)
        .expect("flash boundary should succeed");
    let sdpa = scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA should succeed");

    assert_tensors_close(&flash, &sdpa, 1e-5, "flash_block_boundary");
}

#[test]
fn test_flash_attention_asymmetric_blocks() {
    // Different block_r and block_c
    let (batch, heads, seq, dim) = (1, 1, 10, 4);
    let n = batch * heads * seq * dim;
    let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.14).sin()).collect();
    let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.21).cos()).collect();
    let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.17 + 0.9).sin()).collect();

    let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
    let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
    let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

    // block_r=2, block_c=4
    let flash = flash_attention_with_block_size(&q, &k, &v, None, false, 2, 4)
        .expect("flash asymmetric blocks should succeed");
    let sdpa = scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA should succeed");

    assert_tensors_close(&flash, &sdpa, 1e-5, "flash_asymmetric_blocks");
}

#[test]
fn test_flash_attention_default_block_fallback() {
    // seq < default block (64), should fall through to SDPA
    let (batch, heads, seq, dim) = (1, 1, 4, 8);
    let n = batch * heads * seq * dim;
    let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin()).collect();
    let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.2).cos()).collect();
    let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.15 + 1.0).sin()).collect();

    let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
    let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
    let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

    let flash = flash_attention(&q, &k, &v, None, false).expect("flash default should succeed");
    let sdpa = scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA should succeed");

    assert_tensors_close(&flash, &sdpa, 1e-6, "flash_default_fallback");
}

#[test]
fn test_multi_head_flash_attention_basic() {
    // batch=1, seq=4, embed=8, heads=2
    let (batch, seq, embed, heads) = (1, 4, 8, 2);
    let n = batch * seq * embed;
    let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.08).sin()).collect();
    let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.12).cos()).collect();
    let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1 + 0.5).sin()).collect();

    let q = Tensor::new(q_data, vec![batch, seq, embed]);
    let k = Tensor::new(k_data, vec![batch, seq, embed]);
    let v = Tensor::new(v_data, vec![batch, seq, embed]);

    let mhfa =
        multi_head_flash_attention(&q, &k, &v, None, false, heads).expect("MHFA should succeed");

    assert_eq!(mhfa.shape, vec![batch, seq, embed]);

    // Compare against standard MHA (which uses SDPA internally)
    let mha = multi_head_attention(&q, &k, &v, None, None, None, None, None, heads)
        .expect("MHA should succeed");

    assert_tensors_close(&mhfa, &mha, 1e-5, "mhfa_vs_mha");
}

#[test]
fn test_multi_head_flash_attention_causal() {
    let (batch, seq, embed, heads) = (2, 6, 16, 4);
    let n = batch * seq * embed;
    let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).sin()).collect();
    let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.11).cos()).collect();
    let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.09 + 0.8).sin()).collect();

    let q = Tensor::new(q_data, vec![batch, seq, embed]);
    let k = Tensor::new(k_data, vec![batch, seq, embed]);
    let v = Tensor::new(v_data, vec![batch, seq, embed]);

    let mhfa = multi_head_flash_attention(&q, &k, &v, None, true, heads)
        .expect("MHFA causal should succeed");

    assert_eq!(mhfa.shape, vec![batch, seq, embed]);
    for (i, &val) in mhfa.data.iter().enumerate() {
        assert!(val.is_finite(), "NaN/Inf at index {i}");
    }
}

#[test]
fn test_flash_attention_4d_mask() {
    // 4D mask [batch, heads, seq_q, seq_k]
    let (batch, heads, seq, dim) = (2, 2, 4, 4);
    let n = batch * heads * seq * dim;
    let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.06).sin()).collect();
    let k_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.09).cos()).collect();
    let v_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07 + 1.0).sin()).collect();

    let q = Tensor::new(q_data, vec![batch, heads, seq, dim]);
    let k = Tensor::new(k_data, vec![batch, heads, seq, dim]);
    let v = Tensor::new(v_data, vec![batch, heads, seq, dim]);

    // 4D mask that zeros out last key position
    let mask_n = batch * heads * seq * seq;
    let mut mask_data = vec![0.0f32; mask_n];
    for b in 0..batch {
        for h in 0..heads {
            for i in 0..seq {
                let idx = ((b * heads + h) * seq + i) * seq + (seq - 1);
                mask_data[idx] = -1e9;
            }
        }
    }
    let mask = Tensor::new(mask_data, vec![batch, heads, seq, seq]);

    // block_size=2 to exercise the flash algorithm
    let flash = flash_attention_with_block_size(&q, &k, &v, Some(&mask), false, 2, 2)
        .expect("flash with 4D mask should succeed");
    let sdpa =
        scaled_dot_product_attention(&q, &k, &v, Some(&mask), None).expect("SDPA should succeed");

    assert_tensors_close(&flash, &sdpa, 1e-5, "flash_4d_mask");
}

#[test]
fn test_flash_attention_error_on_wrong_dims() {
    let q = Tensor::new(vec![1.0; 12], vec![3, 4]);
    let k = Tensor::new(vec![1.0; 12], vec![3, 4]);
    let v = Tensor::new(vec![1.0; 12], vec![3, 4]);

    let result = flash_attention(&q, &k, &v, None, false);
    assert!(result.is_err(), "should fail on 2D inputs");
}

#[test]
fn test_flash_attention_uniform_values() {
    // All-ones Q,K,V: output should also be all-ones V
    let (batch, heads, seq, dim) = (1, 1, 8, 4);
    let n = batch * heads * seq * dim;
    let q = Tensor::new(vec![1.0f32; n], vec![batch, heads, seq, dim]);
    let k = Tensor::new(vec![1.0f32; n], vec![batch, heads, seq, dim]);
    let v = Tensor::new(vec![1.0f32; n], vec![batch, heads, seq, dim]);

    let flash = flash_attention_with_block_size(&q, &k, &v, None, false, 3, 3)
        .expect("flash uniform should succeed");

    for (i, &val) in flash.data.iter().enumerate() {
        assert!(
            (val - 1.0).abs() < 1e-5,
            "Expected ~1.0 at index {i}, got {val}",
        );
    }
}
