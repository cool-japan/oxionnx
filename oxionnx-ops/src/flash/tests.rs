//! Flash Attention unit tests — numerical parity against the scalar SDPA
//! reference implementation and edge-case coverage.

// NumPy reference constants are pasted verbatim from the float64 print-out so
// they can be diffed against the generating snippet; keeping digits f32 cannot
// represent is deliberate.
#![allow(clippy::excessive_precision)]

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

// ── Blocked-kernel parity after the scratch-hoisting / rayon rework ──────────
//
// The four per-block `Vec`s (`s_ij`, `m_ij`, `p_ij`, `l_ij`) are now allocated
// once per worker and reused. `m_ij` is a max accumulator and `l_ij` a sum
// accumulator, so they must be reset at the top of every K-block iteration —
// these tests would fail loudly on a stale accumulator, and they also cover
// the `(batch, head)` rayon partition.

/// `sin(i * a + b)` in f64 then narrowed — matches the NumPy generator used
/// for the reference constants.
fn gen_f64(n: usize, a: f64, b: f64) -> Vec<f32> {
    (0..n).map(|i| ((i as f64) * a + b).sin() as f32).collect()
}

fn tensor_f64(shape: &[usize], a: f64, b: f64) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::new(gen_f64(n, a, b), shape.to_vec())
}

#[track_caller]
fn assert_rel_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= tol * (1.0 + w.abs()),
            "{label}: idx {i}: got {g}, want {w} (diff {})",
            (g - w).abs()
        );
    }
}

/// Ragged blocks in *both* axes (10 = 4+4+2 query rows, 13 = 5+5+3 key cols)
/// against a NumPy softmax-attention reference:
///
/// ```text
/// q = sin(arange(1*2*10*6)*0.11 + 0.07).reshape(1,2,10,6)
/// k = sin(arange(1*2*13*6)*0.13 + 0.29).reshape(1,2,13,6)
/// v = sin(arange(1*2*13*6)*0.17 + 0.43).reshape(1,2,13,6)
/// softmax(q @ k.T / sqrt(6) + triu(-inf, 1)) @ v
/// ```
#[test]
fn test_flash_ragged_blocks_causal_matches_numpy() {
    let q = tensor_f64(&[1, 2, 10, 6], 0.11, 0.07);
    let k = tensor_f64(&[1, 2, 13, 6], 0.13, 0.29);
    let v = tensor_f64(&[1, 2, 13, 6], 0.17, 0.43);
    let want = [
        4.16870803e-01,
        5.64642489e-01,
        6.96135223e-01,
        8.07558119e-01,
        8.95698667e-01,
        9.58015859e-01,
        8.10941100e-01,
        8.61745656e-01,
        8.87705684e-01,
        8.88072729e-01,
        8.62836242e-01,
        8.12723756e-01,
        7.62957633e-01,
        7.46136904e-01,
        7.07804739e-01,
        6.49066269e-01,
        5.71614802e-01,
        4.77683485e-01,
        6.24985516e-01,
        5.94075382e-01,
        5.46037793e-01,
        4.82257694e-01,
        4.04573828e-01,
        3.15225929e-01,
        2.47501642e-01,
        2.10359529e-01,
        1.67152643e-01,
        1.19126678e-01,
        6.76662102e-02,
        1.42549025e-02,
        -5.01332223e-01,
        -4.66639817e-01,
        -4.18493956e-01,
        -3.58282715e-01,
        -2.87742049e-01,
        -2.08905622e-01,
        -3.90959084e-01,
        -2.91002154e-01,
        -1.82655543e-01,
        -6.90428689e-02,
        4.65603434e-02,
        1.60821185e-01,
        -2.81911224e-01,
        -1.82534873e-01,
        -7.78959990e-02,
        2.89886761e-02,
        1.35037586e-01,
        2.37193301e-01,
        -1.32215425e-01,
        -6.01096228e-02,
        1.37291476e-02,
        8.71721134e-02,
        1.58101872e-01,
        2.24473462e-01,
        2.10666791e-01,
        2.01415509e-01,
        1.86357334e-01,
        1.65926367e-01,
        1.40711680e-01,
        1.11440197e-01,
        9.01675761e-01,
        9.61834490e-01,
        9.94263113e-01,
        9.98026669e-01,
        9.73016620e-01,
        9.19954062e-01,
        8.66817176e-01,
        8.33742797e-01,
        7.76631296e-01,
        6.97129071e-01,
        5.97528338e-01,
        4.80700552e-01,
        2.72908032e-01,
        1.49916455e-01,
        2.26027388e-02,
        -1.05362624e-01,
        -2.30290353e-01,
        -3.48578691e-01,
        -2.67770439e-01,
        -3.51315200e-01,
        -4.24731404e-01,
        -4.85902399e-01,
        -5.33064604e-01,
        -5.64858317e-01,
        2.20815480e-01,
        1.80865586e-01,
        1.35701254e-01,
        8.66246074e-02,
        3.50505151e-02,
        -1.75340939e-02,
        6.61979616e-01,
        6.34321630e-01,
        5.88375807e-01,
        5.25466919e-01,
        4.47408527e-01,
        3.56451154e-01,
        7.52086282e-01,
        7.30642676e-01,
        6.88134372e-01,
        6.25786781e-01,
        5.45397460e-01,
        4.49284077e-01,
        7.30103016e-01,
        6.96126521e-01,
        6.42080426e-01,
        5.69522798e-01,
        4.80545580e-01,
        3.77714038e-01,
        4.09762740e-01,
        3.53639245e-01,
        2.87320197e-01,
        2.12717593e-01,
        1.31982207e-01,
        4.74417396e-02,
        -1.24993905e-01,
        -1.14182062e-01,
        -1.00078300e-01,
        -8.30892250e-02,
        -6.37046546e-02,
        -4.24834490e-02,
    ];
    let out = flash_attention_with_block_size(&q, &k, &v, None, true, 4, 5).expect("flash ragged");
    assert_eq!(out.shape, vec![1, 2, 10, 6]);
    assert_rel_close(&out.data, &want, 1e-5, "flash ragged causal vs numpy");
}

/// Many `(batch, head)` pairs (rayon path) must equal running each pair on its
/// own (serial path).
#[test]
fn test_flash_parallel_matches_per_head_serial() {
    let (batch, heads, seq, dim) = (2usize, 4usize, 80usize, 16usize);
    let q = tensor_f64(&[batch, heads, seq, dim], 0.031, 0.05);
    let k = tensor_f64(&[batch, heads, seq, dim], 0.037, 0.11);
    let v = tensor_f64(&[batch, heads, seq, dim], 0.041, 0.17);

    for &causal in &[false, true] {
        let full = flash_attention(&q, &k, &v, None, causal).expect("flash full");
        let unit = seq * dim;
        for bh in 0..batch * heads {
            let qs = q.data[bh * unit..(bh + 1) * unit].to_vec();
            let ks = k.data[bh * unit..(bh + 1) * unit].to_vec();
            let vs = v.data[bh * unit..(bh + 1) * unit].to_vec();
            let one = flash_attention(
                &Tensor::new(qs, vec![1, 1, seq, dim]),
                &Tensor::new(ks, vec![1, 1, seq, dim]),
                &Tensor::new(vs, vec![1, 1, seq, dim]),
                None,
                causal,
            )
            .expect("flash one head");
            assert_rel_close(
                &full.data[bh * unit..(bh + 1) * unit],
                &one.data,
                1e-5,
                &format!("flash parallel vs serial bh={bh} causal={causal}"),
            );
        }
    }
}

/// Reused block scratch with an additive mask and ragged blocks, checked
/// against the (independent) SDPA implementation.
#[test]
fn test_flash_reused_scratch_with_mask_matches_sdpa() {
    let (seq_q, seq_k, dim) = (11usize, 17usize, 5usize);
    let q = tensor_f64(&[1, 3, seq_q, dim], 0.023, 0.13);
    let k = tensor_f64(&[1, 3, seq_k, dim], 0.029, 0.19);
    let v = tensor_f64(&[1, 3, seq_k, dim], 0.019, 0.23);
    let mask = tensor_f64(&[1, 3, seq_q, seq_k], 0.011, 0.31);

    let flash =
        flash_attention_with_block_size(&q, &k, &v, Some(&mask), false, 4, 6).expect("flash mask");
    let sdpa = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None).expect("sdpa");
    assert_rel_close(&flash.data, &sdpa.data, 1e-5, "flash masked vs sdpa");

    let flash_c =
        flash_attention_with_block_size(&q, &k, &v, Some(&mask), true, 4, 6).expect("flash causal");
    let mut causal_mask = mask.data.clone();
    for h in 0..3 {
        for i in 0..seq_q {
            for j in 0..seq_k {
                if j > i {
                    causal_mask[(h * seq_q + i) * seq_k + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    let sdpa_c = scaled_dot_product_attention(
        &q,
        &k,
        &v,
        Some(&Tensor::new(causal_mask, vec![1, 3, seq_q, seq_k])),
        None,
    )
    .expect("sdpa causal");
    assert_rel_close(
        &flash_c.data,
        &sdpa_c.data,
        1e-5,
        "flash masked causal vs sdpa",
    );
}

/// A fully-masked (all `-inf`) K block leaves `m_ij` at `-inf`; with reused
/// scratch that state must not leak into the next block.
#[test]
fn test_flash_fully_masked_block_does_not_leak_state() {
    let (seq_q, seq_k, dim) = (9usize, 12usize, 4usize);
    let q = tensor_f64(&[1, 1, seq_q, dim], 0.017, 0.07);
    let k = tensor_f64(&[1, 1, seq_k, dim], 0.021, 0.11);
    let v = tensor_f64(&[1, 1, seq_k, dim], 0.013, 0.29);
    // Mask out the entire first key block (columns 0..4).
    let mut mask_data = vec![0.0f32; seq_q * seq_k];
    for i in 0..seq_q {
        for j in 0..4 {
            mask_data[i * seq_k + j] = f32::NEG_INFINITY;
        }
    }
    let mask = Tensor::new(mask_data, vec![seq_q, seq_k]);
    let flash =
        flash_attention_with_block_size(&q, &k, &v, Some(&mask), false, 4, 4).expect("flash");
    let sdpa = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None).expect("sdpa");
    assert_rel_close(&flash.data, &sdpa.data, 1e-5, "fully masked block");
}
