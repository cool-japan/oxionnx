//! Unit tests for the attention module.
#![allow(clippy::identity_op, clippy::needless_range_loop)]

use super::*;

fn assert_close(a: f32, b: f32, tol: f32, msg: &str) {
    assert!(
        (a - b).abs() < tol,
        "{msg}: got {a}, expected {b} (diff={})",
        (a - b).abs()
    );
}

fn assert_tensor_close(a: &oxionnx_core::Tensor, b: &oxionnx_core::Tensor, tol: f32, msg: &str) {
    assert_eq!(a.shape, b.shape, "{msg}: shape mismatch");
    for (i, (&av, &bv)) in a.data.iter().zip(b.data.iter()).enumerate() {
        assert!(
            (av - bv).abs() < tol,
            "{msg} at idx {i}: got {av}, expected {bv} (diff={})",
            (av - bv).abs()
        );
    }
}

#[test]
fn test_sdpa_basic() {
    use oxionnx_core::Tensor;
    let q = Tensor::new(vec![1.0; 2 * 4], vec![2, 4]);
    let k = Tensor::new(vec![1.0; 3 * 4], vec![3, 4]);
    let v = Tensor::new(
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        vec![3, 4],
    );
    let out = scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA should not fail");
    assert_eq!(out.shape, vec![2, 4]);
    let expected_avg = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0];
    for s in 0..2 {
        for (d, &expected) in expected_avg.iter().enumerate() {
            let val = out.data[s * 4 + d];
            assert!(
                (val - expected).abs() < 1e-4,
                "s={s}, d={d}: got {val}, expected {expected}",
            );
        }
    }
}

#[test]
fn test_sdpa_hand_computed_2x2() {
    use oxionnx_core::Tensor;
    let q = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let k = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let v = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let out = scaled_dot_product_attention(&q, &k, &v, None, None).expect("hand-computed SDPA");
    assert_eq!(out.shape, vec![2, 2]);
    let s = 1.0 / (2.0f32).sqrt();
    let e_s = s.exp();
    let e_0 = 1.0f32;
    let p0 = e_s / (e_s + e_0);
    let p1 = e_0 / (e_s + e_0);
    assert_close(out.data[0], p0 * 1.0 + p1 * 3.0, 1e-4, "row0 col0");
    assert_close(out.data[1], p0 * 2.0 + p1 * 4.0, 1e-4, "row0 col1");
    assert_close(out.data[2], p1 * 1.0 + p0 * 3.0, 1e-4, "row1 col0");
    assert_close(out.data[3], p1 * 2.0 + p0 * 4.0, 1e-4, "row1 col1");
}

#[test]
fn test_sdpa_with_mask() {
    use oxionnx_core::Tensor;
    let q = Tensor::new(vec![1.0; 2 * 4], vec![2, 4]);
    let k = Tensor::new(vec![1.0; 3 * 4], vec![3, 4]);
    let v = Tensor::new(
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        vec![3, 4],
    );
    let mask = Tensor::new(vec![0.0, 0.0, -1e9, 0.0, 0.0, -1e9], vec![2, 3]);
    let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None)
        .expect("SDPA with mask should not fail");
    assert_eq!(out.shape, vec![2, 4]);
    for s in 0..2 {
        assert!((out.data[s * 4] - 0.5).abs() < 1e-4);
        assert!((out.data[s * 4 + 1] - 0.5).abs() < 1e-4);
        assert!(out.data[s * 4 + 2].abs() < 1e-4);
    }
}

#[test]
fn test_sdpa_batched() {
    use oxionnx_core::Tensor;
    let q = Tensor::new(vec![0.5; 2 * 3 * 4], vec![2, 3, 4]);
    let k = Tensor::new(vec![0.5; 2 * 3 * 4], vec![2, 3, 4]);
    let v = Tensor::new(vec![1.0; 2 * 3 * 4], vec![2, 3, 4]);
    let out =
        scaled_dot_product_attention(&q, &k, &v, None, None).expect("batched SDPA should not fail");
    assert_eq!(out.shape, vec![2, 3, 4]);
    for &val in &out.data {
        assert!((val - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_sdpa_causal_mask() {
    use oxionnx_core::Tensor;
    let seq = 4;
    let d = 2;
    let q = Tensor::new(vec![1.0; seq * d], vec![seq, d]);
    let k = Tensor::new(vec![1.0; seq * d], vec![seq, d]);
    let v_data: Vec<f32> = (0..seq * d).map(|i| (i / d) as f32).collect();
    let v = Tensor::new(v_data, vec![seq, d]);
    let mut mask_data = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            mask_data[i * seq + j] = -1e9;
        }
    }
    let mask = Tensor::new(mask_data, vec![seq, seq]);
    let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None).expect("causal SDPA");
    assert_close(out.data[0], 0.0, 1e-4, "causal row0 d0");
    assert_close(out.data[1], 0.0, 1e-4, "causal row0 d1");
    let avg = (0.0 + 1.0 + 2.0 + 3.0) / 4.0;
    assert_close(out.data[6], avg, 1e-3, "causal row3 d0");
}

#[allow(clippy::identity_op)]
#[test]
fn test_multi_head_attention_no_projection() {
    use oxionnx_core::Tensor;
    let q = Tensor::new(vec![1.0; 1 * 2 * 4], vec![1, 2, 4]);
    let k = Tensor::new(vec![1.0; 1 * 2 * 4], vec![1, 2, 4]);
    let v = Tensor::new(vec![1.0; 1 * 2 * 4], vec![1, 2, 4]);
    let out = multi_head_attention(&q, &k, &v, None, None, None, None, None, 2)
        .expect("MHA should not fail");
    assert_eq!(out.shape, vec![1, 2, 4]);
    for &val in &out.data {
        assert!((val - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_mha_4heads_batch2() {
    use oxionnx_core::Tensor;
    let batch = 2;
    let seq = 3;
    let embed_dim = 8;
    let num_heads = 4;
    let q = Tensor::new(
        vec![0.3; batch * seq * embed_dim],
        vec![batch, seq, embed_dim],
    );
    let k = q.clone();
    let v = Tensor::new(
        vec![1.0; batch * seq * embed_dim],
        vec![batch, seq, embed_dim],
    );
    let out = multi_head_attention(&q, &k, &v, None, None, None, None, None, num_heads)
        .expect("MHA 4 heads batch 2");
    assert_eq!(out.shape, vec![batch, seq, embed_dim]);
    for &val in &out.data {
        assert!((val - 1.0).abs() < 1e-4, "MHA batch2: got {val}");
    }
}

#[test]
fn test_multi_head_attention_with_projection() {
    use oxionnx_core::Tensor;
    let batch = 1;
    let seq = 2;
    let embed_dim = 4;
    let num_heads = 2;
    let q = Tensor::new(
        vec![0.5; batch * seq * embed_dim],
        vec![batch, seq, embed_dim],
    );
    let k = q.clone();
    let v = q.clone();
    let mut qkv_w = vec![0.0f32; 3 * embed_dim * embed_dim];
    for i in 0..3 * embed_dim {
        qkv_w[i * embed_dim + (i % embed_dim)] = 1.0;
    }
    let qkv_weight = Tensor::new(qkv_w, vec![3 * embed_dim, embed_dim]);
    let mut out_w = vec![0.0f32; embed_dim * embed_dim];
    for i in 0..embed_dim {
        out_w[i * embed_dim + i] = 1.0;
    }
    let out_weight = Tensor::new(out_w, vec![embed_dim, embed_dim]);
    let out = multi_head_attention(
        &q,
        &k,
        &v,
        Some(&qkv_weight),
        None,
        Some(&out_weight),
        None,
        None,
        num_heads,
    )
    .expect("MHA with proj should not fail");
    assert_eq!(out.shape, vec![1, 2, 4]);
    for &val in &out.data {
        assert!((val - 0.5).abs() < 1e-4, "val={val}");
    }
}

#[test]
fn test_mqa_basic() {
    use oxionnx_core::Tensor;
    let batch = 1;
    let num_heads = 4;
    let seq = 3;
    let hd = 4;
    let q = Tensor::new(
        vec![0.5; batch * num_heads * seq * hd],
        vec![batch, num_heads, seq, hd],
    );
    let k = Tensor::new(vec![0.5; batch * 1 * seq * hd], vec![batch, 1, seq, hd]);
    let v = Tensor::new(vec![1.0; batch * 1 * seq * hd], vec![batch, 1, seq, hd]);
    let out = multi_query_attention(&q, &k, &v, None, None).expect("MQA basic");
    assert_eq!(out.shape, vec![batch, num_heads, seq, hd]);
    for &val in &out.data {
        assert!((val - 1.0).abs() < 1e-4, "MQA basic: got {val}");
    }
}

#[test]
fn test_mqa_equivalent_to_broadcast() {
    use oxionnx_core::Tensor;
    let batch = 1;
    let num_heads = 4;
    let seq = 2;
    let hd = 3;
    let q_data: Vec<f32> = (0..batch * num_heads * seq * hd)
        .map(|i| (i as f32) * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..batch * 1 * seq * hd)
        .map(|i| (i as f32) * 0.2 + 0.1)
        .collect();
    let v_data: Vec<f32> = (0..batch * 1 * seq * hd)
        .map(|i| (i as f32) * 0.15 + 0.05)
        .collect();
    let q = Tensor::new(q_data.clone(), vec![batch, num_heads, seq, hd]);
    let k = Tensor::new(k_data.clone(), vec![batch, 1, seq, hd]);
    let v = Tensor::new(v_data.clone(), vec![batch, 1, seq, hd]);
    let mqa_out = multi_query_attention(&q, &k, &v, None, None).expect("MQA broadcast test");
    for b in 0..batch {
        for h in 0..num_heads {
            let q_off = b * num_heads * seq * hd + h * seq * hd;
            let q_head = Tensor::new(q_data[q_off..q_off + seq * hd].to_vec(), vec![seq, hd]);
            let k_head = Tensor::new(k_data.clone(), vec![seq, hd]);
            let v_head = Tensor::new(v_data.clone(), vec![seq, hd]);
            let ref_out = scaled_dot_product_attention(&q_head, &k_head, &v_head, None, None)
                .expect("ref SDPA for MQA");
            let o_off = b * num_heads * seq * hd + h * seq * hd;
            for i in 0..seq * hd {
                assert_close(
                    mqa_out.data[o_off + i],
                    ref_out.data[i],
                    1e-4,
                    &format!("MQA broadcast b={b} h={h} i={i}"),
                );
            }
        }
    }
}

#[test]
fn test_mqa_error_wrong_kv_heads() {
    use oxionnx_core::Tensor;
    let q = Tensor::new(vec![0.0; 1 * 4 * 2 * 3], vec![1, 4, 2, 3]);
    let k = Tensor::new(vec![0.0; 1 * 2 * 2 * 3], vec![1, 2, 2, 3]);
    let v = Tensor::new(vec![0.0; 1 * 1 * 2 * 3], vec![1, 1, 2, 3]);
    assert!(multi_query_attention(&q, &k, &v, None, None).is_err());
}

#[test]
fn test_gqa_basic() {
    use oxionnx_core::Tensor;
    let batch = 1;
    let num_heads = 8;
    let num_kv = 2;
    let seq = 3;
    let hd = 4;
    let q = Tensor::new(
        vec![0.5; batch * num_heads * seq * hd],
        vec![batch, num_heads, seq, hd],
    );
    let k = Tensor::new(
        vec![0.5; batch * num_kv * seq * hd],
        vec![batch, num_kv, seq, hd],
    );
    let v = Tensor::new(
        vec![1.0; batch * num_kv * seq * hd],
        vec![batch, num_kv, seq, hd],
    );
    let out = grouped_query_attention(&q, &k, &v, num_kv, None, None).expect("GQA basic");
    assert_eq!(out.shape, vec![batch, num_heads, seq, hd]);
    for &val in &out.data {
        assert!((val - 1.0).abs() < 1e-4, "GQA basic: got {val}");
    }
}

#[test]
fn test_gqa_group_mapping() {
    use oxionnx_core::Tensor;
    let batch = 1;
    let num_heads = 8;
    let num_kv = 2;
    let seq = 2;
    let hd = 2;
    let q = Tensor::new(
        vec![1.0; batch * num_heads * seq * hd],
        vec![batch, num_heads, seq, hd],
    );
    let k_data = vec![1.0; batch * num_kv * seq * hd];
    let mut v_data = vec![0.0f32; batch * num_kv * seq * hd];
    for i in 0..seq * hd {
        v_data[i] = 1.0;
    }
    for i in seq * hd..2 * seq * hd {
        v_data[i] = 2.0;
    }
    let k = Tensor::new(k_data, vec![batch, num_kv, seq, hd]);
    let v = Tensor::new(v_data, vec![batch, num_kv, seq, hd]);
    let out = grouped_query_attention(&q, &k, &v, num_kv, None, None).expect("GQA group mapping");
    for h in 0..4 {
        let off = h * seq * hd;
        for i in 0..seq * hd {
            assert_close(
                out.data[off + i],
                1.0,
                1e-4,
                &format!("GQA head {h} should use KV group 0"),
            );
        }
    }
    for h in 4..8 {
        let off = h * seq * hd;
        for i in 0..seq * hd {
            assert_close(
                out.data[off + i],
                2.0,
                1e-4,
                &format!("GQA head {h} should use KV group 1"),
            );
        }
    }
}

#[test]
fn test_gqa_degenerates_to_mha() {
    use oxionnx_core::Tensor;
    let batch = 1;
    let num_heads = 4;
    let seq = 3;
    let hd = 4;
    let q_data: Vec<f32> = (0..batch * num_heads * seq * hd)
        .map(|i| (i as f32) * 0.05)
        .collect();
    let k_data: Vec<f32> = (0..batch * num_heads * seq * hd)
        .map(|i| (i as f32) * 0.03 + 0.1)
        .collect();
    let v_data: Vec<f32> = (0..batch * num_heads * seq * hd)
        .map(|i| (i as f32) * 0.02 + 0.5)
        .collect();
    let q = Tensor::new(q_data.clone(), vec![batch, num_heads, seq, hd]);
    let k = Tensor::new(k_data.clone(), vec![batch, num_heads, seq, hd]);
    let v = Tensor::new(v_data.clone(), vec![batch, num_heads, seq, hd]);
    let gqa_out =
        grouped_query_attention(&q, &k, &v, num_heads, None, None).expect("GQA degenerate MHA");
    let sdpa_out =
        scaled_dot_product_attention(&q, &k, &v, None, None).expect("SDPA for comparison");
    assert_tensor_close(&gqa_out, &sdpa_out, 1e-4, "GQA == MHA");
}

#[test]
fn test_gqa_degenerates_to_mqa() {
    use oxionnx_core::Tensor;
    let batch = 1;
    let num_heads = 4;
    let seq = 3;
    let hd = 4;
    let q_data: Vec<f32> = (0..batch * num_heads * seq * hd)
        .map(|i| (i as f32) * 0.05)
        .collect();
    let k_data: Vec<f32> = (0..batch * 1 * seq * hd)
        .map(|i| (i as f32) * 0.03 + 0.1)
        .collect();
    let v_data: Vec<f32> = (0..batch * 1 * seq * hd)
        .map(|i| (i as f32) * 0.02 + 0.5)
        .collect();
    let q = Tensor::new(q_data.clone(), vec![batch, num_heads, seq, hd]);
    let k = Tensor::new(k_data.clone(), vec![batch, 1, seq, hd]);
    let v = Tensor::new(v_data.clone(), vec![batch, 1, seq, hd]);
    let gqa_out = grouped_query_attention(&q, &k, &v, 1, None, None).expect("GQA degenerate MQA");
    let mqa_out = multi_query_attention(&q, &k, &v, None, None).expect("MQA for comparison");
    assert_tensor_close(&gqa_out, &mqa_out, 1e-6, "GQA(1) == MQA");
}

#[test]
fn test_gqa_error_indivisible() {
    use oxionnx_core::Tensor;
    let q = Tensor::new(vec![0.0; 1 * 5 * 2 * 3], vec![1, 5, 2, 3]);
    let k = Tensor::new(vec![0.0; 1 * 3 * 2 * 3], vec![1, 3, 2, 3]);
    let v = Tensor::new(vec![0.0; 1 * 3 * 2 * 3], vec![1, 3, 2, 3]);
    assert!(grouped_query_attention(&q, &k, &v, 3, None, None).is_err());
}

#[test]
fn test_alibi_slopes() {
    let num_heads = 4;
    let slopes: Vec<f32> = (0..num_heads)
        .map(|h| 2.0f32.powf(-8.0 * (h as f32 + 1.0) / num_heads as f32))
        .collect();
    assert_close(slopes[0], 0.25, 1e-6, "slope 0");
    assert_close(slopes[1], 0.0625, 1e-6, "slope 1");
    assert_close(slopes[2], 0.015625, 1e-6, "slope 2");
    assert_close(slopes[3], 0.00390625, 1e-6, "slope 3");
}

#[test]
fn test_alibi_attention_basic() {
    use oxionnx_core::Tensor;
    let batch = 1;
    let num_heads = 2;
    let seq = 4;
    let hd = 4;
    let q = Tensor::new(
        vec![0.5; batch * num_heads * seq * hd],
        vec![batch, num_heads, seq, hd],
    );
    let k = Tensor::new(
        vec![0.5; batch * num_heads * seq * hd],
        vec![batch, num_heads, seq, hd],
    );
    let v = Tensor::new(
        vec![1.0; batch * num_heads * seq * hd],
        vec![batch, num_heads, seq, hd],
    );
    let out = alibi_attention(&q, &k, &v, num_heads, None).expect("ALiBi basic");
    assert_eq!(out.shape, vec![batch, num_heads, seq, hd]);
    for &val in &out.data {
        assert!((val - 1.0).abs() < 1e-3, "ALiBi basic: got {val}");
    }
}

#[test]
fn test_alibi_bias_pattern() {
    use oxionnx_core::Tensor;
    let batch = 1;
    let num_heads = 1;
    let seq = 4;
    let hd = 2;
    let q = Tensor::new(
        vec![1.0; batch * num_heads * seq * hd],
        vec![batch, num_heads, seq, hd],
    );
    let k = q.clone();
    let mut v_data = vec![0.0f32; batch * num_heads * seq * hd];
    for s in 0..seq {
        for d in 0..hd {
            v_data[s * hd + d] = s as f32;
        }
    }
    let v = Tensor::new(v_data, vec![batch, num_heads, seq, hd]);
    let out = alibi_attention(&q, &k, &v, num_heads, None).expect("ALiBi bias pattern");
    let out_pos0 = out.data[0];
    let out_pos3 = out.data[3 * hd];
    assert!(
        out_pos3 > out_pos0,
        "ALiBi: pos3 output ({out_pos3}) should be > pos0 output ({out_pos0})"
    );
}

#[test]
fn test_alibi_with_mask() {
    use oxionnx_core::Tensor;
    let batch = 1;
    let num_heads = 2;
    let seq = 3;
    let hd = 2;
    let q = Tensor::new(
        vec![1.0; batch * num_heads * seq * hd],
        vec![batch, num_heads, seq, hd],
    );
    let k = q.clone();
    let v = Tensor::new(
        vec![1.0; batch * num_heads * seq * hd],
        vec![batch, num_heads, seq, hd],
    );
    let mut mask_data = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            mask_data[i * seq + j] = -1e9;
        }
    }
    let mask = Tensor::new(mask_data, vec![seq, seq]);
    let out = alibi_attention(&q, &k, &v, num_heads, Some(&mask)).expect("ALiBi + mask");
    assert_eq!(out.shape, vec![batch, num_heads, seq, hd]);
    for &val in &out.data {
        assert!(!val.is_nan(), "ALiBi+mask produced NaN");
    }
}

#[test]
fn test_sdpa_seq_len_1() {
    use oxionnx_core::Tensor;
    let q = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
    let k = Tensor::new(vec![3.0, 4.0], vec![1, 2]);
    let v = Tensor::new(vec![5.0, 6.0], vec![1, 2]);
    let out = scaled_dot_product_attention(&q, &k, &v, None, None).expect("seq_len=1");
    assert_eq!(out.shape, vec![1, 2]);
    assert_close(out.data[0], 5.0, 1e-5, "seq1 d0");
    assert_close(out.data[1], 6.0, 1e-5, "seq1 d1");
}

#[test]
fn test_gqa_head_dim_1() {
    use oxionnx_core::Tensor;
    let batch = 1;
    let num_heads = 4;
    let num_kv = 2;
    let seq = 2;
    let hd = 1;
    let q = Tensor::new(
        vec![1.0; batch * num_heads * seq * hd],
        vec![batch, num_heads, seq, hd],
    );
    let k = Tensor::new(
        vec![1.0; batch * num_kv * seq * hd],
        vec![batch, num_kv, seq, hd],
    );
    let v = Tensor::new(
        vec![2.0; batch * num_kv * seq * hd],
        vec![batch, num_kv, seq, hd],
    );
    let out = grouped_query_attention(&q, &k, &v, num_kv, None, None).expect("head_dim=1");
    assert_eq!(out.shape, vec![batch, num_heads, seq, hd]);
    for &val in &out.data {
        assert_close(val, 2.0, 1e-4, "head_dim=1");
    }
}

#[test]
fn test_mqa_batch_1_seq_1() {
    use oxionnx_core::Tensor;
    let q = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 1, 2]);
    let k = Tensor::new(vec![0.5, 0.5], vec![1, 1, 1, 2]);
    let v = Tensor::new(vec![10.0, 20.0], vec![1, 1, 1, 2]);
    let out = multi_query_attention(&q, &k, &v, None, None).expect("batch1 seq1 MQA");
    assert_eq!(out.shape, vec![1, 2, 1, 2]);
    for h in 0..2 {
        let off = h * 1 * 2;
        assert_close(out.data[off], 10.0, 1e-4, "b1s1 V0");
        assert_close(out.data[off + 1], 20.0, 1e-4, "b1s1 V1");
    }
}

#[test]
fn test_sdpa_large_values_no_nan() {
    use oxionnx_core::Tensor;
    let seq = 4;
    let d = 4;
    let data: Vec<f32> = (0..seq * d).map(|i| (i as f32) * 25.0).collect();
    let q = Tensor::new(data.clone(), vec![seq, d]);
    let k = Tensor::new(data.clone(), vec![seq, d]);
    let v = Tensor::new(vec![1.0; seq * d], vec![seq, d]);
    let out = scaled_dot_product_attention(&q, &k, &v, None, None).expect("large values SDPA");
    for &val in &out.data {
        assert!(!val.is_nan(), "Large values produced NaN");
        assert!(!val.is_infinite(), "Large values produced Inf");
    }
}

#[test]
fn test_gqa_large_values_no_nan() {
    use oxionnx_core::Tensor;
    let batch = 1;
    let num_heads = 4;
    let num_kv = 2;
    let seq = 3;
    let hd = 4;
    let q_data: Vec<f32> = (0..batch * num_heads * seq * hd)
        .map(|i| ((i % 7) as f32) * 14.0 + 2.0)
        .collect();
    let k_data: Vec<f32> = (0..batch * num_kv * seq * hd)
        .map(|i| ((i % 5) as f32) * 20.0)
        .collect();
    let v_data: Vec<f32> = (0..batch * num_kv * seq * hd)
        .map(|i| ((i % 3) as f32) * 33.0 + 1.0)
        .collect();
    let q = Tensor::new(q_data, vec![batch, num_heads, seq, hd]);
    let k = Tensor::new(k_data, vec![batch, num_kv, seq, hd]);
    let v = Tensor::new(v_data, vec![batch, num_kv, seq, hd]);
    let out = grouped_query_attention(&q, &k, &v, num_kv, None, None).expect("GQA large values");
    for &val in &out.data {
        assert!(!val.is_nan(), "GQA large values produced NaN");
        assert!(!val.is_infinite(), "GQA large values produced Inf");
    }
}

#[test]
fn test_alibi_large_values_no_nan() {
    use oxionnx_core::Tensor;
    let batch = 1;
    let num_heads = 4;
    let seq = 4;
    let hd = 4;
    let data: Vec<f32> = (0..batch * num_heads * seq * hd)
        .map(|i| ((i % 10) as f32) * 10.0)
        .collect();
    let q = Tensor::new(data.clone(), vec![batch, num_heads, seq, hd]);
    let k = Tensor::new(data.clone(), vec![batch, num_heads, seq, hd]);
    let v = Tensor::new(
        vec![1.0; batch * num_heads * seq * hd],
        vec![batch, num_heads, seq, hd],
    );
    let out = alibi_attention(&q, &k, &v, num_heads, None).expect("ALiBi large values");
    for &val in &out.data {
        assert!(!val.is_nan(), "ALiBi large values produced NaN");
        assert!(!val.is_infinite(), "ALiBi large values produced Inf");
    }
}

#[test]
fn test_rotary_embedding_basic() {
    use oxionnx_core::Tensor;
    let input = Tensor::new(
        vec![
            1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        ],
        vec![1, 4, 4],
    );
    let pos_ids = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]);
    let out =
        rotary_embedding(&input, &pos_ids, None, None, 10000.0).expect("RoPE should not fail");
    assert_eq!(out.shape, vec![1, 4, 4]);
    assert!((out.data[0] - 1.0).abs() < 1e-5);
    assert!((out.data[1] - 0.0).abs() < 1e-5);
    assert!((out.data[2] - 0.0).abs() < 1e-5);
    assert!((out.data[3] - 1.0).abs() < 1e-5);
    let p1 = &out.data[4..8];
    assert!(
        (p1[0] - 1.0).abs() > 1e-3 || (p1[2] - 0.0).abs() > 1e-3,
        "RoPE should modify values at position 1"
    );
}

#[test]
fn test_rotary_embedding_preserves_norm() {
    use oxionnx_core::Tensor;
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]);
    let pos_ids = Tensor::new(vec![5.0], vec![1, 1]);
    let out = rotary_embedding(&input, &pos_ids, None, None, 10000.0).expect("RoPE norm test");
    let in_norm: f32 = input.data.iter().map(|x| x * x).sum::<f32>().sqrt();
    let out_norm: f32 = out.data.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (in_norm - out_norm).abs() < 1e-4,
        "RoPE should preserve norm: in={in_norm}, out={out_norm}"
    );
}
