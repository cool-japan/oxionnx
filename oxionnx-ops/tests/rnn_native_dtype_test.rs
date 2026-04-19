//! Tests for LSTMOp and GRUOp native typed dispatch (F16/BF16) — Phase D.3.
//!
//! Validates:
//!  - `native_dtypes()` includes DType::F32, F16, BF16.
//!  - F32 typed path returns results matching the reference f32 kernel.
//!  - F16 and BF16 typed paths match the f32 reference within tolerances.
//!
//! Tolerances: F16 ≤ 1e-2, BF16 ≤ 5e-2.

use oxionnx_core::{
    dtype::{DType, TensorStorage, TypedTensor},
    graph::{Attributes, Node, OpKind},
    operator::{Operator, TypedOpContext},
};
use oxionnx_ops::registry::rnn_ops::{GRUOp, LSTMOp};

// ── Conversion helpers ────────────────────────────────────────────────────────

fn f32_to_f16_bits(vals: &[f32]) -> Vec<u16> {
    vals.iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect()
}

fn f32_to_bf16_bits(vals: &[f32]) -> Vec<u16> {
    vals.iter()
        .map(|&x| half::bf16::from_f32(x).to_bits())
        .collect()
}

fn f16_bits_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect()
}

fn bf16_bits_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| half::bf16::from_bits(b).to_f32())
        .collect()
}

// ── Node builders ─────────────────────────────────────────────────────────────

fn lstm_node(hidden_size: usize) -> Node {
    let mut attrs = Attributes::default();
    attrs.ints.insert("hidden_size".into(), hidden_size as i64);
    Node {
        name: "test_lstm".into(),
        op: OpKind::LSTM,
        inputs: vec![],
        outputs: vec![],
        attrs,
    }
}

fn gru_node(hidden_size: usize) -> Node {
    let mut attrs = Attributes::default();
    attrs.ints.insert("hidden_size".into(), hidden_size as i64);
    Node {
        name: "test_gru".into(),
        op: OpKind::GRU,
        inputs: vec![],
        outputs: vec![],
        attrs,
    }
}

// ── TypedOpContext builders ───────────────────────────────────────────────────

fn make_lstm_ctx<'a>(
    node: &'a Node,
    x: &'a TypedTensor,
    w: &'a TypedTensor,
    r: &'a TypedTensor,
) -> TypedOpContext<'a> {
    // Inputs 0=X, 1=W, 2=R; optional 3..7 are None (no bias, seq_lens, h0, c0, peephole).
    TypedOpContext {
        node,
        inputs: vec![Some(x), Some(w), Some(r), None, None, None, None, None],
        outer_scope: None,
        registry: None,
    }
}

fn make_gru_ctx<'a>(
    node: &'a Node,
    x: &'a TypedTensor,
    w: &'a TypedTensor,
    r: &'a TypedTensor,
) -> TypedOpContext<'a> {
    // Inputs 0=X, 1=W, 2=R; optional 3..5 are None (no bias, seq_lens, h0).
    TypedOpContext {
        node,
        inputs: vec![Some(x), Some(w), Some(r), None, None, None],
        outer_scope: None,
        registry: None,
    }
}

// ── Assert helpers ────────────────────────────────────────────────────────────

fn assert_slices_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        assert!(
            diff <= tol,
            "{label}: index {i}: |{av} - {bv}| = {diff} > tol {tol}"
        );
    }
}

// ── Small tensor generators ───────────────────────────────────────────────────

/// Generate a deterministic small f32 vector of given length.
/// Values cycle through 0.1, 0.2, ..., 0.9, 0.1, ... to avoid large magnitudes.
fn small_f32(len: usize) -> Vec<f32> {
    (0..len).map(|i| ((i % 9) as f32 + 1.0) * 0.1).collect()
}

// ── LSTM tests ────────────────────────────────────────────────────────────────

#[test]
fn test_lstm_native_dtypes_includes_all_three() {
    let dtypes = LSTMOp.native_dtypes();
    assert!(
        dtypes.contains(&DType::F32),
        "LSTMOp must advertise DType::F32"
    );
    assert!(
        dtypes.contains(&DType::F16),
        "LSTMOp must advertise DType::F16"
    );
    assert!(
        dtypes.contains(&DType::BF16),
        "LSTMOp must advertise DType::BF16"
    );
}

/// F32 baseline: execute_typed on F32 tensors must return 3 output tensors
/// with shapes matching the ONNX LSTM spec.
#[test]
fn test_lstm_f32_baseline_parity() {
    let seq = 4_usize;
    let batch = 1_usize;
    let input_size = 8_usize;
    let hidden_size = 16_usize;

    // X: [seq, batch, input_size]
    let x_data = small_f32(seq * batch * input_size);
    // W: [1, 4*hidden_size, input_size]
    let w_data = small_f32(4 * hidden_size * input_size);
    // R: [1, 4*hidden_size, hidden_size]
    let r_data = small_f32(4 * hidden_size * hidden_size);

    let x = TypedTensor::new(TensorStorage::F32(x_data), vec![seq, batch, input_size]);
    let w = TypedTensor::new(
        TensorStorage::F32(w_data),
        vec![1, 4 * hidden_size, input_size],
    );
    let r = TypedTensor::new(
        TensorStorage::F32(r_data),
        vec![1, 4 * hidden_size, hidden_size],
    );

    let node = lstm_node(hidden_size);
    let ctx = make_lstm_ctx(&node, &x, &w, &r);

    let outputs = LSTMOp
        .execute_typed(&ctx)
        .expect("LSTM F32 execute_typed failed");

    assert_eq!(outputs.len(), 3, "LSTM must return 3 outputs");
    // Y: [seq, 1, batch, hidden_size]
    assert_eq!(
        outputs[0].shape,
        vec![seq, 1, batch, hidden_size],
        "Y shape mismatch"
    );
    // Y_h: [1, batch, hidden_size]
    assert_eq!(
        outputs[1].shape,
        vec![1, batch, hidden_size],
        "Y_h shape mismatch"
    );
    // Y_c: [1, batch, hidden_size]
    assert_eq!(
        outputs[2].shape,
        vec![1, batch, hidden_size],
        "Y_c shape mismatch"
    );
}

/// F16 parity: F16 execute_typed results must be close to F32 execute_typed.
#[test]
fn test_lstm_f16_parity() {
    let seq = 4_usize;
    let batch = 1_usize;
    let input_size = 8_usize;
    let hidden_size = 16_usize;

    let x_f32 = small_f32(seq * batch * input_size);
    let w_f32 = small_f32(4 * hidden_size * input_size);
    let r_f32 = small_f32(4 * hidden_size * hidden_size);

    // F32 reference
    let x_ref = TypedTensor::new(
        TensorStorage::F32(x_f32.clone()),
        vec![seq, batch, input_size],
    );
    let w_ref = TypedTensor::new(
        TensorStorage::F32(w_f32.clone()),
        vec![1, 4 * hidden_size, input_size],
    );
    let r_ref = TypedTensor::new(
        TensorStorage::F32(r_f32.clone()),
        vec![1, 4 * hidden_size, hidden_size],
    );

    let node = lstm_node(hidden_size);
    let ctx_ref = make_lstm_ctx(&node, &x_ref, &w_ref, &r_ref);
    let ref_out = LSTMOp.execute_typed(&ctx_ref).expect("LSTM F32 ref failed");
    let ref_y = match &ref_out[0].storage {
        TensorStorage::F32(d) => d.clone(),
        _ => panic!("F32 path must return F32 storage"),
    };
    let ref_yh = match &ref_out[1].storage {
        TensorStorage::F32(d) => d.clone(),
        _ => panic!("F32 path must return F32 storage"),
    };

    // F16 typed path
    let x_f16 = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&x_f32)),
        vec![seq, batch, input_size],
    );
    let w_f16 = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&w_f32)),
        vec![1, 4 * hidden_size, input_size],
    );
    let r_f16 = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&r_f32)),
        vec![1, 4 * hidden_size, hidden_size],
    );

    let ctx_f16 = make_lstm_ctx(&node, &x_f16, &w_f16, &r_f16);
    let f16_out = LSTMOp
        .execute_typed(&ctx_f16)
        .expect("LSTM F16 execute_typed failed");

    assert_eq!(f16_out.len(), 3, "LSTM F16 must return 3 outputs");
    assert_eq!(f16_out[0].shape, vec![seq, 1, batch, hidden_size]);

    let got_y = match &f16_out[0].storage {
        TensorStorage::F16(bits) => f16_bits_to_f32(bits),
        _ => panic!("F16 path must return F16 storage for Y"),
    };
    let got_yh = match &f16_out[1].storage {
        TensorStorage::F16(bits) => f16_bits_to_f32(bits),
        _ => panic!("F16 path must return F16 storage for Y_h"),
    };

    assert_slices_close(&ref_y, &got_y, 1e-2, "LSTM F16 Y");
    assert_slices_close(&ref_yh, &got_yh, 1e-2, "LSTM F16 Y_h");
}

/// BF16 parity: BF16 execute_typed results must be close to F32 within 5e-2.
#[test]
fn test_lstm_bf16_parity() {
    let seq = 4_usize;
    let batch = 1_usize;
    let input_size = 8_usize;
    let hidden_size = 16_usize;

    let x_f32 = small_f32(seq * batch * input_size);
    let w_f32 = small_f32(4 * hidden_size * input_size);
    let r_f32 = small_f32(4 * hidden_size * hidden_size);

    // F32 reference
    let x_ref = TypedTensor::new(
        TensorStorage::F32(x_f32.clone()),
        vec![seq, batch, input_size],
    );
    let w_ref = TypedTensor::new(
        TensorStorage::F32(w_f32.clone()),
        vec![1, 4 * hidden_size, input_size],
    );
    let r_ref = TypedTensor::new(
        TensorStorage::F32(r_f32.clone()),
        vec![1, 4 * hidden_size, hidden_size],
    );

    let node = lstm_node(hidden_size);
    let ctx_ref = make_lstm_ctx(&node, &x_ref, &w_ref, &r_ref);
    let ref_out = LSTMOp.execute_typed(&ctx_ref).expect("LSTM F32 ref failed");
    let ref_y = match &ref_out[0].storage {
        TensorStorage::F32(d) => d.clone(),
        _ => panic!("F32 path must return F32"),
    };
    let ref_yh = match &ref_out[1].storage {
        TensorStorage::F32(d) => d.clone(),
        _ => panic!("F32 path must return F32"),
    };

    // BF16 typed path
    let x_bf16 = TypedTensor::new(
        TensorStorage::BF16(f32_to_bf16_bits(&x_f32)),
        vec![seq, batch, input_size],
    );
    let w_bf16 = TypedTensor::new(
        TensorStorage::BF16(f32_to_bf16_bits(&w_f32)),
        vec![1, 4 * hidden_size, input_size],
    );
    let r_bf16 = TypedTensor::new(
        TensorStorage::BF16(f32_to_bf16_bits(&r_f32)),
        vec![1, 4 * hidden_size, hidden_size],
    );

    let ctx_bf16 = make_lstm_ctx(&node, &x_bf16, &w_bf16, &r_bf16);
    let bf16_out = LSTMOp
        .execute_typed(&ctx_bf16)
        .expect("LSTM BF16 execute_typed failed");

    assert_eq!(bf16_out.len(), 3, "LSTM BF16 must return 3 outputs");

    let got_y = match &bf16_out[0].storage {
        TensorStorage::BF16(bits) => bf16_bits_to_f32(bits),
        _ => panic!("BF16 path must return BF16 storage for Y"),
    };
    let got_yh = match &bf16_out[1].storage {
        TensorStorage::BF16(bits) => bf16_bits_to_f32(bits),
        _ => panic!("BF16 path must return BF16 storage for Y_h"),
    };

    assert_slices_close(&ref_y, &got_y, 5e-2, "LSTM BF16 Y");
    assert_slices_close(&ref_yh, &got_yh, 5e-2, "LSTM BF16 Y_h");
}

// ── GRU tests ─────────────────────────────────────────────────────────────────

#[test]
fn test_gru_native_dtypes_includes_all_three() {
    let dtypes = GRUOp.native_dtypes();
    assert!(
        dtypes.contains(&DType::F32),
        "GRUOp must advertise DType::F32"
    );
    assert!(
        dtypes.contains(&DType::F16),
        "GRUOp must advertise DType::F16"
    );
    assert!(
        dtypes.contains(&DType::BF16),
        "GRUOp must advertise DType::BF16"
    );
}

/// GRU F16 parity test.
#[test]
fn test_gru_f16_parity() {
    let seq = 4_usize;
    let batch = 1_usize;
    let input_size = 8_usize;
    let hidden_size = 16_usize;

    let x_f32 = small_f32(seq * batch * input_size);
    // W: [1, 3*hidden_size, input_size]
    let w_f32 = small_f32(3 * hidden_size * input_size);
    // R: [1, 3*hidden_size, hidden_size]
    let r_f32 = small_f32(3 * hidden_size * hidden_size);

    // F32 reference
    let x_ref = TypedTensor::new(
        TensorStorage::F32(x_f32.clone()),
        vec![seq, batch, input_size],
    );
    let w_ref = TypedTensor::new(
        TensorStorage::F32(w_f32.clone()),
        vec![1, 3 * hidden_size, input_size],
    );
    let r_ref = TypedTensor::new(
        TensorStorage::F32(r_f32.clone()),
        vec![1, 3 * hidden_size, hidden_size],
    );

    let node = gru_node(hidden_size);
    let ctx_ref = make_gru_ctx(&node, &x_ref, &w_ref, &r_ref);
    let ref_out = GRUOp.execute_typed(&ctx_ref).expect("GRU F32 ref failed");
    assert_eq!(ref_out.len(), 2, "GRU must return 2 outputs");

    let ref_y = match &ref_out[0].storage {
        TensorStorage::F32(d) => d.clone(),
        _ => panic!("F32 path must return F32 storage"),
    };
    let ref_yh = match &ref_out[1].storage {
        TensorStorage::F32(d) => d.clone(),
        _ => panic!("F32 path must return F32 storage"),
    };

    // F16 typed path
    let x_f16 = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&x_f32)),
        vec![seq, batch, input_size],
    );
    let w_f16 = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&w_f32)),
        vec![1, 3 * hidden_size, input_size],
    );
    let r_f16 = TypedTensor::new(
        TensorStorage::F16(f32_to_f16_bits(&r_f32)),
        vec![1, 3 * hidden_size, hidden_size],
    );

    let ctx_f16 = make_gru_ctx(&node, &x_f16, &w_f16, &r_f16);
    let f16_out = GRUOp
        .execute_typed(&ctx_f16)
        .expect("GRU F16 execute_typed failed");

    assert_eq!(f16_out.len(), 2, "GRU F16 must return 2 outputs");
    assert_eq!(f16_out[0].shape, vec![seq, 1, batch, hidden_size]);

    let got_y = match &f16_out[0].storage {
        TensorStorage::F16(bits) => f16_bits_to_f32(bits),
        _ => panic!("F16 path must return F16 storage for Y"),
    };
    let got_yh = match &f16_out[1].storage {
        TensorStorage::F16(bits) => f16_bits_to_f32(bits),
        _ => panic!("F16 path must return F16 storage for Y_h"),
    };

    assert_slices_close(&ref_y, &got_y, 1e-2, "GRU F16 Y");
    assert_slices_close(&ref_yh, &got_yh, 1e-2, "GRU F16 Y_h");
}

/// GRU BF16 parity test.
#[test]
fn test_gru_bf16_parity() {
    let seq = 4_usize;
    let batch = 1_usize;
    let input_size = 8_usize;
    let hidden_size = 16_usize;

    let x_f32 = small_f32(seq * batch * input_size);
    let w_f32 = small_f32(3 * hidden_size * input_size);
    let r_f32 = small_f32(3 * hidden_size * hidden_size);

    // F32 reference
    let x_ref = TypedTensor::new(
        TensorStorage::F32(x_f32.clone()),
        vec![seq, batch, input_size],
    );
    let w_ref = TypedTensor::new(
        TensorStorage::F32(w_f32.clone()),
        vec![1, 3 * hidden_size, input_size],
    );
    let r_ref = TypedTensor::new(
        TensorStorage::F32(r_f32.clone()),
        vec![1, 3 * hidden_size, hidden_size],
    );

    let node = gru_node(hidden_size);
    let ctx_ref = make_gru_ctx(&node, &x_ref, &w_ref, &r_ref);
    let ref_out = GRUOp.execute_typed(&ctx_ref).expect("GRU F32 ref failed");

    let ref_y = match &ref_out[0].storage {
        TensorStorage::F32(d) => d.clone(),
        _ => panic!("F32 path must return F32"),
    };
    let ref_yh = match &ref_out[1].storage {
        TensorStorage::F32(d) => d.clone(),
        _ => panic!("F32 path must return F32"),
    };

    // BF16 typed path
    let x_bf16 = TypedTensor::new(
        TensorStorage::BF16(f32_to_bf16_bits(&x_f32)),
        vec![seq, batch, input_size],
    );
    let w_bf16 = TypedTensor::new(
        TensorStorage::BF16(f32_to_bf16_bits(&w_f32)),
        vec![1, 3 * hidden_size, input_size],
    );
    let r_bf16 = TypedTensor::new(
        TensorStorage::BF16(f32_to_bf16_bits(&r_f32)),
        vec![1, 3 * hidden_size, hidden_size],
    );

    let ctx_bf16 = make_gru_ctx(&node, &x_bf16, &w_bf16, &r_bf16);
    let bf16_out = GRUOp
        .execute_typed(&ctx_bf16)
        .expect("GRU BF16 execute_typed failed");

    assert_eq!(bf16_out.len(), 2, "GRU BF16 must return 2 outputs");

    let got_y = match &bf16_out[0].storage {
        TensorStorage::BF16(bits) => bf16_bits_to_f32(bits),
        _ => panic!("BF16 path must return BF16 storage for Y"),
    };
    let got_yh = match &bf16_out[1].storage {
        TensorStorage::BF16(bits) => bf16_bits_to_f32(bits),
        _ => panic!("BF16 path must return BF16 storage for Y_h"),
    };

    assert_slices_close(&ref_y, &got_y, 5e-2, "GRU BF16 Y");
    assert_slices_close(&ref_yh, &got_yh, 5e-2, "GRU BF16 Y_h");
}
