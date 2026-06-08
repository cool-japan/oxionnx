//! Output-slot correctness tests for RNN ops (F.12 W2.2):
//! LSTMOp, GRUOp — zero-copy slot reuse with pointer stability checks.

use oxionnx_core::operator::Operator;
use oxionnx_core::{
    graph::{Attributes, Node, OpKind},
    operator::OpContext,
    Tensor,
};
use oxionnx_ops::registry::rnn_ops::{GRUOp, LSTMOp};

// ── Test infrastructure ──────────────────────────────────────────────────────

fn make_ctx<'a>(node: &'a Node, inputs: Vec<Option<&'a Tensor>>) -> OpContext<'a> {
    OpContext {
        node,
        inputs,
        outer_scope: None,
        weights: None,
        registry: None,
    }
}

fn dummy_node(op: OpKind) -> Node {
    Node {
        name: "test".into(),
        op,
        inputs: Vec::new(),
        outputs: Vec::new(),
        attrs: Attributes::default(),
    }
}

fn node_with_int_attrs(op: OpKind, pairs: &[(&str, i64)]) -> Node {
    let mut n = dummy_node(op);
    for &(k, v) in pairs {
        n.attrs.ints.insert(k.to_string(), v);
    }
    n
}

fn assert_near(a: &Tensor, b: &Tensor, label: &str, tol: f32) {
    assert_eq!(a.shape, b.shape, "{label}: shape mismatch");
    assert_eq!(a.data.len(), b.data.len(), "{label}: data len mismatch");
    for (i, (&av, &bv)) in a.data.iter().zip(b.data.iter()).enumerate() {
        assert!(
            (av - bv).abs() <= tol || (av.is_nan() && bv.is_nan()),
            "{label}[{i}]: got {av}, expected {bv} (tol={tol})",
        );
    }
}

fn lstm_node(hidden_size: i64) -> Node {
    node_with_int_attrs(OpKind::LSTM, &[("hidden_size", hidden_size)])
}

fn gru_node(hidden_size: i64) -> Node {
    node_with_int_attrs(OpKind::GRU, &[("hidden_size", hidden_size)])
}

// ── F.12 W2.2 tests: LSTM and GRU zero-copy slots ─────────────────────────────

#[test]
fn test_lstm_slot_reuse() {
    // input [seq=4, batch=1, input_size=8], hidden_size=16, forward
    let seq: usize = 4;
    let batch: usize = 1;
    let input_size: usize = 8;
    let hidden_size: usize = 16;
    let num_dir: usize = 1;

    let x_len = seq * batch * input_size;
    let x = Tensor::new(
        (0..x_len)
            .map(|v| (v as f32 - x_len as f32 / 2.0) * 0.1)
            .collect(),
        vec![seq, batch, input_size],
    );

    // W: [1, 4*hs=64, input_size=8]
    let w_len = num_dir * 4 * hidden_size * input_size;
    let w = Tensor::new(
        (0..w_len)
            .map(|v| (v as f32 - w_len as f32 / 2.0) * 0.02)
            .collect(),
        vec![num_dir, 4 * hidden_size, input_size],
    );

    // R: [1, 4*hs=64, hs=16]
    let r_len = num_dir * 4 * hidden_size * hidden_size;
    let r = Tensor::new(
        (0..r_len)
            .map(|v| (v as f32 - r_len as f32 / 2.0) * 0.01)
            .collect(),
        vec![num_dir, 4 * hidden_size, hidden_size],
    );

    let node = lstm_node(hidden_size as i64);
    // inputs: x, W, R, b=None, seq_lens=None, initial_h=None, initial_c=None, peephole=None
    let ctx = make_ctx(
        &node,
        vec![Some(&x), Some(&w), Some(&r), None, None, None, None, None],
    );

    // Reference via execute()
    let expected = LSTMOp.execute(&ctx).expect("LSTMOp execute failed");
    assert_eq!(expected.len(), 3);
    assert_eq!(expected[0].shape, vec![seq, num_dir, batch, hidden_size]);
    assert_eq!(expected[1].shape, vec![num_dir, batch, hidden_size]);
    assert_eq!(expected[2].shape, vec![num_dir, batch, hidden_size]);

    assert!(
        LSTMOp.supports_output_slots(),
        "LSTMOp must support output slots"
    );

    // Pre-allocate 3 slots with correct buffer sizes.
    let y_len = seq * num_dir * batch * hidden_size;
    let yh_len = num_dir * batch * hidden_size;
    let mut slots = vec![
        Tensor::new(vec![0.0_f32; y_len], vec![seq, num_dir, batch, hidden_size]),
        Tensor::new(vec![0.0_f32; yh_len], vec![num_dir, batch, hidden_size]),
        Tensor::new(vec![0.0_f32; yh_len], vec![num_dir, batch, hidden_size]),
    ];

    // First call
    LSTMOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("LSTMOp execute_into_slots first call failed");

    assert_near(&slots[0], &expected[0], "lstm Y first call", 1e-5);
    assert_near(&slots[1], &expected[1], "lstm Y_h first call", 1e-5);
    assert_near(&slots[2], &expected[2], "lstm Y_c first call", 1e-5);

    // Record raw pointers for all 3 slots.
    let ptr_y = slots[0].data.as_ptr() as usize;
    let ptr_yh = slots[1].data.as_ptr() as usize;
    let ptr_yc = slots[2].data.as_ptr() as usize;

    // Second call: same input — no reallocation should occur.
    LSTMOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("LSTMOp execute_into_slots second call failed");

    assert_eq!(
        slots[0].data.as_ptr() as usize,
        ptr_y,
        "lstm Y slot pointer must be stable on same-shape second call"
    );
    assert_eq!(
        slots[1].data.as_ptr() as usize,
        ptr_yh,
        "lstm Y_h slot pointer must be stable on same-shape second call"
    );
    assert_eq!(
        slots[2].data.as_ptr() as usize,
        ptr_yc,
        "lstm Y_c slot pointer must be stable on same-shape second call"
    );

    assert_near(&slots[0], &expected[0], "lstm Y second call", 1e-5);
    assert_near(&slots[1], &expected[1], "lstm Y_h second call", 1e-5);
    assert_near(&slots[2], &expected[2], "lstm Y_c second call", 1e-5);
}

#[test]
fn test_gru_slot_reuse() {
    // input [seq=4, batch=1, input_size=8], hidden_size=16, forward
    let seq: usize = 4;
    let batch: usize = 1;
    let input_size: usize = 8;
    let hidden_size: usize = 16;
    let num_dir: usize = 1;

    let x_len = seq * batch * input_size;
    let x = Tensor::new(
        (0..x_len)
            .map(|v| (v as f32 - x_len as f32 / 2.0) * 0.1)
            .collect(),
        vec![seq, batch, input_size],
    );

    // W: [1, 3*hs=48, input_size=8]
    let w_len = num_dir * 3 * hidden_size * input_size;
    let w = Tensor::new(
        (0..w_len)
            .map(|v| (v as f32 - w_len as f32 / 2.0) * 0.02)
            .collect(),
        vec![num_dir, 3 * hidden_size, input_size],
    );

    // R: [1, 3*hs=48, hs=16]
    let r_len = num_dir * 3 * hidden_size * hidden_size;
    let r = Tensor::new(
        (0..r_len)
            .map(|v| (v as f32 - r_len as f32 / 2.0) * 0.01)
            .collect(),
        vec![num_dir, 3 * hidden_size, hidden_size],
    );

    let node = gru_node(hidden_size as i64);
    // inputs: x, W, R, b=None, seq_lens=None, initial_h=None
    let ctx = make_ctx(&node, vec![Some(&x), Some(&w), Some(&r), None, None, None]);

    // Reference via execute()
    let expected = GRUOp.execute(&ctx).expect("GRUOp execute failed");
    assert_eq!(expected.len(), 2);
    assert_eq!(expected[0].shape, vec![seq, num_dir, batch, hidden_size]);
    assert_eq!(expected[1].shape, vec![num_dir, batch, hidden_size]);

    assert!(
        GRUOp.supports_output_slots(),
        "GRUOp must support output slots"
    );

    // Pre-allocate 2 slots with correct buffer sizes.
    let y_len = seq * num_dir * batch * hidden_size;
    let yh_len = num_dir * batch * hidden_size;
    let mut slots = vec![
        Tensor::new(vec![0.0_f32; y_len], vec![seq, num_dir, batch, hidden_size]),
        Tensor::new(vec![0.0_f32; yh_len], vec![num_dir, batch, hidden_size]),
    ];

    // First call
    GRUOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("GRUOp execute_into_slots first call failed");

    assert_near(&slots[0], &expected[0], "gru Y first call", 1e-5);
    assert_near(&slots[1], &expected[1], "gru Y_h first call", 1e-5);

    // Record raw pointers for both slots.
    let ptr_y = slots[0].data.as_ptr() as usize;
    let ptr_yh = slots[1].data.as_ptr() as usize;

    // Second call: same input — no reallocation should occur.
    GRUOp
        .execute_into_slots(&ctx, &mut slots)
        .expect("GRUOp execute_into_slots second call failed");

    assert_eq!(
        slots[0].data.as_ptr() as usize,
        ptr_y,
        "gru Y slot pointer must be stable on same-shape second call"
    );
    assert_eq!(
        slots[1].data.as_ptr() as usize,
        ptr_yh,
        "gru Y_h slot pointer must be stable on same-shape second call"
    );

    assert_near(&slots[0], &expected[0], "gru Y second call", 1e-5);
    assert_near(&slots[1], &expected[1], "gru Y_h second call", 1e-5);
}
