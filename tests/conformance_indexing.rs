//! Conformance tests 32–34, 38–39: Indexing, selection, and miscellaneous operators.

mod common;

use oxionnx::{Attributes, OpKind, Tensor};

use common::{assert_close, assert_shape, run_op};

// ═══════════════════════════════════════════════════════════════════════════════
// 32–34: Indexing conformance
// ═══════════════════════════════════════════════════════════════════════════════

/// 32. conformance_gather_axis0 — gather rows
#[test]
fn conformance_gather_axis0() {
    // data = [[1,2],[3,4],[5,6]] shape [3,2]
    // indices = [2, 0] shape [2]
    // gather axis=0 => [[5,6],[1,2]] shape [2,2]
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), 0);

    let data = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let indices = Tensor::new(vec![2.0, 0.0], vec![2]);

    let out = run_op(
        OpKind::Gather,
        vec!["data", "indices"],
        vec!["out"],
        vec!["data", "indices"],
        vec![("data", data), ("indices", indices)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[2, 2], "gather_axis0");
    assert_close(&t.data, &[5.0, 6.0, 1.0, 2.0], 1e-5, "gather_axis0");
}

/// 33. conformance_where_condition — ternary select
#[test]
fn conformance_where_condition() {
    // condition = [1, 0, 1, 0] (treated as bool)
    // x = [10, 20, 30, 40]
    // y = [1, 2, 3, 4]
    // where(cond, x, y) = [10, 2, 30, 4]
    let cond = Tensor::new(vec![1.0, 0.0, 1.0, 0.0], vec![4]);
    let x = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![4]);
    let y = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

    let out = run_op(
        OpKind::Where,
        vec!["cond", "x", "y"],
        vec!["out"],
        vec!["cond", "x", "y"],
        vec![("cond", cond), ("x", x), ("y", y)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[10.0, 2.0, 30.0, 4.0], 1e-5, "where");
}

/// 34. conformance_onehot — one-hot encoding
#[test]
fn conformance_onehot() {
    // indices = [0, 1, 2] shape [3]
    // depth = 4
    // values = [0, 1] (off_value=0, on_value=1)
    // axis = -1 (default)
    // Expected shape [3,4]:
    // [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), -1);

    let indices = Tensor::new(vec![0.0, 1.0, 2.0], vec![3]);
    let depth = Tensor::new(vec![4.0], vec![1]);
    let values = Tensor::new(vec![0.0, 1.0], vec![2]);

    let out = run_op(
        OpKind::OneHot,
        vec!["indices", "depth", "values"],
        vec!["out"],
        vec!["indices", "depth", "values"],
        vec![("indices", indices), ("depth", depth), ("values", values)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[3, 4], "onehot");
    assert_close(
        &t.data,
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        1e-5,
        "onehot",
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// 38–39: Miscellaneous operators
// ═══════════════════════════════════════════════════════════════════════════════

/// 38. conformance_clip — clamp values within [min, max]
#[test]
fn conformance_clip() {
    // Clip uses min/max as additional inputs
    let x = Tensor::new(vec![-5.0, -1.0, 0.0, 3.0, 10.0], vec![5]);
    let min_val = Tensor::new(vec![-2.0], vec![1]);
    let max_val = Tensor::new(vec![5.0], vec![1]);

    let out = run_op(
        OpKind::Clip,
        vec!["x", "min", "max"],
        vec!["out"],
        vec!["x", "min", "max"],
        vec![("x", x), ("min", min_val), ("max", max_val)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_close(&t.data, &[-2.0, -1.0, 0.0, 3.0, 5.0], 1e-5, "clip");
}

/// 39. conformance_identity — passthrough
#[test]
fn conformance_identity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Tensor::new(data.clone(), vec![2, 3]);
    let out = run_op(
        OpKind::Identity,
        vec!["x"],
        vec!["out"],
        vec!["x"],
        vec![("x", x)],
        vec![],
        Attributes::default(),
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[2, 3], "identity");
    assert_close(&t.data, &data, 0.0, "identity");
}
