//! Tests for TypedTensor::from_f32_vec, Session::run_typed, and related helpers.

use oxionnx::{DType, Session, TensorStorage, TypedTensor};
use oxionnx_core::{Attributes, Graph, Node, OpKind, TensorInfo};
use std::collections::HashMap;

#[test]
fn test_typed_tensor_from_f32_vec_i64() {
    let data = vec![1.0f32, 2.0, 100.0];
    let shape = vec![3];
    let tt = TypedTensor::from_f32_vec(data, shape.clone(), DType::I64)
        .expect("from_f32_vec I64 should succeed");

    assert_eq!(tt.dtype(), DType::I64);
    assert_eq!(tt.shape(), shape.as_slice());
    assert_eq!(tt.numel(), 3);

    match &tt.storage {
        TensorStorage::I64(v) => {
            assert_eq!(v.as_slice(), &[1i64, 2, 100]);
        }
        other => panic!("Expected I64 storage, got {:?}", other),
    }
}

#[test]
fn test_typed_tensor_from_f32_vec_bool() {
    let data = vec![1.0f32, 0.0, 0.5];
    let shape = vec![3];
    let tt = TypedTensor::from_f32_vec(data, shape.clone(), DType::Bool)
        .expect("from_f32_vec Bool should succeed");

    assert_eq!(tt.dtype(), DType::Bool);
    assert_eq!(tt.shape(), shape.as_slice());

    match &tt.storage {
        TensorStorage::Bool(v) => {
            assert_eq!(v.as_slice(), &[true, false, true]);
        }
        other => panic!("Expected Bool storage, got {:?}", other),
    }
}

#[test]
fn test_typed_tensor_roundtrip_f16() {
    // Known f32 values that round-trip cleanly through f16
    let f32_vals = [1.0f32, 0.5, 2.0, -1.0];
    let f16_bits: Vec<u16> = f32_vals
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();

    let original = TypedTensor::new(TensorStorage::F16(f16_bits.clone()), vec![4]);
    assert_eq!(original.dtype(), DType::F16);

    // Convert to f32 and back through from_f32_vec
    let f32_vec = original.storage.to_f32_vec();
    let roundtrip = TypedTensor::from_f32_vec(f32_vec, vec![4], DType::F16)
        .expect("roundtrip F16 should succeed");

    assert_eq!(roundtrip.dtype(), DType::F16);
    match &roundtrip.storage {
        TensorStorage::F16(bits) => {
            assert_eq!(bits.as_slice(), f16_bits.as_slice());
        }
        other => panic!("Expected F16 storage, got {:?}", other),
    }
}

#[test]
fn test_run_typed_f32_passthrough() {
    let graph = Graph {
        nodes: vec![Node {
            op: OpKind::Identity,
            name: "id".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        }],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: vec![],
        output_infos: vec![],
        name: String::new(),
    };

    let session = Session::from_graph(graph, HashMap::new()).expect("session creation");

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_shape = vec![2, 2];
    let input_tt = TypedTensor::new(TensorStorage::F32(input_data.clone()), input_shape.clone());

    let mut inputs = HashMap::new();
    inputs.insert("x", input_tt);

    let outputs = session
        .run_typed(&inputs)
        .expect("run_typed should succeed");

    let out = outputs.get("y").expect("output 'y' should be present");
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(out.shape(), input_shape.as_slice());

    match &out.storage {
        TensorStorage::F32(v) => {
            assert_eq!(v.as_slice(), input_data.as_slice());
        }
        other => panic!("Expected F32 storage, got {:?}", other),
    }
}

// ── New Phase D / Phase F tests ────────────────────────────────────────────

/// native_dtypes for a pilot op (AddOp) must include at least F32, F16, I64.
#[test]
fn test_native_dtypes_add_op() {
    use oxionnx_core::{DType, Operator};
    use oxionnx_ops::registry::math_ops::AddOp;

    let op = AddOp;
    let dtypes = op.native_dtypes();
    assert!(
        !dtypes.is_empty(),
        "AddOp must declare at least one native dtype"
    );
    assert!(
        dtypes.contains(&DType::F32),
        "AddOp must support F32 natively"
    );
    assert!(
        dtypes.contains(&DType::F16),
        "AddOp must support F16 natively"
    );
    assert!(
        dtypes.contains(&DType::I64),
        "AddOp must support I64 natively"
    );
}

/// native_dtypes for a non-pilot op (SoftmaxOp) returns empty slice.
#[test]
fn test_native_dtypes_softmax_op_default_empty() {
    use oxionnx_core::Operator;
    use oxionnx_ops::registry::nn_ops::SoftmaxOp;

    let op = SoftmaxOp;
    let dtypes = op.native_dtypes();
    assert!(
        dtypes.is_empty(),
        "SoftmaxOp must return empty native_dtypes (non-pilot op)"
    );
}

/// execute_typed for IdentityOp clones the typed input unchanged.
#[test]
fn test_execute_typed_identity_preserves_storage() {
    use oxionnx_core::{Attributes, Node, OpKind, Operator, TypedOpContext};
    use oxionnx_ops::registry::misc_ops::IdentityOp;

    let input_data: Vec<i64> = vec![10, 20, 30];
    let tt = TypedTensor::new(TensorStorage::I64(input_data.clone()), vec![3]);

    let node = Node {
        op: OpKind::Identity,
        name: "id".to_string(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        attrs: Attributes::default(),
    };

    let ctx = TypedOpContext {
        node: &node,
        inputs: vec![Some(&tt)],
        outer_scope: None,
        registry: None,
    };

    let op = IdentityOp;
    let results = op.execute_typed(&ctx).expect("execute_typed Identity");
    assert_eq!(results.len(), 1);
    let out = &results[0];
    assert_eq!(out.dtype(), DType::I64);
    match &out.storage {
        TensorStorage::I64(v) => assert_eq!(v.as_slice(), input_data.as_slice()),
        other => panic!("Expected I64 storage, got {:?}", other),
    }
}

/// execute_typed for CastOp (to=7 → I64) converts F32 input via f32 intermediate.
/// Values < 2^24 are lossless.
#[test]
fn test_execute_typed_cast_f32_to_i64() {
    use oxionnx_core::{Attributes, Node, OpKind, Operator, TypedOpContext};
    use oxionnx_ops::registry::misc_ops::CastOp;

    // "to=7" = ONNX INT64
    let mut attrs = Attributes::default();
    attrs.ints.insert("to".to_string(), 7);

    let input_data: Vec<f32> = vec![1.0, 42.0, 100.0];
    let tt = TypedTensor::new(TensorStorage::F32(input_data.clone()), vec![3]);

    let node = Node {
        op: OpKind::Cast,
        name: "cast".to_string(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        attrs,
    };

    let ctx = TypedOpContext {
        node: &node,
        inputs: vec![Some(&tt)],
        outer_scope: None,
        registry: None,
    };

    let op = CastOp;
    let results = op.execute_typed(&ctx).expect("execute_typed Cast");
    assert_eq!(results.len(), 1);
    let out = &results[0];
    assert_eq!(out.dtype(), DType::I64);
    match &out.storage {
        TensorStorage::I64(v) => {
            assert_eq!(v.as_slice(), &[1i64, 42, 100]);
        }
        other => panic!("Expected I64 storage after Cast, got {:?}", other),
    }
}

/// run_typed with output_infos specifying F16 output preserves the dtype.
#[test]
fn test_run_typed_preserves_output_dtype_f16() {
    // Use an Identity graph with output_infos declaring F16 dtype.
    // run_typed recovers the dtype from output_infos.
    let graph = Graph {
        nodes: vec![Node {
            op: OpKind::Identity,
            name: "id".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        }],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: vec![],
        output_infos: vec![TensorInfo {
            name: "y".to_string(),
            dtype: DType::F16,
            shape: vec![],
            dim_params: vec![],
        }],
        name: String::new(),
    };

    let session = Session::from_graph(graph, HashMap::new()).expect("session creation");

    // Provide F16 bits for known values: 1.0 and 2.0 in f16
    let f16_bits: Vec<u16> = [1.0f32, 2.0]
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let tt = TypedTensor::new(TensorStorage::F16(f16_bits.clone()), vec![2]);

    let mut inputs = HashMap::new();
    inputs.insert("x", tt);

    let outputs = session.run_typed(&inputs).expect("run_typed");
    let out = outputs.get("y").expect("output 'y'");
    // The output dtype should be recovered from output_infos as F16
    assert_eq!(
        out.dtype(),
        DType::F16,
        "output dtype should be F16 per output_infos"
    );
}

/// run_typed with output_infos specifying I64 output preserves the dtype.
#[test]
fn test_run_typed_preserves_output_dtype_i64() {
    let graph = Graph {
        nodes: vec![Node {
            op: OpKind::Identity,
            name: "id".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        }],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: vec![],
        output_infos: vec![TensorInfo {
            name: "y".to_string(),
            dtype: DType::I64,
            shape: vec![],
            dim_params: vec![],
        }],
        name: String::new(),
    };

    let session = Session::from_graph(graph, HashMap::new()).expect("session creation");

    // Use small integers to avoid f32 precision loss (values < 2^24)
    let data: Vec<i64> = vec![1, 2, 3, 42];
    let tt = TypedTensor::new(TensorStorage::I64(data.clone()), vec![4]);

    let mut inputs = HashMap::new();
    inputs.insert("x", tt);

    let outputs = session.run_typed(&inputs).expect("run_typed");
    let out = outputs.get("y").expect("output 'y'");
    assert_eq!(
        out.dtype(),
        DType::I64,
        "output dtype should be I64 per output_infos"
    );
    match &out.storage {
        TensorStorage::I64(v) => assert_eq!(v.as_slice(), data.as_slice()),
        other => panic!("Expected I64 storage, got {:?}", other),
    }
}

/// run_typed without output_infos: D.1 native dispatch preserves dtype for native ops.
///
/// Identity is a native op (handles all dtypes). With D.1, the output carries the same
/// dtype as the input — I64 in, I64 out — because no f32 round-trip occurs. The
/// old pre-D.1 behaviour (F32 output regardless) has been superseded by native dispatch.
#[test]
fn test_run_typed_defaults_to_f32_without_output_infos() {
    let graph = Graph {
        nodes: vec![Node {
            op: OpKind::Identity,
            name: "id".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attrs: Attributes::default(),
        }],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        input_infos: vec![],
        output_infos: vec![], // no output metadata
        name: String::new(),
    };

    let session = Session::from_graph(graph, HashMap::new()).expect("session creation");

    // D.1: Identity natively handles I64, so the output preserves I64 (no f32 round-trip).
    let tt = TypedTensor::new(TensorStorage::I64(vec![1i64, 2, 3]), vec![3]);

    let mut inputs = HashMap::new();
    inputs.insert("x", tt);

    let outputs = session.run_typed(&inputs).expect("run_typed");
    let out = outputs.get("y").expect("output 'y'");
    assert_eq!(
        out.dtype(),
        DType::I64,
        "Identity on I64 input must produce I64 output via native dispatch (D.1)"
    );
    match &out.storage {
        TensorStorage::I64(v) => assert_eq!(v.as_slice(), &[1i64, 2, 3]),
        other => panic!("Expected I64 storage, got {:?}", other),
    }
}

/// execute_typed for AddOp on I64 inputs produces correct I64 output via typed dispatch.
#[test]
fn test_execute_typed_add_op_i64() {
    use oxionnx_core::{Attributes, Node, OpKind, Operator, TypedOpContext};
    use oxionnx_ops::registry::math_ops::AddOp;

    let a = TypedTensor::new(TensorStorage::I64(vec![10i64, 20, 30]), vec![3]);
    let b = TypedTensor::new(TensorStorage::I64(vec![1i64, 2, 3]), vec![3]);

    let node = Node {
        op: OpKind::Add,
        name: "add".to_string(),
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["c".to_string()],
        attrs: Attributes::default(),
    };

    let ctx = TypedOpContext {
        node: &node,
        inputs: vec![Some(&a), Some(&b)],
        outer_scope: None,
        registry: None,
    };

    let op = AddOp;
    let results = op.execute_typed(&ctx).expect("execute_typed Add I64");
    assert_eq!(results.len(), 1);
    let out = &results[0];
    // AddOp uses typed_add which dispatches natively for I64
    assert_eq!(out.dtype(), DType::I64);
    match &out.storage {
        TensorStorage::I64(v) => assert_eq!(v.as_slice(), &[11i64, 22, 33]),
        other => panic!("Expected I64 storage from Add, got {:?}", other),
    }
}

/// execute_typed fallback: non-pilot op (ReluOp via default_typed_via_f32) works on F32.
#[test]
fn test_execute_typed_fallback_via_f32_relu() {
    use oxionnx_core::{Attributes, Node, OpKind, Operator, TypedOpContext};
    use oxionnx_ops::registry::nn_ops::ReluOp;

    let data: Vec<f32> = vec![-1.0, 0.0, 2.5, -3.0];
    let tt = TypedTensor::new(TensorStorage::F32(data), vec![4]);

    let node = Node {
        op: OpKind::Relu,
        name: "relu".to_string(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        attrs: Attributes::default(),
    };

    let ctx = TypedOpContext {
        node: &node,
        inputs: vec![Some(&tt)],
        outer_scope: None,
        registry: None,
    };

    let op = ReluOp;
    let results = op.execute_typed(&ctx).expect("execute_typed Relu");
    assert_eq!(results.len(), 1);
    let out = &results[0];
    // ReluOp uses default_typed_via_f32
    match &out.storage {
        TensorStorage::F32(v) => {
            assert_eq!(v.as_slice(), &[0.0f32, 0.0, 2.5, 0.0]);
        }
        other => panic!("Expected F32 storage from Relu, got {:?}", other),
    }
}

// ── Phase D: native_dtypes coverage for all pilot ops ─────────────────────────

/// native_dtypes for all NN activation pilot ops must include F32, F16, BF16.
#[test]
fn test_native_dtypes_nn_activation_ops() {
    use oxionnx_core::{DType, Operator};
    use oxionnx_ops::registry::nn_ops::{ErfOp, GeluOp, ReluOp, SiLUOp, SigmoidOp, TanhOp};

    let required = [DType::F32, DType::F16, DType::BF16];

    macro_rules! check_op {
        ($op:expr, $name:expr) => {{
            let dtypes = $op.native_dtypes();
            assert!(
                !dtypes.is_empty(),
                "{} must declare at least one native dtype",
                $name
            );
            for &dt in &required {
                assert!(
                    dtypes.contains(&dt),
                    "{} must support {:?} natively",
                    $name,
                    dt
                );
            }
        }};
    }

    check_op!(ReluOp, "ReluOp");
    check_op!(SigmoidOp, "SigmoidOp");
    check_op!(TanhOp, "TanhOp");
    check_op!(GeluOp, "GeluOp");
    check_op!(SiLUOp, "SiLUOp");
    check_op!(ErfOp, "ErfOp");
}

/// native_dtypes for math pilot ops (Add, Sub, Mul, Div, Sqrt) must include F32, F16, I64.
#[test]
fn test_native_dtypes_math_pilot_ops() {
    use oxionnx_core::{DType, Operator};
    use oxionnx_ops::registry::math_ops::{AddOp, DivOp, MulOp, SqrtOp, SubOp};

    let required = [DType::F32, DType::F16, DType::I64];

    macro_rules! check_op {
        ($op:expr, $name:expr) => {{
            let dtypes = $op.native_dtypes();
            for &dt in &required {
                assert!(
                    dtypes.contains(&dt),
                    "{} must support {:?} natively",
                    $name,
                    dt
                );
            }
        }};
    }

    check_op!(AddOp, "AddOp");
    check_op!(SubOp, "SubOp");
    check_op!(MulOp, "MulOp");
    check_op!(DivOp, "DivOp");
    check_op!(SqrtOp, "SqrtOp");
}

/// native_dtypes for misc pilot ops (Identity, Cast, Reshape) include all ONNX dtypes.
#[test]
fn test_native_dtypes_misc_ops_full_coverage() {
    use oxionnx_core::{DType, Operator};
    use oxionnx_ops::registry::misc_ops::{CastOp, IdentityOp};

    // Identity is fully generic — must support every storage type
    let all_dtypes = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::Bool,
    ];

    let identity_dtypes = IdentityOp.native_dtypes();
    for &dt in &all_dtypes {
        assert!(
            identity_dtypes.contains(&dt),
            "IdentityOp must support {:?} natively",
            dt
        );
    }

    // Cast must support the same set (it can convert between them)
    let cast_dtypes = CastOp.native_dtypes();
    for &dt in &all_dtypes {
        assert!(
            cast_dtypes.contains(&dt),
            "CastOp must support {:?} natively",
            dt
        );
    }
}

/// execute_typed for SubOp on I64 inputs dispatches via typed_sub.
#[test]
fn test_execute_typed_sub_op_i64() {
    use oxionnx_core::{Attributes, Node, OpKind, Operator, TypedOpContext};
    use oxionnx_ops::registry::math_ops::SubOp;

    let a = TypedTensor::new(TensorStorage::I64(vec![100i64, 50, 30]), vec![3]);
    let b = TypedTensor::new(TensorStorage::I64(vec![1i64, 5, 10]), vec![3]);

    let node = Node {
        op: OpKind::Sub,
        name: "sub".to_string(),
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["c".to_string()],
        attrs: Attributes::default(),
    };

    let ctx = TypedOpContext {
        node: &node,
        inputs: vec![Some(&a), Some(&b)],
        outer_scope: None,
        registry: None,
    };

    let results = SubOp.execute_typed(&ctx).expect("execute_typed Sub I64");
    assert_eq!(results.len(), 1);
    let out = &results[0];
    assert_eq!(out.dtype(), DType::I64);
    match &out.storage {
        TensorStorage::I64(v) => assert_eq!(v.as_slice(), &[99i64, 45, 20]),
        other => panic!("Expected I64 from Sub, got {:?}", other),
    }
}

/// execute_typed for MulOp on I64 inputs dispatches via typed_mul.
#[test]
fn test_execute_typed_mul_op_i64() {
    use oxionnx_core::{Attributes, Node, OpKind, Operator, TypedOpContext};
    use oxionnx_ops::registry::math_ops::MulOp;

    let a = TypedTensor::new(TensorStorage::I64(vec![2i64, 3, 4]), vec![3]);
    let b = TypedTensor::new(TensorStorage::I64(vec![5i64, 6, 7]), vec![3]);

    let node = Node {
        op: OpKind::Mul,
        name: "mul".to_string(),
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["c".to_string()],
        attrs: Attributes::default(),
    };

    let ctx = TypedOpContext {
        node: &node,
        inputs: vec![Some(&a), Some(&b)],
        outer_scope: None,
        registry: None,
    };

    let results = MulOp.execute_typed(&ctx).expect("execute_typed Mul I64");
    let out = &results[0];
    assert_eq!(out.dtype(), DType::I64);
    match &out.storage {
        TensorStorage::I64(v) => assert_eq!(v.as_slice(), &[10i64, 18, 28]),
        other => panic!("Expected I64 from Mul, got {:?}", other),
    }
}

/// execute_typed for SigmoidOp via default_typed_via_f32 produces correct F32 output.
#[test]
fn test_execute_typed_sigmoid_via_f32() {
    use oxionnx_core::{Attributes, Node, OpKind, Operator, TypedOpContext};
    use oxionnx_ops::registry::nn_ops::SigmoidOp;

    let data: Vec<f32> = vec![0.0, 2.0, -2.0];
    let tt = TypedTensor::new(TensorStorage::F32(data), vec![3]);

    let node = Node {
        op: OpKind::Sigmoid,
        name: "sig".to_string(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        attrs: Attributes::default(),
    };

    let ctx = TypedOpContext {
        node: &node,
        inputs: vec![Some(&tt)],
        outer_scope: None,
        registry: None,
    };

    let results = SigmoidOp
        .execute_typed(&ctx)
        .expect("execute_typed Sigmoid");
    let out = &results[0];
    match &out.storage {
        TensorStorage::F32(v) => {
            // sigmoid(0) = 0.5
            assert!(
                (v[0] - 0.5f32).abs() < 1e-6,
                "sigmoid(0) ≈ 0.5, got {}",
                v[0]
            );
            // sigmoid(2) ≈ 0.8808
            assert!(
                (v[1] - 0.880_797f32).abs() < 1e-4,
                "sigmoid(2) ≈ 0.8808, got {}",
                v[1]
            );
            // sigmoid(-2) ≈ 0.1192
            assert!(
                (v[2] - 0.119_203f32).abs() < 1e-4,
                "sigmoid(-2) ≈ 0.1192, got {}",
                v[2]
            );
        }
        other => panic!("Expected F32 from Sigmoid, got {:?}", other),
    }
}

/// execute_typed for TanhOp via default_typed_via_f32 produces correct F32 output.
#[test]
fn test_execute_typed_tanh_via_f32() {
    use oxionnx_core::{Attributes, Node, OpKind, Operator, TypedOpContext};
    use oxionnx_ops::registry::nn_ops::TanhOp;

    let data: Vec<f32> = vec![0.0, 1.0, -1.0];
    let tt = TypedTensor::new(TensorStorage::F32(data), vec![3]);

    let node = Node {
        op: OpKind::Tanh,
        name: "tanh".to_string(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        attrs: Attributes::default(),
    };

    let ctx = TypedOpContext {
        node: &node,
        inputs: vec![Some(&tt)],
        outer_scope: None,
        registry: None,
    };

    let results = TanhOp.execute_typed(&ctx).expect("execute_typed Tanh");
    let out = &results[0];
    match &out.storage {
        TensorStorage::F32(v) => {
            assert!((v[0] - 0.0f32).abs() < 1e-6, "tanh(0) = 0, got {}", v[0]);
            // tanh(1) ≈ 0.7616
            assert!(
                (v[1] - 0.761_594f32).abs() < 1e-4,
                "tanh(1) ≈ 0.7616, got {}",
                v[1]
            );
            // tanh(-1) ≈ -0.7616
            assert!(
                (v[2] + 0.761_594f32).abs() < 1e-4,
                "tanh(-1) ≈ -0.7616, got {}",
                v[2]
            );
        }
        other => panic!("Expected F32 from Tanh, got {:?}", other),
    }
}

/// native_dtypes consistency: every op that declares native_dtypes must include F32
/// (F32 is the universal baseline for all ONNX float ops).
#[test]
fn test_native_dtypes_always_includes_f32_for_declared_ops() {
    use oxionnx_core::{DType, Operator};
    use oxionnx_ops::registry::{
        math_ops::{AddOp, DivOp, MulOp, SqrtOp, SubOp},
        misc_ops::{CastOp, IdentityOp},
        nn_ops::{AbsOp, ErfOp, ExpOp, GeluOp, LogOp, ReluOp, SiLUOp, SigmoidOp, TanhOp},
        shape_ops::ReshapeOp,
    };

    // All pilot ops must include F32 in their native_dtypes declaration.
    let ops: &[(&dyn Operator, &str)] = &[
        (&AddOp, "AddOp"),
        (&SubOp, "SubOp"),
        (&MulOp, "MulOp"),
        (&DivOp, "DivOp"),
        (&SqrtOp, "SqrtOp"),
        (&ReluOp, "ReluOp"),
        (&SigmoidOp, "SigmoidOp"),
        (&TanhOp, "TanhOp"),
        (&GeluOp, "GeluOp"),
        (&SiLUOp, "SiLUOp"),
        (&ErfOp, "ErfOp"),
        (&AbsOp, "AbsOp"),
        (&LogOp, "LogOp"),
        (&ExpOp, "ExpOp"),
        (&IdentityOp, "IdentityOp"),
        (&CastOp, "CastOp"),
        (&ReshapeOp, "ReshapeOp"),
    ];

    for (op, name) in ops {
        let dtypes = op.native_dtypes();
        assert!(
            !dtypes.is_empty(),
            "{name} must declare at least one native dtype"
        );
        assert!(
            dtypes.contains(&DType::F32),
            "{name} native_dtypes must include F32"
        );
    }
}

// ── D.1 session-level typed dispatch tests ────────────────────────────────────

/// D.1 + D.2: run_typed with I64 Add graph preserves exactness for large token IDs.
///
/// The value 2^40 = 1_099_511_627_776 cannot be represented exactly in f32
/// (which has only 24 bits of significand). A whole-graph f32 round-trip would
/// corrupt it. With D.1 native dispatch, AddOp operates directly on I64 storage.
#[test]
fn test_run_typed_i64_token_ids_preserved() {
    // Graph: Add(input, const_bias) — both I64
    // const_bias is provided as a weight (initializer)
    let graph = oxionnx_core::Graph {
        nodes: vec![oxionnx_core::Node {
            op: OpKind::Add,
            name: "add".to_string(),
            inputs: vec!["input_ids".to_string(), "bias".to_string()],
            outputs: vec!["output_ids".to_string()],
            attrs: Attributes::default(),
        }],
        input_names: vec!["input_ids".to_string()],
        output_names: vec!["output_ids".to_string()],
        input_infos: vec![],
        output_infos: vec![],
        name: String::new(),
    };

    // Large I64 value that would lose precision through f32
    let large_val: i64 = 1_099_511_627_776; // 2^40

    // Weight bias = [1, 1, 1] as I64 wrapped in a TypedTensor via the weight map.
    // Weights in Session are f32 Tensors, so we must supply the bias as a session input.
    // Instead, provide both as session inputs so they go through typed state.
    let mut inputs = HashMap::new();

    let input_tt = TypedTensor::new(
        TensorStorage::I64(vec![large_val, large_val + 1, large_val + 2]),
        vec![3],
    );
    let bias_tt = TypedTensor::new(TensorStorage::I64(vec![1i64, 2, 3]), vec![3]);

    inputs.insert("input_ids", input_tt);
    inputs.insert("bias", bias_tt);

    let session =
        Session::from_graph(graph, HashMap::new()).expect("session creation for I64 add test");
    let outputs = session.run_typed(&inputs).expect("run_typed I64 add");

    let out = outputs
        .get("output_ids")
        .expect("output 'output_ids' should be present");

    // Native I64 dispatch must preserve exact values
    assert_eq!(out.dtype(), DType::I64, "output must remain I64");
    match &out.storage {
        TensorStorage::I64(v) => {
            assert_eq!(
                v[0],
                large_val + 1,
                "first element: expected {}, got {}",
                large_val + 1,
                v[0]
            );
            assert_eq!(
                v[1],
                large_val + 3,
                "second element: expected {}, got {}",
                large_val + 3,
                v[1]
            );
            assert_eq!(
                v[2],
                large_val + 5,
                "third element: expected {}, got {}",
                large_val + 5,
                v[2]
            );
        }
        other => panic!("Expected I64 storage from I64 Add, got {:?}", other),
    }
}

/// D.1 regression: run_typed on an F32 graph produces results identical to the old path.
///
/// A graph with two Add nodes chained: z = (x + y) + x. All inputs are F32.
/// With D.1, F32 AddOp dispatches natively (F32 is in AddOp::native_dtypes), so
/// there is no behavioural change for pure-F32 graphs.
#[test]
fn test_run_typed_f32_passthrough_unchanged() {
    // Graph: y = Add(x, x)  →  z = Add(y, x)  i.e. z = 3*x
    let graph = oxionnx_core::Graph {
        nodes: vec![
            oxionnx_core::Node {
                op: OpKind::Add,
                name: "add1".to_string(),
                inputs: vec!["x".to_string(), "x".to_string()],
                outputs: vec!["y".to_string()],
                attrs: Attributes::default(),
            },
            oxionnx_core::Node {
                op: OpKind::Add,
                name: "add2".to_string(),
                inputs: vec!["y".to_string(), "x".to_string()],
                outputs: vec!["z".to_string()],
                attrs: Attributes::default(),
            },
        ],
        input_names: vec!["x".to_string()],
        output_names: vec!["z".to_string()],
        input_infos: vec![],
        output_infos: vec![],
        name: String::new(),
    };

    let session = Session::from_graph(graph, HashMap::new()).expect("session creation F32 test");

    let x_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let x_tt = TypedTensor::new(TensorStorage::F32(x_data.clone()), vec![4]);

    let mut inputs = HashMap::new();
    inputs.insert("x", x_tt);

    let outputs = session.run_typed(&inputs).expect("run_typed F32 graph");
    let out = outputs.get("z").expect("output 'z'");

    assert_eq!(out.dtype(), DType::F32, "F32 graph output must stay F32");
    match &out.storage {
        TensorStorage::F32(v) => {
            let expected: Vec<f32> = x_data.iter().map(|&x| 3.0 * x).collect();
            for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-6,
                    "element {i}: expected {exp}, got {got}"
                );
            }
        }
        other => panic!("Expected F32 storage, got {:?}", other),
    }
}
