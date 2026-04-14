//! Tests for TypedTensor::from_f32_vec, Session::run_typed, and related helpers.

use oxionnx::{DType, Session, TensorStorage, TypedTensor};
use oxionnx_core::{Attributes, Graph, Node, OpKind};
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
