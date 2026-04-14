//! Tests for the `inputs!` macro (ort-compatibility layer).

use oxionnx::{inputs, Tensor};
use std::collections::HashMap;

#[test]
fn test_inputs_single_entry() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let result = inputs!["input_ids" => t.clone()];
    assert!(result.is_ok(), "inputs! should return Ok");
    let map: HashMap<&str, Tensor> = result.unwrap();
    assert_eq!(map.len(), 1);
    assert!(map.contains_key("input_ids"));
    assert_eq!(map["input_ids"].data, t.data);
    assert_eq!(map["input_ids"].shape, t.shape);
}

#[test]
fn test_inputs_multiple_entries() {
    let a = Tensor::zeros(&[1, 4]);
    let b = Tensor::zeros(&[1, 4]);
    let map = inputs!["q" => a.clone(), "k" => b.clone()].expect("multiple inputs");
    assert_eq!(map.len(), 2);
    assert!(map.contains_key("q"));
    assert!(map.contains_key("k"));
}

#[test]
fn test_inputs_trailing_comma() {
    let t = Tensor::scalar(42.0);
    let map = inputs!["x" => t,].expect("trailing comma");
    assert_eq!(map.len(), 1);
}

#[test]
fn test_inputs_empty() {
    let map = inputs![].expect("empty inputs");
    assert_eq!(map.len(), 0);
}

#[test]
fn test_inputs_three_entries() {
    let a = Tensor::zeros(&[2]);
    let b = Tensor::zeros(&[3]);
    let c = Tensor::zeros(&[4]);
    let map = inputs![
        "attention_mask" => a,
        "input_ids"      => b,
        "token_type_ids" => c,
    ]
    .expect("three entries");
    assert_eq!(map.len(), 3);
}
