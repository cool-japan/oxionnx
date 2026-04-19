//! Conformance tests 29–31: Convolution and pooling operators.

mod common;

use oxionnx::{Attributes, OpKind, Tensor};

use common::{assert_close, assert_shape, run_op};

// ═══════════════════════════════════════════════════════════════════════════════
// 29–31: Conv conformance
// ═══════════════════════════════════════════════════════════════════════════════

/// 29. conformance_conv2d_1x1 — 1x1 convolution (equivalent to per-pixel matmul)
#[test]
fn conformance_conv2d_1x1() {
    // Input: [1,2,2,2] (batch=1, 2 channels, 2x2 spatial)
    // Kernel: [3,2,1,1] (3 output channels, 2 input channels, 1x1)
    // Each output pixel = dot product of kernel row with input channels at that pixel
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".to_string(), vec![1, 1]);
    attrs.int_lists.insert("pads".to_string(), vec![0, 0, 0, 0]);
    attrs.int_lists.insert("dilations".to_string(), vec![1, 1]);
    attrs.ints.insert("group".to_string(), 1);

    // Input: channel 0 = [[1,2],[3,4]], channel 1 = [[5,6],[7,8]]
    let input = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![1, 2, 2, 2],
    );
    // Kernel: 3 filters, each [2,1,1]
    // filter0 = [1, 0], filter1 = [0, 1], filter2 = [1, 1]
    let kernel = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2, 1, 1]);

    let out = run_op(
        OpKind::Conv,
        vec!["input", "kernel"],
        vec!["out"],
        vec!["input"],
        vec![("input", input)],
        vec![("kernel", kernel)],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[1, 3, 2, 2], "conv2d_1x1");
    // filter0 (takes ch0): [1,2,3,4]
    // filter1 (takes ch1): [5,6,7,8]
    // filter2 (ch0+ch1):   [6,8,10,12]
    assert_close(
        &t.data,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.0, 8.0, 10.0, 12.0],
        1e-5,
        "conv2d_1x1",
    );
}

/// 30. conformance_conv2d_3x3_pad1 — 3x3 with padding=1 (output same spatial dims)
#[test]
fn conformance_conv2d_3x3_pad1() {
    // Input: [1,1,3,3] all ones
    // Kernel: [1,1,3,3] all ones, pad=1
    // With padding, output is [1,1,3,3]
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".to_string(), vec![1, 1]);
    attrs.int_lists.insert("pads".to_string(), vec![1, 1, 1, 1]);
    attrs.int_lists.insert("dilations".to_string(), vec![1, 1]);
    attrs.ints.insert("group".to_string(), 1);

    let input = Tensor::new(vec![1.0; 9], vec![1, 1, 3, 3]);
    let kernel = Tensor::new(vec![1.0; 9], vec![1, 1, 3, 3]);

    let out = run_op(
        OpKind::Conv,
        vec!["input", "kernel"],
        vec!["out"],
        vec!["input"],
        vec![("input", input)],
        vec![("kernel", kernel)],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[1, 1, 3, 3], "conv2d_3x3_pad1");
    // Corner (0,0): 4 elements in receptive field that overlap with input => 4
    // Edge (0,1): 6 elements => 6
    // Center (1,1): all 9 => 9
    assert_close(
        &t.data,
        &[4.0, 6.0, 4.0, 6.0, 9.0, 6.0, 4.0, 6.0, 4.0],
        1e-5,
        "conv2d_3x3_pad1",
    );
}

/// 31. conformance_maxpool_2x2 — 2x2 pool stride 2
#[test]
fn conformance_maxpool_2x2() {
    // Input: [1,1,4,4] with values 1..16
    // kernel_shape=[2,2], strides=[2,2] => [1,1,2,2]
    let mut attrs = Attributes::default();
    attrs
        .int_lists
        .insert("kernel_shape".to_string(), vec![2, 2]);
    attrs.int_lists.insert("strides".to_string(), vec![2, 2]);
    attrs.int_lists.insert("pads".to_string(), vec![0, 0, 0, 0]);

    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Tensor::new(input_data, vec![1, 1, 4, 4]);

    let out = run_op(
        OpKind::MaxPool,
        vec!["input"],
        vec!["out"],
        vec!["input"],
        vec![("input", input)],
        vec![],
        attrs,
    );
    let t = out.get("out").unwrap();
    assert_shape(t, &[1, 1, 2, 2], "maxpool_2x2");
    // max of each 2x2 block:
    // (0,0): max(1,2,5,6)=6
    // (0,1): max(3,4,7,8)=8
    // (1,0): max(9,10,13,14)=14
    // (1,1): max(11,12,15,16)=16
    assert_close(&t.data, &[6.0, 8.0, 14.0, 16.0], 1e-5, "maxpool_2x2");
}
