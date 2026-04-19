//! Convolution and pooling operator integration tests: Conv2D variants,
//! MaxPool, AveragePool.

mod common;

use oxionnx::{Attributes, OpKind, Tensor};

use common::{assert_tensor_approx, run_single_op};

// ═══════════════════════════════════════════════════════════════════════════════
// Conv ops
// ═══════════════════════════════════════════════════════════════════════════════

// 9. test_conv2d_dilated - Conv2D with dilation=[2,2]
#[test]
fn test_conv2d_dilated() {
    // Input: [1,1,5,5] all ones
    // Kernel: [1,1,2,2] all ones, dilation=[2,2]
    // With dilation=2, effective kernel is 3x3 (2 + (2-1)*2 = 3 for each dim, but actually
    // dilated kernel covers positions: (0,0),(0,2),(2,0),(2,2) in a 3x3 receptive field)
    // Output size: (5 + 0 + 0 - 2*(2-1) - 1)/1 + 1 = (5 - 2 - 1)/1 + 1 = 3
    // Each output = sum of 4 input values (all 1s) = 4
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".to_string(), vec![1, 1]);
    attrs.int_lists.insert("pads".to_string(), vec![0, 0, 0, 0]);
    attrs.int_lists.insert("dilations".to_string(), vec![2, 2]);
    attrs.ints.insert("group".to_string(), 1);

    let input = Tensor::new(vec![1.0; 25], vec![1, 1, 5, 5]);
    let kernel = Tensor::new(vec![1.0; 4], vec![1, 1, 2, 2]);

    let outputs = run_single_op(
        OpKind::Conv,
        vec![("input", input)],
        vec![("kernel", kernel)],
        vec!["input"],
        vec!["input", "kernel"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 1, 3, 3]);
    assert_tensor_approx(out, &[4.0; 9], 1e-5);
}

// 10. test_conv2d_grouped - Conv2D with group=2
#[test]
fn test_conv2d_grouped() {
    // Input: [1,4,3,3] (4 channels)
    // Kernel: [4,2,1,1] (4 output channels, 2 input channels per group, 1x1 kernel)
    // group=2: group0 reads channels 0,1 -> outputs 0,1; group1 reads channels 2,3 -> outputs 2,3
    // With all-ones input and all-ones kernel:
    // Each output channel = sum of 2 input channels = 2
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".to_string(), vec![1, 1]);
    attrs.int_lists.insert("pads".to_string(), vec![0, 0, 0, 0]);
    attrs.int_lists.insert("dilations".to_string(), vec![1, 1]);
    attrs.ints.insert("group".to_string(), 2);

    let input = Tensor::new(vec![1.0; 36], vec![1, 4, 3, 3]); // 4 channels, 3x3
    let kernel = Tensor::new(vec![1.0; 8], vec![4, 2, 1, 1]); // 4 out, 2 in/group, 1x1

    let outputs = run_single_op(
        OpKind::Conv,
        vec![("input", input)],
        vec![("kernel", kernel)],
        vec!["input"],
        vec!["input", "kernel"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 4, 3, 3]);
    // Each output element = sum over 2 input channels (each=1) = 2
    assert_tensor_approx(out, &[2.0; 36], 1e-5);
}

// 11. test_conv2d_stride2 - Conv2D with stride=2
#[test]
fn test_conv2d_stride2() {
    // Input: [1,1,4,4] with values 1..16
    // Kernel: [1,1,2,2] all ones
    // stride=2 => output size = (4 - 2)/2 + 1 = 2
    // Output[i,j] = sum of 2x2 block starting at (2i, 2j)
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".to_string(), vec![2, 2]);
    attrs.int_lists.insert("pads".to_string(), vec![0, 0, 0, 0]);
    attrs.int_lists.insert("dilations".to_string(), vec![1, 1]);
    attrs.ints.insert("group".to_string(), 1);

    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Tensor::new(input_data, vec![1, 1, 4, 4]);
    let kernel = Tensor::new(vec![1.0; 4], vec![1, 1, 2, 2]);

    let outputs = run_single_op(
        OpKind::Conv,
        vec![("input", input)],
        vec![("kernel", kernel)],
        vec!["input"],
        vec!["input", "kernel"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    // (0,0): 1+2+5+6=14; (0,1): 3+4+7+8=22
    // (1,0): 9+10+13+14=46; (1,1): 11+12+15+16=54
    assert_tensor_approx(out, &[14.0, 22.0, 46.0, 54.0], 1e-5);
}

// 12. test_maxpool
#[test]
fn test_maxpool() {
    // Input: [1,1,4,4] with values 1..16
    // MaxPool 2x2, stride=2
    let mut attrs = Attributes::default();
    attrs
        .int_lists
        .insert("kernel_shape".to_string(), vec![2, 2]);
    attrs.int_lists.insert("strides".to_string(), vec![2, 2]);
    attrs.int_lists.insert("pads".to_string(), vec![0, 0, 0, 0]);

    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Tensor::new(input_data, vec![1, 1, 4, 4]);

    let outputs = run_single_op(
        OpKind::MaxPool,
        vec![("input", input)],
        vec![],
        vec!["input"],
        vec!["input"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    // 2x2 blocks: max of each
    // (0,0): max(1,2,5,6)=6; (0,1): max(3,4,7,8)=8
    // (1,0): max(9,10,13,14)=14; (1,1): max(11,12,15,16)=16
    assert_tensor_approx(out, &[6.0, 8.0, 14.0, 16.0], 1e-5);
}

// 18. test_conv2d_batch_n - Conv2D with batch=4
#[test]
fn test_conv2d_batch_n() {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("strides".to_string(), vec![1, 1]);
    attrs.int_lists.insert("pads".to_string(), vec![0, 0, 0, 0]);
    attrs.int_lists.insert("dilations".to_string(), vec![1, 1]);
    attrs.ints.insert("group".to_string(), 1);

    // Input: [4,1,3,3] all ones
    let input = Tensor::new(vec![1.0; 36], vec![4, 1, 3, 3]);
    // Kernel: [1,1,2,2] all ones
    let kernel = Tensor::new(vec![1.0; 4], vec![1, 1, 2, 2]);

    let outputs = run_single_op(
        OpKind::Conv,
        vec![("input", input)],
        vec![("kernel", kernel)],
        vec!["input"],
        vec!["input", "kernel"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    // Output: [4,1,2,2] each element = sum of 2x2 = 4
    assert_eq!(out.shape, vec![4, 1, 2, 2]);
    assert_tensor_approx(out, &[4.0; 16], 1e-5);
}

// test_average_pool
#[test]
fn test_average_pool() {
    let mut attrs = Attributes::default();
    attrs
        .int_lists
        .insert("kernel_shape".to_string(), vec![2, 2]);
    attrs.int_lists.insert("strides".to_string(), vec![2, 2]);
    attrs.int_lists.insert("pads".to_string(), vec![0, 0, 0, 0]);

    let input_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = Tensor::new(input_data, vec![1, 1, 4, 4]);

    let outputs = run_single_op(
        OpKind::AveragePool,
        vec![("input", input)],
        vec![],
        vec!["input"],
        vec!["input"],
        "out",
        attrs,
    );
    let out = outputs.get("out").unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    // Avg of each 2x2 block:
    // (0,0): (1+2+5+6)/4 = 3.5
    // (0,1): (3+4+7+8)/4 = 5.5
    // (1,0): (9+10+13+14)/4 = 11.5
    // (1,1): (11+12+15+16)/4 = 13.5
    assert_tensor_approx(out, &[3.5, 5.5, 11.5, 13.5], 1e-5);
}
