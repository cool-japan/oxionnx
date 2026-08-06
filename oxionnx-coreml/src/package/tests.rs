//! Tests for the macOS `macos_impl` runtime.
//!
//! Every test that hits the framework is `#[ignore]`d because the
//! bundles live outside the source tree.  Manual run command:
//!
//!     cargo test -p oxionnx-coreml -- --ignored --test-threads=1

use super::*;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_core_ml::{MLFeatureType, MLFeatureValue, MLSequence};
use objc2_core_video::{
    kCVPixelFormatType_24RGB, kCVPixelFormatType_32BGRA, kCVPixelFormatType_OneComponent16Half,
    kCVPixelFormatType_OneComponent32Float, kCVPixelFormatType_OneComponent8, kCVReturnSuccess,
    CVPixelBuffer, CVPixelBufferCreate, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow,
    CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString};
use oxionnx_core::Tensor;

const ARCFACE_PATH: &str = "/tmp/w600k_r50.mlpackage";

/// Smoke test: model loads and surfaces a single input + single output.
/// Requires `/tmp/w600k_r50.mlpackage` from the OxiFace ArcFace sub-gate.
#[test]
#[ignore]
fn test_load_arcface() {
    let model = MlPackageModel::load(ARCFACE_PATH, MlComputeUnits::All)
        .expect("load /tmp/w600k_r50.mlpackage (run the OxiFace conversion script first)");
    assert_eq!(
        model.input_names().len(),
        1,
        "ArcFace has exactly one input"
    );
    assert_eq!(
        model.output_names().len(),
        1,
        "ArcFace has exactly one output"
    );
}

/// End-to-end roundtrip: 1×3×112×112 input -> 1×512 embedding.
#[test]
#[ignore]
fn test_predict_arcface_returns_512_dim_embedding() {
    let model =
        MlPackageModel::load(ARCFACE_PATH, MlComputeUnits::All).expect("load arcface .mlpackage");
    let input_name = model
        .input_names()
        .into_iter()
        .next()
        .expect("at least one input");
    let output_name = model
        .output_names()
        .into_iter()
        .next()
        .expect("at least one output");

    let n = 3 * 112 * 112;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) / 1000.0).collect();
    let tensor = Tensor::new(data, vec![1, 3, 112, 112]);
    let mut inputs = HashMap::new();
    inputs.insert(input_name, tensor);
    let outputs = model.predict(&inputs).expect("prediction");
    let out = outputs
        .get(&output_name)
        .expect("declared output present in result map");
    assert_eq!(out.data.len(), 512, "ArcFace embedding dimension");
    assert_eq!(out.shape.iter().product::<usize>(), 512, "shape sanity");
}

/// Confirm the compute-plan introspection actually places work on the ANE
/// for the ArcFace model (the headline finding from the sub-gate: 97 %
/// of compute ops on ANE).
#[test]
#[ignore]
fn test_compute_plan_reports_ane_engagement() {
    let model =
        MlPackageModel::load(ARCFACE_PATH, MlComputeUnits::All).expect("load arcface .mlpackage");
    let summary = model.compute_plan_summary().expect("compute plan");
    assert!(
        summary.ane_ops > 0,
        "expected ArcFace to engage the ANE, got {summary:?}",
    );
    let frac = summary.ane_fraction();
    assert!(
        frac > 0.5,
        "ArcFace should run majority on ANE, got fraction {frac}",
    );
}

/// `compute_plan_breakdown`'s per-operator entries must reconcile
/// exactly with `compute_plan_summary`'s flat totals: every key must
/// be a real (non-empty) operator name, and summing every entry's
/// fields across the whole breakdown must equal the flat summary's
/// totals field-for-field — the core correctness invariant the split
/// traversal (`accumulate_program_operations` feeding both a flat
/// accumulator and a per-name map from the identical
/// `classify_operation` calls) exists to guarantee. Requires
/// `/tmp/w600k_r50.mlpackage`, same as the other ArcFace tests.
#[test]
#[ignore]
fn test_compute_plan_breakdown_reconciles_with_summary() {
    let model =
        MlPackageModel::load(ARCFACE_PATH, MlComputeUnits::All).expect("load arcface .mlpackage");
    let summary = model.compute_plan_summary().expect("compute plan summary");
    let breakdown = model
        .compute_plan_breakdown()
        .expect("compute plan breakdown");

    assert!(
        !breakdown.is_empty(),
        "expected at least one operator name in the breakdown"
    );

    let mut reconciled = ComputePlanSummary::default();
    for (opname, per_op) in &breakdown {
        assert!(!opname.is_empty(), "operator name must not be empty");
        reconciled.merge(per_op);
    }

    assert_eq!(
        reconciled.total_ops(),
        summary.total_ops(),
        "breakdown sum must reconcile with the flat summary's total_ops()"
    );
    assert_eq!(reconciled.ane_ops, summary.ane_ops, "ane_ops mismatch");
    assert_eq!(reconciled.gpu_ops, summary.gpu_ops, "gpu_ops mismatch");
    assert_eq!(reconciled.cpu_ops, summary.cpu_ops, "cpu_ops mismatch");
    assert_eq!(
        reconciled.unknown_ops, summary.unknown_ops,
        "unknown_ops mismatch"
    );
}

/// `model_metadata()` must return `Ok` with a map — never panic,
/// never error on a well-formed model. Requires
/// `/tmp/w600k_r50.mlpackage`, same as the other ArcFace tests.
#[test]
#[ignore]
fn test_model_metadata_returns_ok_map() {
    let model =
        MlPackageModel::load(ARCFACE_PATH, MlComputeUnits::All).expect("load arcface .mlpackage");
    let metadata = model
        .model_metadata()
        .expect("model_metadata should not error even with sparse/no metadata");
    // This fixture's actual metadata content is not known ahead of
    // time (it may legitimately be empty), so the only assertable
    // contract here is the key-naming scheme itself: every key must
    // be one of the four well-known names, or a "creator."-prefixed
    // custom entry.
    for key in metadata.keys() {
        assert!(
            matches!(
                key.as_str(),
                "description" | "version" | "author" | "license"
            ) || key.starts_with("creator."),
            "unexpected model_metadata() key: {key:?}"
        );
    }
}

/// `read_raw_bytes`'s raw F32 bytes, reinterpreted as `&[f32]`, must
/// match what `tensor_from_multi_array` (`predict()`'s existing path)
/// returns as a `Tensor` — i.e. refactoring the shared extraction core
/// out from under `tensor_from_multi_array` must not have changed its
/// observable output.  No model file needed — both are exercised
/// directly against an array built by `multi_array_from_f32`.
#[test]
fn read_raw_bytes_f32_matches_tensor_from_multi_array() {
    let data = vec![1.0f32, -2.5, 3.25, 0.0, 42.0, -100.0];
    let shape = vec![2usize, 3];
    let arr = macos_impl::multi_array_from_f32(&data, &shape)
        .expect("multi_array_from_f32 should build a 2x3 Float32 MLMultiArray");

    let raw = macos_impl::read_raw_bytes(&arr).expect("read_raw_bytes");
    assert_eq!(raw.shape, shape);
    assert_eq!(raw.dtype, MlArrayDtype::F32);
    assert_eq!(raw.data.len(), data.len() * std::mem::size_of::<f32>());
    let raw_as_f32: Vec<f32> = raw
        .data
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(raw_as_f32, data);

    let tensor = macos_impl::tensor_from_multi_array(&arr).expect("tensor_from_multi_array");
    assert_eq!(tensor.shape, shape);
    assert_eq!(tensor.data, data);
    assert_eq!(
        raw_as_f32, tensor.data,
        "read_raw_bytes and tensor_from_multi_array must agree on element values"
    );
}

/// `predict_raw` must run end-to-end against a real model and produce
/// dtype-tagged outputs whose values, once converted to `f32`, match
/// `predict()`'s own output for the same input.  Requires
/// `/tmp/w600k_r50.mlpackage`, same as the other ArcFace tests.
#[test]
#[ignore]
fn test_predict_raw_matches_predict_when_converted() {
    let model =
        MlPackageModel::load(ARCFACE_PATH, MlComputeUnits::All).expect("load arcface .mlpackage");
    let input_name = model
        .input_names()
        .into_iter()
        .next()
        .expect("at least one input");
    let output_name = model
        .output_names()
        .into_iter()
        .next()
        .expect("at least one output");

    let n = 3 * 112 * 112;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) / 1000.0).collect();
    let tensor = Tensor::new(data, vec![1, 3, 112, 112]);
    let mut inputs = HashMap::new();
    inputs.insert(input_name, tensor);

    let f32_outputs = model.predict(&inputs).expect("predict");
    let raw_outputs = model.predict_raw(&inputs).expect("predict_raw");

    let f32_out = f32_outputs
        .get(&output_name)
        .expect("predict output present");
    let raw_out = raw_outputs
        .get(&output_name)
        .expect("predict_raw output present");

    assert_eq!(raw_out.shape, f32_out.shape);
    let converted: Vec<f32> = match raw_out.dtype {
        MlArrayDtype::F32 => raw_out
            .data
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        MlArrayDtype::F16 => raw_out
            .data
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_ne_bytes([b[0], b[1]])).to_f32())
            .collect(),
        other => panic!("unexpected predict_raw output dtype {other:?}"),
    };
    assert_eq!(converted.len(), f32_out.data.len());
    for (a, b) in converted.iter().zip(f32_out.data.iter()) {
        assert!(
            (a - b).abs() <= 1e-3,
            "predict_raw (converted) and predict disagree: {a} vs {b}"
        );
    }
}

/// load_from_bytes is intentionally unsupported and must surface a
/// clean error instead of panicking or attempting to parse the bytes.
#[test]
fn load_from_bytes_returns_unsupported_format() {
    let r = MlPackageModel::load_from_bytes(&[], MlComputeUnits::All);
    match r {
        Err(CoreMLError::UnsupportedFormat(_)) => {}
        Err(other) => panic!("expected UnsupportedFormat, got {other:?}"),
        Ok(_) => panic!("expected UnsupportedFormat, got Ok"),
    }
}

/// load() of a non-existent path must produce CoreMLError::Io rather
/// than crossing into Objective-C with a bogus path.
#[test]
fn load_missing_path_returns_io_error() {
    let r = MlPackageModel::load(
        "/tmp/this/path/does/not/exist.mlpackage",
        MlComputeUnits::All,
    );
    match r {
        Err(CoreMLError::Io { .. }) => {}
        Err(other) => panic!("expected Io, got {other:?}"),
        Ok(_) => panic!("expected Io, got Ok"),
    }
}

// ────────────────────────────────────────────────────────────────── //
// predict_features: synthetic MLFeatureValue dispatch — no model     //
// needed, every value below is constructed directly.                //
// ────────────────────────────────────────────────────────────────── //

/// `MLFeatureValue::featureValueWithInt64` must report
/// `MLFeatureType::Int64` and dispatch to `FeatureOutput::Int64`.
#[test]
fn feature_value_int64_dispatches_correctly() {
    let fv = unsafe { MLFeatureValue::featureValueWithInt64(42) };
    assert_eq!(unsafe { fv.r#type() }, MLFeatureType::Int64);
    let out = macos_impl::feature_value_to_output(&fv).expect("Int64 dispatch");
    match out {
        FeatureOutput::Int64(v) => assert_eq!(v, 42),
        other => panic!("expected FeatureOutput::Int64, got {other:?}"),
    }
}

/// `MLFeatureValue::featureValueWithDouble` must report
/// `MLFeatureType::Double` and dispatch to `FeatureOutput::Double`.
#[test]
fn feature_value_double_dispatches_correctly() {
    let fv = unsafe { MLFeatureValue::featureValueWithDouble(3.5) };
    assert_eq!(unsafe { fv.r#type() }, MLFeatureType::Double);
    let out = macos_impl::feature_value_to_output(&fv).expect("Double dispatch");
    match out {
        // 3.5 is exactly representable in f64 and passes through
        // featureValueWithDouble/doubleValue without conversion, so
        // an exact comparison is correct here (no epsilon needed).
        FeatureOutput::Double(v) => assert_eq!(v, 3.5),
        other => panic!("expected FeatureOutput::Double, got {other:?}"),
    }
}

/// `MLFeatureValue::featureValueWithString` must report
/// `MLFeatureType::String` and dispatch to `FeatureOutput::String`.
#[test]
fn feature_value_string_dispatches_correctly() {
    let s = NSString::from_str("hello coreml");
    let fv = unsafe { MLFeatureValue::featureValueWithString(&s) };
    assert_eq!(unsafe { fv.r#type() }, MLFeatureType::String);
    let out = macos_impl::feature_value_to_output(&fv).expect("String dispatch");
    match out {
        FeatureOutput::String(v) => assert_eq!(v, "hello coreml"),
        other => panic!("expected FeatureOutput::String, got {other:?}"),
    }
}

/// `MLFeatureValue::featureValueWithMultiArray` must report
/// `MLFeatureType::MultiArray` and dispatch to
/// `FeatureOutput::MultiArray`, with the same values `predict()`'s
/// own extraction (`tensor_from_multi_array`) would produce.
#[test]
fn feature_value_multi_array_dispatches_correctly() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2usize, 3];
    let arr = macos_impl::multi_array_from_f32(&data, &shape)
        .expect("multi_array_from_f32 should build a 2x3 Float32 MLMultiArray");
    let fv = unsafe { MLFeatureValue::featureValueWithMultiArray(&arr) };
    assert_eq!(unsafe { fv.r#type() }, MLFeatureType::MultiArray);
    let out = macos_impl::feature_value_to_output(&fv).expect("MultiArray dispatch");
    match out {
        FeatureOutput::MultiArray(tensor) => {
            assert_eq!(tensor.shape, shape);
            assert_eq!(tensor.data, data);
        }
        other => panic!("expected FeatureOutput::MultiArray, got {other:?}"),
    }
}

/// `MLFeatureValue::featureValueWithDictionary_error` must report
/// `MLFeatureType::Dictionary` and dispatch to
/// `FeatureOutput::Dictionary`, with `NSString` keys stringified
/// directly and `NSNumber` values read as `f64`.
#[test]
fn feature_value_dictionary_dispatches_correctly() {
    let keys_owned: Vec<Retained<NSString>> =
        vec![NSString::from_str("cat"), NSString::from_str("dog")];
    let vals_owned: Vec<Retained<NSNumber>> = vec![NSNumber::new_f64(0.7), NSNumber::new_f64(0.3)];
    let key_refs: Vec<&NSString> = keys_owned.iter().map(|k| &**k).collect();
    let val_refs: Vec<&NSNumber> = vals_owned.iter().map(|v| &**v).collect();
    let dict: Retained<NSDictionary<NSString, NSNumber>> =
        NSDictionary::from_slices(&key_refs, &val_refs);
    // SAFETY: NSDictionary is structurally identical regardless of
    // its Rust type parameters (Objective-C's lightweight generics
    // are erased at the runtime level) — this mirrors
    // `make_provider`'s own `NSDictionary<NSString, MLFeatureValue>
    // -> NSDictionary<NSString, AnyObject>` reinterpret, just
    // widening the *key* parameter here instead of the value
    // parameter.
    let dict_any: &NSDictionary<AnyObject, NSNumber> = unsafe {
        &*(dict.as_ref() as *const NSDictionary<NSString, NSNumber>
            as *const NSDictionary<AnyObject, NSNumber>)
    };
    let fv = unsafe { MLFeatureValue::featureValueWithDictionary_error(dict_any) }
        .expect("featureValueWithDictionary_error should accept NSString keys / NSNumber values");
    assert_eq!(unsafe { fv.r#type() }, MLFeatureType::Dictionary);

    let out = macos_impl::feature_value_to_output(&fv).expect("Dictionary dispatch");
    match out {
        FeatureOutput::Dictionary(map) => {
            assert_eq!(map.len(), 2);
            assert!((map.get("cat").copied().expect("cat key present") - 0.7).abs() < 1e-9);
            assert!((map.get("dog").copied().expect("dog key present") - 0.3).abs() < 1e-9);
        }
        other => panic!("expected FeatureOutput::Dictionary, got {other:?}"),
    }
}

/// `MLSequence::sequenceWithInt64Array` wrapped in
/// `featureValueWithSequence` must report `MLFeatureType::Sequence`
/// and dispatch to `FeatureOutput::Sequence(SequenceValue::Int64(_))`.
#[test]
fn feature_value_sequence_int64_dispatches_correctly() {
    let nums: Vec<Retained<NSNumber>> = vec![
        NSNumber::new_i64(10),
        NSNumber::new_i64(20),
        NSNumber::new_i64(30),
    ];
    let arr: Retained<NSArray<NSNumber>> = NSArray::from_retained_slice(&nums);
    let seq = unsafe { MLSequence::sequenceWithInt64Array(&arr) };
    assert_eq!(unsafe { seq.r#type() }, MLFeatureType::Int64);
    let fv = unsafe { MLFeatureValue::featureValueWithSequence(&seq) };
    assert_eq!(unsafe { fv.r#type() }, MLFeatureType::Sequence);

    let out = macos_impl::feature_value_to_output(&fv).expect("Sequence dispatch");
    match out {
        FeatureOutput::Sequence(sv) => {
            assert_eq!(sv, SequenceValue::Int64(vec![10, 20, 30]));
        }
        other => panic!("expected FeatureOutput::Sequence, got {other:?}"),
    }
}

/// `MLSequence::sequenceWithStringArray` wrapped in
/// `featureValueWithSequence` must report `MLFeatureType::Sequence`
/// and dispatch to `FeatureOutput::Sequence(SequenceValue::String(_))`.
#[test]
fn feature_value_sequence_string_dispatches_correctly() {
    let strs: Vec<Retained<NSString>> =
        vec![NSString::from_str("alpha"), NSString::from_str("beta")];
    let arr: Retained<NSArray<NSString>> = NSArray::from_retained_slice(&strs);
    let seq = unsafe { MLSequence::sequenceWithStringArray(&arr) };
    assert_eq!(unsafe { seq.r#type() }, MLFeatureType::String);
    let fv = unsafe { MLFeatureValue::featureValueWithSequence(&seq) };
    assert_eq!(unsafe { fv.r#type() }, MLFeatureType::Sequence);

    let out = macos_impl::feature_value_to_output(&fv).expect("Sequence dispatch");
    match out {
        FeatureOutput::Sequence(sv) => {
            assert_eq!(
                sv,
                SequenceValue::String(vec!["alpha".to_string(), "beta".to_string()])
            );
        }
        other => panic!("expected FeatureOutput::Sequence, got {other:?}"),
    }
}

// ────────────────────────────────────────────────────────────────── //
// predict_features: synthetic CVPixelBuffer image dispatch — built   //
// via CVPixelBufferCreate, no model needed.                          //
// ────────────────────────────────────────────────────────────────── //

/// Build a `width`×`height` `CVPixelBuffer` of `pixel_format` with no
/// backing attributes (CoreVideo picks a standard, software-readable
/// layout — possibly with row padding, which is exactly the case
/// `copy_pixel_rows_to_f32` is built to handle correctly regardless).
/// `fill_row` is invoked once per row, write-locked, with that row's
/// *entire* byte span (including any padding past the row's logical
/// content) so the caller can write a deterministic pattern.
/// Test-only infrastructure for exercising `predict_features`'s
/// image decoder without a real CoreML model that produces an image
/// output.
fn make_test_pixel_buffer(
    width: usize,
    height: usize,
    pixel_format: u32,
    mut fill_row: impl FnMut(usize, &mut [u8]),
) -> Retained<CVPixelBuffer> {
    let mut out_ptr: *mut CVPixelBuffer = core::ptr::null_mut();
    // SAFETY: `out_ptr` is a valid, live `*mut CVPixelBuffer` local
    // to receive CoreVideo's "Create Rule" +1 owned reference;
    // `allocator: None` and `pixel_buffer_attributes: None` both
    // request CoreVideo's own defaults, which is valid per
    // `CVPixelBufferCreate`'s documented contract.
    let ret = unsafe {
        CVPixelBufferCreate(
            None,
            width,
            height,
            pixel_format,
            None,
            core::ptr::NonNull::from(&mut out_ptr),
        )
    };
    assert_eq!(ret, kCVReturnSuccess, "CVPixelBufferCreate failed: {ret}");
    // SAFETY: `CVPixelBufferCreate`'s "Create Rule" hands back
    // exactly one owned (+1) reference on success, which
    // `Retained::from_raw` claims without an additional retain — the
    // same pattern `compute_plan_summary`'s completion-handler
    // bridging uses for framework-provided +1 pointers.
    let buffer: Retained<CVPixelBuffer> = unsafe { Retained::from_raw(out_ptr) }
        .expect("CVPixelBufferCreate reported success but returned a null pixel buffer");

    // Read-write lock (the *empty* flag set — not ReadOnly) to fill
    // deterministic test data before handing the buffer to the
    // (read-only) decoder under test.
    let lock_ret =
        unsafe { CVPixelBufferLockBaseAddress(&buffer, CVPixelBufferLockFlags::empty()) };
    assert_eq!(
        lock_ret, kCVReturnSuccess,
        "CVPixelBufferLockBaseAddress (write) failed"
    );

    let bytes_per_row = CVPixelBufferGetBytesPerRow(&buffer);
    let base = CVPixelBufferGetBaseAddress(&buffer).cast::<u8>();
    assert!(
        !base.is_null(),
        "CVPixelBufferGetBaseAddress returned null after lock"
    );
    for row in 0..height {
        // SAFETY: test-only helper. `base` is this freshly-created,
        // now write-locked buffer's own base address;
        // `bytes_per_row * height` is exactly the allocation
        // `CVPixelBufferCreate` reserved for it (Apple's own
        // contract for `CVPixelBufferGetBytesPerRow` on a non-planar
        // buffer), and `row < height`.
        let row_slice = unsafe {
            core::slice::from_raw_parts_mut(base.add(row * bytes_per_row), bytes_per_row)
        };
        fill_row(row, row_slice);
    }

    let unlock_ret =
        unsafe { CVPixelBufferUnlockBaseAddress(&buffer, CVPixelBufferLockFlags::empty()) };
    assert_eq!(
        unlock_ret, kCVReturnSuccess,
        "CVPixelBufferUnlockBaseAddress (write) failed"
    );

    buffer
}

/// `OneComponent8` — the simplest supported image format: 1 byte per
/// pixel, raw value widened to `f32` verbatim, shape `[height,
/// width]`.
#[test]
fn feature_value_image_one_component8_dispatches_correctly() {
    let (width, height) = (4usize, 3usize);
    let buffer = make_test_pixel_buffer(
        width,
        height,
        kCVPixelFormatType_OneComponent8,
        |row, row_slice| {
            for (x, slot) in row_slice.iter_mut().enumerate().take(width) {
                *slot = (row * width + x) as u8;
            }
        },
    );
    let fv = unsafe { MLFeatureValue::featureValueWithPixelBuffer(&buffer) };
    assert_eq!(unsafe { fv.r#type() }, MLFeatureType::Image);

    let out = macos_impl::feature_value_to_output(&fv).expect("Image dispatch");
    match out {
        FeatureOutput::Image(tensor) => {
            assert_eq!(tensor.shape, vec![height, width]);
            let expected: Vec<f32> = (0..(width * height) as u32).map(|v| v as f32).collect();
            assert_eq!(tensor.data, expected);
        }
        other => panic!("expected FeatureOutput::Image, got {other:?}"),
    }
}

/// `32BGRA` — 4 bytes per pixel, channel order B, G, R, A exactly as
/// stored in memory, shape `[height, width, 4]`.
#[test]
fn feature_value_image_32bgra_dispatches_correctly() {
    let (width, height) = (2usize, 2usize);
    let buffer = make_test_pixel_buffer(
        width,
        height,
        kCVPixelFormatType_32BGRA,
        |row, row_slice| {
            for x in 0..width {
                let base = x * 4;
                let pixel = (row * width + x) as u8;
                row_slice[base] = pixel * 10 + 1; // B
                row_slice[base + 1] = pixel * 10 + 2; // G
                row_slice[base + 2] = pixel * 10 + 3; // R
                row_slice[base + 3] = pixel * 10 + 4; // A
            }
        },
    );
    let fv = unsafe { MLFeatureValue::featureValueWithPixelBuffer(&buffer) };
    assert_eq!(unsafe { fv.r#type() }, MLFeatureType::Image);

    let out = macos_impl::feature_value_to_output(&fv).expect("Image dispatch");
    match out {
        FeatureOutput::Image(tensor) => {
            assert_eq!(tensor.shape, vec![height, width, 4]);
            let mut expected = Vec::with_capacity(width * height * 4);
            for row in 0..height {
                for x in 0..width {
                    let pixel = (row * width + x) as u8;
                    expected.push((pixel * 10 + 1) as f32);
                    expected.push((pixel * 10 + 2) as f32);
                    expected.push((pixel * 10 + 3) as f32);
                    expected.push((pixel * 10 + 4) as f32);
                }
            }
            assert_eq!(tensor.data, expected);
        }
        other => panic!("expected FeatureOutput::Image, got {other:?}"),
    }
}

/// `OneComponent16Half` — 2-byte `f16` samples converted to `f32`;
/// values are chosen to round-trip exactly through `f16` (small
/// non-negative integers), so the assertion needs no epsilon.
#[test]
fn feature_value_image_one_component16_half_dispatches_correctly() {
    let (width, height) = (3usize, 2usize);
    let buffer = make_test_pixel_buffer(
        width,
        height,
        kCVPixelFormatType_OneComponent16Half,
        |row, row_slice| {
            for x in 0..width {
                let value = half::f16::from_f32((row * width + x) as f32);
                let bytes = value.to_bits().to_ne_bytes();
                row_slice[x * 2] = bytes[0];
                row_slice[x * 2 + 1] = bytes[1];
            }
        },
    );
    let fv = unsafe { MLFeatureValue::featureValueWithPixelBuffer(&buffer) };
    assert_eq!(unsafe { fv.r#type() }, MLFeatureType::Image);

    let out = macos_impl::feature_value_to_output(&fv).expect("Image dispatch");
    match out {
        FeatureOutput::Image(tensor) => {
            assert_eq!(tensor.shape, vec![height, width]);
            let expected: Vec<f32> = (0..(width * height) as u32).map(|v| v as f32).collect();
            assert_eq!(tensor.data, expected);
        }
        other => panic!("expected FeatureOutput::Image, got {other:?}"),
    }
}

/// `OneComponent32Float` — 4-byte native `f32` samples, read
/// verbatim.
#[test]
fn feature_value_image_one_component32_float_dispatches_correctly() {
    let (width, height) = (3usize, 2usize);
    let buffer = make_test_pixel_buffer(
        width,
        height,
        kCVPixelFormatType_OneComponent32Float,
        |row, row_slice| {
            for x in 0..width {
                let value = (row * width + x) as f32 * 0.5 - 1.0;
                let bytes = value.to_ne_bytes();
                row_slice[x * 4..x * 4 + 4].copy_from_slice(&bytes);
            }
        },
    );
    let fv = unsafe { MLFeatureValue::featureValueWithPixelBuffer(&buffer) };
    assert_eq!(unsafe { fv.r#type() }, MLFeatureType::Image);

    let out = macos_impl::feature_value_to_output(&fv).expect("Image dispatch");
    match out {
        FeatureOutput::Image(tensor) => {
            assert_eq!(tensor.shape, vec![height, width]);
            let expected: Vec<f32> = (0..(width * height) as u32)
                .map(|v| v as f32 * 0.5 - 1.0)
                .collect();
            assert_eq!(tensor.data, expected);
        }
        other => panic!("expected FeatureOutput::Image, got {other:?}"),
    }
}

/// A pixel format outside the four standard ones
/// `predict_features`'s image decoder supports must raise
/// `CoreMLError::UnsupportedPixelFormat`, not silently misinterpret
/// the buffer's bytes.
#[test]
fn feature_value_image_unsupported_format_errors_clearly() {
    let buffer = make_test_pixel_buffer(2, 2, kCVPixelFormatType_24RGB, |_row, _slice| {});
    let fv = unsafe { MLFeatureValue::featureValueWithPixelBuffer(&buffer) };
    assert_eq!(unsafe { fv.r#type() }, MLFeatureType::Image);

    let result = macos_impl::feature_value_to_output(&fv);
    match result {
        Err(CoreMLError::UnsupportedPixelFormat(_)) => {}
        Err(other) => panic!("expected UnsupportedPixelFormat, got {other:?}"),
        Ok(out) => panic!("expected UnsupportedPixelFormat, got Ok({out:?})"),
    }
}

// ────────────────────────────────────────────────────────────────── //
// model_metadata: helper-level unit tests — no model needed.         //
// ────────────────────────────────────────────────────────────────── //

/// `insert_metadata_string` must capture a present `NSString` value
/// under the requested destination key.
#[test]
fn insert_metadata_string_reads_present_string_value() {
    let key = NSString::from_str("com.example.author");
    let val: Retained<NSString> = NSString::from_str("Team Kitasan");
    let val_any: Retained<AnyObject> = Retained::from(val);
    let keys_owned = [NSString::from_str("com.example.author")];
    let vals_owned = [val_any];
    let key_refs: Vec<&NSString> = keys_owned.iter().map(|k| &**k).collect();
    let val_refs: Vec<&AnyObject> = vals_owned.iter().map(|v| &**v).collect();
    let dict: Retained<NSDictionary<NSString, AnyObject>> =
        NSDictionary::from_slices(&key_refs, &val_refs);

    let mut out = HashMap::new();
    macos_impl::insert_metadata_string(&dict, Some(&key), "author", &mut out);
    assert_eq!(out.get("author").map(String::as_str), Some("Team Kitasan"));
}

/// A value present under the key but of the wrong runtime class
/// (`NSNumber` instead of `NSString`) must be skipped, never
/// mis-decoded or panicked on.
#[test]
fn insert_metadata_string_skips_non_string_value() {
    let key = NSString::from_str("com.example.version");
    let val: Retained<NSNumber> = NSNumber::new_i32(3);
    let val_any: Retained<AnyObject> = Retained::from(val);
    let keys_owned = [NSString::from_str("com.example.version")];
    let vals_owned = [val_any];
    let key_refs: Vec<&NSString> = keys_owned.iter().map(|k| &**k).collect();
    let val_refs: Vec<&AnyObject> = vals_owned.iter().map(|v| &**v).collect();
    let dict: Retained<NSDictionary<NSString, AnyObject>> =
        NSDictionary::from_slices(&key_refs, &val_refs);

    let mut out = HashMap::new();
    macos_impl::insert_metadata_string(&dict, Some(&key), "version", &mut out);
    assert!(
        !out.contains_key("version"),
        "a non-NSString value must be skipped, not mis-decoded"
    );
}

/// A key absent from the dictionary must be skipped silently.
#[test]
fn insert_metadata_string_skips_missing_key() {
    let val: Retained<NSString> = NSString::from_str("Team Kitasan");
    let val_any: Retained<AnyObject> = Retained::from(val);
    let keys_owned = [NSString::from_str("com.example.author")];
    let vals_owned = [val_any];
    let key_refs: Vec<&NSString> = keys_owned.iter().map(|k| &**k).collect();
    let val_refs: Vec<&AnyObject> = vals_owned.iter().map(|v| &**v).collect();
    let dict: Retained<NSDictionary<NSString, AnyObject>> =
        NSDictionary::from_slices(&key_refs, &val_refs);

    let missing_key = NSString::from_str("com.example.missing");
    let mut out = HashMap::new();
    macos_impl::insert_metadata_string(&dict, Some(&missing_key), "description", &mut out);
    assert!(
        out.is_empty(),
        "a key absent from the dictionary must be skipped"
    );
}

/// `key: None` (the weakly-linked `MLModel*Key` symbol absent on
/// this CoreML framework revision) must be skipped silently, never
/// panicking on the `None`.
#[test]
fn insert_metadata_string_skips_none_key() {
    let val: Retained<NSString> = NSString::from_str("Team Kitasan");
    let val_any: Retained<AnyObject> = Retained::from(val);
    let keys_owned = [NSString::from_str("com.example.author")];
    let vals_owned = [val_any];
    let key_refs: Vec<&NSString> = keys_owned.iter().map(|k| &**k).collect();
    let val_refs: Vec<&AnyObject> = vals_owned.iter().map(|v| &**v).collect();
    let dict: Retained<NSDictionary<NSString, AnyObject>> =
        NSDictionary::from_slices(&key_refs, &val_refs);

    let mut out = HashMap::new();
    macos_impl::insert_metadata_string(&dict, None, "license", &mut out);
    assert!(
        out.is_empty(),
        "key: None must be skipped without panicking"
    );
}

/// An `NSString` key must stringify directly (no `-description`
/// fallback needed).
#[test]
fn any_object_key_to_string_uses_nsstring_directly() {
    let s: Retained<NSString> = NSString::from_str("plain-string-key");
    let any: Retained<AnyObject> = Retained::from(s);
    assert_eq!(
        macos_impl::any_object_key_to_string(any),
        "plain-string-key"
    );
}

/// A non-`NSString` key (`NSNumber`, exactly the case Apple's
/// `dictionaryValue()` contract allows for non-string dictionary
/// keys) must fall back to its Objective-C `-description`, which for
/// `NSNumber` is precisely its decimal string form.
#[test]
fn any_object_key_to_string_falls_back_to_description_for_non_string() {
    let n: Retained<NSNumber> = NSNumber::new_i32(42);
    let any: Retained<AnyObject> = Retained::from(n);
    assert_eq!(macos_impl::any_object_key_to_string(any), "42");
}
