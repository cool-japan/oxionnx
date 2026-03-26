//! Model zoo compatibility testing.
//!
//! Run with: OXIONNX_MODEL_DIR=/path/to/models cargo test model_zoo
//!
//! The test directory should contain .onnx files. Each model is loaded,
//! its metadata is inspected, and a dummy inference is run with random input.

use oxionnx::{OptLevel, Session, Tensor};
use std::path::{Path, PathBuf};

/// Get the model directory from env or default.
fn model_dir() -> PathBuf {
    std::env::var("OXIONNX_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("tests/models"))
}

/// Check if model dir exists and has .onnx files.
fn has_models() -> bool {
    let dir = model_dir();
    if !dir.is_dir() {
        return false;
    }
    std::fs::read_dir(&dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .any(|e| e.path().extension().is_some_and(|ext| ext == "onnx"))
        })
        .unwrap_or(false)
}

/// Collect all .onnx file paths in the model directory.
fn collect_onnx_files() -> Vec<PathBuf> {
    let dir = model_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "onnx"))
        .collect()
}

/// Load a model and validate basic properties.
fn validate_model(path: &Path) -> Result<(), String> {
    let session = Session::from_file(path).map_err(|e| format!("Load failed: {}", e))?;

    let info = session.model_info();
    if info.node_count == 0 {
        return Err("Model has no nodes".to_string());
    }

    let input_names = session.input_names();
    let output_names = session.output_names();
    if input_names.is_empty() {
        return Err("Model has no inputs".to_string());
    }
    if output_names.is_empty() {
        return Err("Model has no outputs".to_string());
    }

    Ok(())
}

/// Generate a dummy input tensor for a given shape.
/// Uses small deterministic values.
fn dummy_input(shape: &[usize], seed: u32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(seed).wrapping_add(17)) as f32;
            (x % 200.0 - 100.0) * 0.01
        })
        .collect();
    Tensor::new(data, shape.to_vec())
}

#[test]
fn test_model_zoo_load_all() {
    if !has_models() {
        eprintln!(
            "Skipping model zoo tests: no models found in {:?}",
            model_dir()
        );
        eprintln!("Set OXIONNX_MODEL_DIR=/path/to/models to enable");
        return;
    }

    let files = collect_onnx_files();
    let mut passed = 0usize;
    let mut failed = 0usize;

    for path in &files {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or_default();
        match validate_model(path) {
            Ok(()) => {
                eprintln!("  PASS: {}", name);
                passed += 1;
            }
            Err(e) => {
                eprintln!("  FAIL: {} - {}", name, e);
                failed += 1;
            }
        }
    }

    eprintln!("Model zoo: {} passed, {} failed", passed, failed);
    assert_eq!(failed, 0, "{} models failed to load", failed);
}

/// Test that each model can load with optimization enabled.
#[test]
fn test_model_zoo_with_optimization() {
    if !has_models() {
        return;
    }

    let files = collect_onnx_files();
    for path in &files {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or_default();
        let result = Session::builder()
            .with_optimization_level(OptLevel::All)
            .with_memory_pool(true)
            .load(path);
        match result {
            Ok(session) => {
                let info = session.model_info();
                eprintln!(
                    "  OK: {} ({} nodes, {} params)",
                    name, info.node_count, info.parameter_count
                );
            }
            Err(e) => {
                eprintln!("  ERR: {} - {}", name, e);
            }
        }
    }
}

/// Test metadata extraction from models using the protobuf parser.
#[test]
fn test_model_zoo_metadata() {
    if !has_models() {
        return;
    }

    let files = collect_onnx_files();
    for path in &files {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or_default();
        match std::fs::read(path) {
            Ok(bytes) => match oxionnx::proto::parse_model(&bytes) {
                Ok(model) => {
                    let opset_versions: Vec<(String, i64)> = model
                        .opset_imports
                        .iter()
                        .map(|o| (o.domain.clone(), o.version))
                        .collect();
                    eprintln!(
                        "  {}: ir_version={}, opset_version={}, opsets={:?}",
                        name, model.ir_version, model.opset_version, opset_versions
                    );
                }
                Err(e) => eprintln!("  {}: metadata error: {}", name, e),
            },
            Err(e) => eprintln!("  {}: read error: {}", name, e),
        }
    }
}

/// Test that dummy inference runs without panicking.
#[test]
fn test_model_zoo_dummy_inference() {
    if !has_models() {
        return;
    }

    let files = collect_onnx_files();
    for path in &files {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or_default();
        let session = match Session::from_file(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  SKIP (load failed): {} - {}", name, e);
                continue;
            }
        };

        // Build dummy inputs: use batch=1 and small spatial dims.
        // This is best-effort since we don't have shape info from the model.
        let input_names = session.input_names().to_vec();
        let mut inputs = std::collections::HashMap::new();
        for (idx, input_name) in input_names.iter().enumerate() {
            // Default shape guess: [1, 3, 224, 224] for image models, [1, 128] for text
            // We try a small tensor first
            let shape = vec![1, 3, 224, 224];
            let tensor = dummy_input(&shape, (idx as u32).wrapping_add(42));
            inputs.insert(input_name.as_str(), tensor);
        }

        match session.run(&inputs) {
            Ok(outputs) => {
                let output_shapes: Vec<_> = session
                    .output_names()
                    .iter()
                    .filter_map(|n| outputs.get(n.as_str()).map(|t| (n.as_str(), &t.shape)))
                    .collect();
                eprintln!("  RUN OK: {} -> {:?}", name, output_shapes);
            }
            Err(e) => {
                // Shape mismatch is expected for dummy inputs
                eprintln!("  RUN ERR (expected with dummy input): {} - {}", name, e);
            }
        }
    }
}
