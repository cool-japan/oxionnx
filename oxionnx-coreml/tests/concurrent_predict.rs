//! Concurrency stress test for `MlPackageModel` (Threading item, `TODO.md`).
//!
//! `MlPackageModel` carries manual `unsafe impl Send` / `unsafe impl Sync`
//! impls in `src/package.rs` — see the `SAFETY` comment attached to those
//! impls there for the underlying justification (Apple documents `MLModel`
//! as safe for concurrent `predictionFromFeatures_error:` calls on the same
//! instance).  Prior to this file the crate had no test that actually
//! exercised that contract with real concurrent `predict` calls; every
//! existing test drives the model from a single thread.
//!
//! This is the crate's first `tests/` (integration test) file.  Everything
//! below is gated to macOS/iOS/tvOS/visionOS, since `MlPackageModel`'s real
//! implementation only exists on those Apple platforms (see
//! `src/package.rs`'s `stub_impl` module for the non-Apple-platform
//! fallback) — on any other target this file compiles to an empty test
//! binary so `cargo test` / `cargo nextest run` stay green everywhere.
#![cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "visionos"
))]

use std::any::Any;
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use oxionnx_core::Tensor;
use oxionnx_coreml::{MlComputeUnits, MlPackageModel};

/// Environment variable naming a real `.mlpackage` (or already-compiled
/// `.mlmodelc`) bundle to drive this test with.
///
/// Unlike the crate's existing `#[ignore]`d tests (which hardcode
/// `/tmp/w600k_r50.mlpackage` and `.expect()`-panic when it is absent),
/// this test reads the path from the environment and skips gracefully
/// instead — no hardcoded absolute paths, per project policy.
const MODEL_PATH_ENV: &str = "OXIONNX_COREML_TEST_MODEL";

/// Number of worker threads sharing one `Arc<MlPackageModel>`.
const THREAD_COUNT: usize = 8;

/// Number of `predict` calls each worker thread performs.
const PREDICTIONS_PER_THREAD: usize = 20;

/// Element count of the ArcFace-shaped synthetic input (1x3x112x112),
/// matching the fixture convention used by
/// `test_predict_arcface_returns_512_dim_embedding` in `src/package.rs`.
const ARCFACE_ELEMENT_COUNT: usize = 3 * 112 * 112;

/// Expected ArcFace embedding dimension (the flattened output size).
const ARCFACE_EMBEDDING_DIM: usize = 512;

/// Resolve [`MODEL_PATH_ENV`] to an existing filesystem path.
///
/// Returns `None` (after printing a skip note to stdout) when the
/// variable is unset or points at a path that does not exist, so callers
/// can degrade gracefully instead of panicking in environments without
/// the model fixture.
fn resolve_model_path() -> Option<PathBuf> {
    let Ok(raw) = env::var(MODEL_PATH_ENV) else {
        println!(
            "skipping concurrent_predict_from_shared_arc: environment variable \
             {MODEL_PATH_ENV} is not set (export it to a real .mlpackage/.mlmodelc \
             path to run this test)"
        );
        return None;
    };
    let path = PathBuf::from(raw);
    if !path.exists() {
        println!(
            "skipping concurrent_predict_from_shared_arc: {MODEL_PATH_ENV} points at \
             {} which does not exist on disk",
            path.display()
        );
        return None;
    }
    Some(path)
}

/// Build a fresh ArcFace-shaped synthetic input map keyed by `input_name`.
///
/// `seed` perturbs the generated values so distinct threads/iterations are
/// not bit-identical; the runtime has no such requirement — this is purely
/// to make individual calls distinguishable under a debugger.
fn synthetic_inputs(input_name: &str, seed: usize) -> HashMap<String, Tensor> {
    let data: Vec<f32> = (0..ARCFACE_ELEMENT_COUNT)
        .map(|i| ((i + seed) as f32) / 1000.0)
        .collect();
    let tensor = Tensor::new(data, vec![1, 3, 112, 112]);
    let mut inputs = HashMap::new();
    inputs.insert(input_name.to_string(), tensor);
    inputs
}

/// Best-effort extraction of a human-readable message from a
/// `JoinHandle::join` panic payload, for informative assertion failures
/// when a worker thread panics.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// Exercises `MlPackageModel`'s documented `Send + Sync` contract: loads
/// the model once, wraps it in a single `Arc`, and fires real concurrent
/// `predict` calls from [`THREAD_COUNT`] threads (each running
/// [`PREDICTIONS_PER_THREAD`] predictions) against that one shared
/// instance.  Every worker's result is checked for the expected ArcFace
/// output shape, and every worker's `JoinHandle` is joined so a panic on
/// any thread fails this test instead of vanishing silently.
///
/// Assumes an ArcFace-shaped model (single 1x3x112x112 input, single
/// 512-element output), consistent with this crate's other model-backed
/// fixtures.
///
/// Requires a real bundle at the path named by the
/// `OXIONNX_COREML_TEST_MODEL` environment variable — skips gracefully
/// (prints a note, returns) when that variable is unset or the path is
/// missing, and is `#[ignore]`d so default `cargo test` / `cargo nextest
/// run` never attempts it.  Run explicitly with:
///
/// ```text
/// OXIONNX_COREML_TEST_MODEL=/path/to/w600k_r50.mlpackage \
///     cargo test -p oxionnx-coreml --test concurrent_predict -- --ignored
/// ```
#[test]
#[ignore]
fn concurrent_predict_from_shared_arc() {
    let Some(model_path) = resolve_model_path() else {
        return;
    };

    let model = MlPackageModel::load(model_path, MlComputeUnits::All)
        .expect("load model bundle from OXIONNX_COREML_TEST_MODEL");
    let input_name = model
        .input_names()
        .into_iter()
        .next()
        .expect("model declares at least one input");
    let output_name = model
        .output_names()
        .into_iter()
        .next()
        .expect("model declares at least one output");

    let shared_model = Arc::new(model);
    let mut handles = Vec::with_capacity(THREAD_COUNT);
    for thread_idx in 0..THREAD_COUNT {
        let shared_model = Arc::clone(&shared_model);
        let input_name = input_name.clone();
        let output_name = output_name.clone();
        let builder = thread::Builder::new().name(format!("concurrent-predict-{thread_idx}"));
        let handle = builder
            .spawn(move || {
                for iter_idx in 0..PREDICTIONS_PER_THREAD {
                    let seed = thread_idx * PREDICTIONS_PER_THREAD + iter_idx;
                    let inputs = synthetic_inputs(&input_name, seed);
                    let outputs = shared_model
                        .predict(&inputs)
                        .expect("predict call from worker thread");
                    let out = outputs
                        .get(&output_name)
                        .expect("declared output present in result map");
                    assert_eq!(
                        out.data.len(),
                        ARCFACE_EMBEDDING_DIM,
                        "thread {thread_idx} iter {iter_idx}: unexpected embedding length"
                    );
                    assert_eq!(
                        out.shape.iter().product::<usize>(),
                        ARCFACE_EMBEDDING_DIM,
                        "thread {thread_idx} iter {iter_idx}: unexpected output shape"
                    );
                }
            })
            .expect("spawn worker thread");
        handles.push(handle);
    }

    let mut panicked = Vec::new();
    for (thread_idx, handle) in handles.into_iter().enumerate() {
        if let Err(payload) = handle.join() {
            panicked.push(format!(
                "thread {thread_idx} panicked: {}",
                panic_message(&*payload)
            ));
        }
    }
    assert!(
        panicked.is_empty(),
        "{} of {THREAD_COUNT} worker threads panicked:\n{}",
        panicked.len(),
        panicked.join("\n")
    );
}
