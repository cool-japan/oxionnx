//! Model-gated end-to-end proof that `.mlpackage` compilation happens
//! **once**, is served from a persistent on-disk cache afterwards, and
//! leaves nothing behind in `$TMPDIR`.
//!
//! Before the cache existed, `MlPackageModel::load` called
//! `MLModel::compileModelAtURL_error:` on every load.  Apple returns a
//! fresh UUID-named directory under `$TMPDIR` for each such call and
//! never reuses or reaps it, so every process start paid seconds of
//! recompilation *and* leaked the result (7,408 orphaned `.mlmodelc`
//! trees / 857 GB were measured on one developer machine).  This test
//! pins down both halves of the fix.
//!
//! Gated to the Apple platforms `MlPackageModel` actually exists on;
//! everywhere else this file compiles to an empty test binary, exactly
//! like `concurrent_predict.rs`.
#![cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "visionos"
))]

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use oxionnx_coreml::{MlComputeUnits, MlPackageModel};

/// Environment variable naming a real `.mlpackage` bundle to drive this
/// test with — the same one every other model-gated test in this crate
/// reads.
const MODEL_PATH_ENV: &str = "OXIONNX_COREML_TEST_MODEL";

/// Environment variable the runtime reads to relocate its compile
/// cache.  Setting it to a fresh directory is what makes this test
/// observable (and keeps it from disturbing the developer's real
/// cache).
const CACHE_DIR_ENV: &str = "OXIONNX_COREML_CACHE_DIR";

/// Suffix identifying a compiled CoreML bundle, both in the cache and
/// in `$TMPDIR`.
const COMPILED_SUFFIX: &str = ".mlmodelc";

/// Resolve [`MODEL_PATH_ENV`] to an existing `.mlpackage`.
///
/// Returns `None` (after printing a skip note) when unset, missing, or
/// already compiled — an input that is itself a `.mlmodelc` bypasses
/// compilation entirely and would make every assertion below vacuous.
fn resolve_package_path() -> Option<PathBuf> {
    let Ok(raw) = env::var(MODEL_PATH_ENV) else {
        println!(
            "skipping compile-cache test: {MODEL_PATH_ENV} is not set (export it to a \
             real .mlpackage path to run this test)"
        );
        return None;
    };
    let path = PathBuf::from(raw);
    if !path.exists() {
        println!(
            "skipping compile-cache test: {MODEL_PATH_ENV} points at {} which does not \
             exist on disk",
            path.display()
        );
        return None;
    }
    if path.extension().and_then(|s| s.to_str()) == Some("mlmodelc") {
        println!(
            "skipping compile-cache test: {MODEL_PATH_ENV} names an already-compiled \
             bundle ({}); the cache only engages for .mlpackage sources",
            path.display()
        );
        return None;
    }
    Some(path)
}

/// Names of the compiled-bundle directories sitting directly in the
/// process temporary directory.
///
/// This is the framework's own scratch space, so the set is shared with
/// anything else on the machine that compiles a CoreML model — but
/// nothing else in this test binary does, and the window being observed
/// is a few hundred milliseconds wide.
fn temp_dir_compiled_bundles() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let Ok(entries) = fs::read_dir(env::temp_dir()) else {
        return names;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(COMPILED_SUFFIX) {
            names.insert(name);
        }
    }
    names
}

/// Names of the cache entries under `root`.
fn cache_entries(root: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(COMPILED_SUFFIX))
        .collect();
    names.sort();
    names
}

/// A fresh, uniquely named cache root for this run.
fn fresh_cache_root() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    env::temp_dir().join(format!(
        "oxionnx-coreml-cache-e2e-{}-{stamp}",
        std::process::id()
    ))
}

/// Two consecutive resolutions of the same `.mlpackage` must compile it
/// exactly once:
///
/// * both return the *same* path, inside the configured cache root;
/// * the cache holds exactly one entry for the bundle, before and after;
/// * the second resolution is dramatically faster (no compile at all);
/// * no compiled bundle is left behind in `$TMPDIR` by either — the
///   leak this cache exists to fix;
/// * a full `MlPackageModel::load` on top of the warm cache still works
///   and still adds no entries.
///
/// Requires a real `.mlpackage` at `OXIONNX_COREML_TEST_MODEL`; skips
/// gracefully otherwise, and is `#[ignore]`d so default runs never
/// attempt it.  Run explicitly with:
///
/// ```text
/// OXIONNX_COREML_TEST_MODEL=/path/to/w600k_r50.mlpackage \
///     cargo test -p oxionnx-coreml --test compile_cache -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn second_load_is_served_from_the_persistent_compile_cache() {
    let Some(package) = resolve_package_path() else {
        return;
    };

    let cache_root = fresh_cache_root();
    let _ = fs::remove_dir_all(&cache_root);
    // Safe on this crate's edition (2021): `set_var`'s unsafety is an
    // edition-2024 change.  This binary runs exactly one test, so
    // there is no concurrent reader of the environment.
    env::set_var(CACHE_DIR_ENV, &cache_root);

    let temp_before = temp_dir_compiled_bundles();

    let started = Instant::now();
    let first = MlPackageModel::ensure_compiled(&package).expect("first ensure_compiled");
    let first_elapsed = started.elapsed();

    let temp_after_first = temp_dir_compiled_bundles();

    let started = Instant::now();
    let second = MlPackageModel::ensure_compiled(&package).expect("second ensure_compiled");
    let second_elapsed = started.elapsed();

    let temp_after_second = temp_dir_compiled_bundles();

    println!(
        "compile cache: first ensure_compiled {first_elapsed:?}, second {second_elapsed:?} \
         (entry: {})",
        first.display()
    );

    assert_eq!(
        first, second,
        "both resolutions must name the same cache entry"
    );
    assert!(
        first.starts_with(&cache_root),
        "the compiled bundle must live under {} , got {}",
        cache_root.display(),
        first.display()
    );
    assert!(
        first.is_dir(),
        "the cache entry must be a real compiled bundle directory"
    );

    assert_eq!(
        cache_entries(&cache_root),
        vec![first
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()],
        "the cache must hold exactly one entry for one bundle"
    );

    assert!(
        second_elapsed < first_elapsed,
        "the cached resolution must be faster than the compiling one \
         (first {first_elapsed:?}, second {second_elapsed:?})"
    );

    // The leak assertions.  Installation *moves* the framework's
    // temporary directory into the cache, so the first resolution must
    // not add one either — and the second never compiles at all.
    let new_after_first: Vec<&String> = temp_after_first.difference(&temp_before).collect();
    assert!(
        new_after_first.is_empty(),
        "compiling must not leave a bundle in the temp dir, found {new_after_first:?}"
    );
    let new_after_second: Vec<&String> = temp_after_second.difference(&temp_after_first).collect();
    assert!(
        new_after_second.is_empty(),
        "a cached resolution must not compile at all, found {new_after_second:?}"
    );

    // A real load on the warm cache must still work end to end, and
    // must not add a second entry for the same bundle.
    let started = Instant::now();
    let model = MlPackageModel::load(&package, MlComputeUnits::All)
        .expect("load must succeed against a warm cache");
    println!(
        "compile cache: warm MlPackageModel::load {:?}",
        started.elapsed()
    );
    assert!(
        !model.input_names().is_empty(),
        "a loaded model must declare at least one input"
    );
    assert!(
        !model.output_names().is_empty(),
        "a loaded model must declare at least one output"
    );
    assert_eq!(
        cache_entries(&cache_root).len(),
        1,
        "loading against a warm cache must not add entries"
    );
    assert!(
        temp_dir_compiled_bundles()
            .difference(&temp_after_first)
            .next()
            .is_none(),
        "loading against a warm cache must not compile"
    );

    let _ = fs::remove_dir_all(&cache_root);
    env::remove_var(CACHE_DIR_ENV);
}
