//! Persistent on-disk cache for CoreML-compiled `.mlmodelc` bundles.
//!
//! ## The problem
//!
//! `MLModel::compileModelAtURL_error:` compiles a `.mlpackage` into a
//! **fresh, UUID-named directory under `$TMPDIR`** on every single call.
//! The framework neither reuses a previous compile for the same bundle
//! nor deletes the one it just produced, so a process that loads models
//! at startup pays the full multi-second compile every launch *and*
//! leaves the result behind forever.  Measured on one developer machine
//! running OxiFace: 7,408 orphaned `.mlmodelc` trees, 857 GB, in
//! `$TMPDIR`.
//!
//! ## The fix
//!
//! Compile once into Apple's temporary directory, then **move** that
//! directory into a stable, content-keyed location under the user's
//! cache directory.  Subsequent loads find the entry and skip the
//! compile entirely; the temporary directory is either renamed away or
//! deleted on every path, so nothing accumulates.
//!
//! ## Layout and keying
//!
//! ```text
//! <cache root>/<sanitized file stem>-<fingerprint:016x>.mlmodelc
//! ```
//!
//! The fingerprint is a 64-bit FNV-1a fold over the sorted
//! `(relative path, length, mtime)` triple of every file in the source
//! bundle, plus [`CACHE_FORMAT_VERSION`].  It is deliberately
//! *metadata-only*: a `.mlpackage` carries hundreds of megabytes of
//! weight blobs, and content-hashing them on every load would cost more
//! than the compile the cache exists to skip.  Every ordinary writer
//! (conversion script, download, `cp`, `git`) bumps a file's mtime, so
//! in-place edits that preserve both length and mtime are the only way
//! to defeat the key.
//!
//! ## Concurrency
//!
//! Two processes compiling the same bundle at the same time each
//! produce their own temporary directory and then attempt an atomic
//! `rename` into the shared entry name.  POSIX `rename` onto a
//! non-empty directory fails, so exactly one wins; the loser observes
//! the now-existing entry, deletes its own copy and uses the winner's.
//! No lock file, no partially visible entry: a reader either sees a
//! complete `.mlmodelc` or nothing at all.  The cross-device fallback
//! (`$TMPDIR` and the cache root on different volumes, where `rename`
//! cannot work) copies into a hidden staging directory *first* and only
//! then renames it into place, preserving that same guarantee.
//!
//! ## Degradation
//!
//! Every failure to use the cache — unwritable root, full disk, a
//! cross-device copy that fails halfway — falls back to returning the
//! framework's own temporary directory, i.e. exactly the pre-cache
//! behavior, rather than failing a load that would otherwise succeed.

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::Result;

/// Environment variable overriding the cache root, for callers that
/// want the cache on a specific volume (or an isolated one per test).
const CACHE_DIR_ENV: &str = "OXIONNX_COREML_CACHE_DIR";

/// Directory name appended to the platform cache root.
const CACHE_DIR_NAME: &str = "oxionnx-coreml";

/// Suffix every cache entry carries — also what makes an entry
/// self-describing enough to load directly (`MlPackageModel::load`
/// treats a `.mlmodelc` path as already compiled).
const ENTRY_SUFFIX: &str = ".mlmodelc";

/// Bumping this invalidates every previously written entry without
/// needing to delete the cache directory: it is folded into the
/// fingerprint, so old entries simply stop being looked up.
const CACHE_FORMAT_VERSION: u32 = 1;

/// Upper bound on the human-readable part of an entry name, so a
/// pathological bundle name cannot blow past the filesystem's
/// per-component length limit.
const MAX_KEY_CHARS: usize = 64;

/// Depth limit for the fingerprint walk — a `.mlpackage` is two or
/// three levels deep; anything past this is a symlink loop or a
/// mis-pointed path, not a model bundle.
const MAX_WALK_DEPTH: u32 = 16;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold `bytes` into an in-progress FNV-1a hash.
fn fold(hash: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *hash ^= u64::from(*b);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

/// Resolve the cache root, honoring `$OXIONNX_COREML_CACHE_DIR` first
/// and falling back to the platform cache directory, then the system
/// temporary directory.
///
/// Never fails: the directory is created lazily at install time, and an
/// unwritable root degrades to the no-cache path rather than erroring.
fn cache_root() -> PathBuf {
    if let Some(dir) = env::var_os(CACHE_DIR_ENV) {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    if let Some(home) = env::var_os("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home)
                .join("Library")
                .join("Caches")
                .join(CACHE_DIR_NAME);
        }
    }
    env::temp_dir().join(CACHE_DIR_NAME)
}

/// Reduce `name` to a filesystem-safe, bounded ASCII component.
///
/// Every emitted character is ASCII, so the length cap can never split
/// a multi-byte `char`.
fn sanitize(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    out.truncate(MAX_KEY_CHARS);
    if out.is_empty() {
        out.push_str("model");
    }
    out
}

/// Fold one file's identity into `hash`: its path relative to the
/// bundle root, its length, and its modification time in nanoseconds.
///
/// An unreadable mtime folds as 0 rather than failing the whole
/// fingerprint — a file whose timestamp cannot be read still
/// contributes its path and length.
fn fold_file(hash: &mut u64, rel: &str, len: u64, mtime_nanos: u128) {
    fold(hash, rel.as_bytes());
    fold(hash, &len.to_le_bytes());
    fold(hash, &mtime_nanos.to_le_bytes());
}

/// Collect `(relative path, length, mtime nanos)` for every regular
/// file under `dir`, recursively.
///
/// Directory iteration order is not stable across filesystems, so the
/// caller sorts before folding.  Symlinks are recorded by name/length
/// but never followed, which is what bounds the walk together with
/// `depth`.
fn collect_files(dir: &Path, prefix: &str, depth: u32, out: &mut Vec<(String, u64, u128)>) {
    if depth > MAX_WALK_DEPTH {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let rel = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            collect_files(&entry.path(), &rel, depth + 1, out);
        } else if let Ok(meta) = entry.metadata() {
            out.push((rel, meta.len(), mtime_nanos(&meta)));
        }
    }
}

/// Modification time as nanoseconds since the Unix epoch, or 0 when the
/// platform cannot report one.
fn mtime_nanos(meta: &fs::Metadata) -> u128 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos())
}

/// Cheap content fingerprint for a model bundle — see the module header
/// for why this is metadata-only rather than a content hash.
fn fingerprint(path: &Path) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    fold(&mut hash, &CACHE_FORMAT_VERSION.to_le_bytes());

    match fs::metadata(path) {
        Ok(meta) if meta.is_dir() => {
            let mut files: Vec<(String, u64, u128)> = Vec::new();
            collect_files(path, "", 0, &mut files);
            files.sort();
            for (rel, len, mtime) in &files {
                fold_file(&mut hash, rel, *len, *mtime);
            }
        }
        Ok(meta) => {
            // Not a directory bundle — fingerprint the single file.
            fold_file(&mut hash, "", meta.len(), mtime_nanos(&meta));
        }
        Err(_) => {
            // Unreadable: fold the path itself so the key is at least
            // stable, and let the compile step report the real error.
            fold(&mut hash, path.as_os_str().as_encoded_bytes());
        }
    }
    hash
}

/// Cache entry path for `source` under an explicit `root`.
fn entry_path_in(root: &Path, source: &Path) -> PathBuf {
    let stem = source
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("model");
    let key = sanitize(stem);
    let fp = fingerprint(source);
    root.join(format!("{key}-{fp:016x}{ENTRY_SUFFIX}"))
}

/// Cache entry path for `source` under the resolved [`cache_root`].
fn entry_path(source: &Path) -> PathBuf {
    entry_path_in(&cache_root(), source)
}

/// Best-effort recursive delete — every caller is on a cleanup path
/// where failing to remove is strictly less bad than propagating.
fn remove_tree(path: &Path) {
    let _ = fs::remove_dir_all(path);
}

/// Unique, hidden sibling of `target` to assemble a cross-device copy
/// in, so a partially written tree never carries the final entry name.
fn staging_path(target: &Path) -> PathBuf {
    static NONCE: AtomicU64 = AtomicU64::new(0);
    let nonce = NONCE.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let name = target
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or("entry");
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!(".{name}.staging-{pid}-{stamp}-{nonce}"))
}

/// Recursively copy `src` into `dst`, creating `dst` and every
/// intermediate directory.
fn copy_dir_all(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_all(&entry.path(), &to)?;
        } else {
            fs::copy(entry.path(), &to)?;
        }
    }
    Ok(())
}

/// Move `temp` into the cache as `target`, atomically with respect to
/// any other process racing on the same key.
///
/// On success `temp` no longer exists — it was either renamed away or
/// deleted — which is the whole point of this function: the framework's
/// temporary directory must never survive a load.
fn install(temp: &Path, target: &Path) -> io::Result<()> {
    let parent = target.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "cache entry path has no parent directory",
        )
    })?;
    fs::create_dir_all(parent)?;

    // Cheap path: a same-volume rename is atomic, so no other process
    // can ever observe a half-written entry.
    if fs::rename(temp, target).is_ok() {
        return Ok(());
    }
    // Lost a race — another process installed this key between our
    // lookup and this rename.  Its tree was compiled from the same
    // (path, fingerprint) pair, so ours is redundant.
    if target.is_dir() {
        remove_tree(temp);
        return Ok(());
    }

    // Otherwise assume the rename failed because `temp` and the cache
    // root are on different volumes (the framework compiles into
    // `$TMPDIR`, which need not share a device with `$HOME`) and copy
    // instead.  Staging first keeps the "never a partial entry name"
    // guarantee; probing `errno` for `EXDEV` would add a platform
    // constant for no extra safety, since a copy that fails for any
    // other reason lands in the same degradation path anyway.
    let staging = staging_path(target);
    remove_tree(&staging);
    if let Err(err) = copy_dir_all(temp, &staging) {
        remove_tree(&staging);
        return Err(err);
    }
    match fs::rename(&staging, target) {
        Ok(()) => {
            remove_tree(temp);
            Ok(())
        }
        Err(err) => {
            remove_tree(&staging);
            if target.is_dir() {
                // Same race as above, just resolved later.
                remove_tree(temp);
                Ok(())
            } else {
                Err(err)
            }
        }
    }
}

/// Resolve `source` to a compiled `.mlmodelc` directory, invoking
/// `compile` at most once and caching its result.
///
/// `compile` receives `source` and must return a directory it owns
/// outright — this function will move or delete it.
pub(super) fn compile_cached<F>(source: &Path, compile: F) -> Result<PathBuf>
where
    F: FnOnce(&Path) -> Result<PathBuf>,
{
    compile_cached_in(&cache_root(), source, compile)
}

/// [`compile_cached`] against an explicit cache root — the seam the
/// unit tests drive, so they never touch the real cache directory or
/// the process environment.
fn compile_cached_in<F>(root: &Path, source: &Path, compile: F) -> Result<PathBuf>
where
    F: FnOnce(&Path) -> Result<PathBuf>,
{
    let target = entry_path_in(root, source);
    if target.is_dir() {
        return Ok(target);
    }
    let temp = compile(source)?;
    match install(&temp, &target) {
        Ok(()) => Ok(target),
        // The cache is unusable (read-only volume, quota, a root that
        // is actually a file, ...).  Degrade to the framework's own
        // temporary directory — exactly the pre-cache behavior —
        // rather than failing a load that would otherwise succeed.
        Err(_) => Ok(temp),
    }
}

/// Evict `compiled` when it is *our* cache entry for `source`, so the
/// caller may recompile and retry exactly once.
///
/// Returns `false` — meaning "do not retry" — when `compiled` is not a
/// cache entry at all: a caller-supplied `.mlmodelc`, or the degraded
/// framework-temp fallback above.  Recompiling those would only
/// reproduce whatever failure prompted the call.
pub(super) fn evict(source: &Path, compiled: &Path) -> bool {
    if compiled != entry_path(source) {
        return false;
    }
    remove_tree(compiled);
    !compiled.exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CoreMLError;

    /// Fresh, uniquely named scratch directory under the system temp
    /// dir — these tests create real trees, so they cannot share one.
    fn scratch(tag: &str) -> PathBuf {
        static NONCE: AtomicU64 = AtomicU64::new(0);
        let nonce = NONCE.fetch_add(1, Ordering::Relaxed);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir = env::temp_dir().join(format!(
            "oxionnx-coreml-cachetest-{tag}-{}-{stamp}-{nonce}",
            std::process::id()
        ));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    /// Build a minimal stand-in for a `.mlpackage`: a directory with a
    /// manifest and a nested weight blob.
    fn fake_bundle(root: &Path, name: &str, weights: &[u8]) -> PathBuf {
        let bundle = root.join(name);
        let data = bundle.join("Data").join("com.apple.CoreML");
        let _ = fs::create_dir_all(&data);
        let _ = fs::write(bundle.join("Manifest.json"), b"{}");
        let _ = fs::write(data.join("weights.bin"), weights);
        bundle
    }

    /// Stand-in for `MLModel::compileModelAtURL_error:`: materializes a
    /// fresh directory under `temp_root` (as the framework does) and
    /// counts its invocations.
    fn fake_compile(temp_root: &Path, calls: &std::cell::Cell<usize>) -> PathBuf {
        let n = calls.get();
        calls.set(n + 1);
        let dir = temp_root.join(format!("compiled-{n}.mlmodelc"));
        let _ = fs::create_dir_all(dir.join("model"));
        let _ = fs::write(dir.join("model").join("coremldata.bin"), b"compiled");
        dir
    }

    #[test]
    fn second_resolve_reuses_the_entry_without_recompiling() {
        let root = scratch("reuse");
        let cache = root.join("cache");
        let temp_root = root.join("tmp");
        let _ = fs::create_dir_all(&temp_root);
        let bundle = fake_bundle(&root, "demo.mlpackage", b"weights");
        let calls = std::cell::Cell::new(0usize);

        let first = compile_cached_in(&cache, &bundle, |_| Ok(fake_compile(&temp_root, &calls)))
            .expect("first resolve");
        let second = compile_cached_in(&cache, &bundle, |_| Ok(fake_compile(&temp_root, &calls)))
            .expect("second resolve");

        assert_eq!(first, second, "both resolves must name the same entry");
        assert_eq!(calls.get(), 1, "the second resolve must not recompile");
        assert!(first.starts_with(&cache), "entry must live in the cache");
        assert!(
            first.join("model").join("coremldata.bin").is_file(),
            "the compiled tree must have been moved intact"
        );

        // The compile leak is the thing this cache exists to fix: the
        // directory the "framework" produced must be gone.
        let leftovers: Vec<_> = fs::read_dir(&temp_root)
            .into_iter()
            .flatten()
            .flatten()
            .collect();
        assert!(
            leftovers.is_empty(),
            "the temporary compile directory must not survive installation"
        );

        let entries: Vec<_> = fs::read_dir(&cache)
            .into_iter()
            .flatten()
            .flatten()
            .collect();
        assert_eq!(entries.len(), 1, "exactly one cache entry for one bundle");

        remove_tree(&root);
    }

    #[test]
    fn changing_the_bundle_produces_a_new_entry() {
        let root = scratch("refingerprint");
        let cache = root.join("cache");
        let temp_root = root.join("tmp");
        let _ = fs::create_dir_all(&temp_root);
        let bundle = fake_bundle(&root, "demo.mlpackage", b"weights");
        let calls = std::cell::Cell::new(0usize);

        let first = compile_cached_in(&cache, &bundle, |_| Ok(fake_compile(&temp_root, &calls)))
            .expect("first resolve");

        // Rewrite the weight blob at a different length: the
        // fingerprint must change even if the mtime clock is coarse.
        let _ = fs::write(
            bundle
                .join("Data")
                .join("com.apple.CoreML")
                .join("weights.bin"),
            b"different weights entirely",
        );

        let second = compile_cached_in(&cache, &bundle, |_| Ok(fake_compile(&temp_root, &calls)))
            .expect("second resolve");

        assert_ne!(first, second, "a changed bundle must not reuse its entry");
        assert_eq!(calls.get(), 2, "a changed bundle must recompile");

        remove_tree(&root);
    }

    #[test]
    fn lost_install_race_adopts_the_existing_entry_and_drops_our_copy() {
        let root = scratch("race");
        let cache = root.join("cache");
        let temp_root = root.join("tmp");
        let _ = fs::create_dir_all(&temp_root);
        let bundle = fake_bundle(&root, "demo.mlpackage", b"weights");
        let calls = std::cell::Cell::new(0usize);

        // Simulate the winner of the race having already installed a
        // complete entry under the exact key we are about to compute.
        let target = entry_path_in(&cache, &bundle);
        let _ = fs::create_dir_all(target.join("model"));
        let _ = fs::write(target.join("model").join("coremldata.bin"), b"winner");

        // `install` is what resolves the race, so drive it directly —
        // `compile_cached_in`'s own existence probe would short-circuit
        // before ever reaching it.
        let ours = fake_compile(&temp_root, &calls);
        install(&ours, &target).expect("install must resolve the race, not fail");

        assert!(!ours.exists(), "our redundant copy must be removed");
        assert_eq!(
            fs::read(target.join("model").join("coremldata.bin")).unwrap_or_default(),
            b"winner".to_vec(),
            "the winner's entry must survive untouched"
        );

        remove_tree(&root);
    }

    #[test]
    fn unusable_cache_root_degrades_to_the_compiled_temp_dir() {
        let root = scratch("degraded");
        let temp_root = root.join("tmp");
        let _ = fs::create_dir_all(&temp_root);
        let bundle = fake_bundle(&root, "demo.mlpackage", b"weights");
        let calls = std::cell::Cell::new(0usize);

        // A *file* where the cache root should be: `create_dir_all` on
        // its child fails, so installation cannot proceed.
        let blocked = root.join("not-a-directory");
        let _ = fs::write(&blocked, b"");

        let resolved =
            compile_cached_in(&blocked, &bundle, |_| Ok(fake_compile(&temp_root, &calls)))
                .expect("an unusable cache must not fail the load");
        assert!(
            resolved.starts_with(&temp_root),
            "degraded resolve must return the compiled temp dir, got {}",
            resolved.display()
        );
        assert!(resolved.is_dir());

        remove_tree(&root);
    }

    #[test]
    fn compile_failure_propagates_and_writes_nothing() {
        let root = scratch("failure");
        let cache = root.join("cache");
        let bundle = fake_bundle(&root, "demo.mlpackage", b"weights");

        let err = compile_cached_in(&cache, &bundle, |_| {
            Err(CoreMLError::Internal("compile exploded".to_string()))
        })
        .expect_err("a compile failure must surface");
        assert!(matches!(err, CoreMLError::Internal(_)));
        assert!(!cache.exists(), "a failed compile must not create entries");

        remove_tree(&root);
    }

    #[test]
    fn evict_only_removes_our_own_entries() {
        let root = scratch("evict");
        let cache = root.join("cache");
        let temp_root = root.join("tmp");
        let _ = fs::create_dir_all(&temp_root);
        let bundle = fake_bundle(&root, "demo.mlpackage", b"weights");

        // A path that is not the cache entry for this bundle must be
        // left alone and must not authorize a retry.
        let foreign = root.join("elsewhere.mlmodelc");
        let _ = fs::create_dir_all(&foreign);
        assert!(!evict(&bundle, &foreign));
        assert!(foreign.is_dir(), "a foreign path must never be deleted");

        remove_tree(&cache);
        remove_tree(&root);
    }

    #[test]
    fn sanitize_bounds_and_escapes_entry_names() {
        assert_eq!(sanitize("w600k_r50"), "w600k_r50");
        // Every path separator and dot collapses to `_`, so no entry
        // name can ever escape the cache root.
        assert_eq!(sanitize("../../etc/passwd"), "______etc_passwd");
        // Non-ASCII collapses one `char` to one `_` — never a partial
        // UTF-8 sequence, which is what makes the length cap safe.
        assert_eq!(sanitize("モデル"), "___");
        assert_eq!(sanitize(""), "model");
        assert_eq!(sanitize(&"x".repeat(200)).len(), MAX_KEY_CHARS);
    }

    #[test]
    fn cache_root_honors_the_environment_override_shape() {
        // `cache_root` reads process-global state, so assert only the
        // shape of the fallbacks that cannot race: the platform
        // directory always ends in the crate's own subdirectory name.
        let root = cache_root();
        assert!(
            root.file_name().and_then(OsStr::to_str) == Some(CACHE_DIR_NAME)
                || env::var_os(CACHE_DIR_ENV).is_some(),
            "unoverridden cache root must end in {CACHE_DIR_NAME}, got {}",
            root.display()
        );
    }
}
