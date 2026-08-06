//! Error types for the CoreML runtime.
//!
//! All public APIs in this crate return [`Result<T, CoreMLError>`].  Every
//! variant either wraps a concrete failure mode (file I/O, NSError from the
//! Apple framework, dtype mismatch) or signals an unrecoverable runtime
//! invariant violation (missing output, dtype the runtime cannot project to
//! `f32`).
//!
//! On non-macOS targets only the [`UnsupportedFormat`](CoreMLError::UnsupportedFormat),
//! [`Io`](CoreMLError::Io) and [`UnsupportedPlatform`](CoreMLError::UnsupportedPlatform)
//! variants are reachable; the Apple-specific variants are present for API
//! parity but never constructed.

use thiserror::Error;

/// Convenience alias used by every fallible function in this crate.
pub type Result<T> = core::result::Result<T, CoreMLError>;

/// Failure modes returned by the CoreML runtime.
///
/// `#[non_exhaustive]`: new failure modes may be added as more of the
/// CoreML/MLPackage surface (feature types, pixel formats, …) is covered.
/// Downstream `match`es need a wildcard arm; existing variants remain
/// constructible as before.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CoreMLError {
    /// File-system error — usually the supplied `.mlpackage` path is missing,
    /// not readable or not a directory.
    #[error("CoreML I/O error at {path}: {source}")]
    Io {
        /// Path that failed to load (best-effort, possibly empty).
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// CoreML framework returned an `NSError` from a load/predict/compile call.
    ///
    /// The `code` is the Apple `NSError.code` (typically a `MLModelError`
    /// constant) and `message` is the framework-supplied
    /// `localizedDescription`.
    #[error("CoreML framework error ({code}): {message}")]
    Framework {
        /// `NSError.code` value reported by the framework.
        code: i64,
        /// `NSError.localizedDescription` projected to UTF-8.
        message: String,
    },

    /// The supplied input map is missing a feature the model requires, or
    /// supplies a feature that the model does not declare.
    #[error("CoreML input mismatch: {0}")]
    InputMismatch(String),

    /// The model returned an output dtype this runtime cannot project to
    /// `f32` (only `Float32` and `Float16` are supported today).
    #[error("CoreML unsupported output dtype: {0}")]
    UnsupportedOutputDtype(String),

    /// The supplied bundle format is not loadable through this entry point —
    /// most commonly produced by `load_from_bytes`, which requires a
    /// directory bundle, not a single file.
    #[error("CoreML unsupported format: {0}")]
    UnsupportedFormat(&'static str),

    /// The current target OS cannot host the CoreML runtime (anything other
    /// than macOS / iOS / tvOS / visionOS).  Only ever returned by the
    /// non-macOS stub crate body.
    #[error("CoreML is not supported on this platform")]
    UnsupportedPlatform,

    /// CoreML reported a successful prediction but the requested output
    /// feature is missing from the response — should never happen with a
    /// well-formed model, but we surface it rather than panic.
    #[error("CoreML output missing: {0}")]
    MissingOutput(String),

    /// MLComputePlan introspection failed (timeout, non-MLProgram model,
    /// missing `main` function, etc.).  Diagnostics-only; never raised by
    /// `predict`.
    #[error("CoreML compute-plan introspection failed: {0}")]
    ComputePlan(String),

    /// [`MlPackageModel::predict_features`](crate::MlPackageModel::predict_features)
    /// encountered an `MLFeatureType` it does not decode — either
    /// `MLFeatureTypeInvalid`, `MLFeatureTypeState` (no portable
    /// representation exists for either), or (for `MLFeatureTypeSequence`)
    /// an `MLSequence` element type other than `Int64`/`String`.
    #[error("CoreML unsupported feature type: {0}")]
    UnsupportedFeatureType(String),

    /// [`MlPackageModel::predict_features`](crate::MlPackageModel::predict_features)'s
    /// image decoder does not recognize a `CVPixelBuffer`'s
    /// `pixelFormatType` — either because it is a planar layout, or
    /// because it is a packed format outside the standard set
    /// (`OneComponent8`, `32BGRA`, `OneComponent16Half`,
    /// `OneComponent32Float`) that decoder supports.
    #[error("CoreML unsupported CVPixelBuffer format: {0}")]
    UnsupportedPixelFormat(String),

    /// Catch-all for invariant violations — unreachable in normal use, but
    /// surfaced rather than panicking so callers always get a `Result`.
    #[error("CoreML internal error: {0}")]
    Internal(String),
}

impl From<CoreMLError> for oxionnx_core::OnnxError {
    fn from(e: CoreMLError) -> Self {
        Self::Internal(e.to_string())
    }
}
