//! Non-macOS stub — every `MlPackageModel` method short-circuits to
//! [`CoreMLError::UnsupportedPlatform`](crate::error::CoreMLError::UnsupportedPlatform),
//! preserving the API surface so dependent crates compile portably.

use std::path::PathBuf;

use super::*;

/// Stub that always fails on non-macOS targets.  Present so callers can
/// share code between platforms behind `#[cfg(feature = "coreml")]`.
pub struct MlPackageModel {
    _private: (),
}

impl MlPackageModel {
    /// Always returns [`CoreMLError::UnsupportedPlatform`].
    pub fn load(_path: impl AsRef<Path>, _compute_units: MlComputeUnits) -> Result<Self> {
        Err(CoreMLError::UnsupportedPlatform)
    }

    /// Always returns [`CoreMLError::UnsupportedPlatform`] — there is
    /// no CoreML compiler on this target, so there is nothing to
    /// compile and nothing to cache.  Present for API parity with the
    /// Apple-platform pre-warm entry point.
    pub fn ensure_compiled(_path: impl AsRef<Path>) -> Result<PathBuf> {
        Err(CoreMLError::UnsupportedPlatform)
    }

    /// Always returns [`CoreMLError::UnsupportedPlatform`] — the
    /// CoreML runtime itself is unavailable on this target,
    /// independent of the `.mlpackage`-is-a-directory-bundle
    /// limitation documented on the macOS implementation of this
    /// method (see that doc comment for why bytes-based loading is
    /// inherently unsupported even where CoreML *is* available).
    pub fn load_from_bytes(_bytes: &[u8], _compute_units: MlComputeUnits) -> Result<Self> {
        Err(CoreMLError::UnsupportedPlatform)
    }

    /// Always returns an empty list — the stub holds no model.
    pub fn input_names(&self) -> Vec<String> {
        Vec::new()
    }

    /// Always returns an empty list — the stub holds no model.
    pub fn output_names(&self) -> Vec<String> {
        Vec::new()
    }

    /// Always returns [`CoreMLError::UnsupportedPlatform`].
    pub fn predict(
        &self,
        _inputs: &HashMap<String, oxionnx_core::Tensor>,
    ) -> Result<HashMap<String, oxionnx_core::Tensor>> {
        Err(CoreMLError::UnsupportedPlatform)
    }

    /// Always returns [`CoreMLError::UnsupportedPlatform`].
    pub fn predict_raw(
        &self,
        _inputs: &HashMap<String, oxionnx_core::Tensor>,
    ) -> Result<HashMap<String, RawArray>> {
        Err(CoreMLError::UnsupportedPlatform)
    }

    /// Always returns [`CoreMLError::UnsupportedPlatform`].
    pub fn predict_features(
        &self,
        _inputs: &HashMap<String, oxionnx_core::Tensor>,
    ) -> Result<HashMap<String, FeatureOutput>> {
        Err(CoreMLError::UnsupportedPlatform)
    }

    /// Always returns [`CoreMLError::UnsupportedPlatform`].
    pub fn warm_up(&self, _input_template: &HashMap<String, oxionnx_core::Tensor>) -> Result<()> {
        Err(CoreMLError::UnsupportedPlatform)
    }

    /// Always returns [`CoreMLError::UnsupportedPlatform`].
    pub fn compute_plan_summary(&self) -> Result<ComputePlanSummary> {
        Err(CoreMLError::UnsupportedPlatform)
    }

    /// Always returns [`CoreMLError::UnsupportedPlatform`].
    pub fn compute_plan_breakdown(&self) -> Result<HashMap<String, ComputePlanSummary>> {
        Err(CoreMLError::UnsupportedPlatform)
    }

    /// Always returns [`CoreMLError::UnsupportedPlatform`].
    pub fn model_metadata(&self) -> Result<HashMap<String, String>> {
        Err(CoreMLError::UnsupportedPlatform)
    }
}

/// Unit tests for instance methods (`predict_features`,
/// `model_metadata`) that need an actual `MlPackageModel` value to
/// call `&self` on. Nested inside `stub_impl` (rather than the
/// crate's usual sibling `mod stub_tests`) because `MlPackageModel`'s
/// `_private` field is not `pub` — only `stub_impl` and its
/// descendants may construct `Self { _private: () }` directly; this
/// mirrors `macos_impl::owned_array_tests`'s reason for existing as
/// a descendant module rather than a sibling one.
#[cfg(test)]
mod inner_tests {
    use super::*;

    /// Every operation on the non-macOS stub must return
    /// `UnsupportedPlatform`, including the two instance methods
    /// added alongside `predict`/`predict_raw`/`compute_plan_summary`
    /// above.
    #[test]
    fn stub_predict_features_returns_unsupported_platform() {
        let model = MlPackageModel { _private: () };
        let r = model.predict_features(&HashMap::new());
        assert!(matches!(r, Err(CoreMLError::UnsupportedPlatform)));
    }

    #[test]
    fn stub_model_metadata_returns_unsupported_platform() {
        let model = MlPackageModel { _private: () };
        let r = model.model_metadata();
        assert!(matches!(r, Err(CoreMLError::UnsupportedPlatform)));
    }

    #[test]
    fn stub_compute_plan_breakdown_returns_unsupported_platform() {
        let model = MlPackageModel { _private: () };
        let r = model.compute_plan_breakdown();
        assert!(matches!(r, Err(CoreMLError::UnsupportedPlatform)));
    }
}
