//! On-device numeric format for the invariant (weight) operands, and the
//! opt-in half-precision compute mode that selects it.
//!
//! # What is and is not half precision here
//!
//! Nothing about a tensor's *storage* changes: every activation buffer, every
//! kernel output, every read-back is `f32`, exactly as before, and the ONNX
//! initializers on disk are untouched. What this module adds is a second
//! on-device *copy* of the weight operands, converted host-side once at upload
//! time, plus the WGSL variants that read it.
//!
//! The prize is bandwidth, not arithmetic. `conv2d`'s implicit-GEMM re-reads
//! its weight slice once per N-tile: for InSwapper's dominant layer
//! (`M = 1024`, `N = 1024`, `K = 9216`) that is ~600 MB of weight traffic per
//! layer per frame at `f32`, against a kernel already measured at 7-10% of the
//! device's `f32` ALU peak — i.e. bound on memory, not on multiplies. Halving
//! the weight bytes halves that stream, halves the resident weight footprint,
//! and (because the staged tiles become `vec4<f16>`) halves workgroup memory
//! per tile, which is what lets more workgroups be resident at once.
//!
//! # Rounding points
//!
//! Enabling this mode changes results. Every place it does so is enumerated
//! here and mirrored in the WGSL by [`crate::shaders`]'s `f16_variant`:
//!
//! 1. **Weight upload** — `f32 -> f16` once per operand per session, in
//!    [`WeightFormat::convert`], round-to-nearest-even via `half`.
//! 2. **Activation stage** — the input tile is read from its `f32` buffer and
//!    narrowed to `f16` as it is written into workgroup memory.
//! 3. **Product** — `f16 * f16`, evaluated at half precision.
//!
//! The accumulator is **`f32` in every kernel**: each product is widened
//! before it reaches the running sum, so a `K = 9216` reduction still
//! accumulates at single precision and the error stays that of the inputs
//! rather than growing with depth. Bias, the fused activation epilogue and the
//! `alpha`/`beta` terms are all `f32` as well. Measured on an Apple M3, that
//! design lands at 75-82 dB PSNR against the `f32` kernel on InSwapper shapes.

use std::sync::atomic::{AtomicBool, Ordering};

/// Which numeric format a weight operand takes on the device.
///
/// Part of the residency cache's key, not just a property of a buffer: one
/// initializer can legitimately be resident in *both* formats when a caller
/// flips [`crate::GpuContext::set_f16_compute`] mid-session, and serving the
/// wrong one to a kernel would reinterpret the bytes rather than convert them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum WeightFormat {
    /// The bytes the caller handed over, verbatim — 4 bytes per element.
    #[default]
    F32,
    /// Narrowed to IEEE binary16 at upload — 2 bytes per element.
    F16,
}

impl WeightFormat {
    /// The format an `f16`-capable dispatch should use, given the context's
    /// effective toggle state.
    #[must_use]
    pub fn for_f16(enabled: bool) -> Self {
        if enabled {
            Self::F16
        } else {
            Self::F32
        }
    }

    /// Bytes on the device for `elements` values in this format.
    ///
    /// Saturating rather than wrapping: the callers feed the result to the byte
    /// budget and to `checked_storage_bytes`, both of which decline on a number
    /// too large, and a wrap would turn a decline into an under-allocation.
    #[must_use]
    pub fn byte_len(self, elements: usize) -> u64 {
        (elements as u64).saturating_mul(self.bytes_per_element())
    }

    /// Bytes one element occupies.
    #[must_use]
    pub fn bytes_per_element(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }

    /// Whether this is the half-precision format.
    #[must_use]
    pub fn is_f16(self) -> bool {
        matches!(self, Self::F16)
    }

    /// A short, stable name for labels and diagnostics.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }

    /// Narrow `data` into this format's device bytes.
    ///
    /// [`Self::F32`] borrows — `bytemuck::cast_slice` over the caller's slice,
    /// no copy at all, which is exactly what every kernel did before this
    /// module existed. [`Self::F16`] allocates a `Vec<u16>` of half the size and
    /// converts element-wise.
    ///
    /// Conversion is `half::f16::from_f32`, i.e. round-to-nearest-even, and
    /// the rounding reaches past the largest finite `binary16`: 65504 is not
    /// the overflow boundary, 65520 is. Magnitudes in `(65504, 65520)` round
    /// *back down* to 65504 — the neighbouring representable values are 65504
    /// and 65536, so 65520 is the midpoint — and only magnitudes at or above
    /// 65520 become an infinity (the tie goes to 65536, whose significand is
    /// the even one, which is out of range). At the other end, magnitudes
    /// below ~6e-8 flush to zero. Convolution weights live in neither region
    /// in any model this crate targets; a caller worried about one should
    /// leave the toggle off, which is the default.
    ///
    /// **Called only on a residency-cache miss.** Converting 9.4M weights on
    /// every dispatch would cost more host time than the dispatch saves, so the
    /// cache lookup happens on the *element count* (via [`Self::byte_len`])
    /// and this runs only when the bytes are actually about to be uploaded.
    /// See `GpuContext::operand_buffer_typed`.
    #[must_use]
    pub fn convert(self, data: &[f32]) -> WeightBytes<'_> {
        match self {
            Self::F32 => WeightBytes::Borrowed(bytemuck::cast_slice(data)),
            Self::F16 => WeightBytes::Owned(
                data.iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect(),
            ),
        }
    }
}

/// The device-ready bytes of one weight operand.
///
/// A `Cow`-shaped pair rather than a `Vec` in both arms so the `f32` path keeps
/// costing nothing: it is the caller's own slice, reinterpreted.
///
/// The owned arm holds `u16` rather than `half::f16` deliberately — `u16` is
/// `bytemuck::Pod`, so casting to bytes needs no extra feature on `half` and no
/// `unsafe` here.
pub enum WeightBytes<'a> {
    /// The caller's `f32` slice, viewed as bytes.
    Borrowed(&'a [u8]),
    /// Freshly narrowed `f16` bit patterns.
    Owned(Vec<u16>),
}

impl WeightBytes<'_> {
    /// The bytes to upload.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Owned(halves) => bytemuck::cast_slice(halves),
        }
    }
}

/// The half-precision compute toggle: what the caller asked for, and what the
/// device can actually do.
///
/// Split out from `GpuContext` so the state machine — which is the whole of
/// mandate "a numerics-changing mode must never be silently on" — is testable
/// without an adapter.
#[derive(Debug)]
pub(crate) struct F16Compute {
    /// Whether the device was created with `wgpu::Features::SHADER_F16`.
    supported: bool,
    /// What the caller last asked for. Off until someone says otherwise.
    requested: AtomicBool,
}

impl F16Compute {
    /// A toggle for a device with (or without) the feature. Always starts off.
    pub(crate) fn new(supported: bool) -> Self {
        Self {
            supported,
            requested: AtomicBool::new(false),
        }
    }

    /// Whether the device has the feature at all.
    pub(crate) fn supported(&self) -> bool {
        self.supported
    }

    /// Ask for half-precision compute; returns the *effective* state.
    ///
    /// Asking on a device without the feature is not an error and not a
    /// decline — it simply leaves the effective state off, which the getter
    /// then reports honestly.
    pub(crate) fn set(&self, requested: bool) -> bool {
        self.requested.store(requested, Ordering::Relaxed);
        self.enabled()
    }

    /// Whether kernels should actually take the half-precision path.
    pub(crate) fn enabled(&self) -> bool {
        effective(self.requested.load(Ordering::Relaxed), self.supported)
    }
}

/// The effective half-precision state for a `requested` / `supported` pair.
///
/// A free function so the rule — *both* must hold, and support is the
/// authority — can be pinned by a test on a machine with no GPU at all, which
/// is the only way to check the unsupported branch on hardware that supports
/// the feature.
#[must_use]
pub(crate) fn effective(requested: bool, supported: bool) -> bool {
    requested && supported
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole state machine, including the branch an M3 cannot reach with a
    /// real device.
    #[test]
    fn support_is_the_authority_over_the_request() {
        assert!(!effective(false, false));
        assert!(!effective(true, false), "unsupported must stay off");
        assert!(!effective(false, true), "default is off, not on");
        assert!(effective(true, true));
    }

    #[test]
    fn a_toggle_on_an_unsupported_device_never_turns_on() {
        let toggle = F16Compute::new(false);
        assert!(!toggle.supported());
        assert!(!toggle.enabled(), "must start off");
        assert!(!toggle.set(true), "set must report the effective state");
        assert!(!toggle.enabled(), "and it must still be off");
        assert!(!toggle.set(false));
    }

    #[test]
    fn a_toggle_on_a_supported_device_still_starts_off() {
        let toggle = F16Compute::new(true);
        assert!(toggle.supported());
        assert!(
            !toggle.enabled(),
            "a numerics-changing mode is never silently on"
        );
        assert!(toggle.set(true));
        assert!(toggle.enabled());
        assert!(!toggle.set(false));
        assert!(!toggle.enabled());
    }

    #[test]
    fn byte_lengths_halve_and_the_format_says_so() {
        assert_eq!(WeightFormat::F32.byte_len(1024), 4096);
        assert_eq!(WeightFormat::F16.byte_len(1024), 2048);
        assert_eq!(WeightFormat::F32.bytes_per_element(), 4);
        assert_eq!(WeightFormat::F16.bytes_per_element(), 2);
        assert!(!WeightFormat::F32.is_f16());
        assert!(WeightFormat::F16.is_f16());
        assert_eq!(WeightFormat::default(), WeightFormat::F32);
        assert_eq!(WeightFormat::for_f16(true), WeightFormat::F16);
        assert_eq!(WeightFormat::for_f16(false), WeightFormat::F32);
        // A length that would overflow the multiply saturates rather than
        // wrapping to a small number the budget would happily admit.
        assert_eq!(WeightFormat::F32.byte_len(usize::MAX), u64::MAX);
    }

    #[test]
    fn conversion_borrows_for_f32_and_halves_for_f16() {
        let data = [1.0f32, -2.5, 0.125, 65_600.0, 3.0e-9];
        let wide = WeightFormat::F32.convert(&data);
        assert_eq!(wide.as_bytes().len(), 20);
        assert_eq!(
            bytemuck::cast_slice::<u8, f32>(wide.as_bytes()),
            &data[..],
            "the f32 path must reinterpret, never transform"
        );

        let narrow = WeightFormat::F16.convert(&data);
        assert_eq!(narrow.as_bytes().len(), 10, "half the bytes");
        let back: Vec<f32> = bytemuck::cast_slice::<u8, u16>(narrow.as_bytes())
            .iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect();
        assert_eq!(back[0], 1.0);
        assert_eq!(back[1], -2.5);
        assert_eq!(back[2], 0.125, "an exact binary16 value round-trips");
        assert!(
            back[3].is_infinite(),
            "over-range saturates to infinity, as documented"
        );
        assert_eq!(back[4], 0.0, "under-range flushes to zero, as documented");

        // The overflow boundary is the rounding midpoint, not the largest
        // finite value: round-to-nearest-even sends everything below 65520
        // back to 65504, and only 65520 and up become infinite. Pinned here
        // because the doc comment above states it.
        assert_eq!(
            half::f16::from_f32(65_510.0).to_f32(),
            65_504.0,
            "inside the rounding band, over-range rounds back to the maximum"
        );
        assert!(
            half::f16::from_f32(65_520.0).to_f32().is_infinite(),
            "at the midpoint the tie goes to the out-of-range even value"
        );
    }

    #[test]
    fn an_empty_operand_converts_to_no_bytes() {
        assert!(WeightFormat::F16.convert(&[]).as_bytes().is_empty());
        assert!(WeightFormat::F32.convert(&[]).as_bytes().is_empty());
    }
}
