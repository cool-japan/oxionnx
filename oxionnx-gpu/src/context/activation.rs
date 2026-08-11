//! Run-scoped device tensors: the values a kernel may consume in place and
//! leave in place.
//!
//! # What this adds on top of `super::resident`
//!
//! `super::resident` keeps *invariant* operands — a convolution's weight, a
//! `Gemm`'s `B` — for the lifetime of the [`GpuContext`](crate::GpuContext), keyed by an identity
//! the caller owns. This module is the other half: the values that change on
//! every frame, produced by one kernel and consumed by the next, whose lifetime
//! is one *run* rather than one session.
//!
//! Before it existed, every kernel here ended in a read-back and every kernel
//! started with an upload, so a `Conv -> Relu -> Conv` chain moved each
//! activation across the bus four times (down, up, down, up) to compute two
//! dependencies that never needed to leave the device. InSwapper-128 moved
//! ~488 MiB per frame that way.
//!
//! # Ownership, and why it is not an `Arc`
//!
//! A [`DeviceTensor`] owns its [`TrackedBuffer`] outright. That is deliberate:
//!
//! * the buffer's bytes are counted by the same [`GpuMemoryBudget`] every other
//!   allocation in this crate is counted by, and `TrackedBuffer`'s `Drop`
//!   releases them, so "an activation that nobody will read again is gone" is a
//!   property of ordinary Rust scoping rather than of a cache eviction policy;
//!   the caller drops the value at its last consumer and the device memory goes
//!   back immediately;
//! * a kernel only ever *binds* an activation, which needs `&wgpu::Buffer` and
//!   nothing more, so shared ownership would buy nothing;
//! * `super::resident`'s `OperandBuffer::Resident` is an `Arc` for a reason that
//!   does not apply here — it must be impossible to hand a session-lifetime
//!   weight to `read_back_and_recycle_async`, which takes a `TrackedBuffer` by
//!   value. An activation, by contrast, *may* legitimately be recycled or
//!   destroyed; it is the caller's to spend.
//!
//! [`GpuMemoryBudget`]: super::budget::GpuMemoryBudget

use oxionnx_core::Tensor;

use super::budget::TrackedBuffer;

/// A tensor that lives in a device buffer for the duration of one run.
///
/// Produced by a kernel asked for [`OutputPlacement::Device`], consumed by the
/// next kernel as [`TensorSource::Device`], and destroyed when the caller drops
/// it — which is what returns its bytes to the byte budget.
#[derive(Debug)]
pub struct DeviceTensor {
    /// The allocation. Destroyed, and its bytes released, when this drops.
    buffer: TrackedBuffer,
    /// Logical shape, so a caller need not track it beside the handle.
    shape: Vec<usize>,
    /// `f32` element count — `shape.iter().product()`, computed once.
    len: usize,
    /// Bytes of the *live* range.
    ///
    /// Not `buffer.size()`: an output buffer may come from the pool, which
    /// hands back anything within 2x of the request, so the allocation is
    /// frequently larger than the tensor. Binding `as_entire_binding()` on such
    /// a buffer would bind the slack too and can exceed
    /// `max_storage_buffer_binding_size` for a tensor that was itself validated
    /// — the same trap `conv2d`'s `output_binding` documents.
    bytes: u64,
}

impl DeviceTensor {
    /// Wrap an allocation that already holds `shape`'s worth of `f32`s.
    pub(crate) fn new(buffer: TrackedBuffer, shape: Vec<usize>, len: usize, bytes: u64) -> Self {
        Self {
            buffer,
            shape,
            len,
            bytes,
        }
    }

    /// Logical shape of the tensor in this buffer.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// `f32` elements held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the tensor holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes of the live range — what a binding must cover.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        self.bytes
    }

    /// Bytes the allocation actually reserves, which the byte budget counts.
    ///
    /// May exceed [`Self::byte_len`] when the buffer came from the pool.
    #[must_use]
    pub fn reserved_bytes(&self) -> u64 {
        self.buffer.reserved_bytes()
    }

    /// The underlying buffer, for binding and copying.
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Give the allocation back to the caller, dropping the shape.
    ///
    /// The one path by which an activation's buffer can be recycled into the
    /// context's pool instead of destroyed.
    pub(crate) fn into_buffer(self) -> TrackedBuffer {
        self.buffer
    }

    /// Bind exactly this tensor's range — never the allocation's slack.
    pub(crate) fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            size: wgpu::BufferSize::new(self.bytes),
        })
    }
}

/// Where one of a kernel's operands currently lives.
///
/// The whole point of the enum is that a kernel body does not branch on it: it
/// hands the source to `GpuContext::operand_source` and binds whatever comes
/// back, so the resident and the transferring paths are the same code.
#[derive(Clone, Copy, Debug)]
pub enum TensorSource<'a> {
    /// Host memory; this dispatch uploads it.
    Host {
        /// The values.
        data: &'a [f32],
        /// Their logical shape.
        shape: &'a [usize],
    },
    /// Already on the device; this dispatch binds it in place.
    Device(&'a DeviceTensor),
}

impl<'a> TensorSource<'a> {
    /// A host operand from its parts — the form every pre-residency entry point
    /// still uses.
    #[must_use]
    pub fn host(data: &'a [f32], shape: &'a [usize]) -> Self {
        Self::Host { data, shape }
    }

    /// A host operand from a [`Tensor`].
    #[must_use]
    pub fn tensor(tensor: &'a Tensor) -> Self {
        Self::Host {
            data: &tensor.data,
            shape: &tensor.shape,
        }
    }

    /// The operand's logical shape.
    #[must_use]
    pub fn shape(&self) -> &'a [usize] {
        match self {
            Self::Host { shape, .. } => shape,
            Self::Device(tensor) => tensor.shape(),
        }
    }

    /// The operand's `f32` element count.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Host { data, .. } => data.len(),
            Self::Device(tensor) => tensor.len(),
        }
    }

    /// Whether the operand holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether these bytes are already on the device.
    ///
    /// Kernels consult this for exactly one thing: their own
    /// "too small to be worth a dispatch" threshold, which is measured against
    /// a *round trip* and therefore does not describe this case at all. See
    /// [`skips_size_threshold`].
    #[must_use]
    pub fn is_device(&self) -> bool {
        matches!(self, Self::Device(_))
    }

    /// The same operand restricted to its first `len` elements, or `None` when
    /// it does not hold that many.
    ///
    /// Kernels use it where they used to write `data.get(..n)?`: the plan says
    /// how many elements the shader will index, and binding exactly that many
    /// keeps a tensor whose buffer is longer than its shape (a malformed model,
    /// or a pooled allocation) from binding its slack. A device operand cannot
    /// be sliced — its buffer is bound whole — so it must already be the right
    /// length.
    #[must_use]
    pub fn truncated(self, len: usize) -> Option<Self> {
        match self {
            Self::Host { data, shape } => Some(Self::Host {
                data: data.get(..len)?,
                shape,
            }),
            Self::Device(tensor) => (tensor.len() == len).then_some(Self::Device(tensor)),
        }
    }

    /// The host values, when there are any.
    #[must_use]
    pub fn host_data(&self) -> Option<&'a [f32]> {
        match self {
            Self::Host { data, .. } => Some(data),
            Self::Device(_) => None,
        }
    }
}

/// Whether a kernel's own minimum-size gate applies to a dispatch with these
/// operands.
///
/// The thresholds in `shaders::common` (`EW_GPU_THRESHOLD` and friends, 100_000
/// elements) answer one question: *is uploading these bytes, computing, and
/// reading the result back cheaper than the CPU kernel?* Below the threshold it
/// is not, because the round trip dominates.
///
/// When an operand is already on the device that question is not the one being
/// asked. Declining does not run the CPU kernel on bytes the host already has —
/// it forces the caller to read the operand back first, and (for anything that
/// feeds a later GPU node) to upload the result again afterwards. That is
/// strictly more traffic than dispatching, at every size. So the threshold is
/// skipped, and the size decision moves to the caller, which is where this
/// crate's module docs say a placement heuristic belongs.
#[must_use]
pub fn skips_size_threshold(sources: &[TensorSource<'_>]) -> bool {
    sources.iter().any(TensorSource::is_device)
}

/// What the caller wants done with a kernel's result.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OutputPlacement {
    /// Copy it to a staging buffer, read it back, and recycle the output
    /// buffer — the behaviour every entry point had before this module existed.
    #[default]
    Host,
    /// Leave it in its device buffer and hand the buffer to the caller. No
    /// staging buffer is allocated and no copy is encoded.
    Device,
}

impl OutputPlacement {
    /// Bytes the read-back staging buffer will need — zero when the result
    /// stays on the device.
    ///
    /// Kernels feed this to `budget_admits` instead of the raw output size, so
    /// a node that keeps its result resident is not declined for room it will
    /// never ask for.
    #[must_use]
    pub fn staging_bytes(self, output_bytes: u64) -> u64 {
        match self {
            Self::Host => output_bytes,
            Self::Device => 0,
        }
    }

    /// Whether the result stays on the device.
    #[must_use]
    pub fn keeps_device(self) -> bool {
        matches!(self, Self::Device)
    }
}

/// What a kernel produced.
#[derive(Debug)]
pub enum GpuOutput {
    /// Read back into host memory.
    Host(Tensor),
    /// Left in a device buffer.
    Device(DeviceTensor),
}

impl GpuOutput {
    /// The result's shape, wherever it lives.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Host(tensor) => &tensor.shape,
            Self::Device(tensor) => tensor.shape(),
        }
    }

    /// The result's `f32` element count.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Host(tensor) => tensor.data.len(),
            Self::Device(tensor) => tensor.len(),
        }
    }

    /// Whether the result holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The host tensor, or `None` when the result stayed on the device.
    ///
    /// The pre-residency entry points call this and `?` it: they asked for
    /// [`OutputPlacement::Host`], so `None` is unreachable for them and would
    /// mean the kernel ignored its placement argument.
    #[must_use]
    pub fn into_tensor(self) -> Option<Tensor> {
        match self {
            Self::Host(tensor) => Some(tensor),
            Self::Device(_) => None,
        }
    }

    /// The host values, or `None` when the result stayed on the device.
    #[must_use]
    pub fn into_vec(self) -> Option<Vec<f32>> {
        self.into_tensor().map(|tensor| tensor.data)
    }

    /// The device tensor, or `None` when the result was read back.
    #[must_use]
    pub fn into_device(self) -> Option<DeviceTensor> {
        match self {
            Self::Host(_) => None,
            Self::Device(tensor) => Some(tensor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Placement arithmetic is the part of this module a budget decision
    /// depends on, and it needs no device.
    #[test]
    fn a_device_placement_asks_for_no_staging_bytes() {
        assert_eq!(OutputPlacement::Host.staging_bytes(4096), 4096);
        assert_eq!(OutputPlacement::Device.staging_bytes(4096), 0);
        assert!(OutputPlacement::Device.keeps_device());
        assert!(!OutputPlacement::Host.keeps_device());
        assert_eq!(OutputPlacement::default(), OutputPlacement::Host);
    }

    #[test]
    fn a_host_source_reports_its_own_shape_and_length() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2usize, 3];
        let source = TensorSource::host(&data, &shape);
        assert_eq!(source.shape(), &shape);
        assert_eq!(source.len(), 6);
        assert!(!source.is_empty());
        assert!(!source.is_device());
        assert_eq!(source.host_data(), Some(&data[..]));
        assert!(
            !skips_size_threshold(&[source]),
            "a host-only dispatch keeps its kernel's own size gate"
        );
    }

    #[test]
    fn a_tensor_source_borrows_rather_than_copies() {
        let tensor = Tensor::new(vec![0.5f32; 12], vec![3, 4]);
        let source = TensorSource::tensor(&tensor);
        assert_eq!(source.shape(), &[3, 4]);
        assert_eq!(source.len(), 12);
        assert_eq!(source.host_data().map(<[f32]>::len), Some(12));
    }

    #[test]
    fn a_host_output_converts_and_a_device_one_declines() {
        let out = GpuOutput::Host(Tensor::new(vec![1.0f32, 2.0], vec![2]));
        assert_eq!(out.shape(), &[2]);
        assert_eq!(out.len(), 2);
        assert!(!out.is_empty());
        assert_eq!(out.into_vec(), Some(vec![1.0, 2.0]));

        let out = GpuOutput::Host(Tensor::new(Vec::new(), vec![0]));
        assert!(out.is_empty());
        assert!(out.into_device().is_none());
    }
}
