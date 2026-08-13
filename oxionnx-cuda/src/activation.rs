//! Device-resident **activations**: the half of residency
//! [`mod@crate::residency`] deliberately left out.
//!
//! `residency` caches *initializers* — bytes that are invariant for the
//! session — and its module docs state the reason the activation half could
//! not live there: `try_cuda_dispatch` was handed host tensors per node and
//! returned host tensors per node, so two consecutive CUDA-claimed ops A → B
//! made B re-upload A's output from host memory, unconditionally. Measured
//! across the three face-pipeline models that was **1327 MB/frame** of PCIe
//! traffic and **237 blocking `stream.synchronize()` calls**, one per claimed
//! node.
//!
//! This module supplies the pieces that let a node's output stay where the
//! kernel wrote it:
//!
//! * [`CudaDeviceTensor`] — an *owned* device buffer plus the shape it holds.
//!   Owned rather than [`PooledBuffer`](crate::residency::PooledBuffer),
//!   because a pooled borrow dies at the end of the dispatch that took it and
//!   an activation has to outlive its producing node.
//! * [`InputBinding`] — one operand as the dispatcher sees it: host bytes to
//!   upload, or a buffer the run already left on the device.
//! * [`CudaDispatchOutcome`] / [`CudaOutputPlacement`] — what a dispatch
//!   produced, and what the session asked it to produce.
//! * [`ResidentActivations`] — the name → buffer lookup, implemented by the
//!   *session* (which is the only layer that knows node order, last use and
//!   graph outputs) and consumed here.
//!
//! # Why recycling a buffer with work still queued is safe here
//!
//! [`crate::residency::PooledBuffer`] refuses to recycle a borrow whose stream
//! work has not been confirmed finished, and its docs name the reason: this
//! crate used **two** streams (`DnnHandle`'s own and its `BlasHandle`'s), so a
//! buffer released by a convolution and picked up by a GEMM changed stream and
//! stream order stopped protecting it.
//!
//! That is no longer true. `oxicuda-dnn`'s `DnnHandle` now builds its BLAS
//! sub-handle on its *own* stream (`DnnHandle::build`), so every launch and
//! every copy this crate issues rides one queue, and
//! [`DnnHandle::streams_unified`](oxicuda_dnn::DnnHandle::streams_unified)
//! reports it. On one queue, a buffer handed back to the pool while a kernel
//! is still reading it is still safe: the next borrower's kernel is enqueued
//! *behind* that read, and the device executes the queue in order.
//!
//! Every recycle-without-a-fence in this module is therefore gated on
//! [`CudaContext::streams_unified`](crate::CudaContext::streams_unified)
//! rather than assumed — a handle built with `with_split_blas_stream` keeps
//! the old, conservative behaviour and loses reuse, not correctness.

use std::sync::Arc;

use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_driver::Stream;
use oxicuda_memory::DeviceBuffer;
use oxionnx_core::Tensor;

use crate::context::CudaContext;
use crate::error::CudaDispatchError;
use crate::residency::{Operand, PooledBuffer};

// ─── the tensor ────────────────────────────────────────────────────────────

/// One activation living in device memory between two nodes of a run.
///
/// # Length against capacity
///
/// The backing allocation comes from the scratch pool, whose size classes are
/// powers of two, so `buffer.len()` is **at least** — and usually more than —
/// the tensor's element count. [`Self::len`] is the logical count and is what
/// every descriptor, launch geometry and read-back must be built from; the
/// tail beyond it holds a previous borrower's numbers.
///
/// # Sharing
///
/// The buffer is behind an [`Arc`] for two reasons, both real:
///
/// * a kernel binds it as an operand while the session's activation map still
///   owns it, and
/// * [`Self::alias`] rebinds the same allocation under a second name with a
///   different shape — which is all a `Reshape`/`Unsqueeze`/`Squeeze` is on a
///   contiguous row-major buffer.
///
/// Recycling therefore only happens when the last handle goes
/// ([`Self::into_unique_buffer`]): an alias that outlives its origin keeps the
/// allocation alive rather than reading freed memory.
pub struct CudaDeviceTensor {
    /// The allocation. Capacity `>= len`; see the type docs.
    buffer: Arc<DeviceBuffer<f32>>,
    /// The ONNX shape this buffer holds.
    shape: Vec<usize>,
    /// `shape.iter().product()`, cached because every bind reads it.
    len: usize,
}

impl CudaDeviceTensor {
    /// Take ownership of `buffer` as a tensor of `shape`.
    ///
    /// Returns `None` when the allocation cannot hold `shape` — a caller bug
    /// rather than a model one, reported as a decline so it can never become a
    /// kernel reading past the end of an allocation.
    pub(crate) fn from_owned(buffer: DeviceBuffer<f32>, shape: Vec<usize>) -> Option<Self> {
        let len = shape
            .iter()
            .try_fold(1usize, |acc, d| acc.checked_mul(*d))?;
        if buffer.len() < len {
            return None;
        }
        Some(Self {
            buffer: Arc::new(buffer),
            shape,
            len,
        })
    }

    /// The ONNX shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Logical element count — **not** the allocation's capacity.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the tensor holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Device bytes this tensor's *allocation* occupies.
    ///
    /// The reserved size rather than `4 * len`, so a caller summing it gets a
    /// number directly comparable with what the pool and the driver report.
    #[must_use]
    pub fn reserved_bytes(&self) -> u64 {
        self.buffer.byte_size() as u64
    }

    /// A second handle to the same allocation, under a different shape.
    ///
    /// This is what makes a metadata-only reshape (`Reshape`, `Unsqueeze`,
    /// `Squeeze`, `Flatten`, `Identity`) free on the device: the bytes are
    /// already contiguous and row-major, so only the interpretation changes.
    /// Returns `None` when the new shape does not fit the allocation, or when
    /// it does not describe the *same number of elements* — a reshape that
    /// changed the element count would be a different tensor wearing the same
    /// buffer, and silently serving it would corrupt every later node.
    #[must_use]
    pub fn alias(&self, shape: Vec<usize>) -> Option<Self> {
        let len = shape
            .iter()
            .try_fold(1usize, |acc, d| acc.checked_mul(*d))?;
        if len != self.len {
            return None;
        }
        Some(Self {
            buffer: Arc::clone(&self.buffer),
            shape,
            len,
        })
    }

    /// The raw device pointer, for kernel launch arguments.
    pub(crate) fn device_ptr(&self) -> CUdeviceptr {
        self.buffer.as_device_ptr()
    }

    /// Another handle to the allocation, for binding as a kernel operand.
    pub(crate) fn share(&self) -> Arc<DeviceBuffer<f32>> {
        Arc::clone(&self.buffer)
    }

    /// The allocation, if this is the last handle to it.
    ///
    /// `None` when an [`alias`](Self::alias) (or a live kernel binding) still
    /// refers to it — in which case dropping this handle is exactly right: the
    /// allocation stays alive for whoever else holds it, and *that* handle
    /// recycles it.
    pub(crate) fn into_unique_buffer(self) -> Option<DeviceBuffer<f32>> {
        Arc::try_unwrap(self.buffer).ok()
    }

    /// Read the tensor back into a host [`Tensor`], fencing once.
    ///
    /// The single place a resident activation crosses the bus back. The
    /// session memoizes the result into its run state, so a value read for one
    /// CPU consumer is not read again for the next.
    ///
    /// # Errors
    ///
    /// Propagates the driver's error from the copy or the fence.
    pub fn read_back(&self, ctx: &CudaContext) -> Result<Tensor, CudaDispatchError> {
        let stream = ctx.dnn.stream();
        let mut host = vec![0.0_f32; self.len];
        // SAFETY: a non-owning view of exactly `self.len` elements over an
        // allocation `from_owned` proved holds at least that many. Built by
        // `from_raw`, so its drop does not free the allocation.
        let view = unsafe { DeviceBuffer::<f32>::from_raw(self.device_ptr(), self.len) };
        view.copy_to_host_async(&mut host, stream)?;
        ctx.sync_stream(stream)?;
        ctx.caches.note_download(self.len);
        Ok(Tensor::new(host, self.shape.clone()))
    }
}

// ─── what the session asks for, and what it gets ───────────────────────────

/// Where the session wants this node's result left.
///
/// [`Self::Device`] is a **request**, not an instruction: an arm whose
/// epilogue only exists on the host (a `Gemm` with a bias to add, a
/// convolution whose engine owes a host-side bias) answers with
/// [`CudaDispatchOutcome::Host`] anyway, and the caller stores whichever it
/// gets. Mirrors `oxionnx::session::gpu_dispatch`'s `OutputPlacement`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaOutputPlacement {
    /// Read back into host tensors — the pre-residency behaviour, and still
    /// what a graph output and every CPU-consumed value gets.
    Host,
    /// Leave it in a device buffer for the next CUDA node.
    Device,
}

/// What one dispatch produced.
///
/// The predecessor of this type was `Vec<Tensor>` unconditionally, which was
/// the same statement as "every CUDA node ends in a read-back and a fence".
pub enum CudaDispatchOutcome {
    /// Read back into host tensors, positionally aligned with `node.outputs`.
    Host(Vec<Tensor>),
    /// Left in a device buffer. Single-output nodes only — the session's
    /// placement rule is what enforces that, and no op with a resident-capable
    /// arm produces more than one result.
    Device(CudaDeviceTensor),
}

/// The run's name → device-buffer map, as this crate needs to read it.
///
/// Implemented by `oxionnx`'s session-owned activation map. It is a trait
/// rather than a concrete type because the map's *policy* — which names may
/// stay resident, when each is released — belongs to the layer that knows the
/// node order, and this crate must not grow a second copy of it.
pub trait ResidentActivations {
    /// The device buffer holding `name`, if this run left one there.
    fn resident(&self, name: &str) -> Option<&CudaDeviceTensor>;

    /// Whether a *node in this graph* produced `name` onto the device.
    ///
    /// Consulted by the initializer-identity rule: a name a node has written
    /// must never be keyed into the weight cache, or one tensor's bytes would
    /// be served for another's. Mirrors `initializer_key`'s `holds_node_output`
    /// guard on the wgpu path.
    fn holds_node_output(&self, name: &str) -> bool;
}

/// The empty map: nothing is ever resident.
///
/// What [`crate::try_cuda_dispatch`] passes, so the pre-residency entry point
/// keeps its exact pre-residency behaviour rather than a re-implementation of
/// it.
pub struct NoActivations;

impl ResidentActivations for NoActivations {
    fn resident(&self, _name: &str) -> Option<&CudaDeviceTensor> {
        None
    }

    fn holds_node_output(&self, _name: &str) -> bool {
        false
    }
}

// ─── operand binding ───────────────────────────────────────────────────────

/// One operand as the dispatcher sees it.
///
/// The two variants are the whole point of this wave: [`Self::Device`] costs
/// nothing to bind, [`Self::Host`] costs an upload.
#[derive(Clone, Copy)]
pub(crate) enum InputBinding<'a> {
    /// Bytes on the host, to be uploaded into a pooled buffer.
    Host(&'a [f32]),
    /// A buffer this run already left on the device.
    Device(&'a CudaDeviceTensor),
}

impl<'a> InputBinding<'a> {
    /// Element count this operand holds.
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Host(data) => data.len(),
            Self::Device(tensor) => tensor.len(),
        }
    }

    /// The host bytes, when they exist.
    ///
    /// `None` for a resident operand, whose bytes are only on the device. The
    /// shadow-verification oracle is the caller that cares: it needs the exact
    /// inputs the kernel read, so a dispatch it cannot reconstruct must not be
    /// "verified" against a substitute. Residency is switched off wholesale
    /// under `OXIONNX_CUDA_VERIFY` for exactly this reason — see
    /// [`crate::reference::verify_enabled`].
    pub(crate) fn host(&self) -> Option<&'a [f32]> {
        match self {
            Self::Host(data) => Some(data),
            Self::Device(_) => None,
        }
    }

    /// Get this operand onto the device, uploading only if it is not there.
    ///
    /// `needed` is the element count the kernel will actually read: a host
    /// slice longer than that is truncated rather than uploaded whole, and a
    /// shorter one is a decline rather than an out-of-bounds read.
    ///
    /// # Errors
    ///
    /// Propagates allocation and upload failures.
    pub(crate) fn bind<'c>(
        &self,
        ctx: &'c CudaContext,
        label: &'static str,
        needed: usize,
        stream: &Stream,
    ) -> Result<Option<Operand<'c>>, CudaDispatchError> {
        match self {
            Self::Host(data) => {
                let Some(slice) = data.get(..needed) else {
                    return Ok(None);
                };
                Ok(Some(ctx.operand(None, label, slice, stream)?))
            }
            Self::Device(tensor) => {
                if tensor.len() < needed {
                    return Ok(None);
                }
                ctx.caches.note_resident_bind();
                Ok(Some(Operand::Resident(tensor.share())))
            }
        }
    }
}

// ─── finishing a dispatch ──────────────────────────────────────────────────

/// What a kernel left behind, before it is wrapped in a [`Tensor`].
pub(crate) enum KernelOutput {
    /// Read back to the host; the dispatch has fenced.
    Host(Vec<f32>),
    /// Left on the device; the dispatch has **not** fenced.
    Device(CudaDeviceTensor),
}

/// Finish a dispatch whose result is sitting in `d_output`.
///
/// This is the single decision point residency introduces, and both halves of
/// it are here rather than repeated in each op module:
///
/// * [`CudaOutputPlacement::Host`] — queue the read-back behind the kernel,
///   fence once, and hand back the numbers. Exactly what every op did before.
/// * [`CudaOutputPlacement::Device`] — take the allocation out of the pool and
///   hand it to the caller as a [`CudaDeviceTensor`]. **No read-back and no
///   fence**: the next node's kernel is enqueued behind this one on the same
///   queue, so the data is ordered without the host ever seeing it.
///
/// A `Device` request that cannot be honoured — the allocation is too small
/// for the declared shape — falls back to the host form rather than failing,
/// because the numbers are correct either way and the caller stores whichever
/// it gets.
///
/// # Errors
///
/// Propagates the driver's error from the read-back or the fence.
pub(crate) fn finish_output(
    ctx: &CudaContext,
    mut d_output: PooledBuffer<'_>,
    out_len: usize,
    shape: &[usize],
    placement: CudaOutputPlacement,
    stream: &Stream,
) -> Result<KernelOutput, CudaDispatchError> {
    if placement == CudaOutputPlacement::Device {
        // The borrow is handed over whole: `into_owned` disarms its `Drop`, so
        // the allocation is not returned to the pool here. It goes back through
        // `CudaContext::recycle_activation` when the session's activation map
        // releases it — see this module's header for why that needs no fence.
        let buffer = d_output.into_owned();
        match CudaDeviceTensor::from_owned(buffer, shape.to_vec()) {
            Some(tensor) => {
                ctx.caches.note_device_handoff();
                return Ok(KernelOutput::Device(tensor));
            }
            None => {
                // Unreachable for a correctly-sized acquisition, and handled
                // rather than trusted: the allocation is already out of the
                // pool, so it is simply dropped (freed) and the node falls back
                // to a host result computed from the same kernel output... which
                // no longer exists. Report a shape error instead of inventing
                // numbers.
                return Err(CudaDispatchError::Shape {
                    op: "activation_residency",
                    msg: format!(
                        "output buffer of {out_len} elements cannot hold the declared shape \
                         {shape:?}"
                    ),
                });
            }
        }
    }

    let mut out = vec![0.0_f32; out_len];
    d_output.download(&mut out, stream)?;
    ctx.sync_stream(stream)?;
    d_output.retire();
    Ok(KernelOutput::Host(out))
}

/// Retire a borrow whose queued work outlives this dispatch.
///
/// Called on the operands of a dispatch that did **not** fence, which the
/// pre-residency code never had to do because every dispatch fenced. Safe only
/// on a unified queue — see this module's header — and a split-stream handle
/// simply keeps the conservative behaviour: the borrow is freed on drop rather
/// than pooled, which costs reuse and nothing else.
pub(crate) fn retire_queued(ctx: &CudaContext, operand: &mut Operand<'_>) {
    if ctx.streams_unified() {
        operand.retire();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Everything that needs a real allocation is covered by the on-device
    // suites; these exercise the pure shape/aliasing arithmetic, which is where
    // a mistake would silently mis-describe a buffer.

    #[test]
    fn an_empty_map_never_reports_a_resident_value() {
        let map = NoActivations;
        assert!(map.resident("anything").is_none());
        assert!(!map.holds_node_output("anything"));
    }

    #[test]
    fn a_host_binding_reports_its_own_length_and_bytes() {
        let data = [1.0_f32, 2.0, 3.0];
        let binding = InputBinding::Host(&data);
        assert_eq!(binding.len(), 3);
        assert_eq!(binding.host(), Some(&data[..]));
    }

    #[test]
    fn placement_device_and_host_are_distinct_requests() {
        assert_ne!(CudaOutputPlacement::Host, CudaOutputPlacement::Device);
    }

    #[test]
    fn a_host_kernel_output_carries_the_numbers_it_read_back() {
        match KernelOutput::Host(vec![1.0, 2.0]) {
            KernelOutput::Host(data) => assert_eq!(data, vec![1.0_f32, 2.0]),
            KernelOutput::Device(_) => unreachable!("constructed as Host"),
        }
    }
}
