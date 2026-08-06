//! The compute root signature, and the pipeline-state cache.
//!
//! # Binding model — read this before changing anything
//!
//! **Root descriptors and root constants only.**  No descriptor heap, no descriptor
//! table, no CBV, anywhere in the HLSL path:
//!
//! | Root param | Type | HLSL register | Bound to |
//! |------------|------|---------------|----------|
//! | [`ROOT_PARAM_CONSTANTS`] = 0 | `32BIT_CONSTANTS`, `Num32BitValues = 8` | `b0` | [`crate::plan::MatMulConstants`] / [`crate::plan::ElementwiseConstants`] |
//! | [`ROOT_PARAM_SRV_A`] = 1 | `SRV` (root descriptor) | `t0` | operand `A` |
//! | [`ROOT_PARAM_SRV_B`] = 2 | `SRV` (root descriptor) | `t1` | operand `B` |
//! | [`ROOT_PARAM_UAV_C`] = 3 | `UAV` (root descriptor) | `u0` | output `C` |
//!
//! This eliminates **two entire hazard classes by construction**:
//!
//! * the 256-byte constant-buffer alignment rule — there *is* no constant buffer;
//!   `SetComputeRoot32BitConstants` takes the eight `u32`s directly; and
//! * descriptor-heap handle arithmetic — there *is* no heap;
//!   `SetComputeRootShaderResourceView` takes a raw GPU virtual address.
//!
//! Root SRV/UAV descriptors are legal for **raw and structured buffers**, which is all
//! this crate binds ([`crate::hlsl`] declares `StructuredBuffer<float>` and
//! `RWStructuredBuffer<float>`, neither with a counter — a UAV *with* a counter cannot
//! be a root descriptor).  They are not legal for typed buffers or textures, which this
//! crate never uses.
//!
//! Batch slicing is done with **root constants**, not by offsetting a root descriptor's
//! GPU address — a root descriptor's address has its own alignment rules, and there is
//! no reason to go near them when the shader can just index.
//!
//! # Why one root signature serves all eight entry points
//!
//! A root signature may expose *more* than a given shader reads.  The unary kernels
//! declare no `t1`, and share this signature happily.  But **every root parameter must
//! still be *set* before `Dispatch`** — the D3D12 debug layer errors on an unset root
//! parameter even when the bound shader never reads it — so the unary path in
//! [`super::hlsl_backend`] binds `A` to both `t0` and `t1`.
//!
//! # The shader-register hazard this file owns
//!
//! A mismatch between a root parameter's `ShaderRegister`/`ParameterType` and the
//! `register(b0)` / `register(t0)` / `register(t1)` / `register(u0)` declarations in
//! [`crate::hlsl`] is **not** a compile error.  It is a `CreateComputePipelineState`
//! failure at best, and garbage output or a device-removal at worst — on a user's
//! machine.  `root_signature_covers_every_register_the_shaders_declare` in this
//! module's tests parses the register declarations straight out of the HLSL and checks
//! them against the array this file actually serialises, which is the only mechanical
//! check of that join that exists.
//!
//! The DirectML backend *does* need a real descriptor heap; that lives in
//! [`super::device::DescriptorHeap`] and is used by nothing here.

use core::cell::RefCell;
use core::mem::ManuallyDrop;
use std::collections::HashMap;

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::{
    D3D12SerializeRootSignature, ID3D12PipelineState, ID3D12RootSignature,
    D3D12_CACHED_PIPELINE_STATE, D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_PIPELINE_STATE_FLAG_NONE,
    D3D12_ROOT_CONSTANTS, D3D12_ROOT_DESCRIPTOR, D3D12_ROOT_PARAMETER, D3D12_ROOT_PARAMETER_0,
    D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, D3D12_ROOT_PARAMETER_TYPE_SRV,
    D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_ROOT_SIGNATURE_DESC, D3D12_ROOT_SIGNATURE_FLAG_NONE,
    D3D12_SHADER_VISIBILITY_ALL, D3D_ROOT_SIGNATURE_VERSION_1,
};

use super::device::D3d12Core;
use super::shader::{blob_bytes, blob_text, ShaderBlob, ShaderKind};
use crate::error::{DirectMLError, HrExt, Result};
use crate::plan::ROOT_CONSTANT_COUNT;

// ─── root parameter indices ──────────────────────────────────────────────────

/// Root parameter index of the 8 × 32-bit root constants (`b0`).
pub(crate) const ROOT_PARAM_CONSTANTS: u32 = 0;
/// Root parameter index of the `A` operand's SRV (`t0`).
pub(crate) const ROOT_PARAM_SRV_A: u32 = 1;
/// Root parameter index of the `B` operand's SRV (`t1`).
pub(crate) const ROOT_PARAM_SRV_B: u32 = 2;
/// Root parameter index of the output's UAV (`u0`).
pub(crate) const ROOT_PARAM_UAV_C: u32 = 3;

/// Number of root parameters in the shared compute root signature.
const ROOT_PARAM_COUNT: usize = 4;

/// [`ROOT_PARAM_COUNT`] as the `u32` `NumParameters` wants.
///
/// Spelled as its own constant, and pinned to [`ROOT_PARAM_COUNT`] by the assertion
/// below, so that no `as u32` — which would silently truncate — appears at the call site.
const ROOT_PARAM_COUNT_U32: u32 = 4;
const _: [(); ROOT_PARAM_COUNT] = [(); ROOT_PARAM_COUNT_U32 as usize];

// The four indices above are used both as `SetComputeRoot*` arguments *and* as slots in
// the array `root_parameters` builds.  If they ever drift apart, the root signature
// would describe one layout while the backend bound another — a silent wrong answer.
// These four assertions pin each index to its slot at **compile** time, so the drift
// cannot survive a build (and, because `backend::d3d12` is cross-compiled for Windows
// from Linux under `-D warnings`, they are checked in this repository's CI).
const _: [(); ROOT_PARAM_CONSTANTS as usize] = [(); 0];
const _: [(); ROOT_PARAM_SRV_A as usize] = [(); 1];
const _: [(); ROOT_PARAM_SRV_B as usize] = [(); 2];
const _: [(); ROOT_PARAM_UAV_C as usize] = [(); 3];
const _: [(); ROOT_PARAM_COUNT] = [(); 1 + ROOT_PARAM_UAV_C as usize];

// ─── shader registers (the HLSL side of the contract) ────────────────────────

/// `register(b0)` — the shared constant block declared by every kernel.
const SHADER_REGISTER_CONSTANTS: u32 = 0;
/// `register(t0)` — operand `A`.
const SHADER_REGISTER_SRV_A: u32 = 0;
/// `register(t1)` — operand `B`.  Declared by the MatMul and binary kernels only.
const SHADER_REGISTER_SRV_B: u32 = 1;
/// `register(u0)` — the output `C`.
const SHADER_REGISTER_UAV_C: u32 = 0;

/// Every resource in [`crate::hlsl`] is declared without an explicit `space`, which
/// Shader Model 5.1 resolves to `space0`.
const REGISTER_SPACE: u32 = 0;

/// [`crate::plan::ROOT_CONSTANT_COUNT`] as the `u32` `D3D12_ROOT_CONSTANTS` wants.
///
/// Not an `as` cast at the use site: a cast would silently truncate if the neutral
/// constant ever grew, and the root signature would then describe fewer constants than
/// the backend pushes.  The assertion below makes the two provably equal at compile
/// time instead.
const ROOT_CONSTANT_COUNT_U32: u32 = 8;
const _: [(); ROOT_CONSTANT_COUNT] = [(); ROOT_CONSTANT_COUNT_U32 as usize];

// ─── root signature ──────────────────────────────────────────────────────────

/// One root SRV descriptor at `register(tN, space0)`.
fn root_srv(shader_register: u32) -> D3D12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR {
                ShaderRegister: shader_register,
                RegisterSpace: REGISTER_SPACE,
            },
        },
        // Compute shaders live entirely in the "all" visibility bucket; every other
        // `D3D12_SHADER_VISIBILITY_*` is a graphics stage and is rejected outright in a
        // compute root signature.
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

/// One root UAV descriptor at `register(uN, space0)`.
fn root_uav(shader_register: u32) -> D3D12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR {
                ShaderRegister: shader_register,
                RegisterSpace: REGISTER_SPACE,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

/// The `Num32BitValues`-wide root-constant block at `register(bN, space0)`.
fn root_constants(shader_register: u32) -> D3D12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: shader_register,
                RegisterSpace: REGISTER_SPACE,
                Num32BitValues: ROOT_CONSTANT_COUNT_U32,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

/// The shared compute root signature's parameter array.
///
/// Each parameter is written **at the index of its own `ROOT_PARAM_*` constant**, so the
/// slot a `SetComputeRoot*` call targets and the slot this array describes cannot
/// disagree.  Pure — no device, no FFI — so the tests below can inspect the exact array
/// that gets serialised.
fn root_parameters() -> [D3D12_ROOT_PARAMETER; ROOT_PARAM_COUNT] {
    let mut params = [D3D12_ROOT_PARAMETER::default(); ROOT_PARAM_COUNT];
    params[ROOT_PARAM_CONSTANTS as usize] = root_constants(SHADER_REGISTER_CONSTANTS);
    params[ROOT_PARAM_SRV_A as usize] = root_srv(SHADER_REGISTER_SRV_A);
    params[ROOT_PARAM_SRV_B as usize] = root_srv(SHADER_REGISTER_SRV_B);
    params[ROOT_PARAM_UAV_C as usize] = root_uav(SHADER_REGISTER_UAV_C);
    params
}

/// The single compute root signature shared by every entry point.
pub(crate) struct RootSig(ID3D12RootSignature);

impl RootSig {
    /// `D3D12SerializeRootSignature(D3D_ROOT_SIGNATURE_VERSION_1)` +
    /// `ID3D12Device::CreateRootSignature`.
    ///
    /// Version 1_0, not 1_1: the 1_1 descriptor *flags* (`DATA_STATIC`,
    /// `DATA_VOLATILE`, …) are a driver optimisation hint, and getting one wrong —
    /// promising a buffer is static and then writing it — is undefined behaviour that
    /// nothing here could catch.  1_0 has no such flags and no such hazard, and this
    /// crate's dispatches are far too coarse for the hint to matter.
    ///
    /// Flags: `D3D12_ROOT_SIGNATURE_FLAG_NONE` — never
    /// `ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT`; this is a compute-only signature and the IA
    /// flag would only waste root-signature space.
    ///
    /// # Errors
    /// [`DirectMLError::Win32`] when serialisation or creation fails.  The serialiser's
    /// error blob (which names the offending root parameter) is folded into the message.
    pub(crate) fn new(core: &D3d12Core) -> Result<Self> {
        let params = root_parameters();
        let desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: ROOT_PARAM_COUNT_U32,
            pParameters: params.as_ptr(),
            NumStaticSamplers: 0,
            pStaticSamplers: core::ptr::null(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };

        let mut blob: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;

        // SAFETY: `desc` is a live local; `desc.pParameters` points at `params`, another
        // live local of exactly `desc.NumParameters` elements, and both outlive this
        // synchronous call.  `pStaticSamplers` is null, which D3D12 accepts precisely
        // because `NumStaticSamplers` is 0.  `blob` and `errors` are out-parameters:
        // the callee writes an owned COM pointer into each (or leaves it null), and the
        // `Option<ID3DBlob>`s release them on drop, on both the success and failure
        // paths.  The serialiser copies everything it needs into `blob` before
        // returning, so `params` may die at the end of this function.
        let serialized = unsafe {
            D3D12SerializeRootSignature(
                &desc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                Some(&mut errors),
            )
        };

        if let Err(e) = serialized {
            // Not routed through `HrExt::ctx`: the serialiser's blob names the offending
            // root parameter, and an `HRESULT` alone ("E_INVALIDARG") would not.
            #[allow(clippy::cast_sign_loss)]
            let hresult = e.code().0 as u32;
            return Err(DirectMLError::Win32 {
                context: "D3D12SerializeRootSignature",
                hresult,
                message: format!("{}: {}", e.message(), blob_text(errors.as_ref())),
            });
        }

        let Some(blob) = blob else {
            return Err(DirectMLError::DeviceInitFailed(
                "D3D12SerializeRootSignature reported success but produced no blob".to_owned(),
            ));
        };

        let bytes = blob_bytes(&blob);
        if bytes.is_empty() {
            return Err(DirectMLError::DeviceInitFailed(
                "D3D12SerializeRootSignature produced an empty root-signature blob".to_owned(),
            ));
        }

        // SAFETY: `bytes` borrows `blob`'s allocation, which is live for this whole
        // scope, and holds exactly the serialised root signature the call above
        // produced.  `CreateRootSignature` parses it synchronously and returns an owned
        // `ID3D12RootSignature`; node mask 0 is the single-GPU-node case, which is the
        // only one this crate supports.
        let signature: ID3D12RootSignature = unsafe { core.device.CreateRootSignature(0, bytes) }
            .ctx("ID3D12Device::CreateRootSignature")?;

        Ok(Self(signature))
    }

    /// The underlying COM interface, for `SetComputeRootSignature` and for the PSO desc.
    pub(crate) fn raw(&self) -> &ID3D12RootSignature {
        &self.0
    }
}

// ─── PSO cache ───────────────────────────────────────────────────────────────

/// Lazily-populated compute PSO cache, one entry per [`ShaderKind`].
///
/// # Why this type exists
///
/// `D3DCompile` is *slow* — tens of milliseconds. Compiling a shader per node would
/// make the GPU path slower than the CPU path it is supposed to accelerate, on every
/// model with more than a handful of nodes. A [`PsoCache`] is built once per
/// [`crate::DirectMLContext`] and each of the eight entry points is compiled at most
/// once, on first use, for the lifetime of the process.
///
/// `RefCell`, not `Mutex`: the whole `Backend` already sits behind
/// `DirectMLContext`'s mutex, so there is never more than one thread in here, and a
/// second lock would be pure overhead. Nothing in this type escapes that mutex.
pub(crate) struct PsoCache {
    /// The one root signature every PSO in `psos` was created against.
    root: RootSig,
    /// Compiled pipeline states, keyed by entry point.
    psos: RefCell<HashMap<ShaderKind, ID3D12PipelineState>>,
}

impl PsoCache {
    /// Build the shared root signature; compile nothing yet.
    ///
    /// Shaders are compiled on first use rather than eagerly, so a model that only ever
    /// runs `MatMul` and `Relu` never pays for the other six.
    ///
    /// # Errors
    /// [`DirectMLError::Win32`] when the root signature cannot be created.
    pub(crate) fn new(core: &D3d12Core) -> Result<Self> {
        Ok(Self {
            root: RootSig::new(core)?,
            psos: RefCell::new(HashMap::new()),
        })
    }

    /// Get-or-compile the PSO for `kind`.
    ///
    /// Returns a **clone** of the COM handle — a refcount bump — so the `RefCell` borrow
    /// is released before the caller records anything into a command list. Holding a
    /// `Ref` across a `SetPipelineState` would be sound but would make any future
    /// re-entrant `get` a panic, and this crate does not panic.
    ///
    /// # Errors
    /// [`DirectMLError::ShaderCompile`] when `D3DCompile` rejects the HLSL — a hard
    /// error, never a decline; see [`super::shader`].
    /// [`DirectMLError::Win32`] when `CreateComputePipelineState` fails.
    pub(crate) fn get(&self, core: &D3d12Core, kind: ShaderKind) -> Result<ID3D12PipelineState> {
        if let Some(pso) = self.psos.borrow().get(&kind) {
            return Ok(pso.clone());
        }

        // Compile *without* holding any borrow: `create` performs FFI that can take tens
        // of milliseconds, and a live `Ref` here would turn any future re-entrancy into
        // a panic. A benign race is impossible — the enclosing mutex serialises us — so
        // the worst case is that this recomputes what a concurrent caller already
        // inserted, which cannot happen anyway.
        let pso = self.create(core, kind)?;
        self.psos.borrow_mut().insert(kind, pso.clone());
        Ok(pso)
    }

    /// The root signature every cached PSO was created against.
    ///
    /// The caller must bind *this* signature with `SetComputeRootSignature` before
    /// dispatching any PSO from this cache: a PSO records the signature it was created
    /// with, and D3D12 requires the bound signature to match it.
    pub(crate) fn root(&self) -> &RootSig {
        &self.root
    }

    /// Compile `kind` and create its compute PSO.  Always a cache miss.
    fn create(&self, core: &D3d12Core, kind: ShaderKind) -> Result<ID3D12PipelineState> {
        let blob = ShaderBlob::compile(kind)?;

        // `pRootSignature` is a `ManuallyDrop<Option<ID3D12RootSignature>>`. Filling it
        // with `Some(clone)` performs an `AddRef`; because the field is `ManuallyDrop`,
        // dropping `desc` will *not* `Release` it. We must therefore drop it by hand
        // after the call — otherwise every PSO leaks one reference to the root
        // signature, and nothing in this repository (not rustc, not clippy, not Miri,
        // not any test we can run) would ever say so.
        let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(self.root.raw().clone())),
            CS: blob.bytecode(),
            NodeMask: 0,
            CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
            Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
        };

        // SAFETY: `desc` is fully initialised. `desc.CS` holds raw pointers into `blob`,
        // which is a live local for the whole call (and is dropped explicitly below,
        // *after* it). `desc.pRootSignature` holds an owned reference obtained by the
        // `clone` above. `CreateComputePipelineState` reads `desc` synchronously and
        // copies the bytecode into the driver's own storage, so nothing here needs to
        // outlive the call.
        let created = unsafe { core.device.CreateComputePipelineState(&desc) }
            .ctx("ID3D12Device::CreateComputePipelineState");

        // SAFETY: `desc.pRootSignature` was initialised with `ManuallyDrop::new(Some(..))`
        // immediately above and has not been dropped or moved out of since.
        // `D3D12_COMPUTE_PIPELINE_STATE_DESC` has no `Drop` impl (verified against
        // `windows-0.62.2`), so this is the *only* release of that reference: it runs
        // exactly once, on both the success and the failure path, balancing the `clone`.
        // `desc` is not used again afterwards.
        unsafe { ManuallyDrop::drop(&mut desc.pRootSignature) };

        // `blob` must outlive the call above — `desc.CS` pointed into it. Dropping it
        // here, explicitly, keeps that ordering visible to whoever edits this next.
        drop(blob);

        created
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::{
        root_parameters, ROOT_CONSTANT_COUNT_U32, ROOT_PARAM_CONSTANTS, ROOT_PARAM_COUNT,
        ROOT_PARAM_SRV_A, ROOT_PARAM_SRV_B, ROOT_PARAM_UAV_C, SHADER_REGISTER_CONSTANTS,
        SHADER_REGISTER_SRV_A, SHADER_REGISTER_SRV_B, SHADER_REGISTER_UAV_C,
    };
    use crate::backend::d3d12::shader::ShaderKind;
    use crate::plan::ROOT_CONSTANT_COUNT;
    use std::collections::BTreeSet;
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, D3D12_ROOT_PARAMETER_TYPE_SRV,
        D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_SHADER_VISIBILITY_ALL,
    };

    const ALL: [ShaderKind; 13] = [
        ShaderKind::MatMul,
        ShaderKind::Add,
        ShaderKind::Sub,
        ShaderKind::Mul,
        ShaderKind::Div,
        ShaderKind::Relu,
        ShaderKind::Sigmoid,
        ShaderKind::Tanh,
        ShaderKind::Softmax,
        ShaderKind::ReduceSum,
        ShaderKind::ReduceMean,
        ShaderKind::ReduceMax,
        ShaderKind::ReduceMin,
    ];

    /// Every `register(X#)` an HLSL source declares, as `('b' | 't' | 'u', number)`.
    ///
    /// Deliberately parsed out of the shader text rather than restated: the point is to
    /// catch an edit to [`crate::hlsl`] that this file was not told about.
    fn declared_registers(src: &str) -> BTreeSet<(char, u32)> {
        let mut out = BTreeSet::new();
        for tail in src.split("register(").skip(1) {
            let decl: String = tail.chars().take_while(|c| *c != ')').collect();
            let mut chars = decl.trim().chars();
            let Some(class) = chars.next() else { continue };
            let digits: String = chars.take_while(char::is_ascii_digit).collect();
            if let Ok(number) = digits.parse::<u32>() {
                out.insert((class, number));
            }
        }
        out
    }

    #[test]
    fn the_register_parser_actually_finds_things() {
        // A parser that silently returns nothing would make every test below vacuous.
        let matmul = declared_registers(ShaderKind::MatMul.source());
        assert_eq!(
            matmul,
            BTreeSet::from([('b', 0), ('t', 0), ('t', 1), ('u', 0)]),
            "MATMUL_HLSL must declare exactly b0, t0, t1, u0"
        );
    }

    #[test]
    fn root_signature_covers_every_register_the_shaders_declare() {
        // THE hazard this file owns. A root signature that does not cover a register the
        // shader declares is not a compile error — it is a PSO-creation failure, garbage
        // output, or a device-removal, on a user's machine. Nothing else checks this.
        let params = root_parameters();

        // What the root signature actually provides, derived from the array we serialise.
        let mut provided: BTreeSet<(char, u32)> = BTreeSet::new();
        for param in &params {
            match param.ParameterType {
                D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS => {
                    // SAFETY: the union's `Constants` arm is the one `root_constants`
                    // wrote, and `ParameterType` is what says so.
                    let c = unsafe { param.Anonymous.Constants };
                    provided.insert(('b', c.ShaderRegister));
                }
                D3D12_ROOT_PARAMETER_TYPE_SRV => {
                    // SAFETY: `ParameterType` says this is the `Descriptor` arm.
                    let d = unsafe { param.Anonymous.Descriptor };
                    provided.insert(('t', d.ShaderRegister));
                }
                D3D12_ROOT_PARAMETER_TYPE_UAV => {
                    // SAFETY: as above.
                    let d = unsafe { param.Anonymous.Descriptor };
                    provided.insert(('u', d.ShaderRegister));
                }
                other => panic!("unexpected root parameter type {other:?}"),
            }
        }

        assert_eq!(
            provided,
            BTreeSet::from([('b', 0), ('t', 0), ('t', 1), ('u', 0)]),
            "the root signature must provide exactly b0, t0, t1, u0"
        );

        for kind in ALL {
            let declared = declared_registers(kind.source());
            assert!(
                !declared.is_empty(),
                "{}: parsed no registers",
                kind.as_str()
            );
            for reg in &declared {
                assert!(
                    provided.contains(reg),
                    "{}: the shader declares register({}{}) but the root signature does not \
                     provide it — this is a device-removal, not a compile error",
                    kind.as_str(),
                    reg.0,
                    reg.1
                );
            }
        }
    }

    #[test]
    fn unary_kernels_are_a_strict_subset_and_still_need_t1_bound() {
        // The unary sources declare no `t1`. That is *fine* for the shared root
        // signature — but root parameter 2 must still be SET before Dispatch, or the
        // debug layer errors. This test documents the asymmetry that `hlsl_backend`'s
        // "bind A to both t0 and t1" exists to satisfy.
        let unary = declared_registers(ShaderKind::Relu.source());
        assert_eq!(unary, BTreeSet::from([('b', 0), ('t', 0), ('u', 0)]));
        assert!(
            !unary.contains(&('t', 1)),
            "unary kernels must not declare t1"
        );

        let binary = declared_registers(ShaderKind::Add.source());
        assert!(binary.contains(&('t', 1)), "binary kernels must declare t1");
    }

    #[test]
    fn each_root_parameter_sits_at_its_own_constants_index_with_the_right_type() {
        let params = root_parameters();
        assert_eq!(params.len(), ROOT_PARAM_COUNT);

        assert_eq!(
            params[ROOT_PARAM_CONSTANTS as usize].ParameterType,
            D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS
        );
        assert_eq!(
            params[ROOT_PARAM_SRV_A as usize].ParameterType,
            D3D12_ROOT_PARAMETER_TYPE_SRV
        );
        assert_eq!(
            params[ROOT_PARAM_SRV_B as usize].ParameterType,
            D3D12_ROOT_PARAMETER_TYPE_SRV
        );
        assert_eq!(
            params[ROOT_PARAM_UAV_C as usize].ParameterType,
            D3D12_ROOT_PARAMETER_TYPE_UAV
        );

        // SAFETY: each read below matches the `ParameterType` asserted immediately above.
        unsafe {
            let constants = params[ROOT_PARAM_CONSTANTS as usize].Anonymous.Constants;
            assert_eq!(constants.ShaderRegister, SHADER_REGISTER_CONSTANTS);
            assert_eq!(constants.Num32BitValues, ROOT_CONSTANT_COUNT_U32);

            assert_eq!(
                params[ROOT_PARAM_SRV_A as usize]
                    .Anonymous
                    .Descriptor
                    .ShaderRegister,
                SHADER_REGISTER_SRV_A
            );
            assert_eq!(
                params[ROOT_PARAM_SRV_B as usize]
                    .Anonymous
                    .Descriptor
                    .ShaderRegister,
                SHADER_REGISTER_SRV_B
            );
            assert_eq!(
                params[ROOT_PARAM_UAV_C as usize]
                    .Anonymous
                    .Descriptor
                    .ShaderRegister,
                SHADER_REGISTER_UAV_C
            );
        }
    }

    #[test]
    fn every_root_parameter_is_visible_to_the_compute_stage() {
        // Any other visibility is a graphics stage and is rejected in a compute root
        // signature — a `D3D12SerializeRootSignature` failure at context creation.
        for param in &root_parameters() {
            assert_eq!(param.ShaderVisibility, D3D12_SHADER_VISIBILITY_ALL);
        }
    }

    #[test]
    fn the_root_constant_block_is_exactly_as_wide_as_the_neutral_plan_says() {
        // `SetComputeRoot32BitConstants` pushes `ROOT_CONSTANT_COUNT` u32s; the root
        // signature must declare exactly that many, or D3D12 rejects the call. The
        // `const _` assertion above already proves this at compile time — this test is
        // the runtime witness for anyone reading the failure output.
        assert_eq!(ROOT_CONSTANT_COUNT_U32 as usize, ROOT_CONSTANT_COUNT);
        assert_eq!(ROOT_CONSTANT_COUNT_U32, 8);
    }
}
