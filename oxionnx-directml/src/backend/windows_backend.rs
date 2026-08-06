//! The Windows backend: a [`D3d12Core`] plus whichever engine came up on it.
//!
//! # Resolution order, and why it is this order
//!
//! 1. [`D3d12Core::try_new`] — a D3D12 device, a COMPUTE queue, an allocator, a closed
//!    command list, a fence and an auto-reset event.  No adapter → `None` → the session
//!    runs on the CPU, which is not a failure.
//! 2. [`DmlEngine::new`] → [`BackendKind::DirectMl`].  **Preferred**, because DirectML's
//!    operators are vendor-tuned (they hit the tensor cores / matrix units that a naive
//!    `cs_5_0` shader cannot reach), and because its 0-stride tensor descriptors express
//!    broadcasting without copying a byte.
//! 3. else [`HlslEngine::new`] → [`BackendKind::Hlsl`].  `DirectML.dll` is **not** present
//!    on every supported Windows SKU; when it is missing, `DMLCreateDevice` cannot be
//!    resolved and this is the fallback that keeps the provider useful.
//! 4. else `None`.
//!
//! **Never panics.**  **Never returns `Err`**: a machine with no GPU is not an error, it
//! is a machine that runs on the CPU.
//!
//! # Threading
//!
//! Everything reachable from here is `!Send + !Sync` COM plus `Cell`/`RefCell` interior
//! mutability.  That is sound only because the whole `Backend` sits behind
//! [`crate::DirectMLContext`]'s mutex; see the `SAFETY` block on `context.rs`'s
//! `BackendCell` for the argument, which is load-bearing and not a formality.

use crate::backend::d3d12::device::D3d12Core;
use crate::backend::d3d12::hlsl_backend::HlslEngine;
use crate::backend::dml::dml_backend::DmlEngine;
use crate::backend::BackendKind;
use crate::error::{DirectMLError, Result};
use crate::plan::{
    BinaryOp, ConvPlan, ElementwisePlan, MatMulPlan, ReducePlan, SoftmaxPlan, UnaryOp,
};

/// The Windows backend.
pub(crate) struct Backend {
    /// The D3D12 foundation both engines record into.  Shared, never duplicated: a second
    /// device would mean a second command queue and a second fence, and nothing here needs
    /// either.
    core: D3d12Core,
    /// Whichever engine came up.  Fixed for the life of the backend — there is no
    /// per-node re-selection, because a machine does not grow a `DirectML.dll` halfway
    /// through an inference.
    engine: Engine,
    /// Cached [`Engine`] discriminant, so `kind()` needs no match.
    kind: BackendKind,
}

/// The resolved execution engine.
enum Engine {
    /// Genuine DirectML operators.
    Dml(DmlEngine),
    /// `D3DCompile`d compute shaders — the fallback when `DirectML.dll` is absent.
    Hlsl(HlslEngine),
}

impl Backend {
    /// Acquire the best available backend: DirectML, else HLSL, else nothing.
    ///
    /// **Never panics.**
    pub(crate) fn try_new() -> Option<Self> {
        let core = D3d12Core::try_new()?;

        if let Some(engine) = DmlEngine::new(&core) {
            tracing::info!(
                adapter = %core.adapter_name,
                backend = BackendKind::DirectMl.as_str(),
                "DirectML execution provider active"
            );
            return Some(Self {
                core,
                engine: Engine::Dml(engine),
                kind: BackendKind::DirectMl,
            });
        }

        match HlslEngine::new(&core) {
            Ok(engine) => {
                tracing::info!(
                    adapter = %core.adapter_name,
                    backend = BackendKind::Hlsl.as_str(),
                    "DirectML.dll unavailable; falling back to D3D12 compute shaders"
                );
                Some(Self {
                    core,
                    engine: Engine::Hlsl(engine),
                    kind: BackendKind::Hlsl,
                })
            }
            Err(e) => {
                // D3D12 came up but we could not even build a root signature.  That is a
                // genuine failure, and it is the last thing that happens before we hand
                // the whole graph back to the CPU — so it is said out loud exactly once,
                // rather than discarded into the `None` that the signature forces.
                tracing::warn!(
                    adapter = %core.adapter_name,
                    error = %e,
                    "D3D12 device acquired but no execution engine could be built; \
                     the DirectML provider is disabled for this process"
                );
                None
            }
        }
    }

    /// Which backend this is.
    pub(crate) fn kind(&self) -> BackendKind {
        self.kind
    }

    /// The DXGI adapter description, e.g. `"NVIDIA GeForce RTX 4090"`.
    pub(crate) fn adapter_name(&self) -> String {
        self.core.adapter_name.clone()
    }

    /// Execute a planned MatMul / Gemm.
    ///
    /// `a` / `b` / `c` are the dense f32 operands exactly as they sit in the caller's
    /// `Tensor::data`.  Returns the dense `[batch, m, n]` output.
    ///
    /// # Errors
    /// [`crate::DirectMLError::Declined`] when this backend cannot express the plan — the
    /// router turns that into `Ok(None)` (a correct CPU fallback), *not* an error.  Any
    /// other variant is a genuine GPU failure and is reported as one.
    pub(crate) fn matmul(
        &self,
        plan: &MatMulPlan,
        a: &[f32],
        b: &[f32],
        c: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        match &self.engine {
            Engine::Dml(engine) => engine.matmul(&self.core, plan, a, b, c),
            Engine::Hlsl(engine) => engine.matmul(&self.core, plan, a, b, c),
        }
    }

    /// Execute a planned binary elementwise op.
    ///
    /// # Errors
    /// As [`Self::matmul`].
    pub(crate) fn binary(
        &self,
        plan: &ElementwisePlan,
        op: BinaryOp,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>> {
        match &self.engine {
            Engine::Dml(engine) => engine.binary(&self.core, plan, op, a, b),
            Engine::Hlsl(engine) => engine.binary(&self.core, plan, op, a, b),
        }
    }

    /// Execute a planned unary elementwise op.
    ///
    /// # Errors
    /// As [`Self::matmul`].
    pub(crate) fn unary(&self, plan: &ElementwisePlan, op: UnaryOp, a: &[f32]) -> Result<Vec<f32>> {
        match &self.engine {
            Engine::Dml(engine) => engine.unary(&self.core, plan, op, a),
            Engine::Hlsl(engine) => engine.unary(&self.core, plan, op, a),
        }
    }

    /// Execute a planned single-axis Softmax.
    ///
    /// Both engines implement `softmax`, so — exactly as [`Self::matmul`] — this simply
    /// hands the plan to whichever one came up.  The two op-specific facts about Softmax
    /// are both encoded *inside* the engines, not here:
    ///
    /// * On the **DirectML** engine a non-innermost axis returns
    ///   [`crate::DirectMLError::Declined`] (the axis-less `DML_ACTIVATION_SOFTMAX`
    ///   normalises the last dimension only).  That decline propagates unchanged through
    ///   this match to `dispatch::route`, which turns it into a CPU fallback — a decline is
    ///   "try elsewhere", never a failure.  A genuine `DmlEngine` `Err` (a broken GPU) is
    ///   *not* a `Declined` and so propagates as the failure it is: the two stay distinct.
    /// * The **HLSL** engine, when it is the live one, handles any axis itself, so it never
    ///   declines on the axis.
    ///
    /// # Errors
    /// As [`Self::matmul`].
    pub(crate) fn softmax(&self, plan: &SoftmaxPlan, a: &[f32]) -> Result<Vec<f32>> {
        match &self.engine {
            Engine::Dml(engine) => engine.softmax(&self.core, plan, a),
            Engine::Hlsl(engine) => engine.softmax(&self.core, plan, a),
        }
    }

    /// Execute a planned single-axis Reduce (`Sum` / `Mean` / `Max` / `Min`).
    ///
    /// # Errors
    /// As [`Self::matmul`].
    pub(crate) fn reduce(&self, plan: &ReducePlan, a: &[f32]) -> Result<Vec<f32>> {
        match &self.engine {
            Engine::Dml(engine) => engine.reduce(&self.core, plan, a),
            Engine::Hlsl(engine) => engine.reduce(&self.core, plan, a),
        }
    }

    /// Execute a planned 2-D Conv.
    ///
    /// # Conv is DirectML-only — the HLSL engine has no convolution kernel
    ///
    /// Unlike every other op, there is deliberately no `HlslEngine::conv`: a correct,
    /// performant convolution shader is a wholly different animal from the naive
    /// elementwise kernels (see [`crate::plan::ConvPlan`]).  So when the live engine is the
    /// HLSL fallback (i.e. `DirectML.dll` was absent), this **declines** with
    /// [`crate::DirectMLError::Declined`] rather than failing — the router turns that into a
    /// correct CPU convolution.  Only the genuine DirectML metacommand executes `Conv` on
    /// the GPU.  A decline is not a failure, and this is the former.
    ///
    /// # Errors
    /// As [`Self::matmul`], plus the HLSL-path decline described above.
    pub(crate) fn conv(
        &self,
        plan: &ConvPlan,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        match &self.engine {
            Engine::Dml(engine) => engine.conv(&self.core, plan, input, weight, bias),
            Engine::Hlsl(_) => Err(DirectMLError::Declined(
                "Conv: the HLSL fallback engine has no convolution kernel; declining to the \
                 CPU operator (only the genuine DirectML backend executes Conv on the GPU)"
                    .to_owned(),
            )),
        }
    }
}
