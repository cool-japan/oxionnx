//! BGL helper functions and WGSL shader constants for oxionnx-gpu.

pub(super) fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
pub(super) fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
pub(super) fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
pub(super) const MATMUL_SHADER: &str = r#"
struct Params {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= params.M || col >= params.N) { return; }

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.K; i++) {
        sum += A[row * params.K + i] * B[i * params.N + col];
    }
    C[row * params.N + col] = sum;
}
"#;
/// Softmax WGSL shader — one workgroup per row, shared-memory tree reduction.
///
/// [a7-18] The previous kernel gave one *thread* to each row and ran two
/// dispatches over it: a serial max scan plus an exp write, then a serial sum
/// plus a normalize. Because `SOFTMAX_DIM_THRESHOLD` is 1000, the GPU was only
/// ever used when every thread had to make at least 1000 strided, dependent
/// global-memory accesses — adjacent lanes read addresses `row_len` apart, so
/// essentially every access touched its own cache line. A
/// `[1, 32, 1024, 1024]` attention softmax was 32_768 threads each walking
/// ~4096 elements four times.
///
/// This version assigns one 256-thread *workgroup* to each row and fuses both
/// passes into a single dispatch, mirroring `LAYER_NORM_SHADER`:
///
/// 1. each thread scans its strided slice for a local max, then a shared-memory
///    tree reduction produces the row max;
/// 2. each thread writes `exp(x - row_max)` for its slice and accumulates a
///    local sum, then a second tree reduction produces the row sum;
/// 3. each thread scales its own slice by `1 / sum`.
///
/// Adjacent lanes now read adjacent addresses, so the accesses coalesce, and
/// the row is traversed three times instead of four across one dispatch
/// instead of two.
///
/// Layout: input[num_rows * row_len], output[num_rows * row_len],
/// params = { num_rows, row_len, wg_per_row, _pad }.
///
/// `wg_per_row` is the grid's X extent: more rows than the device allows along
/// a single dimension are dispatched as a 2-D grid and the row index is rebuilt
/// as `wid.y * wg_per_row + wid.x`, exactly like the LayerNorm kernel. The old
/// kernel indexed rows with `gid.x` alone and had to decline outright whenever
/// a second dimension was needed.
pub(super) const SOFTMAX_SHADER: &str = r#"
struct Params {
    num_rows: u32,
    row_len: u32,
    wg_per_row: u32,
    _pad: u32,
}

const WG_SIZE: u32 = 256u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax_rows(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.y * params.wg_per_row + wid.x;
    if (row >= params.num_rows) { return; }
    let tid = lid.x;
    let n = params.row_len;
    let base = row * n;

    // Phase 1: row max. Seeding every thread with element 0 (always present —
    // the host declines rows shorter than SOFTMAX_DIM_THRESHOLD) keeps threads
    // with no slice of their own from contributing a sentinel, and matches the
    // old kernel's `v > max_val` comparison so NaNs are dropped identically.
    var local_max: f32 = input[base];
    for (var i: u32 = tid; i < n; i = i + WG_SIZE) {
        let v = input[base + i];
        if (v > local_max) { local_max = v; }
    }
    shared_data[tid] = local_max;
    workgroupBarrier();

    for (var s: u32 = WG_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            let other = shared_data[tid + s];
            if (other > shared_data[tid]) { shared_data[tid] = other; }
        }
        workgroupBarrier();
    }

    let row_max = shared_data[0];
    // Every thread has read shared_data[0]; the barrier keeps the phase-2
    // writes below from racing ahead of a slower lane's read.
    workgroupBarrier();

    // Phase 2: exp(x - row_max) into the output, accumulating the row sum.
    var local_sum: f32 = 0.0;
    for (var i: u32 = tid; i < n; i = i + WG_SIZE) {
        let e = exp(input[base + i] - row_max);
        output[base + i] = e;
        local_sum = local_sum + e;
    }
    shared_data[tid] = local_sum;
    workgroupBarrier();

    for (var s: u32 = WG_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + s];
        }
        workgroupBarrier();
    }

    let inv_sum = 1.0 / shared_data[0];
    workgroupBarrier();

    // Phase 3: normalize. Each thread touches exactly the indices it wrote in
    // phase 2, so no cross-thread ordering is involved.
    for (var i: u32 = tid; i < n; i = i + WG_SIZE) {
        output[base + i] = output[base + i] * inv_sum;
    }
}
"#;
/// Element-wise WGSL shader — relu, sigmoid, gelu, tanh, exp, sqrt, abs, neg, log, silu, leaky_relu.
///
/// Layout: input[len], output[len], params = { len, alpha, row_threads, _pad }.
///
/// `row_threads` is `grid_x * 256`: dispatches wider than
/// `max_compute_workgroups_per_dimension` are issued as a 2-D grid and the flat
/// element index is rebuilt as `gid.y * row_threads + gid.x` (which degenerates
/// to `gid.x` for the common single-row case). `alpha` carries the LeakyRelu
/// slope from the node's attribute instead of baking a constant into the kernel.
pub(super) const ELEMENTWISE_SHADER: &str = r#"
struct Params {
    len: u32,
    alpha: f32,
    row_threads: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn flat_index(gid: vec3<u32>) -> u32 {
    return gid.y * params.row_threads + gid.x;
}

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.len) { return; }
    output[idx] = max(input[idx], 0.0);
}

@compute @workgroup_size(256)
fn sigmoid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.len) { return; }
    output[idx] = 1.0 / (1.0 + exp(-input[idx]));
}

@compute @workgroup_size(256)
fn gelu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.len) { return; }
    let x = input[idx];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c = 0.7978845608; // sqrt(2/pi)
    let inner = c * (x + 0.044715 * x * x * x);
    output[idx] = 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(256)
fn op_tanh(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.len) { return; }
    output[idx] = tanh(input[idx]);
}

@compute @workgroup_size(256)
fn op_exp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.len) { return; }
    output[idx] = exp(input[idx]);
}

@compute @workgroup_size(256)
fn op_sqrt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.len) { return; }
    output[idx] = sqrt(input[idx]);
}

@compute @workgroup_size(256)
fn op_abs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.len) { return; }
    output[idx] = abs(input[idx]);
}

@compute @workgroup_size(256)
fn op_neg(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.len) { return; }
    output[idx] = -input[idx];
}

@compute @workgroup_size(256)
fn op_log(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.len) { return; }
    output[idx] = log(input[idx]);
}

@compute @workgroup_size(256)
fn silu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.len) { return; }
    let x = input[idx];
    output[idx] = x / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn leaky_relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.len) { return; }
    let x = input[idx];
    output[idx] = select(params.alpha * x, x, x >= 0.0);
}
"#;
/// Reduction WGSL shader — reduce_sum and reduce_max along an axis.
///
/// The input is conceptualized as [outer_size, axis_len, inner_size].
/// Each thread handles one (outer, inner) pair, reducing over axis_len.
/// Layout: input[total], output[outer_size * inner_size], params = { outer_size, axis_len, inner_size }.
pub(super) const REDUCE_SHADER: &str = r#"
struct Params {
    outer_size: u32,
    axis_len: u32,
    inner_size: u32,
    row_threads: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn flat_index(gid: vec3<u32>) -> u32 {
    return gid.y * params.row_threads + gid.x;
}

@compute @workgroup_size(256)
fn reduce_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_idx = flat_index(gid);
    let total_out = params.outer_size * params.inner_size;
    if (flat_idx >= total_out || params.axis_len == 0u || params.inner_size == 0u) { return; }

    let outer = flat_idx / params.inner_size;
    let inner = flat_idx % params.inner_size;
    let in_base = outer * params.axis_len * params.inner_size + inner;

    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < params.axis_len; i++) {
        acc += input[in_base + i * params.inner_size];
    }
    output[flat_idx] = acc;
}

@compute @workgroup_size(256)
fn reduce_max(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_idx = flat_index(gid);
    let total_out = params.outer_size * params.inner_size;
    if (flat_idx >= total_out || params.axis_len == 0u || params.inner_size == 0u) { return; }

    let outer = flat_idx / params.inner_size;
    let inner = flat_idx % params.inner_size;
    let in_base = outer * params.axis_len * params.inner_size + inner;

    var acc: f32 = input[in_base];
    for (var i: u32 = 1u; i < params.axis_len; i++) {
        let v = input[in_base + i * params.inner_size];
        if (v > acc) { acc = v; }
    }
    output[flat_idx] = acc;
}

@compute @workgroup_size(256)
fn reduce_min(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_idx = flat_index(gid);
    let total_out = params.outer_size * params.inner_size;
    if (flat_idx >= total_out || params.axis_len == 0u || params.inner_size == 0u) { return; }

    let outer = flat_idx / params.inner_size;
    let inner = flat_idx % params.inner_size;
    let in_base = outer * params.axis_len * params.inner_size + inner;

    var acc: f32 = input[in_base];
    for (var i: u32 = 1u; i < params.axis_len; i++) {
        let v = input[in_base + i * params.inner_size];
        if (v < acc) { acc = v; }
    }
    output[flat_idx] = acc;
}

@compute @workgroup_size(256)
fn reduce_mean(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_idx = flat_index(gid);
    let total_out = params.outer_size * params.inner_size;
    if (flat_idx >= total_out || params.axis_len == 0u || params.inner_size == 0u) { return; }

    let outer = flat_idx / params.inner_size;
    let inner = flat_idx % params.inner_size;
    let in_base = outer * params.axis_len * params.inner_size + inner;

    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < params.axis_len; i++) {
        acc += input[in_base + i * params.inner_size];
    }
    output[flat_idx] = acc / f32(params.axis_len);
}
"#;
/// Binary element-wise WGSL shader — add, mul.
///
/// Layout: a[len], b[len], output[len], params = { len, alpha (unused), row_threads, _pad }.
///
/// Shares the [`EwParams`](crate::shaders) uniform layout with the unary kernels;
/// `row_threads` reconstructs the flat index for 2-D dispatch grids.
pub(super) const BINARY_ELEMENTWISE_SHADER: &str = r#"
struct Params {
    len: u32,
    alpha: f32,
    row_threads: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn flat_index(gid: vec3<u32>) -> u32 {
    return gid.y * params.row_threads + gid.x;
}

@compute @workgroup_size(256)
fn op_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.len) { return; }
    output[idx] = a[idx] + b[idx];
}

@compute @workgroup_size(256)
fn op_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = flat_index(gid);
    if (idx >= params.len) { return; }
    output[idx] = a[idx] * b[idx];
}
"#;
/// Register-blocked tiled matrix multiply WGSL shader.
///
/// Each workgroup owns a 64(M) x 64(N) macro-tile of the output. Its 256
/// threads (`@workgroup_size(16, 16)`) each compute a 4x4 register tile
/// within that macro-tile, iterating over K in `KTILE`-deep chunks staged
/// through workgroup shared memory. Register blocking does not change the
/// FLOP count — it changes how many *distinct-address* shared-memory reads
/// pay for each FMA: the one-thread-one-element predecessor of this kernel
/// had every shared element re-read (broadcast) by 16 threads before this
/// change; here each loaded element feeds 4 FMAs before the next shared
/// read, which is what lets the kernel approach f32 peak instead of being
/// shared-memory-bandwidth-bound. Bounds handling (ragged M/N/K) is
/// unconditional zero-padding on load, as the predecessor kernel did.
///
/// [register tile: 16 named scalars, not a `[4][4]` array] The 16
/// accumulators (`acc00`..`acc33`) and the 4+4 per-`kk` operands
/// (`a_reg0`..`3`, `b_reg0`..`3`) are individually-named `f32` locals, not
/// `array<f32, 4>` / `array<array<f32, 4>, 4>`. This was measured, not
/// stylistic: an earlier array-based version of this exact algorithm ran
/// *slower* than the one-thread-one-element predecessor it was meant to
/// replace (naga's WGSL-\>MSL lowering did not promote the small
/// loop-indexed arrays to registers on this backend, so every accumulator
/// read/write went through real memory instead) — rewriting the identical
/// math with named scalars in place of the two small arrays measured
/// 3-4x faster than that array version, and is what actually delivers the
/// register-blocking win described above. If a future edit reintroduces an
/// array here (e.g. to shrink this file), re-benchmark before assuming it
/// is a neutral refactor.
///
/// [dispatch contract] `oxionnx-gpu/src/compute.rs` (a parallel agent's file
/// — not touched here) dispatches `ceil(N/16) x ceil(M/16)` workgroups; that
/// formula was sized for this kernel's *predecessor*, whose macro-tile was
/// 16x16. It is intentionally left as-is rather than recomputed for the new
/// 64x64 macro-tile, because `ceil(X/16) >= ceil(X/64)` for every X >= 1 (16
/// is a divisor of 64), so that dispatch always launches *at least* as many
/// workgroups along each axis as this kernel needs — it only ever
/// over-provisions, never under-provisions. The `tiles_m` / `tiles_n` bounds
/// check below turns the surplus (up to 16x fewer workgroups would actually
/// be needed at large M/N) into an immediate, uniform-control-flow return —
/// `workgroup_id` is identical for every invocation in a workgroup, so this
/// is not a divergent exit ahead of the `workgroupBarrier()` calls later in
/// the function (same reasoning `softmax_rows` / `layer_norm` in this file
/// already rely on for their own early returns). This is why the entry
/// point name, bind group layout and dispatch-count formula did not need to
/// change anywhere else: anyone lowering that divisor below 64 must keep it
/// a divisor of 64 (e.g. 32, 8, ...), or this bound under-dispatches.
/// (Measured: the 16x surplus this produces at large M/N is not a
/// meaningful cost — every early-returning workgroup retires in a handful
/// of instructions.)
pub(super) const TILED_MATMUL_SHADER: &str = r#"
struct Dims {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}

// Macro-tile: output rows/cols owned by one workgroup.
const MACRO_M: u32 = 64u;
const MACRO_N: u32 = 64u;
// K-chunk staged into shared memory per outer-loop iteration.
const KTILE: u32 = 16u;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

// k-major storage (tile_a[k][m], tile_b[k][n]): for a fixed k, a thread's 4
// register-tile elements (`m_base..m_base+3` or `n_base..n_base+3`) are
// contiguous within one row, and same-k/same-tile-row(col) reads across
// threads land on shared addresses rather than a strided pattern.
var<workgroup> tile_a: array<array<f32, 64>, 16>;
var<workgroup> tile_b: array<array<f32, 64>, 16>;

@compute @workgroup_size(16, 16)
fn tiled_matmul(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let M = dims.M;
    let K = dims.K;
    let N = dims.N;

    // See the "[dispatch contract]" note on the Rust doc comment above this
    // shader: `wid` is uniform across the whole workgroup, so this is not a
    // divergent early exit relative to the workgroupBarrier() calls below.
    let tiles_n = (N + MACRO_N - 1u) / MACRO_N;
    let tiles_m = (M + MACRO_M - 1u) / MACRO_M;
    if (wid.x >= tiles_n || wid.y >= tiles_m) {
        return;
    }

    let tile_row = wid.y * MACRO_M;
    let tile_col = wid.x * MACRO_N;
    let tx = lid.x;
    let ty = lid.y;

    // See the "[register tile: 16 named scalars]" doc comment above: this
    // block intentionally does not use `array<f32, 4>` / `array<array<f32,
    // 4>, 4>` in place of these 16 declarations.
    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc02: f32 = 0.0;
    var acc03: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;
    var acc12: f32 = 0.0;
    var acc13: f32 = 0.0;
    var acc20: f32 = 0.0;
    var acc21: f32 = 0.0;
    var acc22: f32 = 0.0;
    var acc23: f32 = 0.0;
    var acc30: f32 = 0.0;
    var acc31: f32 = 0.0;
    var acc32: f32 = 0.0;
    var acc33: f32 = 0.0;

    let num_k_tiles = (K + KTILE - 1u) / KTILE;
    for (var t: u32 = 0u; t < num_k_tiles; t = t + 1u) {
        let k0 = t * KTILE;

        // Cooperatively load A's [MACRO_M x KTILE] sub-block (k-major),
        // zero-padding any (row, k) outside [M, K) so the accumulate loop
        // below needs no bounds check of its own. Division-free 2-D
        // grid-stride loop: ty walks k in steps of 16, tx walks m in steps
        // of 16 (16x16 threads cover the 16x64 sub-block in 4 inner steps).
        for (var kr: u32 = ty; kr < KTILE; kr = kr + 16u) {
            let g_col = k0 + kr;
            for (var mc: u32 = tx; mc < MACRO_M; mc = mc + 16u) {
                let g_row = tile_row + mc;
                if (g_row < M && g_col < K) {
                    tile_a[kr][mc] = a[g_row * K + g_col];
                } else {
                    tile_a[kr][mc] = 0.0;
                }
            }
        }

        // Cooperatively load B's [KTILE x MACRO_N] sub-block (k-major, same
        // orientation as B's own row-major global layout), same zero-pad.
        for (var kr: u32 = ty; kr < KTILE; kr = kr + 16u) {
            let g_row = k0 + kr;
            for (var nc: u32 = tx; nc < MACRO_N; nc = nc + 16u) {
                let g_col = tile_col + nc;
                if (g_row < K && g_col < N) {
                    tile_b[kr][nc] = b[g_row * N + g_col];
                } else {
                    tile_b[kr][nc] = 0.0;
                }
            }
        }

        workgroupBarrier();

        for (var kk: u32 = 0u; kk < KTILE; kk = kk + 1u) {
            let a_reg0 = tile_a[kk][ty * 4u + 0u];
            let a_reg1 = tile_a[kk][ty * 4u + 1u];
            let a_reg2 = tile_a[kk][ty * 4u + 2u];
            let a_reg3 = tile_a[kk][ty * 4u + 3u];
            let b_reg0 = tile_b[kk][tx * 4u + 0u];
            let b_reg1 = tile_b[kk][tx * 4u + 1u];
            let b_reg2 = tile_b[kk][tx * 4u + 2u];
            let b_reg3 = tile_b[kk][tx * 4u + 3u];
            acc00 = acc00 + a_reg0 * b_reg0;
            acc01 = acc01 + a_reg0 * b_reg1;
            acc02 = acc02 + a_reg0 * b_reg2;
            acc03 = acc03 + a_reg0 * b_reg3;
            acc10 = acc10 + a_reg1 * b_reg0;
            acc11 = acc11 + a_reg1 * b_reg1;
            acc12 = acc12 + a_reg1 * b_reg2;
            acc13 = acc13 + a_reg1 * b_reg3;
            acc20 = acc20 + a_reg2 * b_reg0;
            acc21 = acc21 + a_reg2 * b_reg1;
            acc22 = acc22 + a_reg2 * b_reg2;
            acc23 = acc23 + a_reg2 * b_reg3;
            acc30 = acc30 + a_reg3 * b_reg0;
            acc31 = acc31 + a_reg3 * b_reg1;
            acc32 = acc32 + a_reg3 * b_reg2;
            acc33 = acc33 + a_reg3 * b_reg3;
        }

        workgroupBarrier();
    }

    let r0 = tile_row + ty * 4u + 0u;
    let r1 = tile_row + ty * 4u + 1u;
    let r2 = tile_row + ty * 4u + 2u;
    let r3 = tile_row + ty * 4u + 3u;
    let col0 = tile_col + tx * 4u + 0u;
    let col1 = tile_col + tx * 4u + 1u;
    let col2 = tile_col + tx * 4u + 2u;
    let col3 = tile_col + tx * 4u + 3u;

    // Ragged M/N edges clip individual rows/cols of this thread's 4x4 tile;
    // each of the 16 writes is bounds-checked independently rather than
    // split into a fast/slow path, which measured indistinguishably from a
    // fast-path version at every shape tested (the K-loop above dominates
    // total cost by orders of magnitude).
    if (r0 < M) {
        if (col0 < N) { c[r0 * N + col0] = acc00; }
        if (col1 < N) { c[r0 * N + col1] = acc01; }
        if (col2 < N) { c[r0 * N + col2] = acc02; }
        if (col3 < N) { c[r0 * N + col3] = acc03; }
    }
    if (r1 < M) {
        if (col0 < N) { c[r1 * N + col0] = acc10; }
        if (col1 < N) { c[r1 * N + col1] = acc11; }
        if (col2 < N) { c[r1 * N + col2] = acc12; }
        if (col3 < N) { c[r1 * N + col3] = acc13; }
    }
    if (r2 < M) {
        if (col0 < N) { c[r2 * N + col0] = acc20; }
        if (col1 < N) { c[r2 * N + col1] = acc21; }
        if (col2 < N) { c[r2 * N + col2] = acc22; }
        if (col3 < N) { c[r2 * N + col3] = acc23; }
    }
    if (r3 < M) {
        if (col0 < N) { c[r3 * N + col0] = acc30; }
        if (col1 < N) { c[r3 * N + col1] = acc31; }
        if (col2 < N) { c[r3 * N + col2] = acc32; }
        if (col3 < N) { c[r3 * N + col3] = acc33; }
    }
}
"#;
/// LayerNorm WGSL shader — parallel reduction in shared memory.
///
/// Each workgroup (256 threads) processes one normalization instance.
/// Three phases: compute mean, compute variance, normalize + scale + bias.
/// Layout: input(ro), scale(ro), bias(ro), output(rw), params(uniform).
pub(super) const LAYER_NORM_SHADER: &str = r#"
struct Params {
    n_elements: u32,
    batch_count: u32,
    eps: f32,
    wg_per_row: u32,
}

const WG_SIZE: u32 = 256u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> ln_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn layer_norm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    // One workgroup per normalization instance. More instances than the device
    // allows along a single dimension are dispatched as a 2-D grid, so rebuild
    // the instance index from both components (`wg_per_row` is the X extent).
    let instance = wid.y * params.wg_per_row + wid.x;
    if (instance >= params.batch_count) { return; }
    let tid = lid.x;
    let n = params.n_elements;
    let base = instance * n;

    // Phase 1: parallel sum for mean
    var local_sum: f32 = 0.0;
    for (var step: u32 = 0u; step < n; step = step + WG_SIZE) {
        let i = tid + step;
        if (i < n) {
            local_sum = local_sum + input[base + i];
        }
    }
    shared_data[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction for sum
    for (var s: u32 = WG_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + s];
        }
        workgroupBarrier();
    }

    let mean_val = shared_data[0] / f32(n);
    workgroupBarrier();

    // Phase 2: parallel sum for variance
    var local_var: f32 = 0.0;
    for (var step: u32 = 0u; step < n; step = step + WG_SIZE) {
        let i = tid + step;
        if (i < n) {
            let diff = input[base + i] - mean_val;
            local_var = local_var + diff * diff;
        }
    }
    shared_data[tid] = local_var;
    workgroupBarrier();

    for (var s: u32 = WG_SIZE / 2u; s > 0u; s = s / 2u) {
        if (tid < s) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + s];
        }
        workgroupBarrier();
    }

    let variance = shared_data[0] / f32(n);
    let inv_std = 1.0 / sqrt(variance + params.eps);
    workgroupBarrier();

    // Phase 3: normalize + scale + bias
    for (var step: u32 = 0u; step < n; step = step + WG_SIZE) {
        let i = tid + step;
        if (i < n) {
            let norm_val = (input[base + i] - mean_val) * inv_std;
            output[base + i] = norm_val * scale[i] + ln_bias[i];
        }
    }
}
"#;
/// BatchNorm WGSL shader — inference mode per-channel normalization.
///
/// out = scale * (x - mean) / sqrt(variance + eps) + bias
/// Input is [N,C,H,W] flattened; per-channel mean/var/scale/bias.
/// Each thread handles one element.
pub(super) const BATCH_NORM_SHADER: &str = r#"
struct Params {
    total_elements: u32,
    channels: u32,
    spatial_size: u32,
    eps: f32,
    row_threads: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> bn_scale: array<f32>;
@group(0) @binding(2) var<storage, read> bn_bias: array<f32>;
@group(0) @binding(3) var<storage, read> bn_mean: array<f32>;
@group(0) @binding(4) var<storage, read> bn_var: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(256)
fn batch_norm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.y * params.row_threads + gid.x;
    if (idx >= params.total_elements) { return; }
    if (params.spatial_size == 0u || params.channels == 0u) { return; }

    // Determine channel: layout [N, C, spatial_size]
    let channel = (idx / params.spatial_size) % params.channels;

    let x = input[idx];
    let m = bn_mean[channel];
    let v = bn_var[channel];
    let s = bn_scale[channel];
    let b = bn_bias[channel];

    output[idx] = s * (x - m) / sqrt(v + params.eps) + b;
}
"#;
/// Transpose WGSL shader — general permutation via precomputed strides.
///
/// perm_data buffer layout: [input_strides..., output_strides..., perm...] (3*ndim u32 values).
/// Each thread computes one output element by converting its flat index to
/// multi-dimensional coordinates, permuting dimensions, and computing the source index.
pub(super) const TRANSPOSE_SHADER: &str = r#"
struct Params {
    total_elements: u32,
    ndim: u32,
    row_threads: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> perm_data: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn transpose_op(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_idx = gid.y * params.row_threads + gid.x;
    if (out_idx >= params.total_elements) { return; }

    let ndim = params.ndim;

    // perm_data layout: input_strides[0..ndim], output_strides[ndim..2*ndim], perm[2*ndim..3*ndim]
    // Convert flat output index → output coords → input coords → flat input index
    var remaining = out_idx;
    var in_flat: u32 = 0u;

    for (var d: u32 = 0u; d < ndim; d = d + 1u) {
        let out_stride = perm_data[ndim + d];
        // The host validates that every dimension is non-zero, so strides are
        // always >= 1; guard anyway so a corrupt buffer cannot divide by zero.
        if (out_stride == 0u) { return; }
        let coord = remaining / out_stride;
        remaining = remaining % out_stride;

        // This output dimension d corresponds to input dimension perm[d]
        let in_dim = perm_data[2u * ndim + d];
        let in_stride = perm_data[in_dim];
        in_flat = in_flat + coord * in_stride;
    }

    output[out_idx] = input[in_flat];
}
"#;
