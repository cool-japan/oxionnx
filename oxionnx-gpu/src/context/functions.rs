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
/// Tiled matrix multiply WGSL shader using workgroup shared memory.
///
/// Uses 16x16 tiles loaded into shared memory for improved cache locality.
/// Each workgroup computes a 16x16 tile of the output matrix C by iterating
/// over tiles along the K dimension, loading A and B tiles into shared memory,
/// and accumulating partial dot products.
pub(super) const TILED_MATMUL_SHADER: &str = r#"
struct Dims {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}

const TILE_SIZE: u32 = 16u;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn tiled_matmul(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let local_row = lid.y;
    let local_col = lid.x;
    let M = dims.M;
    let K = dims.K;
    let N = dims.N;

    var sum: f32 = 0.0;
    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of A into shared memory
        let a_col = t * TILE_SIZE + local_col;
        if (row < M && a_col < K) {
            tile_a[local_row][local_col] = a[row * K + a_col];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }

        // Load tile of B into shared memory
        let b_row = t * TILE_SIZE + local_row;
        if (b_row < K && col < N) {
            tile_b[local_row][local_col] = b[b_row * N + col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product for this tile
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            sum = sum + tile_a[local_row][i] * tile_b[i][local_col];
        }

        workgroupBarrier();
    }

    if (row < M && col < N) {
        c[row * N + col] = sum;
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
