//! InSwapper forward-pass transfer counters for the data-movement CUDA op
//! wave (`MaxPool`/`AveragePool`/`Resize`/`Pad`/`Slice`/`Concat`, plus the
//! zero-cost `Reshape`/`Squeeze`/`Unsqueeze`/`Flatten` residency alias).
//!
//! Reports the **current** steady-state per-frame byte count via
//! `Session::cuda_cache_counters()` — the number the wave's audit staged a
//! projection for (465 MB/frame -> ~150 MB from this wave's ops alone, before
//! the pinned-staging floor's further reduction on top). There is no
//! synthetic "before" session to diff against in this same binary without
//! reverting the dispatch arms; read the printed total and compare against
//! that projection by hand.
//!
//! ```text
//! cargo run --release --features cuda --example data_ops_wave_transfer_counters
//! ```
//!
//! Point `OXIONNX_INSWAPPER_MODEL` at the real `.onnx` file if it is not at
//! `/tmp/oxiface-models/inswapper_128.onnx`.

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("built without --features cuda; nothing to measure");
}

#[cfg(feature = "cuda")]
fn main() {
    cuda_impl::run();
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::Instant;

    use oxionnx::execution_providers::OpPlacement;
    use oxionnx::{Session, Tensor};

    /// Matches `oxiface-swap`/`oxiface-detect`'s own production threshold
    /// (`session_loader.rs::GPU_THRESHOLD_BYTES`) exactly, so this example
    /// measures the placement policy real `oxiface --device cuda` runs use,
    /// not an arbitrary one.
    const GPU_THRESHOLD_BYTES: usize = 16_384;

    fn model_path() -> Option<PathBuf> {
        if let Ok(explicit) = std::env::var("OXIONNX_INSWAPPER_MODEL") {
            let path = PathBuf::from(explicit);
            return path.is_file().then_some(path);
        }
        let default = PathBuf::from("/tmp/oxiface-models/inswapper_128.onnx");
        default.is_file().then_some(default)
    }

    /// Deterministic pseudo-random tensor -- a fixed LCG, matching every
    /// other on-device fixture in this workspace (no `rand` dependency,
    /// bit-reproducible).
    fn lcg_tensor(shape: &[usize], seed: u64) -> Tensor {
        let n: usize = shape.iter().product();
        let mut state = seed;
        let data = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect();
        Tensor::new(data, shape.to_vec())
    }

    fn mib(bytes: u64) -> f64 {
        bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn run() {
        // SAFETY: nothing else has touched the environment or spawned a
        // thread yet (this is the first statement of a single-threaded
        // `main`) -- `set_var`'s safety contract is about concurrent access,
        // not this sequential, startup-time use, and every call site in this
        // workspace that sets this variable programmatically (`oxiface-cli`'s
        // `--device cuda` handling) does so under the identical precondition.
        unsafe {
            std::env::set_var("OXIONNX_CUDA", "1");
        }

        let Some(path) = model_path() else {
            println!(
                "inswapper_128.onnx not found (looked at $OXIONNX_INSWAPPER_MODEL and \
                 /tmp/oxiface-models/inswapper_128.onnx); nothing to measure"
            );
            return;
        };

        let session = match Session::builder()
            .with_op_placement(OpPlacement::Auto {
                gpu_threshold_bytes: GPU_THRESHOLD_BYTES,
            })
            .load(&path)
        {
            Ok(s) => s,
            Err(e) => {
                println!("could not load {}: {e}", path.display());
                return;
            }
        };
        if session.cuda_cache_counters().is_none() {
            println!(
                "no CUDA context attached to this session -- no CUDA driver/device, or the \
                 self-test at load time failed; nothing to measure"
            );
            return;
        }

        let target = lcg_tensor(&[1, 3, 128, 128], 0x00D4_740F_0AD5_1001);
        let source = lcg_tensor(&[1, 512], 0x00D4_740F_0AD5_1002);
        let mut inputs: HashMap<&str, Tensor> = HashMap::new();
        inputs.insert("target", target);
        inputs.insert("source", source);

        // Warm-up: JIT-compiles every PTX kernel this graph touches and
        // uploads every graph-initializer weight into the residency cache (a
        // one-time, ~503 MB-per-model cost the wave's audit already counted
        // separately from the per-frame number below).
        if let Err(e) = session.run(&inputs) {
            println!("warm-up run failed: {e}");
            return;
        }

        let before = session
            .cuda_cache_counters()
            .expect("a CUDA context was already confirmed attached above");

        let t0 = Instant::now();
        if let Err(e) = session.run(&inputs) {
            println!("steady-state run failed: {e}");
            return;
        }
        let wall = t0.elapsed();

        let after = session
            .cuda_cache_counters()
            .expect("a CUDA context was already confirmed attached above");
        let delta = after.since(before);

        println!("== InSwapper steady-state forward pass (frame 2, warm caches) ==");
        println!(
            "  wall clock                {:>10.2} ms",
            wall.as_secs_f64() * 1e3
        );
        println!(
            "  host -> device             {:>10.2} MiB  ({} bytes)",
            mib(delta.host_to_device_bytes),
            delta.host_to_device_bytes
        );
        println!(
            "  device -> host             {:>10.2} MiB  ({} bytes)",
            mib(delta.device_to_host_bytes),
            delta.device_to_host_bytes
        );
        println!(
            "  total PCIe traffic         {:>10.2} MiB",
            mib(delta.host_to_device_bytes + delta.device_to_host_bytes)
        );
        println!(
            "  ...of which staged         {:>10.2} MiB  (upload {:.2} MiB / download {:.2} MiB)",
            mib(delta.staged_upload_bytes + delta.staged_download_bytes),
            mib(delta.staged_upload_bytes),
            mib(delta.staged_download_bytes),
        );
        println!("  blocking stream syncs      {:>10}", delta.stream_syncs);
        println!(
            "  weight bytes uploaded      {:>10.2} MiB  (must be 0 in steady state)",
            mib(delta.weight_bytes_uploaded)
        );
        println!(
            "  pool allocs                {:>10}  (must be 0 in steady state)",
            delta.pool_allocs
        );
        println!(
            "  resident activation binds  {:>10}",
            delta.resident_activation_binds
        );
        println!("  device handoffs            {:>10}", delta.device_handoffs);
    }
}
