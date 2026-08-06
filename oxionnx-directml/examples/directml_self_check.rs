//! The hardware acceptance gate for `oxionnx-directml`.
//!
//! # Why this program exists
//!
//! This crate's GPU path **has never been executed**.  The repository it lives in has no
//! Windows host and no D3D12 adapter; every line under `src/backend/d3d12/` and
//! `src/backend/dml/` is type-checked and lint-checked by a cross-target `cargo clippy`,
//! and by nothing else.  What that cross-check *cannot* see:
//!
//! * the HLSL source text — `rustc` sees a `&'static str`; `D3DCompile` is the first thing
//!   that ever parses it, at run time, on your machine;
//! * whether the root signature's registers agree with the shader's;
//! * whether the resource barriers are complete (a missing UAV barrier is *correct* on
//!   some IHVs and garbage on others — this is the single most likely bug in the crate);
//! * whether the fence and event actually serialise the readback;
//! * whether DirectML's validator accepts our tensor descriptors;
//! * whether the numbers that come back are **right**.
//!
//! A GPU kernel bug does not crash.  It returns a buffer of exactly the right length and
//! shape, full of plausible-looking wrong numbers, which then propagate silently through
//! the rest of an inference graph.  This program is the only thing that can catch that: it
//! runs every op the provider claims on fixed, deterministic inputs and diffs each result,
//! element by element, against the CPU oracle in `oxionnx_directml::reference`.
//!
//! **If you have Windows and a D3D12 GPU, running this and pasting the output is the single
//! most valuable thing you can do for this crate.**
//!
//! # Running it
//!
//! ```text
//! # On a machine with a real GPU:
//! set OXIONNX_DIRECTML=1
//! cargo run -p oxionnx-directml --example directml_self_check
//!
//! # On a Windows VM with no GPU — the only environment where this code can be exercised
//! # at all without hardware.  WARP is Microsoft's *conformant* software D3D12
//! # implementation, so it will catch a wrong index, a wrong root-signature slot or a bad
//! # tensor descriptor exactly as well as silicon will.  It will simply do it slowly, and
//! # it cannot catch the class of bug that is correct on one vendor's part and wrong on
//! # another's.
//! set OXIONNX_DIRECTML=1
//! set OXIONNX_DIRECTML_ALLOW_WARP=1
//! cargo run -p oxionnx-directml --example directml_self_check
//!
//! # Tighter than the default 1e-4 gate:
//! cargo run -p oxionnx-directml --example directml_self_check -- 1e-6
//! ```
//!
//! # Exit codes
//!
//! | Code | Meaning |
//! |---|---|
//! | 0 | every op matched the oracle |
//! | 1 | **at least one op disagreed with the oracle** — the GPU path is wrong; paste this output into an issue |
//! | 2 | no context: not opted in, not Windows, or no D3D12 adapter.  Nothing was tested. |
//! | 3 | the GPU path *failed* (a shader would not compile, a dispatch broke, an operator was declined).  Nothing was proved either way. |
//!
//! Note that 2 is **not** a pass.  A program that prints "no GPU found" and exits 0 is how
//! an untested backend gets mistaken for a working one.

use std::process::ExitCode;

use oxionnx_directml::{
    reference::ComparisonReport, Activation, BackendKind, DirectMLContext, SelfCheckReport,
};

/// The default blunt gate, applied *in addition to* the per-op numerical policy in
/// `reference::Tolerance` (which holds `Add`, `Sub`, `Mul` and `Relu` to bit-exactness
/// regardless of what is passed here — those kernels do no accumulation, so there is no
/// legitimate source of drift in them at all).
///
/// `1e-4` is loose on purpose for a first run: it is tight enough to catch every structural
/// bug (a transposed index, a half-empty dispatch grid, a wrong stride) by orders of
/// magnitude, and loose enough not to cry wolf over an `fma` contraction in the MatMul
/// accumulator.  Tighten it with the command-line argument once a machine has passed.
const DEFAULT_TOLERANCE: f32 = 1.0e-4;

fn main() -> ExitCode {
    let tolerance = match parse_tolerance() {
        Ok(t) => t,
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!("usage: directml_self_check [TOLERANCE]   (default {DEFAULT_TOLERANCE:e})");
            return ExitCode::from(3);
        }
    };

    // `Activation::Enabled` rather than `try_new()`: running this program *is* the opt-in.
    // Making the user also set `OXIONNX_DIRECTML=1` before the tool whose entire purpose is
    // to test the GPU would agree to touch a GPU is a papercut with no safety value —
    // nothing here writes to a model or returns numbers to an inference.
    let Some(context) = DirectMLContext::try_new_with(Activation::Enabled) else {
        print_no_context();
        return ExitCode::from(2);
    };

    println!("DirectML self-check");
    println!("  backend:   {}", context.backend_kind().as_str());
    println!("  adapter:   {}", context.adapter_name());
    println!("  tolerance: {tolerance:e}");
    println!("  note:      {}", backend_caveat(context.backend_kind()));
    println!();

    match context.self_check_reports(tolerance) {
        Ok((report, comparisons)) => {
            print_comparisons(&comparisons);
            println!();
            print!("{report}");
            print_verdict(&report);
            if report.passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(e) => {
            // Not a mismatch — the GPU path never got far enough to produce numbers to
            // compare.  This is a different fact from "the numbers are wrong", and it gets
            // a different exit code, because it proves nothing about correctness.
            eprintln!("SELF-CHECK COULD NOT RUN: {e}");
            eprintln!();
            eprintln!("This is a FAILURE of the GPU path, not a numerical mismatch: the");
            eprintln!("backend never produced a result to compare.  Likely causes, in order:");
            eprintln!();
            eprintln!("  * D3DCompile rejected one of the shaders in `oxionnx_directml::hlsl`");
            eprintln!("    (this is the most likely bug in the crate — the HLSL has never been");
            eprintln!("    parsed by a compiler);");
            eprintln!("  * the root signature and the shader registers disagree;");
            eprintln!("  * DirectML's validator rejected a tensor descriptor;");
            eprintln!("  * the device was removed mid-dispatch.");
            eprintln!();
            eprintln!("Please paste this message, the adapter name above, and your Windows");
            eprintln!("and driver versions into an issue.");
            ExitCode::from(3)
        }
    }
}

/// The single optional positional argument: the blunt tolerance.
fn parse_tolerance() -> Result<f32, String> {
    let mut args = std::env::args().skip(1);
    let Some(raw) = args.next() else {
        return Ok(DEFAULT_TOLERANCE);
    };
    if args.next().is_some() {
        return Err("too many arguments".to_string());
    }
    let tolerance: f32 = raw
        .parse()
        .map_err(|_| format!("`{raw}` is not a number"))?;
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(format!(
            "tolerance must be finite and non-negative, got `{raw}`"
        ));
    }
    Ok(tolerance)
}

/// Per-op detail.  This is where a failure is actually *diagnosed*.
fn print_comparisons(comparisons: &[ComparisonReport]) {
    println!("per-op comparison against the CPU oracle:");
    for comparison in comparisons {
        println!("  {comparison}");
    }

    // The first mismatched *index* is the most diagnostic single number available, and it
    // is worth spelling out rather than leaving buried in the line above: the elementwise
    // shaders compute `i = (gid.y * GroupsX + gid.x) * 256 + lid.x`, so a first mismatch at
    // 256 says "thread group 1 is wrong and group 0 is fine", and one at N/2 says "half the
    // dispatch grid never ran at all".  Those are different bugs with different fixes.
    for comparison in comparisons.iter().filter(|c| !c.passed) {
        if let Some(first) = comparison.first_mismatch {
            println!();
            println!(
                "  {} FIRST MISMATCH at element {} of {}: gpu={:e} oracle={:e}",
                comparison.op, first.index, comparison.elem_count, first.gpu, first.cpu
            );
            println!(
                "      {} of {} elements disagree.",
                comparison.mismatches, comparison.elem_count
            );
            println!("      {}", diagnose(first.index, comparison.elem_count));
        }
    }
}

/// Turn a first-mismatch index into the hypothesis it most strongly suggests.
///
/// Heuristics, and labelled as such — but a heuristic that points at the right thread group
/// beats a bare integer, and every one of these corresponds to a bug this crate could
/// plausibly have.
fn diagnose(first_index: usize, elem_count: usize) -> &'static str {
    const GROUP: usize = 256; // `ELEMENTWISE_THREADS_PER_GROUP`
    if first_index == 0 {
        "index 0: nothing is right.  Suspect a wrong buffer binding (t0/t1/u0), an \
         unwritten output, or a root-constant layout that does not match the cbuffer."
    } else if first_index % GROUP == 0 {
        "a multiple of the 256-wide thread group: suspect the 2-D group grid — i.e. \
         `GroupsX` in the root constants disagreeing with DispatchGrid::x."
    } else if first_index * 2 >= elem_count {
        "in the second half of the buffer: suspect a dispatch grid that is too small, or \
         a missing UAV barrier before the readback copy (correct on NVIDIA, garbage on AMD)."
    } else {
        "mid-buffer: suspect an indexing error inside the kernel (a transposed row/col, \
         or a stride computed from the wrong dimension)."
    }
}

/// What a pass on *this* backend does and does not prove.
fn backend_caveat(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::DirectMl => {
            "genuine DirectML operators.  A pass here says nothing about the HLSL fallback \
             — that is a separate code path with separate bugs, and it is what runs on a \
             box without DirectML.dll."
        }
        BackendKind::Hlsl => {
            "D3D12 compute shaders (DirectML.dll was not found).  A pass here says nothing \
             about the DirectML path, which is what most machines will actually use."
        }
        // Unreachable: `try_new_with` returns `None` rather than an inactive context.
        BackendKind::Unavailable => "no GPU backend.",
    }
}

fn print_verdict(report: &SelfCheckReport) {
    println!();
    if report.passed {
        println!("PASS — every op agreed with the CPU oracle.");
        println!();
        println!("Please paste this output into the OxiONNX issue tracker.  It is the only");
        println!("evidence that exists that this code works, and it is specific to your");
        println!("adapter: a pass on one vendor's part does not carry to another's.");
    } else {
        println!("FAIL — the GPU path returned WRONG NUMBERS.");
        println!();
        println!("This is the failure mode the whole design is built around: nothing crashed,");
        println!("every buffer was the right length and the right shape, and the values are");
        println!("simply incorrect.  Do NOT set OXIONNX_DIRECTML=1 on this machine for real");
        println!("work; inference would be silently wrong.");
        println!();
        println!("Please paste this entire output, plus the adapter, Windows version and");
        println!("driver version, into the OxiONNX issue tracker.");
    }
}

fn print_no_context() {
    eprintln!("No DirectML context could be acquired — NOTHING WAS TESTED.");
    eprintln!();
    eprintln!("This is not a pass.  In order, the possible reasons:");
    eprintln!();
    eprintln!("  1. This is not Windows.  The DirectML provider is Windows + D3D12 only,");
    eprintln!("     and on every other target it is a typed no-op by construction.");
    eprintln!("  2. No D3D12 adapter was found.  Hardware adapters are enumerated via DXGI;");
    eprintln!("     the software (WARP) adapter is SKIPPED unless you opt in:");
    eprintln!();
    eprintln!("         set OXIONNX_DIRECTML_ALLOW_WARP=1");
    eprintln!();
    eprintln!("     WARP is a CPU rasteriser, so it is skipped by default — a \"GPU\" backend");
    eprintln!("     silently running on it would be slower than the tuned CPU kernels it was");
    eprintln!("     meant to beat.  For *this* program it is exactly what you want, though:");
    eprintln!("     it is the only way to exercise the D3D12 path on a machine with no GPU.");
    eprintln!("  3. D3D12 came up but neither engine could be built.  A `tracing` subscriber");
    eprintln!("     would have shown a warning explaining which; run with one attached.");
}
