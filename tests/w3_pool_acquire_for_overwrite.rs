//! Wave-3: [`SizeClassPool::acquire_for_overwrite`] — the pool entry point that
//! skips the zero-fill.
//!
//! # What it is for
//!
//! `acquire` returns a buffer that is `size` elements long **and zeroed**.  For
//! an output slot that the kernel is about to overwrite completely — a
//! convolution, a matmul — that zero-fill is a full pass over the output tensor
//! whose every byte is immediately replaced.  `acquire_for_overwrite` skips it.
//!
//! # What must stay true
//!
//! Skipping the fill changes exactly one thing: the *contents*.  Every other
//! contract has to hold identically, because a caller that gets a short buffer,
//! or a pool whose accounting drifts, breaks in ways that have nothing to do
//! with the values.  So these tests pin the length, the reuse behaviour, the
//! statistics and the interaction with `release`, and pin the contents only as
//! far as they can be pinned: an element is always a valid `f32`, never
//! uninitialised memory (there is no `unsafe` on this path at all).

use oxionnx::{SizeClassPool, Tensor};
use std::collections::HashMap;

/// The length contract is the same as `acquire`'s, across every size class and
/// every reuse route (exact bucket, larger bucket, fresh allocation).
#[test]
fn the_returned_buffer_is_always_exactly_the_requested_length() {
    let mut pool = SizeClassPool::new();
    // tiny / small / medium / large boundaries, plus the degenerate zero.
    for size in [0usize, 1, 64, 127, 128, 1023, 1024, 16383, 16384, 40000] {
        let buf = pool.acquire_for_overwrite(size);
        assert_eq!(buf.len(), size, "fresh allocation of {size}");
        pool.release(buf);
        let reused = pool.acquire_for_overwrite(size);
        assert_eq!(reused.len(), size, "recycled buffer of {size}");
        pool.release(reused);
    }
}

/// A buffer recycled for a **smaller** request is truncated, not left long — the
/// caller indexes `0..size` and a longer buffer would silently carry a tail that
/// `Tensor::new`'s length check would then reject.
#[test]
fn a_larger_recycled_buffer_is_truncated_to_the_request() {
    let mut pool = SizeClassPool::new();
    let big = pool.acquire(20_000);
    pool.release(big);

    let small = pool.acquire_for_overwrite(500);
    assert_eq!(small.len(), 500);
    assert!(
        small.capacity() >= 20_000,
        "it really is the recycled buffer, not a fresh allocation: capacity {}",
        small.capacity(),
    );
}

/// Growing a short recycled buffer has nothing to grow it *with* except zeros,
/// so the tail is zeroed even here.  The elements that were already there are
/// the ones left untouched.
#[test]
fn a_shorter_recycled_buffer_is_grown_with_zeros() {
    let mut pool = SizeClassPool::new();
    let mut buf = pool.acquire(4);
    buf.copy_from_slice(&[7.0, 7.0, 7.0, 7.0]);
    pool.release(buf);

    // 4 -> 8 stays inside the `Tiny` class, so this is the same buffer.
    let grown = pool.acquire_for_overwrite(8);
    assert_eq!(grown.len(), 8);
    assert_eq!(
        &grown[4..],
        &[0.0, 0.0, 0.0, 0.0],
        "the appended tail is zeroed",
    );
}

/// Every element is a valid, readable `f32`.  This is the property that makes
/// the API safe rather than merely fast: no `unsafe`, no uninitialised memory,
/// no `MaybeUninit` — the contents are *unspecified*, which is a much weaker
/// claim than *undefined*.
#[test]
fn every_element_is_a_readable_finite_or_not_but_valid_f32() {
    let mut pool = SizeClassPool::new();
    let mut buf = pool.acquire(1024);
    for (i, v) in buf.iter_mut().enumerate() {
        *v = i as f32;
    }
    pool.release(buf);

    let reused = pool.acquire_for_overwrite(1024);
    // Reading every element must be sound; the values themselves are not part
    // of the contract, so the assertion is about readability, not content.
    let total: f32 = reused.iter().sum();
    assert!(
        total.is_finite() || total.is_nan(),
        "every element was readable"
    );
    assert_eq!(reused.len(), 1024);
}

/// Pool accounting must not diverge between the two entry points: a recycled
/// buffer counts as a reuse and a fresh one as an allocation, exactly as with
/// `acquire`.
#[test]
fn the_statistics_match_acquires_accounting() {
    let mut zeroing = SizeClassPool::new();
    let mut overwriting = SizeClassPool::new();

    for _ in 0..3 {
        let a = zeroing.acquire(2048);
        zeroing.release(a);
        let b = overwriting.acquire_for_overwrite(2048);
        overwriting.release(b);
    }

    assert_eq!(
        (
            overwriting.stats().alloc_count,
            overwriting.stats().reuse_count
        ),
        (zeroing.stats().alloc_count, zeroing.stats().reuse_count),
        "one allocation then two reuses, either way",
    );
    assert_eq!(overwriting.stats().alloc_count, 1);
    assert_eq!(overwriting.stats().reuse_count, 2);
}

/// A cleared pool has nothing to recycle, so the next request allocates — and an
/// allocation is zeroed, because `vec![0.0; n]` is how one is made.
#[test]
fn a_fresh_allocation_is_still_zeroed() {
    let mut pool = SizeClassPool::new();
    let mut buf = pool.acquire(64);
    buf.iter_mut().for_each(|v| *v = 5.0);
    pool.release(buf);
    pool.clear();

    let fresh = pool.acquire_for_overwrite(64);
    assert_eq!(
        fresh,
        vec![0.0_f32; 64],
        "nothing was left to recycle, so this is a fresh zeroed allocation",
    );
}

/// The zeroing entry point is unchanged — this is an *additional* API, not a
/// change to the existing one.  A caller that never opts in cannot be affected.
#[test]
fn acquire_still_zeroes() {
    let mut pool = SizeClassPool::new();
    let mut buf = pool.acquire(256);
    buf.iter_mut().for_each(|v| *v = 9.0);
    pool.release(buf);

    assert_eq!(
        pool.acquire(256),
        vec![0.0_f32; 256],
        "acquire's zeroing guarantee is load-bearing and must not have moved",
    );
}

/// The engine still uses the zeroing path for its output slots, so inference
/// results cannot have changed.  Two runs of one pooled session must agree
/// exactly — the case where a stale buffer would show up first.
#[test]
fn pooled_inference_is_unaffected() {
    use oxionnx::{Attributes, Graph, Node, OpKind, Session, Tensor as T};

    let graph = Graph {
        nodes: vec![
            Node {
                op: OpKind::Relu,
                name: "r".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["a".to_string()],
                attrs: Attributes::default(),
            },
            Node {
                op: OpKind::Mul,
                name: "m".to_string(),
                inputs: vec!["a".to_string(), "a".to_string()],
                outputs: vec!["y".to_string()],
                attrs: Attributes::default(),
            },
        ],
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
        ..Default::default()
    };
    let session = Session::builder()
        .with_memory_pool(true)
        .build_from_graph(graph, HashMap::new())
        .expect("build");

    let mut inputs = HashMap::new();
    inputs.insert("x", T::new(vec![-1.0, 2.0, -3.0, 4.0], vec![4]));
    let first = session
        .run(&inputs)
        .expect("run")
        .get("y")
        .expect("y")
        .clone();
    for _ in 0..5 {
        let again = session
            .run(&inputs)
            .expect("run")
            .get("y")
            .expect("y")
            .clone();
        assert_eq!(again.data, first.data);
        assert_eq!(again.shape, first.shape);
    }
    assert_eq!(first.data, vec![0.0, 4.0, 0.0, 16.0]);
    assert!(
        session.pool_stats().is_some(),
        "the pool really was enabled for this session",
    );
}

/// `Tensor` is the consumer this exists for: a slot is a `Tensor` built over a
/// pool buffer, so the buffer's length must satisfy `Tensor::new`'s invariant.
#[test]
fn a_buffer_from_it_satisfies_tensors_length_invariant() {
    let mut pool = SizeClassPool::new();
    let spare = pool.acquire(64);
    pool.release(spare);

    let data = pool.acquire_for_overwrite(12);
    let tensor = Tensor::new(data, vec![3, 4]);
    assert_eq!(tensor.data.len(), 12);
    assert_eq!(tensor.shape, vec![3, 4]);
}
