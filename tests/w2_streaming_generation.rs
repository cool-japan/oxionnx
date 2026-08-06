//! Streaming token generation ([`oxionnx::TokenStream`]) against a synthetic,
//! GPT-2-shaped autoregressive model with a real key/value cache.
//!
//! # The model
//!
//! One decoder block, shaped exactly like a GPT-2 export's per-layer structure,
//! shrunk to numbers a reference implementation can reproduce by hand:
//!
//! ```text
//! input_ids [1,S] ──Gather(wte)──► hidden [1,S,3]
//!                                    ├── MatMul(wk) ─┐
//!                                    │               Concat(past_key,  ·, axis=1) ──► present_key  [1,P+S,3]
//!                                    ├── MatMul(wv) ─┐
//!                                    │               Concat(past_value,·, axis=1) ──► present_value[1,P+S,3]
//!                                    │
//!                                    └─ MatMul(Transpose(present_key)) ─► Softmax ─► MatMul(present_value)
//!                                                                                        └─ MatMul(wo) ─► logits [1,S,5]
//! ```
//!
//! Vocabulary 5, embedding width 3, one attention head, no positional encoding
//! and no causal mask — the last position attends over the whole cache, which is
//! precisely the regime an incremental decoder runs in, and is what makes the
//! cached and uncached paths comparable below.
//!
//! # The reference
//!
//! Every expected number here was produced by NumPy (float32 throughout), not by
//! this engine. The generator script is reproduced in
//! `numpy_reference_program()` so the constants can be re-derived.

use oxionnx::{
    Attributes, CancellationToken, GenerationConfig, Graph, KvCacheBinding, Node, OnnxError,
    OpKind, Session, Tensor,
};
use std::collections::HashMap;

const VOCAB: usize = 5;
const EMBED: usize = 3;

/// The exact NumPy program the constants in this file come from.
///
/// Kept as text (rather than a comment) so that a future change to the model
/// shape has an executable reference to regenerate against:
///
/// ```text
/// import numpy as np
/// wte = np.array([[ .1,-.2,.3],[.4,.5,-.6],[-.7,.8,.9],[1.,-1.1,.2],[.3,.6,-.9]], np.float32)
/// wk  = np.array([[ .2,.1,-.3],[-.4,.5,.6],[.7,-.8,.9]], np.float32)
/// wv  = np.array([[ .5,-.1,.2],[.3,.4,-.5],[-.6,.7,.8]], np.float32)
/// wo  = np.array([[ .1,.2,-.3,.4,-.5],[.6,-.7,.8,-.9,1.],[-.2,.3,.5,-.4,.1]], np.float32)
/// def softmax(x): m = x.max(-1, keepdims=True); e = np.exp(x-m); return e/e.sum(-1, keepdims=True)
/// def forward(ids, pk, pv):
///     h = wte[ids]
///     k = np.concatenate([pk, h @ wk], 1); v = np.concatenate([pv, h @ wv], 1)
///     p = softmax(h @ np.transpose(k, (0,2,1)))
///     return (p @ v) @ wo, k, v
/// pk = pv = np.zeros((1,0,3), np.float32); ids = np.array([[2,1,1]], np.int64)
/// for _ in range(4):
///     lg, pk, pv = forward(ids, pk, pv)
///     t = int(np.argmax(lg[0,-1])); ids = np.array([[t]], np.int64); print(t, lg[0,-1])
/// ```
fn numpy_reference_program() {}

/// The prompt every reference below was generated from.
const PROMPT: [i64; 3] = [2, 1, 1];

/// Greedy continuation NumPy produces for [`PROMPT`] over four decode steps.
const EXPECTED_TOKENS: [i64; 4] = [3, 2, 4, 3];

/// Last-position logits row NumPy produces at each of those steps.
const EXPECTED_LOGITS: [[f32; VOCAB]; 4] = [
    [0.0956065, 0.0291988, -0.5454203, 0.5577642, -0.4581352],
    [-0.0374166, 0.0265238, 0.4254616, -0.4033565, 0.2758979],
    [0.2928391, -0.3690054, 0.288556, -0.354281, 0.4524892],
    [-0.0341204, 0.1881729, -0.5614933, 0.6064805, -0.5785047],
];

/// What the same loop produces if the `present.* → past.*` feedback is dropped
/// (i.e. every step starts from an empty cache).
///
/// This is the control: it must **differ** from [`EXPECTED_TOKENS`], otherwise
/// the test could pass with the cache entirely unwired.
const TOKENS_IF_CACHE_NEVER_FED_BACK: [i64; 4] = [3, 1, 3, 1];

/// float32 arithmetic reordered by a different kernel is worth ~1e-6 here; the
/// tensors are tiny and the reference is also float32.
const TOL: f32 = 2e-6;

fn weight(rows: &[[f32; EMBED]]) -> Tensor {
    let data: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::new(data, vec![rows.len(), EMBED])
}

fn node(op: OpKind, name: &str, inputs: &[&str], outputs: &[&str], attrs: Attributes) -> Node {
    Node {
        op,
        name: name.to_string(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs,
    }
}

fn axis_attr(axis: i64) -> Attributes {
    let mut attrs = Attributes::default();
    attrs.ints.insert("axis".to_string(), axis);
    attrs
}

fn perm_attr(perm: Vec<i64>) -> Attributes {
    let mut attrs = Attributes::default();
    attrs.int_lists.insert("perm".to_string(), perm);
    attrs
}

/// Weights shared by the cached and uncached variants of the model.
fn weights() -> HashMap<String, Tensor> {
    let mut weights = HashMap::new();
    weights.insert(
        "wte".to_string(),
        weight(&[
            [0.10, -0.20, 0.30],
            [0.40, 0.50, -0.60],
            [-0.70, 0.80, 0.90],
            [1.00, -1.10, 0.20],
            [0.30, 0.60, -0.90],
        ]),
    );
    weights.insert(
        "wk".to_string(),
        weight(&[
            [0.20, 0.10, -0.30],
            [-0.40, 0.50, 0.60],
            [0.70, -0.80, 0.90],
        ]),
    );
    weights.insert(
        "wv".to_string(),
        weight(&[
            [0.50, -0.10, 0.20],
            [0.30, 0.40, -0.50],
            [-0.60, 0.70, 0.80],
        ]),
    );
    weights.insert(
        "wo".to_string(),
        Tensor::new(
            vec![
                0.10, 0.20, -0.30, 0.40, -0.50, //
                0.60, -0.70, 0.80, -0.90, 1.00, //
                -0.20, 0.30, 0.50, -0.40, 0.10,
            ],
            vec![EMBED, VOCAB],
        ),
    );
    weights
}

/// The decoder block, with its key/value cache exposed as graph I/O.
fn cached_graph() -> Graph {
    Graph {
        name: "tiny_gpt".to_string(),
        nodes: vec![
            node(
                OpKind::Gather,
                "embed",
                &["wte", "input_ids"],
                &["hidden"],
                axis_attr(0),
            ),
            node(
                OpKind::MatMul,
                "proj_k",
                &["hidden", "wk"],
                &["new_key"],
                Attributes::default(),
            ),
            node(
                OpKind::MatMul,
                "proj_v",
                &["hidden", "wv"],
                &["new_value"],
                Attributes::default(),
            ),
            node(
                OpKind::Concat,
                "cat_k",
                &["past_key", "new_key"],
                &["present_key"],
                axis_attr(1),
            ),
            node(
                OpKind::Concat,
                "cat_v",
                &["past_value", "new_value"],
                &["present_value"],
                axis_attr(1),
            ),
            node(
                OpKind::Transpose,
                "key_t",
                &["present_key"],
                &["key_t"],
                perm_attr(vec![0, 2, 1]),
            ),
            node(
                OpKind::MatMul,
                "scores",
                &["hidden", "key_t"],
                &["scores"],
                Attributes::default(),
            ),
            node(
                OpKind::Softmax,
                "attn",
                &["scores"],
                &["probs"],
                axis_attr(-1),
            ),
            node(
                OpKind::MatMul,
                "context",
                &["probs", "present_value"],
                &["context"],
                Attributes::default(),
            ),
            node(
                OpKind::MatMul,
                "lm_head",
                &["context", "wo"],
                &["logits"],
                Attributes::default(),
            ),
        ],
        input_names: vec![
            "input_ids".to_string(),
            "past_key".to_string(),
            "past_value".to_string(),
        ],
        output_names: vec![
            "logits".to_string(),
            "present_key".to_string(),
            "present_value".to_string(),
        ],
        input_infos: Vec::new(),
        output_infos: Vec::new(),
    }
}

/// The same block with the cache inputs replaced by zero-length constants, i.e.
/// a model that has to be re-fed the entire sequence on every step.
fn uncached_graph() -> Graph {
    let mut graph = cached_graph();
    graph
        .nodes
        .retain(|n| n.name != "cat_k" && n.name != "cat_v");
    for node in &mut graph.nodes {
        for input in &mut node.inputs {
            if input == "present_key" {
                *input = "new_key".to_string();
            } else if input == "present_value" {
                *input = "new_value".to_string();
            }
        }
    }
    graph.input_names = vec!["input_ids".to_string()];
    graph.output_names = vec!["logits".to_string()];
    graph
}

fn cached_session() -> Session {
    Session::from_graph(cached_graph(), weights()).expect("cached model builds")
}

fn kv_bindings() -> Vec<KvCacheBinding> {
    vec![
        KvCacheBinding::empty("past_key", "present_key", vec![1, 0, EMBED]),
        KvCacheBinding::empty("past_value", "present_value", vec![1, 0, EMBED]),
    ]
}

fn base_config() -> GenerationConfig {
    GenerationConfig::default()
        .with_kv_cache(kv_bindings())
        .with_max_new_tokens(EXPECTED_TOKENS.len())
}

fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= TOL,
            "{what}[{i}]: engine {a} vs NumPy {e} (tolerance {TOL})"
        );
    }
}

#[test]
fn the_generated_tokens_and_logits_match_the_numpy_reference() {
    numpy_reference_program();
    let session = cached_session();
    let config = base_config().with_emit_logits(true);

    let steps: Vec<_> = session
        .generate(&PROMPT, config)
        .expect("stream starts")
        .collect::<Result<Vec<_>, OnnxError>>()
        .expect("every step succeeds");

    assert_eq!(steps.len(), EXPECTED_TOKENS.len());
    for (i, step) in steps.iter().enumerate() {
        assert_eq!(step.index, i, "steps are numbered in order");
        assert_eq!(
            step.token, EXPECTED_TOKENS[i],
            "token {i}: engine {} vs NumPy {}",
            step.token, EXPECTED_TOKENS[i]
        );
        let logits = step.logits.as_ref().expect("emit_logits was requested");
        assert_close(logits, &EXPECTED_LOGITS[i], &format!("step {i} logits"));
    }
}

/// The load-bearing property of a KV cache: feeding the cache forward and
/// re-feeding the whole sequence must agree token for token.
///
/// Two *different graphs* are run here — one with `Concat(past, new)` cache
/// edges, one with the cache removed entirely — so this cannot be satisfied by
/// a stream that quietly ignores the cache.
#[test]
fn the_cached_and_full_resequence_paths_agree_token_for_token() {
    let cached = cached_session();
    let cached_tokens = cached
        .generate_tokens(&PROMPT, base_config())
        .expect("cached generation");

    let uncached = Session::from_graph(uncached_graph(), weights()).expect("uncached model builds");
    let uncached_tokens = uncached
        .generate_tokens(
            &PROMPT,
            GenerationConfig::default().with_max_new_tokens(EXPECTED_TOKENS.len()),
        )
        .expect("uncached generation");

    assert_eq!(cached_tokens, EXPECTED_TOKENS.to_vec());
    assert_eq!(uncached_tokens, cached_tokens);
}

/// Guards the guard: the reference sequence must be *distinguishable* from what
/// a broken (never-fed-back) cache would produce, or the test above proves
/// nothing.
#[test]
fn a_never_fed_back_cache_would_produce_a_different_sequence() {
    assert_ne!(
        EXPECTED_TOKENS.to_vec(),
        TOKENS_IF_CACHE_NEVER_FED_BACK.to_vec(),
        "the reference and the broken-cache control must differ, or the cache is untested"
    );

    // And the control is reproducible: a stream whose bindings are re-seeded
    // empty before every step is exactly the broken case.
    let session = cached_session();
    let mut ids = PROMPT.to_vec();
    let mut broken = Vec::new();
    for _ in 0..EXPECTED_TOKENS.len() {
        let tokens = session
            .generate_tokens(&ids, base_config().with_max_new_tokens(1))
            .expect("single step");
        let token = tokens[0];
        broken.push(token);
        ids = vec![token];
    }
    assert_eq!(broken, TOKENS_IF_CACHE_NEVER_FED_BACK.to_vec());
}

#[test]
fn the_stream_stops_at_the_eos_token_and_yields_it() {
    let session = cached_session();
    // The reference's second token is 2; stopping on it must yield exactly two.
    let config = base_config().with_eos_token_id(EXPECTED_TOKENS[1]);
    let tokens = session
        .generate_tokens(&PROMPT, config)
        .expect("generation");
    assert_eq!(tokens, vec![EXPECTED_TOKENS[0], EXPECTED_TOKENS[1]]);
}

#[test]
fn the_token_cap_is_honoured_exactly() {
    let session = cached_session();
    for cap in 0..=EXPECTED_TOKENS.len() {
        let tokens = session
            .generate_tokens(&PROMPT, base_config().with_max_new_tokens(cap))
            .expect("generation");
        assert_eq!(tokens.len(), cap, "max_new_tokens = {cap}");
        assert_eq!(tokens, EXPECTED_TOKENS[..cap].to_vec());
    }
}

#[test]
fn generation_is_lazy_and_reports_its_progress() {
    let session = cached_session();
    let mut stream = session
        .generate(&PROMPT, base_config())
        .expect("stream starts");
    assert_eq!(
        stream.generated(),
        &[] as &[i64],
        "nothing runs until polled"
    );
    assert_eq!(stream.prompt(), &PROMPT);

    let first = stream.next().expect("a first token").expect("step ok");
    assert_eq!(first.token, EXPECTED_TOKENS[0]);
    assert_eq!(stream.generated(), &EXPECTED_TOKENS[..1]);
    assert!(!stream.is_finished());
    assert!(first.logits.is_none(), "logits are opt-in");
}

#[test]
fn a_cancelled_generation_stops_between_steps_with_a_typed_error() {
    let session = cached_session();
    let token = CancellationToken::new();
    let mut stream = session
        .generate(&PROMPT, base_config().with_cancellation(token.clone()))
        .expect("stream starts");

    let first = stream.next().expect("a first token").expect("step ok");
    assert_eq!(first.token, EXPECTED_TOKENS[0]);

    token.cancel();
    match stream.next() {
        Some(Err(OnnxError::Cancelled(msg))) => {
            assert!(msg.contains("1 token"), "unexpected message: {msg}");
        }
        other => panic!("expected a Cancelled error, got {other:?}"),
    }
    assert!(stream.is_finished());
    assert!(stream.next().is_none(), "a cancelled stream stays ended");
}

/// Two generations on one session, cancelled independently — the property a
/// session-scoped token cannot provide.
#[test]
fn two_streams_on_one_session_are_cancelled_independently() {
    let session = cached_session();
    let doomed_token = CancellationToken::new();

    let mut doomed = session
        .generate(
            &PROMPT,
            base_config().with_cancellation(doomed_token.clone()),
        )
        .expect("stream starts");
    let mut survivor = session
        .generate(
            &PROMPT,
            base_config().with_cancellation(CancellationToken::new()),
        )
        .expect("stream starts");

    assert!(doomed.next().expect("token").is_ok());
    assert!(survivor.next().expect("token").is_ok());

    doomed_token.cancel();
    assert!(matches!(doomed.next(), Some(Err(OnnxError::Cancelled(_)))));

    // The survivor must complete its full budget, unaffected.
    let mut survivor_tokens = vec![EXPECTED_TOKENS[0]];
    for step in survivor {
        survivor_tokens.push(step.expect("step ok").token);
    }
    assert_eq!(survivor_tokens, EXPECTED_TOKENS.to_vec());
}

#[test]
fn a_misconfigured_stream_fails_up_front_rather_than_on_the_first_step() {
    let session = cached_session();

    let missing_input = session.generate(
        &PROMPT,
        base_config().with_input_ids("tokens_that_do_not_exist"),
    );
    assert!(matches!(missing_input, Err(OnnxError::InvalidModel(_))));

    let missing_output = session.generate(&PROMPT, base_config().with_logits("scores_maybe"));
    assert!(matches!(missing_output, Err(OnnxError::InvalidModel(_))));

    let missing_cache = session.generate(
        &PROMPT,
        base_config().with_kv_cache(vec![KvCacheBinding::empty(
            "past_nothing",
            "present_key",
            vec![1, 0, EMBED],
        )]),
    );
    assert!(matches!(missing_cache, Err(OnnxError::InvalidModel(_))));

    let empty_prompt = session.generate(&[], base_config());
    assert!(matches!(empty_prompt, Err(OnnxError::InvalidModel(_))));
}
