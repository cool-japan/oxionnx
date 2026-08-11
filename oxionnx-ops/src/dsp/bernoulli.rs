//! Bernoulli operator: elementwise Bernoulli sampling with a fast PRNG.

use oxionnx_core::{OnnxError, OpContext, Operator, Tensor};

use crate::time_compat::{SystemTime, UNIX_EPOCH};

/// A simple xorshift64* PRNG — no external crates.
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // xorshift must not start with 0.
        let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self(s)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        // Use upper 24 bits for mantissa (avoids lower-bit patterns in xorshift).
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

pub struct BernoulliOp;
impl Operator for BernoulliOp {
    fn op_type(&self) -> &str {
        "Bernoulli"
    }

    fn execute(&self, ctx: &OpContext<'_>) -> Result<Vec<Tensor>, OnnxError> {
        let probs = ctx.input(0)?;

        // Resolve seed: if attr "seed" is present and non-zero, use it.
        // Otherwise seed from system time.
        let seed_attr = ctx.attrs().f("seed", 0.0);
        let rng_seed: u64 = if seed_attr != 0.0 {
            // Reinterpret the f32 bits as a u64 seed.
            (seed_attr.to_bits() as u64).wrapping_mul(0x9E3779B97F4A7C15)
        } else {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9E3779B97F4A7C15)
        };

        let mut rng = Xorshift64::new(rng_seed);
        let data: Vec<f32> = probs
            .data
            .iter()
            .map(|&p| if rng.next_f32() < p { 1.0 } else { 0.0 })
            .collect();

        Ok(vec![Tensor::new(data, probs.shape.clone())])
    }

    fn supports_output_slots(&self) -> bool {
        true
    }
}
