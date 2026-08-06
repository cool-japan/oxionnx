//! `RandomNormal`, `RandomUniform`, `RandomNormalLike`, `RandomUniformLike`
//! and `Multinomial` -- the ONNX random-generator operator family.
//!
//! All five share one seeded PRNG (this module's private `rng` submodule);
//! see that submodule's doc comment for the "distributional, not
//! bitwise-ORT-compatible" contract every op here inherits.

mod multinomial;
mod random_normal_uniform;
mod rng;

pub use multinomial::MultinomialOp;
pub use random_normal_uniform::{
    RandomNormalLikeOp, RandomNormalOp, RandomUniformLikeOp, RandomUniformOp,
};
