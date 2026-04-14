//! Convenience macros for building inference input maps.
//!
//! Provides an `inputs!` macro compatible with the `ort` 2.x API surface,
//! allowing callers to construct a `HashMap<&str, Tensor>` wrapped in a `Result`
//! using the same `"name" => tensor` syntax.

/// Build a `Result<HashMap<&str, Tensor>, OnnxError>` input map.
///
/// Compatible with the `ort` 2.x `inputs!` macro pattern used throughout
/// the COOLJAPAN ecosystem (torsh, oxigdal, oxify).
///
/// # Examples
///
/// ```
/// use oxionnx::{inputs, Tensor};
///
/// let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
/// let map = inputs!["input_ids" => t].expect("build inputs");
/// assert_eq!(map.len(), 1);
/// ```
///
/// Multiple inputs:
///
/// ```
/// use oxionnx::{inputs, Tensor};
///
/// let a = Tensor::zeros(&[1, 4]);
/// let b = Tensor::zeros(&[1, 4]);
/// let map = inputs!["q" => a, "k" => b].expect("build inputs");
/// assert_eq!(map.len(), 2);
/// ```
#[macro_export]
macro_rules! inputs {
    ($($key:expr => $val:expr),* $(,)?) => {
        (|| -> ::std::result::Result<
            ::std::collections::HashMap<&str, $crate::Tensor>,
            $crate::OnnxError,
        > {
            let mut map = ::std::collections::HashMap::new();
            $(
                map.insert($key, $val);
            )*
            ::std::result::Result::Ok(map)
        })()
    };
}

/// Build a typed input map for use with [`crate::Session::run_typed`].
///
/// Each value must be a [`crate::TypedTensor`]. Returns a
/// `Result<HashMap<&str, TypedTensor>, OnnxError>` for consistency with
/// the `inputs!` macro.
///
/// # Example
///
/// ```ignore
/// let map = inputs_typed!("input_ids" => TypedTensor::new(TensorStorage::I64(ids), vec![1, 32]))?;
/// ```
#[macro_export]
macro_rules! inputs_typed {
    ($($key:expr => $val:expr),* $(,)?) => {
        (|| -> ::std::result::Result<
            ::std::collections::HashMap<&str, $crate::TypedTensor>,
            $crate::OnnxError,
        > {
            let mut map = ::std::collections::HashMap::new();
            $(map.insert($key, $val);)*
            ::std::result::Result::Ok(map)
        })()
    };
}
