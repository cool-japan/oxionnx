//! AES-GCM model encryption and decryption.
//!
//! Encrypt ONNX model files at rest. The key is provided at load time.
//! Format: 12-byte nonce || encrypted\_data || 16-byte auth\_tag

#[cfg(feature = "encryption")]
use aes_gcm::{
    aead::{Aead, Generate, KeyInit},
    Aes256Gcm, Nonce,
};
use oxionnx_core::OnnxError;
use std::path::Path;

/// Encrypt an ONNX model file and write to the output path.
///
/// Uses AES-256-GCM with a 12-byte nonce, freshly drawn from the OS CSPRNG for
/// every call (see [`encrypt_bytes`] for why this matters), prepended to the
/// ciphertext. Key must be exactly 32 bytes.
#[cfg(feature = "encryption")]
pub fn encrypt_model(
    input_path: &Path,
    output_path: &Path,
    key: &[u8; 32],
) -> Result<(), OnnxError> {
    let plaintext = std::fs::read(input_path)
        .map_err(|e| OnnxError::Parse(format!("Cannot read model file: {}", e)))?;

    let ciphertext_with_nonce = encrypt_bytes(&plaintext, key)?;

    std::fs::write(output_path, &ciphertext_with_nonce)
        .map_err(|e| OnnxError::Internal(format!("Cannot write encrypted file: {}", e)))?;

    Ok(())
}

/// Encrypt raw bytes in memory, returning nonce || ciphertext (includes auth tag).
///
/// The 96-bit nonce is drawn fresh from the OS CSPRNG (via `getrandom`) for every
/// call, using [`aead`](aes_gcm::aead)'s [`Generate::try_generate`]. AES-GCM's security guarantee
/// collapses the moment a `(key, nonce)` pair repeats -- an attacker who observes two
/// ciphertexts encrypted under the same key and nonce can XOR them to recover the
/// plaintext XOR and forge authenticated ciphertexts (the GCM nonce-reuse "forbidden
/// attack"). A 96-bit CSPRNG nonce keeps the collision probability for any realistic
/// number of encryptions under the same key astronomically small (birthday bound on
/// 2^96), unlike a nonce derived from wall-clock time, which can collide whenever two
/// calls (e.g. concurrent callers, or platforms/containers with coarse clock
/// resolution) land on the same timestamp tick and see same-length plaintext.
#[cfg(feature = "encryption")]
pub fn encrypt_bytes(plaintext: &[u8], key: &[u8; 32]) -> Result<Vec<u8>, OnnxError> {
    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| OnnxError::Internal(format!("Failed to create cipher: {}", e)))?;

    // Cryptographically random nonce (OS CSPRNG via `getrandom`), never derived from
    // wall-clock time or any other predictable/collidable source.
    let nonce_bytes: [u8; 12] = Generate::try_generate()
        .map_err(|e| OnnxError::Internal(format!("Failed to generate nonce: {}", e)))?;

    let nonce = <&Nonce<_>>::try_from(nonce_bytes.as_slice())
        .map_err(|e| OnnxError::Internal(format!("Invalid nonce length: {}", e)))?;

    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| OnnxError::Internal(format!("Encryption failed: {}", e)))?;

    let mut output = Vec::with_capacity(12 + ciphertext.len());
    output.extend_from_slice(&nonce_bytes);
    output.extend_from_slice(&ciphertext);
    Ok(output)
}

/// Decrypt an encrypted ONNX model file and return the plaintext bytes.
#[cfg(feature = "encryption")]
pub fn decrypt_model(encrypted_path: &Path, key: &[u8; 32]) -> Result<Vec<u8>, OnnxError> {
    let data = std::fs::read(encrypted_path)
        .map_err(|e| OnnxError::Parse(format!("Cannot read encrypted file: {}", e)))?;

    decrypt_bytes(&data, key)
}

/// Decrypt raw bytes (nonce || ciphertext) in memory.
#[cfg(feature = "encryption")]
pub fn decrypt_bytes(data: &[u8], key: &[u8; 32]) -> Result<Vec<u8>, OnnxError> {
    if data.len() < 12 {
        return Err(OnnxError::Parse("Encrypted data too short".into()));
    }

    let (nonce_bytes, ciphertext) = data.split_at(12);
    let nonce = <&Nonce<_>>::try_from(nonce_bytes)
        .map_err(|e| OnnxError::Parse(format!("Invalid nonce length: {}", e)))?;

    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| OnnxError::Internal(format!("Failed to create cipher: {}", e)))?;

    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|e| OnnxError::Internal(format!("Decryption failed (wrong key?): {}", e)))
}

/// Load a [`crate::Session`] from an encrypted ONNX model file.
#[cfg(feature = "encryption")]
pub fn load_encrypted(path: &Path, key: &[u8; 32]) -> Result<crate::session::Session, OnnxError> {
    let plaintext = decrypt_model(path, key)?;
    crate::session::Session::from_bytes(&plaintext)
}

#[cfg(test)]
#[cfg(feature = "encryption")]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = [0x42u8; 32];
        let plaintext = b"ONNX model content for testing roundtrip encryption";

        let encrypted = encrypt_bytes(plaintext, &key).expect("encryption should succeed");
        let decrypted = decrypt_bytes(&encrypted, &key).expect("decryption should succeed");

        assert_eq!(decrypted, plaintext);
    }

    /// [a9-1] regression: the nonce used to be derived from `SystemTime::now()`
    /// nanoseconds plus `plaintext.len()`, so repeated same-length encryptions could
    /// reuse a nonce whenever timestamp ticks collided (plausible on platforms with
    /// coarser-than-nanosecond clock resolution, or under fast back-to-back calls,
    /// which is exactly what this tight loop exercises). AES-GCM's security
    /// guarantee requires every `(key, nonce)` pair to be unique; with the fix (a
    /// 96-bit nonce freshly drawn from the OS CSPRNG on every call), a collision
    /// across a few hundred draws is astronomically unlikely (birthday bound on
    /// 2^96), so this loop must never observe a repeated nonce.
    #[test]
    fn test_nonce_is_unique_across_same_length_encryptions() {
        let key = [0x11u8; 32];
        let plaintext = b"fixed-length plaintext used to isolate the nonce source";

        let mut seen_nonces = std::collections::HashSet::new();
        for _ in 0..256 {
            let encrypted = encrypt_bytes(plaintext, &key).expect("encryption should succeed");
            let nonce = encrypted[..12].to_vec();
            assert!(
                seen_nonces.insert(nonce),
                "nonce reuse detected across same-length encryptions under the same key"
            );
        }
    }

    #[test]
    fn test_decrypt_wrong_key() {
        let key = [0x42u8; 32];
        let wrong_key = [0x99u8; 32];
        let plaintext = b"secret model data";

        let encrypted = encrypt_bytes(plaintext, &key).expect("encryption should succeed");
        let result = decrypt_bytes(&encrypted, &wrong_key);

        assert!(result.is_err(), "decryption with wrong key should fail");
    }

    #[test]
    fn test_encrypt_model_file() {
        let key = [0xABu8; 32];
        let plaintext = b"fake ONNX model bytes for file-based test";

        let tmp = std::env::temp_dir();
        let input_path = tmp.join("oxionnx_test_encrypt_input.onnx");
        let output_path = tmp.join("oxionnx_test_encrypt_output.enc");

        std::fs::write(&input_path, plaintext).expect("should write test input");

        encrypt_model(&input_path, &output_path, &key).expect("encrypt_model should succeed");

        let decrypted = decrypt_model(&output_path, &key).expect("decrypt_model should succeed");

        assert_eq!(decrypted, plaintext);

        // Cleanup
        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&output_path);
    }
}
