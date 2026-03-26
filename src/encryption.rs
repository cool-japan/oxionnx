//! AES-GCM model encryption and decryption.
//!
//! Encrypt ONNX model files at rest. The key is provided at load time.
//! Format: 12-byte nonce || encrypted\_data || 16-byte auth\_tag

#[cfg(feature = "encryption")]
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use oxionnx_core::OnnxError;
use std::path::Path;

/// Encrypt an ONNX model file and write to the output path.
///
/// Uses AES-256-GCM with a random 12-byte nonce prepended to the ciphertext.
/// Key must be exactly 32 bytes.
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
#[cfg(feature = "encryption")]
pub fn encrypt_bytes(plaintext: &[u8], key: &[u8; 32]) -> Result<Vec<u8>, OnnxError> {
    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| OnnxError::Internal(format!("Failed to create cipher: {}", e)))?;

    // Generate nonce from timestamp and data length for entropy.
    // For production, callers should consider providing their own nonce source.
    let mut nonce_bytes = [0u8; 12];
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    nonce_bytes[..8].copy_from_slice(&timestamp.to_le_bytes()[..8]);
    let size_bytes = (plaintext.len() as u32).to_le_bytes();
    nonce_bytes[8..12].copy_from_slice(&size_bytes);

    let nonce = Nonce::from_slice(&nonce_bytes);

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
    let nonce = Nonce::from_slice(nonce_bytes);

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
