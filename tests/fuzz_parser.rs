//! Fuzz-like tests for the protobuf parser.
//! Feeds various malformed inputs to verify graceful error handling.

use oxionnx::Session;

/// Feed completely random bytes.
#[test]
fn test_fuzz_random_bytes() {
    // Generate pseudo-random bytes using a simple LCG
    let mut rng_state: u64 = 12345;
    for _ in 0..100 {
        let len = (rng_state % 256) as usize;
        let bytes: Vec<u8> = (0..len)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng_state >> 33) as u8
            })
            .collect();
        // Should not panic - may return Ok or Err
        let _ = Session::from_bytes(&bytes);
    }
}

/// Feed truncated protobuf (cut at various offsets).
#[test]
fn test_fuzz_truncated() {
    // Start with valid-ish protobuf header bytes
    let header: Vec<u8> = vec![
        0x08, 0x07, // ir_version = 7
        0x12, 0x04, 0x74, 0x65, 0x73, 0x74, // producer_name = "test"
        0x3A, 0x10, // graph field
    ];
    for cut_at in 0..header.len() {
        let _ = Session::from_bytes(&header[..cut_at]);
    }
}

/// Feed very large varint values.
#[test]
fn test_fuzz_large_varints() {
    // Varint encoding of very large numbers
    let bytes: Vec<u8> = vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01];
    let _ = Session::from_bytes(&bytes);
}

/// Feed bytes with invalid wire types.
#[test]
fn test_fuzz_invalid_wire_types() {
    for wire_type in 3..=7u8 {
        let bytes = vec![(1 << 3) | wire_type]; // field 1, invalid wire type
        let _ = Session::from_bytes(&bytes);
    }
}

/// Feed nested messages with incorrect lengths.
#[test]
fn test_fuzz_bad_lengths() {
    // Length-delimited field claiming more bytes than available
    let bytes: Vec<u8> = vec![0x12, 0xFF, 0x01]; // field 2, length 127, but only 1 byte follows
    let _ = Session::from_bytes(&bytes);
}

/// Feed zero-length fields.
#[test]
fn test_fuzz_zero_length_fields() {
    let bytes: Vec<u8> = vec![0x12, 0x00]; // field 2, length 0
    let _ = Session::from_bytes(&bytes);

    let bytes2: Vec<u8> = vec![0x12, 0x00, 0x12, 0x00, 0x12, 0x00]; // multiple empty fields
    let _ = Session::from_bytes(&bytes2);
}

/// Feed deeply nested messages.
#[test]
fn test_fuzz_deep_nesting() {
    // Create deeply nested length-delimited fields
    let mut bytes = vec![0x42u8]; // some data
    for _ in 0..50 {
        let len = bytes.len();
        let mut new = vec![0x3A]; // field 7 (graph), length-delimited
                                  // encode length as varint
        let mut l = len;
        loop {
            let byte = (l & 0x7F) as u8;
            l >>= 7;
            if l > 0 {
                new.push(byte | 0x80);
            } else {
                new.push(byte);
                break;
            }
        }
        new.extend_from_slice(&bytes);
        bytes = new;
    }
    let _ = Session::from_bytes(&bytes);
}

/// Feed maximum-size allocations (try to trigger OOM protection).
#[test]
fn test_fuzz_huge_claimed_size() {
    // Claim a string of 2GB but provide no data
    let bytes: Vec<u8> = vec![
        0x12, // field 2, wire type 2 (length-delimited)
        0x80, 0x80, 0x80, 0x80, 0x08, // varint = 2^31
    ];
    let result = Session::from_bytes(&bytes);
    assert!(result.is_ok() || result.is_err()); // just don't panic
}

/// Repeated field flood.
#[test]
fn test_fuzz_repeated_flood() {
    // Many repeated small fields
    let mut bytes = Vec::new();
    for _ in 0..1000 {
        bytes.push(0x08); // field 1, varint
        bytes.push(0x01); // value 1
    }
    let _ = Session::from_bytes(&bytes);
}
