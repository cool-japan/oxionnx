//! Low-level protobuf wire-protocol helpers.
//!
//! Stream-based varint reading, byte skipping, and field-header parsing.

use std::io::Read;

use crate::parser::MAX_NESTING_DEPTH;

/// Upper bound on a single allocation made ahead of reading (16 MiB).
///
/// A declared length at or below this is allocated in one shot (the common case —
/// no growth, no copying). Anything larger is grown geometrically as bytes actually
/// arrive, so a bogus length taken from the wire (up to `u64::MAX`) can neither
/// abort with "capacity overflow" nor get the process OOM-killed allocating memory
/// that was never backed by real data.
const MAX_PREALLOC: usize = 16 * 1024 * 1024;

// ─────────────────────────────────────────────────────────────────
// Wire types
// ─────────────────────────────────────────────────────────────────

/// Wire types used in protobuf encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WireType {
    Varint,     // 0
    Fixed64,    // 1
    LenDelim,   // 2
    StartGroup, // 3
    EndGroup,   // 4
    Fixed32,    // 5
}

impl WireType {
    pub(super) fn from_u8(wt: u8) -> Result<Self, String> {
        match wt {
            0 => Ok(Self::Varint),
            1 => Ok(Self::Fixed64),
            2 => Ok(Self::LenDelim),
            3 => Ok(Self::StartGroup),
            4 => Ok(Self::EndGroup),
            5 => Ok(Self::Fixed32),
            other => Err(format!("unknown wire type {other}")),
        }
    }
}

/// A field header: field number + wire type.
pub(super) struct FieldHeader {
    pub(super) field_no: u32,
    pub(super) wire_type: WireType,
}

// ─────────────────────────────────────────────────────────────────
// Stream reader helpers
// ─────────────────────────────────────────────────────────────────

/// Read a single varint from a `Read` source.
/// Returns `None` on clean EOF (first byte), `Err` on truncation.
///
/// Mirrors `parser::read_varint`: the 10th byte may only carry bit 63, so a
/// non-canonical encoding cannot silently decode to a different value.
pub(super) fn read_varint_from_reader<R: Read>(reader: &mut R) -> Result<Option<u64>, String> {
    let mut result = 0u64;
    let mut shift = 0u32;
    let mut one = [0u8; 1];

    loop {
        match reader.read_exact(&mut one) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                if shift == 0 {
                    return Ok(None); // clean EOF
                }
                return Err("varint: unexpected EOF mid-varint".into());
            }
            Err(e) => return Err(format!("varint read error: {e}")),
        }
        let byte = one[0];
        if shift == 63 {
            if byte > 0x01 {
                return Err("varint: overflow (value exceeds 64 bits)".into());
            }
            return Ok(Some(result | ((byte as u64) << 63)));
        }
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(Some(result));
        }
        shift += 7;
    }
}

/// Convert a wire-format length to `usize`, rejecting values this target cannot address.
pub(super) fn len_to_usize(len: u64, what: &str) -> Result<usize, String> {
    usize::try_from(len).map_err(|_| format!("{what}: length {len} exceeds addressable memory"))
}

/// Read exactly `len` bytes from a reader into a new Vec.
///
/// Lengths up to [`MAX_PREALLOC`] take the fast path: one exact allocation and a
/// single `read_exact`. Beyond that the buffer is grown geometrically, and each step
/// is only as large as the data already proven to exist — a stream that claims
/// `u64::MAX` bytes fails with a read error after allocating [`MAX_PREALLOC`],
/// instead of aborting on a capacity overflow or being OOM-killed.
pub(super) fn read_exact_vec<R: Read>(reader: &mut R, len: usize) -> Result<Vec<u8>, String> {
    if len <= MAX_PREALLOC {
        let mut buf = vec![0u8; len];
        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("read_exact({len} bytes) failed: {e}"))?;
        return Ok(buf);
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut remaining = len;
    while remaining > 0 {
        let filled = buf.len();
        // Double the buffer each round (never more than what is still declared),
        // so growth is proportional to the bytes already read, not to `len`.
        let chunk = remaining.min(filled.max(MAX_PREALLOC));
        buf.reserve_exact(chunk);
        buf.resize(filled + chunk, 0);
        reader
            .read_exact(&mut buf[filled..])
            .map_err(|e| format!("read_exact({len} bytes) failed after {filled}: {e}"))?;
        remaining -= chunk;
    }
    Ok(buf)
}

/// Skip exactly `len` bytes by reading and discarding in chunks.
pub(super) fn skip_bytes<R: Read>(reader: &mut R, mut len: usize) -> Result<(), String> {
    let mut discard = [0u8; 8192];
    while len > 0 {
        let chunk = len.min(discard.len());
        reader
            .read_exact(&mut discard[..chunk])
            .map_err(|e| format!("skip({len} bytes) failed: {e}"))?;
        len -= chunk;
    }
    Ok(())
}

/// Read a field header from a reader. Returns None on clean EOF.
pub(super) fn read_field_header<R: Read>(reader: &mut R) -> Result<Option<FieldHeader>, String> {
    let tag = match read_varint_from_reader(reader)? {
        Some(t) => t,
        None => return Ok(None),
    };
    let field_no = (tag >> 3) as u32;
    let wire_type = WireType::from_u8((tag & 0x7) as u8)?;
    Ok(Some(FieldHeader {
        field_no,
        wire_type,
    }))
}

/// Skip a field value based on wire type. For LenDelim, also reads the length.
///
/// `field_no` is only needed to match a group's end tag.
pub(super) fn skip_field_value<R: Read>(
    reader: &mut R,
    wire_type: WireType,
    field_no: u32,
) -> Result<(), String> {
    skip_field_value_at(reader, wire_type, field_no, 0)
}

fn skip_field_value_at<R: Read>(
    reader: &mut R,
    wire_type: WireType,
    field_no: u32,
    depth: u32,
) -> Result<(), String> {
    match wire_type {
        WireType::Varint => {
            // Just consume the varint
            read_varint_from_reader(reader)?
                .ok_or_else(|| "unexpected EOF skipping varint".to_string())?;
        }
        WireType::Fixed64 => {
            skip_bytes(reader, 8)?;
        }
        WireType::Fixed32 => {
            skip_bytes(reader, 4)?;
        }
        WireType::LenDelim => {
            let len = read_varint_from_reader(reader)?
                .ok_or_else(|| "unexpected EOF reading length".to_string())?;
            skip_bytes(reader, len_to_usize(len, "skip length-delimited")?)?;
        }
        WireType::StartGroup => {
            skip_group(reader, field_no, depth + 1)?;
        }
        WireType::EndGroup => {
            return Err(format!("unexpected end-group tag for field {field_no}"));
        }
    }
    Ok(())
}

/// Consume a group body up to and including its matching end-group tag.
fn skip_group<R: Read>(reader: &mut R, group_field_no: u32, depth: u32) -> Result<(), String> {
    if depth >= MAX_NESTING_DEPTH {
        return Err(format!(
            "group nesting exceeds maximum depth {MAX_NESTING_DEPTH}"
        ));
    }
    loop {
        let hdr = read_field_header(reader)?.ok_or_else(|| {
            format!("group {group_field_no}: unterminated (EOF before end-group tag)")
        })?;
        match hdr.wire_type {
            WireType::EndGroup => {
                if hdr.field_no != group_field_no {
                    return Err(format!(
                        "group {group_field_no}: mismatched end-group tag for field {}",
                        hdr.field_no
                    ));
                }
                return Ok(());
            }
            wt => skip_field_value_at(reader, wt, hdr.field_no, depth)?,
        }
    }
}

/// Read a varint field value from a reader.
pub(super) fn read_varint_value<R: Read>(reader: &mut R) -> Result<u64, String> {
    read_varint_from_reader(reader)?
        .ok_or_else(|| "unexpected EOF reading varint value".to_string())
}

/// Read a length-delimited field value into a Vec<u8>.
pub(super) fn read_len_delim_value<R: Read>(reader: &mut R) -> Result<Vec<u8>, String> {
    let len = read_varint_from_reader(reader)?
        .ok_or_else(|| "unexpected EOF reading length prefix".to_string())?;
    read_exact_vec(reader, len_to_usize(len, "length-delimited")?)
}

#[cfg(test)]
mod wire_tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn read_exact_vec_fast_path_is_exact() {
        let data: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let got = read_exact_vec(&mut Cursor::new(data.clone()), data.len())
            .expect("exact read must succeed");
        assert_eq!(got, data);
    }

    #[test]
    fn read_exact_vec_grows_past_the_prealloc_cap() {
        // Exercise the geometric-growth path (> MAX_PREALLOC) with real data.
        let len = MAX_PREALLOC + 3 * 1024 * 1024 + 7;
        let mut data = vec![0u8; len];
        data[0] = 0xAB;
        data[MAX_PREALLOC] = 0xCD;
        data[len - 1] = 0xEF;

        let got = read_exact_vec(&mut Cursor::new(data), len).expect("large read must succeed");
        assert_eq!(got.len(), len);
        assert_eq!(got[0], 0xAB);
        assert_eq!(got[MAX_PREALLOC], 0xCD);
        assert_eq!(got[len - 1], 0xEF);
    }

    #[test]
    fn read_exact_vec_rejects_a_length_the_stream_cannot_back() {
        // A 4-byte stream claiming ~4 GiB: must error, not allocate the declared size.
        let err = read_exact_vec(&mut Cursor::new(vec![0u8; 4]), 4 * 1024 * 1024 * 1024)
            .expect_err("bogus length must be rejected");
        assert!(err.contains("read_exact"), "unexpected message: {err}");
    }

    #[test]
    fn varint_rejects_non_canonical_ten_byte_encoding() {
        let mut over_long = vec![0xFFu8; 9];
        over_long.push(0x7F);
        let err = read_varint_from_reader(&mut Cursor::new(over_long))
            .expect_err("over-long varint must be rejected");
        assert!(err.contains("overflow"), "unexpected message: {err}");

        let mut canonical = vec![0xFFu8; 9];
        canonical.push(0x01);
        let v = read_varint_from_reader(&mut Cursor::new(canonical))
            .expect("canonical 10-byte varint must parse");
        assert_eq!(v, Some(u64::MAX));
    }
}
