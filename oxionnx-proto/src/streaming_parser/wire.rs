//! Low-level protobuf wire-protocol helpers.
//!
//! Stream-based varint reading, byte skipping, and field-header parsing.

use std::io::Read;

// ─────────────────────────────────────────────────────────────────
// Wire types
// ─────────────────────────────────────────────────────────────────

/// Wire types used in protobuf encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WireType {
    Varint,   // 0
    Fixed64,  // 1
    LenDelim, // 2
    Fixed32,  // 5
}

impl WireType {
    pub(super) fn from_u8(wt: u8) -> Result<Self, String> {
        match wt {
            0 => Ok(Self::Varint),
            1 => Ok(Self::Fixed64),
            2 => Ok(Self::LenDelim),
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
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(Some(result));
        }
        shift += 7;
        if shift >= 64 {
            return Err("varint: overflow".into());
        }
    }
}

/// Read exactly `len` bytes from a reader into a new Vec.
pub(super) fn read_exact_vec<R: Read>(reader: &mut R, len: usize) -> Result<Vec<u8>, String> {
    let mut buf = vec![0u8; len];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("read_exact({len} bytes) failed: {e}"))?;
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
pub(super) fn skip_field_value<R: Read>(reader: &mut R, wire_type: WireType) -> Result<(), String> {
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
                .ok_or_else(|| "unexpected EOF reading length".to_string())?
                as usize;
            skip_bytes(reader, len)?;
        }
    }
    Ok(())
}

/// Read a varint field value from a reader.
pub(super) fn read_varint_value<R: Read>(reader: &mut R) -> Result<u64, String> {
    read_varint_from_reader(reader)?
        .ok_or_else(|| "unexpected EOF reading varint value".to_string())
}

/// Read a length-delimited field value into a Vec<u8>.
pub(super) fn read_len_delim_value<R: Read>(reader: &mut R) -> Result<Vec<u8>, String> {
    let len = read_varint_from_reader(reader)?
        .ok_or_else(|| "unexpected EOF reading length prefix".to_string())? as usize;
    read_exact_vec(reader, len)
}
