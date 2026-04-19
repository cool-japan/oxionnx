//! Protobuf message-level helper parsers for opset imports and value names.

use crate::parser;

/// Parse an OperatorSetIdProto from bytes.
pub(super) fn parse_opset_import_bytes(buf: &[u8]) -> Result<(String, i64), String> {
    let mut domain = String::new();
    let mut version: i64 = 0;
    let mut pos = 0;
    while pos < buf.len() {
        let (tag, next_pos) = parser::read_varint(buf, pos)?;
        let field_no = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u8;
        pos = next_pos;
        match (field_no, wire_type) {
            (1, 2) => {
                let (len, next_pos) = parser::read_varint(buf, pos)?;
                let len = len as usize;
                pos = next_pos;
                domain = String::from_utf8_lossy(&buf[pos..pos + len]).into_owned();
                pos += len;
            }
            (2, 0) => {
                let (v, next_pos) = parser::read_varint(buf, pos)?;
                version = v as i64;
                pos = next_pos;
            }
            (_, 0) => {
                let (_v, next_pos) = parser::read_varint(buf, pos)?;
                pos = next_pos;
            }
            (_, 1) => pos += 8,
            (_, 2) => {
                let (len, next_pos) = parser::read_varint(buf, pos)?;
                pos = next_pos + len as usize;
            }
            (_, 5) => pos += 4,
            (_, wt) => return Err(format!("opset_import: unknown wire type {wt}")),
        }
    }
    Ok((domain, version))
}

/// Extract the name (field 1, string) from a ValueInfoProto buffer.
pub(super) fn extract_name_from_bytes(buf: &[u8]) -> String {
    let mut pos = 0;
    while pos < buf.len() {
        if let Ok((tag, next_pos)) = parser::read_varint(buf, pos) {
            let field_no = (tag >> 3) as u32;
            let wire_type = (tag & 0x7) as u8;
            pos = next_pos;
            match (field_no, wire_type) {
                (1, 2) => {
                    if let Ok((len, next_pos)) = parser::read_varint(buf, pos) {
                        let len = len as usize;
                        pos = next_pos;
                        if pos + len <= buf.len() {
                            return String::from_utf8_lossy(&buf[pos..pos + len]).into_owned();
                        }
                    }
                    break;
                }
                (_, 0) => {
                    if let Ok((_v, np)) = parser::read_varint(buf, pos) {
                        pos = np;
                    } else {
                        break;
                    }
                }
                (_, 1) => pos += 8,
                (_, 2) => {
                    if let Ok((len, np)) = parser::read_varint(buf, pos) {
                        pos = np + len as usize;
                    } else {
                        break;
                    }
                }
                (_, 5) => pos += 4,
                _ => break,
            }
        } else {
            break;
        }
    }
    String::new()
}
