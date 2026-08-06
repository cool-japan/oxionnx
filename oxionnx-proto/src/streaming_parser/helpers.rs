//! Protobuf message-level helper parsers for opset imports.

use crate::parser;

/// Parse an OperatorSetIdProto from bytes.
///
/// Every length prefix is consumed through `parser::read_len_delim_at`, which uses
/// checked arithmetic and validates the end offset against the buffer: a truncated
/// sub-message (e.g. a 3-byte payload declaring a 5-byte domain) now yields a typed
/// error instead of panicking on an out-of-range slice.
pub(super) fn parse_opset_import_bytes(buf: &[u8]) -> Result<(String, i64), String> {
    let mut domain = String::new();
    let mut version: i64 = 0;
    let mut pos = 0;
    while pos < buf.len() {
        let (field_no, wire_type, next_pos) = parser::read_tag(buf, pos)?;
        pos = next_pos;
        match (field_no, wire_type) {
            (1, 2) => {
                let (body, next) = parser::read_len_delim_at(buf, pos, "opset_import domain")?;
                domain = String::from_utf8_lossy(body).into_owned();
                pos = next;
            }
            (2, 0) => {
                let (v, next) = parser::read_varint(buf, pos)?;
                version = v as i64;
                pos = next;
            }
            (_, wt) => {
                pos = parser::skip_field_value_in_buf(buf, pos, field_no, wt)?;
            }
        }
    }
    Ok((domain, version))
}
