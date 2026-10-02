//! One WebSocket binary message = a JSON header plus raw binary blobs.
//!
//! ```text
//! [u32 LE header_len][header JSON][u32 LE blob_count]([u64 LE blob_len])*[blob bytes]*
//! ```
//!
//! Structured data goes through JSON - the same format the project files use,
//! so every settings type round-trips exactly. Pixel data travels as blobs so
//! millions of samples aren't encoded as individual JSON numbers.

use evanalyzer_cfg::core_types::InternalErrors;
use serde::Serialize;
use serde::de::DeserializeOwned;

pub(crate) struct Frame<T> {
    pub msg: T,
    pub blobs: Vec<Vec<u8>>,
}

pub(crate) fn encode<T: Serialize>(msg: &T, blobs: &[Vec<u8>]) -> Result<Vec<u8>, InternalErrors> {
    let header = serde_json::to_vec(msg)
        .map_err(|e| InternalErrors::Internal(format!("failed to encode message: {e}")))?;
    let blob_bytes: usize = blobs.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(8 + header.len() + blobs.len() * 8 + blob_bytes);
    out.extend_from_slice(&len_u32(header.len())?.to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&len_u32(blobs.len())?.to_le_bytes());
    for blob in blobs {
        out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
    }
    for blob in blobs {
        out.extend_from_slice(blob);
    }
    Ok(out)
}

/// Fails instead of panicking on any malformed input - it comes from the
/// network.
pub(crate) fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<Frame<T>, InternalErrors> {
    let mut reader = Reader { bytes, pos: 0 };
    let header_len = reader.u32()? as usize;
    let header = reader.take(header_len)?;
    let msg = serde_json::from_slice(header)
        .map_err(|e| InternalErrors::Internal(format!("malformed message: {e}")))?;
    let blob_count = reader.u32()? as usize;
    // Every declared blob needs at least its 8-byte length - reject absurd
    // counts before allocating for them.
    if blob_count > reader.remaining() / 8 {
        return Err(malformed("blob count exceeds message size"));
    }
    let lens = (0..blob_count)
        .map(|_| reader.u64())
        .collect::<Result<Vec<_>, _>>()?;
    let blobs = lens
        .into_iter()
        .map(|len| {
            let len = usize::try_from(len).map_err(|_| malformed("blob too large"))?;
            Ok(reader.take(len)?.to_vec())
        })
        .collect::<Result<Vec<_>, InternalErrors>>()?;
    if reader.remaining() != 0 {
        return Err(malformed("trailing bytes"));
    }
    Ok(Frame { msg, blobs })
}

fn len_u32(len: usize) -> Result<u32, InternalErrors> {
    u32::try_from(len).map_err(|_| InternalErrors::Internal("message part too large".into()))
}

fn malformed(what: &str) -> InternalErrors {
    InternalErrors::Internal(format!("malformed message: {what}"))
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], InternalErrors> {
        if len > self.remaining() {
            return Err(malformed("truncated"));
        }
        let slice = &self.bytes[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, InternalErrors> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, InternalErrors> {
        let b = self.take(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(b);
        Ok(u64::from_le_bytes(arr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Msg {
        id: u64,
        name: String,
    }

    #[test]
    fn header_and_blobs_round_trip() {
        let msg = Msg {
            id: 7,
            name: "tile".into(),
        };
        let blobs = vec![vec![1, 2, 3], vec![], vec![255; 1000]];
        let frame: Frame<Msg> = decode(&encode(&msg, &blobs).unwrap()).unwrap();
        assert_eq!(frame.msg, msg);
        assert_eq!(frame.blobs, blobs);
    }

    #[test]
    fn truncated_or_padded_input_is_an_error_not_a_panic() {
        let bytes = encode(
            &Msg {
                id: 1,
                name: "x".into(),
            },
            &[vec![9; 16]],
        )
        .unwrap();
        for cut in 0..bytes.len() {
            assert!(decode::<Msg>(&bytes[..cut]).is_err(), "cut at {cut}");
        }
        let mut padded = bytes.clone();
        padded.push(0);
        assert!(decode::<Msg>(&padded).is_err());
    }

    #[test]
    fn an_absurd_blob_count_is_rejected_before_allocating() {
        let mut bytes = encode(
            &Msg {
                id: 1,
                name: "x".into(),
            },
            &[],
        )
        .unwrap();
        let count_at = bytes.len() - 4;
        bytes[count_at..].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode::<Msg>(&bytes).is_err());
    }
}
