//! Read-only parser for [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
//! headers, metadata, and tensor descriptors.
//!
//! GGUF is the container format used by `llama.cpp`/`ggml` for local inference.
//! This module reads the **header**: the key/value metadata table (architecture,
//! hyper-parameters, tokenizer vocab, chat template, quantization info, …) and
//! the per-tensor descriptors (name, shape, ggml type, offset). It does **not**
//! decode quantized tensor payloads — it is a metadata/inspection reader that
//! complements the safetensors side of this crate.
//!
//! Only little-endian GGUF versions 2 and 3 are supported (this covers every
//! GGUF file produced by mainstream tooling). Version 1 and big-endian files are
//! rejected with a clear error.
//!
//! # Example
//!
//! ```no_run
//! use tpt_safetensors_io::gguf::GgufFile;
//!
//! let f = GgufFile::open("model.gguf")?;
//! println!("version {}", f.version());
//! if let Some(arch) = f.get("general.architecture").and_then(|v| v.as_str()) {
//!     println!("architecture: {arch}");
//! }
//! for t in f.tensors() {
//!     println!("{} {:?}", t.name, t.dimensions);
//! }
//! # Ok::<(), tpt_safetensors_io::gguf::GgufError>(())
//! ```

use std::fmt;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const DEFAULT_ALIGNMENT: u64 = 32;

/// Errors that can occur while reading a GGUF file.
#[derive(Debug)]
#[non_exhaustive]
pub enum GgufError {
    /// An I/O error, most commonly from opening or mapping the file.
    Io(std::io::Error),
    /// The file does not start with the `GGUF` magic bytes.
    BadMagic,
    /// The GGUF version is not supported (only 2 and 3 are).
    UnsupportedVersion(u32),
    /// The file was truncated: a read ran past the end of the mapping.
    UnexpectedEof,
    /// A metadata value used a type tag this reader does not understand.
    UnknownValueType(u32),
    /// A string field was not valid UTF-8.
    InvalidUtf8,
    /// A declared count or length is implausibly large / would overflow.
    Malformed(String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "I/O error: {e}"),
            GgufError::BadMagic => write!(f, "not a GGUF file (bad magic)"),
            GgufError::UnsupportedVersion(v) => {
                write!(
                    f,
                    "unsupported GGUF version {v} (only 2 and 3 are supported)"
                )
            }
            GgufError::UnexpectedEof => write!(f, "unexpected end of file while parsing header"),
            GgufError::UnknownValueType(t) => write!(f, "unknown metadata value type {t}"),
            GgufError::InvalidUtf8 => write!(f, "string field is not valid UTF-8"),
            GgufError::Malformed(m) => write!(f, "malformed GGUF header: {m}"),
        }
    }
}

impl std::error::Error for GgufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GgufError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        GgufError::Io(e)
    }
}

/// A single GGUF metadata value.
///
/// GGUF metadata is a flat key/value table where each value is one of a small
/// set of scalar types, a UTF-8 string, or a homogeneous array of the above.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Signed 8-bit integer.
    I8(i8),
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Signed 16-bit integer.
    I16(i16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Signed 32-bit integer.
    I32(i32),
    /// 32-bit floating point.
    F32(f32),
    /// Boolean.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 64-bit integer.
    I64(i64),
    /// 64-bit floating point.
    F64(f64),
    /// A homogeneous array of values.
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Borrows the value as a string slice, if it is a [`GgufValue::String`].
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Interprets any integer variant as `u64` (unsigned) if it fits.
    ///
    /// Signed variants are converted with `try_into`; negative values yield
    /// `None`.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            GgufValue::U8(v) => Some(u64::from(v)),
            GgufValue::U16(v) => Some(u64::from(v)),
            GgufValue::U32(v) => Some(u64::from(v)),
            GgufValue::U64(v) => Some(v),
            GgufValue::I8(v) => u64::try_from(v).ok(),
            GgufValue::I16(v) => u64::try_from(v).ok(),
            GgufValue::I32(v) => u64::try_from(v).ok(),
            GgufValue::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    /// Interprets any integer variant as `i64`.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            GgufValue::U8(v) => Some(i64::from(v)),
            GgufValue::U16(v) => Some(i64::from(v)),
            GgufValue::U32(v) => Some(i64::from(v)),
            GgufValue::U64(v) => i64::try_from(v).ok(),
            GgufValue::I8(v) => Some(i64::from(v)),
            GgufValue::I16(v) => Some(i64::from(v)),
            GgufValue::I32(v) => Some(i64::from(v)),
            GgufValue::I64(v) => Some(v),
            _ => None,
        }
    }

    /// Interprets any float variant as `f64`.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            GgufValue::F32(v) => Some(f64::from(v)),
            GgufValue::F64(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the boolean value, if this is a [`GgufValue::Bool`].
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            GgufValue::Bool(b) => Some(b),
            _ => None,
        }
    }

    /// Borrows the elements, if this is a [`GgufValue::Array`].
    #[must_use]
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

/// A ggml tensor element type (the numeric tags used in a GGUF tensor
/// descriptor).
///
/// Only the tag is decoded; quantized block layouts are not interpreted by this
/// crate. Unknown tags are preserved as [`GgmlType::Unknown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)] // Preserve the canonical ggml type names (Q4_0, Q6_K, …).
#[non_exhaustive]
pub enum GgmlType {
    /// 32-bit float.
    F32,
    /// 16-bit float.
    F16,
    /// Legacy 4-bit quantization (`Q4_0`).
    Q4_0,
    /// Legacy 4-bit quantization (`Q4_1`).
    Q4_1,
    /// 5-bit quantization (`Q5_0`).
    Q5_0,
    /// 5-bit quantization (`Q5_1`).
    Q5_1,
    /// 8-bit quantization (`Q8_0`).
    Q8_0,
    /// 8-bit quantization (`Q8_1`).
    Q8_1,
    /// k-quant 2-bit.
    Q2_K,
    /// k-quant 3-bit.
    Q3_K,
    /// k-quant 4-bit.
    Q4_K,
    /// k-quant 5-bit.
    Q5_K,
    /// k-quant 6-bit.
    Q6_K,
    /// k-quant 8-bit.
    Q8_K,
    /// 8-bit integer.
    I8,
    /// 16-bit integer.
    I16,
    /// 32-bit integer.
    I32,
    /// 64-bit integer.
    I64,
    /// 64-bit float.
    F64,
    /// `bfloat16`.
    BF16,
    /// Any tag not otherwise recognized (preserved verbatim).
    Unknown(u32),
}

impl GgmlType {
    /// Maps a raw ggml type tag to a [`GgmlType`].
    #[must_use]
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2_K,
            11 => GgmlType::Q3_K,
            12 => GgmlType::Q4_K,
            13 => GgmlType::Q5_K,
            14 => GgmlType::Q6_K,
            15 => GgmlType::Q8_K,
            24 => GgmlType::I8,
            25 => GgmlType::I16,
            26 => GgmlType::I32,
            27 => GgmlType::I64,
            28 => GgmlType::F64,
            30 => GgmlType::BF16,
            other => GgmlType::Unknown(other),
        }
    }
}

/// Descriptor for a single tensor stored in a GGUF file.
///
/// The `offset` is relative to the start of the tensor-data section
/// ([`GgufFile::tensor_data_offset`]); this reader does not read the payload.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name.
    pub name: String,
    /// Dimensions, fastest-moving axis first (ggml convention).
    pub dimensions: Vec<u64>,
    /// The ggml element/quantization type.
    pub ggml_type: GgmlType,
    /// Byte offset of the tensor within the tensor-data section.
    pub offset: u64,
}

/// A memory-mapped, read-only GGUF file (header + metadata + tensor list).
pub struct GgufFile {
    // Kept alive so callers could extend this to read tensor bytes later.
    _mmap: Mmap,
    version: u32,
    metadata: Vec<(String, GgufValue)>,
    tensors: Vec<GgufTensorInfo>,
    alignment: u64,
    tensor_data_offset: u64,
}

impl GgufFile {
    /// Opens and parses the header of the GGUF file at `path`.
    ///
    /// # Errors
    /// Returns [`GgufError`] if the file cannot be opened/mapped, is not a GGUF
    /// file, uses an unsupported version, or has a truncated/malformed header.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let file = File::open(path)?;
        // SAFETY: see the equivalent note in `reader.rs`. We only ever read
        // through the map, and every offset derived from the (untrusted) header
        // is bounds-checked by the `Cursor` before use.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse(mmap)
    }

    /// Parses a GGUF header out of an in-memory mapping.
    fn parse(mmap: Mmap) -> Result<Self, GgufError> {
        let mut cur = Cursor::new(&mmap);

        let magic = cur.u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic);
        }
        let version = cur.u32()?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = cur.u64()?;
        let kv_count = cur.u64()?;
        // Guard against absurd counts before allocating.
        let max_reasonable = mmap.len() as u64;
        if tensor_count > max_reasonable || kv_count > max_reasonable {
            return Err(GgufError::Malformed(
                "declared tensor/metadata count exceeds file size".to_string(),
            ));
        }

        let mut metadata = Vec::with_capacity(kv_count.min(1024) as usize);
        for _ in 0..kv_count {
            let key = cur.string()?;
            let value = cur.value()?;
            metadata.push((key, value));
        }

        let mut tensors = Vec::with_capacity(tensor_count.min(4096) as usize);
        for _ in 0..tensor_count {
            let name = cur.string()?;
            let n_dims = cur.u32()?;
            if u64::from(n_dims) > max_reasonable {
                return Err(GgufError::Malformed(
                    "tensor has too many dimensions".to_string(),
                ));
            }
            let mut dimensions = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dimensions.push(cur.u64()?);
            }
            let ggml_type = GgmlType::from_u32(cur.u32()?);
            let offset = cur.u64()?;
            tensors.push(GgufTensorInfo {
                name,
                dimensions,
                ggml_type,
                offset,
            });
        }

        // The tensor-data section begins after the header, padded up to the
        // alignment declared in `general.alignment` (default 32).
        let alignment = metadata
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| v.as_u64())
            .filter(|&a| a != 0)
            .unwrap_or(DEFAULT_ALIGNMENT);
        let pos = cur.pos() as u64;
        let tensor_data_offset = pos.div_ceil(alignment) * alignment;

        Ok(Self {
            _mmap: mmap,
            version,
            metadata,
            tensors,
            alignment,
            tensor_data_offset,
        })
    }

    /// The GGUF format version (2 or 3).
    #[must_use]
    pub fn version(&self) -> u32 {
        self.version
    }

    /// The declared tensor-data alignment (`general.alignment`, default 32).
    #[must_use]
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Absolute byte offset where the tensor-data section begins.
    #[must_use]
    pub fn tensor_data_offset(&self) -> u64 {
        self.tensor_data_offset
    }

    /// All metadata key/value pairs, in file order.
    #[must_use]
    pub fn metadata(&self) -> &[(String, GgufValue)] {
        &self.metadata
    }

    /// Looks up a metadata value by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Convenience accessor for `general.architecture` (e.g. `"llama"`).
    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.get("general.architecture").and_then(GgufValue::as_str)
    }

    /// All tensor descriptors, in file order.
    #[must_use]
    pub fn tensors(&self) -> &[GgufTensorInfo] {
        &self.tensors
    }

    /// Looks up a tensor descriptor by name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Number of tensors described by the header.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the file describes no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// A bounds-checked little-endian byte cursor over the mapping.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let end = self.pos.checked_add(n).ok_or(GgufError::UnexpectedEof)?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or(GgufError::UnexpectedEof)?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::InvalidUtf8)
    }

    /// Reads a metadata value: a `u32` type tag followed by its payload.
    fn value(&mut self) -> Result<GgufValue, GgufError> {
        let ty = self.u32()?;
        self.value_of_type(ty)
    }

    fn value_of_type(&mut self, ty: u32) -> Result<GgufValue, GgufError> {
        Ok(match ty {
            0 => GgufValue::U8(self.u8()?),
            1 => GgufValue::I8(self.u8()? as i8),
            2 => GgufValue::U16(self.u16()?),
            3 => GgufValue::I16(self.u16()? as i16),
            4 => GgufValue::U32(self.u32()?),
            5 => GgufValue::I32(self.u32()? as i32),
            6 => GgufValue::F32(f32::from_bits(self.u32()?)),
            7 => GgufValue::Bool(self.u8()? != 0),
            8 => GgufValue::String(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                if elem_ty == 9 {
                    // Nested arrays are not part of the GGUF spec.
                    return Err(GgufError::Malformed(
                        "nested arrays are not allowed".to_string(),
                    ));
                }
                let len = self.u64()? as usize;
                // Cap the pre-allocation; `value_of_type` still bounds-checks
                // every element against the mapping length.
                let mut items = Vec::with_capacity(len.min(4096));
                for _ in 0..len {
                    items.push(self.value_of_type(elem_ty)?);
                }
                GgufValue::Array(items)
            }
            10 => GgufValue::U64(self.u64()?),
            11 => GgufValue::I64(self.u64()? as i64),
            12 => GgufValue::F64(f64::from_bits(self.u64()?)),
            other => return Err(GgufError::UnknownValueType(other)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Minimal GGUF v3 byte-buffer writer for building test fixtures.
    struct GgufWriter {
        buf: Vec<u8>,
    }

    impl GgufWriter {
        fn new() -> Self {
            Self { buf: Vec::new() }
        }
        fn u32(&mut self, v: u32) {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }
        fn u64(&mut self, v: u64) {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }
        fn string(&mut self, s: &str) {
            self.u64(s.len() as u64);
            self.buf.extend_from_slice(s.as_bytes());
        }
        fn kv_string(&mut self, key: &str, val: &str) {
            self.string(key);
            self.u32(8); // STRING
            self.string(val);
        }
        fn kv_u32(&mut self, key: &str, val: u32) {
            self.string(key);
            self.u32(4); // UINT32
            self.u32(val);
        }
        fn kv_str_array(&mut self, key: &str, vals: &[&str]) {
            self.string(key);
            self.u32(9); // ARRAY
            self.u32(8); // element type STRING
            self.u64(vals.len() as u64);
            for v in vals {
                self.string(v);
            }
        }
    }

    /// Writes `bytes` to a uniquely-named temp file and returns the path.
    fn write_temp(bytes: &[u8]) -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("tpt-gguf-test-{}-{}.gguf", std::process::id(), n));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn fixture_bytes() -> Vec<u8> {
        let mut w = GgufWriter::new();
        w.u32(GGUF_MAGIC);
        w.u32(3); // version
        w.u64(1); // tensor_count
        w.u64(4); // kv_count
        w.kv_string("general.architecture", "llama");
        w.kv_u32("llama.block_count", 32);
        w.kv_str_array("tokenizer.ggml.tokens", &["<s>", "hello", "café"]);
        w.kv_u32("general.alignment", 32);
        // one tensor descriptor
        w.string("token_embd.weight");
        w.u32(2); // n_dims
        w.u64(4096);
        w.u64(128_256);
        w.u32(12); // Q4_K
        w.u64(0); // offset
        w.buf
    }

    #[test]
    fn parses_metadata_and_tensors() {
        let path = write_temp(&fixture_bytes());
        let g = GgufFile::open(&path).unwrap();
        assert_eq!(g.version(), 3);
        assert_eq!(g.architecture(), Some("llama"));
        assert_eq!(
            g.get("llama.block_count").and_then(GgufValue::as_u64),
            Some(32)
        );

        let toks = g
            .get("tokenizer.ggml.tokens")
            .and_then(GgufValue::as_array)
            .unwrap();
        assert_eq!(toks.len(), 3);
        // UTF-8 (accented) survives round-trip.
        assert_eq!(toks[2].as_str(), Some("café"));

        assert_eq!(g.len(), 1);
        let t = g.tensor("token_embd.weight").unwrap();
        assert_eq!(t.dimensions, vec![4096, 128_256]);
        assert_eq!(t.ggml_type, GgmlType::Q4_K);
        assert_eq!(g.alignment(), 32);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_bad_magic() {
        let path = write_temp(b"NOPExxxxxxxxxxxx");
        assert!(matches!(GgufFile::open(&path), Err(GgufError::BadMagic)));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut w = GgufWriter::new();
        w.u32(GGUF_MAGIC);
        w.u32(1); // v1 unsupported
        w.u64(0);
        w.u64(0);
        let path = write_temp(&w.buf);
        assert!(matches!(
            GgufFile::open(&path),
            Err(GgufError::UnsupportedVersion(1))
        ));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_truncated_header() {
        let mut w = GgufWriter::new();
        w.u32(GGUF_MAGIC);
        w.u32(3);
        w.u64(0); // tensor_count
        w.u64(1); // kv_count but no kv data follows
        let path = write_temp(&w.buf);
        assert!(matches!(
            GgufFile::open(&path),
            Err(GgufError::UnexpectedEof)
        ));
        std::fs::remove_file(&path).ok();
    }
}
