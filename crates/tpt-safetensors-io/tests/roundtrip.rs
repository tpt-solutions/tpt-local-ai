//! Round-trip (write -> read) tests for `tpt-safetensors-io`.

use std::sync::atomic::{AtomicU64, Ordering};

use tpt_safetensors_io::{Dtype, SafetensorsBuilder, SafetensorsError, SafetensorsFile};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn write_temp(bytes: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir();
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!(
        "tpt_st_test_{}_{}.safetensors",
        std::process::id(),
        n
    ));
    std::fs::write(&path, bytes).unwrap();
    path
}

/// Assemble a raw safetensors file from a JSON header string and a data blob,
/// applying the mandatory 8-byte alignment padding.
fn raw_file(header_json: &str, data: &[u8]) -> Vec<u8> {
    let mut header = header_json.as_bytes().to_vec();
    let aligned = (header.len() + 7) & !7;
    header.resize(aligned, b' ');
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(data);
    out
}

#[test]
fn round_trip_f32_matrix() {
    let mut builder = SafetensorsBuilder::new();
    builder
        .add_f32("weight", vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    builder
        .add_metadata("description", serde_json::json!("test tensor"))
        .add_f32("bias", vec![3], vec![0.5f32, -0.5, 1.5])
        .unwrap();

    let bytes = builder.build().unwrap();
    let path = write_temp(&bytes);
    let file = SafetensorsFile::open(&path).unwrap();

    assert_eq!(file.len(), 2);
    let names: Vec<&str> = file.tensor_names().collect();
    assert!(names.contains(&"weight"));
    assert!(names.contains(&"bias"));

    let w = file.get_tensor("weight").unwrap();
    assert_eq!(w.dtype, Dtype::F32);
    assert_eq!(w.shape, vec![2, 3]);
    assert_eq!(w.to_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let b = file.get_tensor("bias").unwrap();
    assert_eq!(b.to_f32().unwrap(), vec![0.5, -0.5, 1.5]);

    std::fs::remove_file(&path).ok();
}

#[test]
fn header_is_eight_byte_aligned() {
    let mut builder = SafetensorsBuilder::new();
    builder.add_f32("x", vec![1], vec![1.0f32]).unwrap();
    let bytes = builder.build().unwrap();
    // The 8-byte length prefix plus the JSON header must be a multiple of 8.
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    assert_eq!((8 + header_len) % 8, 0);
    // The data section therefore begins at an 8-byte boundary.
    assert_eq!(bytes.len() - (8 + header_len), 4);
}

#[test]
fn rejects_wrong_data_length() {
    let mut builder = SafetensorsBuilder::new();
    let err = builder
        .add_f32("bad", vec![2], vec![1.0f32, 2.0, 3.0])
        .unwrap_err();
    assert!(matches!(
        err,
        tpt_safetensors_io::SafetensorsError::BadDataLength { .. }
    ));
}

#[test]
fn round_trips_every_dtype() {
    let cases: Vec<(Dtype, Vec<u8>)> = vec![
        (Dtype::F16, vec![0x00, 0x3c, 0x00, 0x40]),  // 1.0, 2.0
        (Dtype::BF16, vec![0x80, 0x3f, 0x00, 0x40]), // 1.0, 2.0
        (Dtype::I8, vec![0xff, 0x02]),
        (Dtype::I16, vec![0x01, 0x00, 0xff, 0xff]),
        (Dtype::I32, vec![1, 0, 0, 0, 2, 0, 0, 0]),
        (
            Dtype::I64,
            vec![1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        ),
        (Dtype::U8, vec![7, 8]),
        (Dtype::U16, vec![0x01, 0x00, 0x02, 0x00]),
        (Dtype::U32, vec![1, 0, 0, 0, 2, 0, 0, 0]),
        (
            Dtype::U64,
            vec![1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        ),
        (Dtype::BOOL, vec![1, 0]),
        (Dtype::F8E4M3, vec![0x38, 0x40]),
        (Dtype::F8E5M2, vec![0x3c, 0x40]),
    ];

    for (dtype, data) in cases {
        let mut builder = SafetensorsBuilder::new();
        builder
            .add_tensor("t", dtype, vec![2], data.clone())
            .unwrap();
        let bytes = builder.build().unwrap();
        let path = write_temp(&bytes);
        let file = SafetensorsFile::open(&path).unwrap();
        let view = file.get_tensor("t").unwrap();
        assert_eq!(view.dtype, dtype, "dtype mismatch for {dtype:?}");
        assert_eq!(view.shape, vec![2]);
        assert_eq!(view.data, data.as_slice(), "bytes mismatch for {dtype:?}");
        std::fs::remove_file(&path).ok();
    }
}

#[test]
fn f16_and_bf16_convert_to_f32() {
    let mut builder = SafetensorsBuilder::new();
    builder
        .add_tensor("h", Dtype::F16, vec![2], vec![0x00, 0x3c, 0x00, 0x40])
        .unwrap();
    builder
        .add_tensor("b", Dtype::BF16, vec![2], vec![0x80, 0x3f, 0x00, 0x40])
        .unwrap();
    let bytes = builder.build().unwrap();
    let path = write_temp(&bytes);
    let file = SafetensorsFile::open(&path).unwrap();
    assert_eq!(
        file.get_tensor("h").unwrap().to_f32().unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        file.get_tensor("b").unwrap().to_f32().unwrap(),
        vec![1.0, 2.0]
    );
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Malformed / adversarial header rejection
// ---------------------------------------------------------------------------

#[test]
fn rejects_header_length_overflow() {
    // 8-byte prefix declares a header length of u64::MAX.
    let mut bytes = u64::MAX.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    let path = write_temp(&bytes);
    assert!(matches!(
        SafetensorsFile::open(&path),
        Err(SafetensorsError::InvalidHeader(_))
    ));
    std::fs::remove_file(&path).ok();
}

#[test]
fn rejects_declared_header_len_exceeding_file() {
    let mut bytes = 1000u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    let path = write_temp(&bytes);
    assert!(matches!(
        SafetensorsFile::open(&path),
        Err(SafetensorsError::InvalidHeader(_))
    ));
    std::fs::remove_file(&path).ok();
}

#[test]
fn rejects_non_object_header() {
    let bytes = raw_file("123", &[]);
    let path = write_temp(&bytes);
    assert!(matches!(
        SafetensorsFile::open(&path),
        Err(SafetensorsError::InvalidHeader(_))
    ));
    std::fs::remove_file(&path).ok();
}

#[test]
fn rejects_missing_fields() {
    let bytes = raw_file(r#"{"t":{"shape":[2],"data_offsets":[0,2]}}"#, &[0, 0]);
    let path = write_temp(&bytes);
    assert!(matches!(
        SafetensorsFile::open(&path),
        Err(SafetensorsError::InvalidHeader(_))
    ));
    std::fs::remove_file(&path).ok();
}

#[test]
fn rejects_start_greater_than_end() {
    let bytes = raw_file(
        r#"{"t":{"dtype":"U8","shape":[2],"data_offsets":[10,5]}}"#,
        &[0; 16],
    );
    let path = write_temp(&bytes);
    assert!(matches!(
        SafetensorsFile::open(&path),
        Err(SafetensorsError::InvalidHeader(_))
    ));
    std::fs::remove_file(&path).ok();
}

#[test]
fn rejects_span_not_matching_shape() {
    // shape [2] * U8 => 2 bytes expected, but the span is 3.
    let bytes = raw_file(
        r#"{"t":{"dtype":"U8","shape":[2],"data_offsets":[0,3]}}"#,
        &[0, 0, 0],
    );
    let path = write_temp(&bytes);
    assert!(matches!(
        SafetensorsFile::open(&path),
        Err(SafetensorsError::InvalidHeader(_))
    ));
    std::fs::remove_file(&path).ok();
}

#[test]
fn rejects_offsets_exceeding_file() {
    let bytes = raw_file(
        r#"{"t":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#,
        &[0, 0, 0], // only 3 data bytes, header promised 8
    );
    let path = write_temp(&bytes);
    assert!(matches!(
        SafetensorsFile::open(&path),
        Err(SafetensorsError::InvalidHeader(_))
    ));
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Patch / streaming APIs
// ---------------------------------------------------------------------------

#[test]
fn patch_existing_file_replaces_one_tensor() {
    let mut builder = SafetensorsBuilder::new();
    builder.add_f32("a", vec![2], vec![1.0f32, 2.0]).unwrap();
    builder.add_f32("b", vec![2], vec![3.0f32, 4.0]).unwrap();
    builder.add_metadata("format", serde_json::json!("pt"));
    let src = write_temp(&builder.build().unwrap());

    let mut patched = SafetensorsBuilder::from_file(&src).unwrap();
    patched
        .replace_tensor("b", Dtype::F32, vec![2], {
            let mut v = Vec::new();
            v.extend_from_slice(&9.0f32.to_le_bytes());
            v.extend_from_slice(&9.0f32.to_le_bytes());
            v
        })
        .unwrap();
    let out = write_temp(&patched.build().unwrap());

    let file = SafetensorsFile::open(&out).unwrap();
    assert_eq!(
        file.get_tensor("a").unwrap().to_f32().unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        file.get_tensor("b").unwrap().to_f32().unwrap(),
        vec![9.0, 9.0]
    );
    assert_eq!(
        file.metadata().get("format").and_then(|v| v.as_str()),
        Some("pt")
    );

    std::fs::remove_file(&src).ok();
    std::fs::remove_file(&out).ok();
}

#[test]
fn streaming_write_matches_build() {
    let mut builder = SafetensorsBuilder::new();
    builder
        .add_f32("w", vec![2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
        .unwrap();
    builder.add_metadata("k", serde_json::json!("v"));

    let in_memory = builder.build().unwrap();
    let mut streamed = Vec::new();
    builder.write_to(&mut streamed).unwrap();
    assert_eq!(in_memory, streamed);
}
