//! Round-trip (write -> read) tests for `tpt-safetensors-io`.

use tpt_safetensors_io::{Dtype, SafetensorsBuilder, SafetensorsFile};

fn write_temp(bytes: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("tpt_st_test_{}.safetensors", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    path
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
