//! Unit and library tests for `tpt-lora-merge`.

use std::sync::atomic::{AtomicU64, Ordering};

use tpt_lora_merge::{merge_linear, merge_lora, merge_loras, MergeError};
use tpt_safetensors_io::{Dtype, SafetensorsBuilder, SafetensorsFile};

static COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn merge_linear_math() {
    // base (2×2), lora_a (r=2, in=2), lora_b (out=2, r=2) — flat row-major
    let base = vec![1.0_f32, 1.0, 1.0, 1.0];
    let a = vec![1.0_f32, 0.0, 0.0, 1.0]; // (r, in)
    let b = vec![2.0_f32, 0.0, 0.0, 2.0]; // (out, r)
    let merged = merge_linear(&base, &a, &b, 2, 2, 2, 0.5);
    assert_eq!(merged, vec![2.0_f32, 1.0, 1.0, 2.0]);
}

fn temp_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    p.push(format!(
        "tpt_lora_{}_{}_{}.safetensors",
        tag,
        std::process::id(),
        n
    ));
    p
}

/// Write bytes to a fresh temp file and open it as a safetensors file.
fn write_and_open(tag: &str, bytes: &[u8]) -> (std::path::PathBuf, SafetensorsFile) {
    let path = temp_path(tag);
    std::fs::write(&path, bytes).unwrap();
    let file = SafetensorsFile::open(&path).unwrap();
    (path, file)
}

#[test]
fn merge_lora_folds_adapter_and_copies_rest() {
    let mut base_b = SafetensorsBuilder::new();
    base_b
        .add_f32("w.weight", vec![2, 2], vec![1.0f32, 1.0, 1.0, 1.0])
        .unwrap();
    base_b.add_f32("bias", vec![2], vec![0.0f32, 0.0]).unwrap();
    let base_bytes = base_b.build().unwrap();

    let mut lora_b = SafetensorsBuilder::new();
    lora_b
        .add_f32("w.lora_A.weight", vec![2, 2], vec![1.0f32, 0.0, 0.0, 1.0])
        .unwrap();
    lora_b
        .add_f32("w.lora_B.weight", vec![2, 2], vec![2.0f32, 0.0, 0.0, 2.0])
        .unwrap();
    let lora_bytes = lora_b.build().unwrap();

    let base_path = temp_path("base");
    let lora_path = temp_path("lora");
    let out_path = temp_path("out");
    std::fs::write(&base_path, &base_bytes).unwrap();
    std::fs::write(&lora_path, &lora_bytes).unwrap();

    let base = SafetensorsFile::open(&base_path).unwrap();
    let lora = SafetensorsFile::open(&lora_path).unwrap();
    let merged = merge_lora(&base, &lora, 0.5).unwrap();
    merged.write_to_file(&out_path).unwrap();

    let out = SafetensorsFile::open(&out_path).unwrap();
    let w = out.get_tensor("w.weight").unwrap();
    assert_eq!(w.to_f32().unwrap(), vec![2.0f32, 1.0, 1.0, 2.0]);
    // Un-adapted tensor is copied through unchanged.
    let bias = out.get_tensor("bias").unwrap();
    assert_eq!(bias.to_f32().unwrap(), vec![0.0f32, 0.0]);

    std::fs::remove_file(&base_path).ok();
    std::fs::remove_file(&lora_path).ok();
    std::fs::remove_file(&out_path).ok();
}

#[test]
fn partial_adapter_pair_errors() {
    let mut base_b = SafetensorsBuilder::new();
    base_b
        .add_f32("w.weight", vec![2, 2], vec![1.0f32; 4])
        .unwrap();
    let (base_path, base) = write_and_open("pp_base", &base_b.build().unwrap());

    let mut lora_b = SafetensorsBuilder::new();
    // Only the A side is present.
    lora_b
        .add_f32("w.lora_A.weight", vec![2, 2], vec![1.0f32, 0.0, 0.0, 1.0])
        .unwrap();
    let (lora_path, lora) = write_and_open("pp_lora", &lora_b.build().unwrap());

    let err = merge_lora(&base, &lora, 1.0).unwrap_err();
    assert!(
        matches!(err, MergeError::PartialAdapterPair { .. }),
        "{err:?}"
    );

    std::fs::remove_file(&base_path).ok();
    std::fs::remove_file(&lora_path).ok();
}

#[test]
fn unused_adapter_tensors_error() {
    let mut base_b = SafetensorsBuilder::new();
    base_b
        .add_f32("w.weight", vec![2, 2], vec![1.0f32; 4])
        .unwrap();
    let (base_path, base) = write_and_open("uu_base", &base_b.build().unwrap());

    // Adapter targets a module ("other") that does not exist in the base.
    let mut lora_b = SafetensorsBuilder::new();
    lora_b
        .add_f32(
            "other.lora_A.weight",
            vec![2, 2],
            vec![1.0f32, 0.0, 0.0, 1.0],
        )
        .unwrap();
    lora_b
        .add_f32(
            "other.lora_B.weight",
            vec![2, 2],
            vec![2.0f32, 0.0, 0.0, 2.0],
        )
        .unwrap();
    let (lora_path, lora) = write_and_open("uu_lora", &lora_b.build().unwrap());

    let err = merge_lora(&base, &lora, 1.0).unwrap_err();
    assert!(
        matches!(err, MergeError::UnusedAdapterTensors(_)),
        "{err:?}"
    );

    std::fs::remove_file(&base_path).ok();
    std::fs::remove_file(&lora_path).ok();
}

#[test]
fn shape_mismatch_errors() {
    let mut base_b = SafetensorsBuilder::new();
    base_b
        .add_f32("w.weight", vec![2, 2], vec![1.0f32; 4])
        .unwrap();
    let (base_path, base) = write_and_open("sm_base", &base_b.build().unwrap());

    // A has in-dim 3 but the base in-dim is 2.
    let mut lora_b = SafetensorsBuilder::new();
    lora_b
        .add_f32("w.lora_A.weight", vec![2, 3], vec![0.0f32; 6])
        .unwrap();
    lora_b
        .add_f32("w.lora_B.weight", vec![2, 2], vec![0.0f32; 4])
        .unwrap();
    let (lora_path, lora) = write_and_open("sm_lora", &lora_b.build().unwrap());

    let err = merge_lora(&base, &lora, 1.0).unwrap_err();
    assert!(matches!(err, MergeError::Shape(_)), "{err:?}");

    std::fs::remove_file(&base_path).ok();
    std::fs::remove_file(&lora_path).ok();
}

#[test]
fn metadata_is_preserved() {
    let mut base_b = SafetensorsBuilder::new();
    base_b.add_metadata("format", serde_json::json!("pt"));
    base_b.add_metadata("license", serde_json::json!("apache-2.0"));
    base_b
        .add_f32("w.weight", vec![2, 2], vec![1.0f32; 4])
        .unwrap();
    let (base_path, base) = write_and_open("md_base", &base_b.build().unwrap());

    let mut lora_b = SafetensorsBuilder::new();
    lora_b
        .add_f32("w.lora_A.weight", vec![2, 2], vec![1.0f32, 0.0, 0.0, 1.0])
        .unwrap();
    lora_b
        .add_f32("w.lora_B.weight", vec![2, 2], vec![2.0f32, 0.0, 0.0, 2.0])
        .unwrap();
    let (lora_path, lora) = write_and_open("md_lora", &lora_b.build().unwrap());

    let merged = merge_lora(&base, &lora, 0.5).unwrap();
    let md = merged.metadata();
    assert_eq!(md.get("format").and_then(|v| v.as_str()), Some("pt"));
    assert_eq!(
        md.get("license").and_then(|v| v.as_str()),
        Some("apache-2.0")
    );
    assert_eq!(
        md.get("merged_by").and_then(|v| v.as_str()),
        Some("tpt-lora-merge")
    );

    std::fs::remove_file(&base_path).ok();
    std::fs::remove_file(&lora_path).ok();
}

#[test]
fn multiple_modules_and_non_2d_copy_through() {
    let mut base_b = SafetensorsBuilder::new();
    base_b
        .add_f32("a.weight", vec![2, 2], vec![1.0f32; 4])
        .unwrap();
    base_b
        .add_f32("b.weight", vec![2, 2], vec![1.0f32; 4])
        .unwrap();
    base_b
        .add_f32("norm.weight", vec![2], vec![5.0f32, 6.0])
        .unwrap();
    let (base_path, base) = write_and_open("mm_base", &base_b.build().unwrap());

    let mut lora_b = SafetensorsBuilder::new();
    for m in ["a", "b"] {
        lora_b
            .add_f32(
                format!("{m}.lora_A.weight"),
                vec![2, 2],
                vec![1.0f32, 0.0, 0.0, 1.0],
            )
            .unwrap();
        lora_b
            .add_f32(
                format!("{m}.lora_B.weight"),
                vec![2, 2],
                vec![2.0f32, 0.0, 0.0, 2.0],
            )
            .unwrap();
    }
    let (lora_path, lora) = write_and_open("mm_lora", &lora_b.build().unwrap());

    let merged = merge_lora(&base, &lora, 0.5).unwrap();
    assert_eq!(merged.merged_modules().len(), 2);
    let out_path = temp_path("mm_out");
    merged.write_to_file(&out_path).unwrap();

    let out = SafetensorsFile::open(&out_path).unwrap();
    assert_eq!(
        out.get_tensor("a.weight").unwrap().to_f32().unwrap(),
        vec![2.0, 1.0, 1.0, 2.0]
    );
    assert_eq!(
        out.get_tensor("b.weight").unwrap().to_f32().unwrap(),
        vec![2.0, 1.0, 1.0, 2.0]
    );
    // 1-D norm copied through unchanged.
    assert_eq!(
        out.get_tensor("norm.weight").unwrap().to_f32().unwrap(),
        vec![5.0, 6.0]
    );

    std::fs::remove_file(&base_path).ok();
    std::fs::remove_file(&lora_path).ok();
    std::fs::remove_file(&out_path).ok();
}

#[test]
fn weighted_sum_of_two_adapters() {
    let mut base_b = SafetensorsBuilder::new();
    base_b
        .add_f32("w.weight", vec![2, 2], vec![0.0f32; 4])
        .unwrap();
    let (base_path, base) = write_and_open("ws_base", &base_b.build().unwrap());

    // Each adapter contributes B@A = identity.
    let make_lora = |tag: &str| {
        let mut b = SafetensorsBuilder::new();
        b.add_f32("w.lora_A.weight", vec![2, 2], vec![1.0f32, 0.0, 0.0, 1.0])
            .unwrap();
        b.add_f32("w.lora_B.weight", vec![2, 2], vec![1.0f32, 0.0, 0.0, 1.0])
            .unwrap();
        write_and_open(tag, &b.build().unwrap())
    };
    let (l1_path, l1) = make_lora("ws_l1");
    let (l2_path, l2) = make_lora("ws_l2");

    // 1.0 * I + 2.0 * I = 3 on the diagonal.
    let merged = merge_loras(&base, &[(&l1, 1.0), (&l2, 2.0)]).unwrap();
    let out_path = temp_path("ws_out");
    merged.write_to_file(&out_path).unwrap();
    let out = SafetensorsFile::open(&out_path).unwrap();
    assert_eq!(
        out.get_tensor("w.weight").unwrap().to_f32().unwrap(),
        vec![3.0, 0.0, 0.0, 3.0]
    );

    std::fs::remove_file(&base_path).ok();
    std::fs::remove_file(&l1_path).ok();
    std::fs::remove_file(&l2_path).ok();
    std::fs::remove_file(&out_path).ok();
}

#[test]
fn preserves_bf16_base_dtype() {
    // BF16 encoding of 1.0 is 0x3f80 => little-endian [0x80, 0x3f].
    let ones_bf16: Vec<u8> = std::iter::repeat_n([0x80u8, 0x3f], 4).flatten().collect();
    let mut base_b = SafetensorsBuilder::new();
    base_b
        .add_tensor("w.weight", Dtype::BF16, vec![2, 2], ones_bf16)
        .unwrap();
    let (base_path, base) = write_and_open("bf_base", &base_b.build().unwrap());

    let mut lora_b = SafetensorsBuilder::new();
    lora_b
        .add_f32("w.lora_A.weight", vec![2, 2], vec![1.0f32, 0.0, 0.0, 1.0])
        .unwrap();
    lora_b
        .add_f32("w.lora_B.weight", vec![2, 2], vec![2.0f32, 0.0, 0.0, 2.0])
        .unwrap();
    let (lora_path, lora) = write_and_open("bf_lora", &lora_b.build().unwrap());

    let merged = merge_lora(&base, &lora, 0.5).unwrap();
    let out_path = temp_path("bf_out");
    merged.write_to_file(&out_path).unwrap();

    let out = SafetensorsFile::open(&out_path).unwrap();
    let w = out.get_tensor("w.weight").unwrap();
    assert_eq!(
        w.dtype,
        Dtype::BF16,
        "output should keep the base BF16 dtype"
    );
    assert_eq!(w.to_f32().unwrap(), vec![2.0, 1.0, 1.0, 2.0]);

    std::fs::remove_file(&base_path).ok();
    std::fs::remove_file(&lora_path).ok();
    std::fs::remove_file(&out_path).ok();
}

#[test]
fn kohya_naming_with_alpha() {
    let mut base_b = SafetensorsBuilder::new();
    base_b
        .add_f32("w.weight", vec![2, 2], vec![1.0f32; 4])
        .unwrap();
    let (base_path, base) = write_and_open("ko_base", &base_b.build().unwrap());

    // Kohya-style down/up + a per-layer alpha tensor. alpha=4, r=2 => x2 scale.
    let mut lora_b = SafetensorsBuilder::new();
    lora_b
        .add_f32(
            "w.lora_down.weight",
            vec![2, 2],
            vec![1.0f32, 0.0, 0.0, 1.0],
        )
        .unwrap();
    lora_b
        .add_f32("w.lora_up.weight", vec![2, 2], vec![1.0f32, 0.0, 0.0, 1.0])
        .unwrap();
    lora_b.add_f32("w.alpha", vec![1], vec![4.0f32]).unwrap();
    let (lora_path, lora) = write_and_open("ko_lora", &lora_b.build().unwrap());

    // user scale 1.0 * (alpha/r = 2.0) * (B@A = I) added to ones => diag 3.
    let merged = merge_lora(&base, &lora, 1.0).unwrap();
    let out_path = temp_path("ko_out");
    merged.write_to_file(&out_path).unwrap();
    let out = SafetensorsFile::open(&out_path).unwrap();
    assert_eq!(
        out.get_tensor("w.weight").unwrap().to_f32().unwrap(),
        vec![3.0, 1.0, 1.0, 3.0]
    );

    std::fs::remove_file(&base_path).ok();
    std::fs::remove_file(&lora_path).ok();
    std::fs::remove_file(&out_path).ok();
}
