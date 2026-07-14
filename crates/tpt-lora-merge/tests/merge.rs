//! Unit and library tests for `tpt-lora-merge`.

use ndarray::array;
use tpt_lora_merge::{merge_linear, merge_lora};
use tpt_safetensors_io::{SafetensorsBuilder, SafetensorsFile};

#[test]
fn merge_linear_math() {
    let base = array![[1.0_f32, 1.0], [1.0, 1.0]];
    let a = array![[1.0_f32, 0.0], [0.0, 1.0]]; // (r, in)
    let b = array![[2.0_f32, 0.0], [0.0, 2.0]]; // (out, r)
    let merged = merge_linear(base.view(), a.view(), b.view(), 0.5);
    assert_eq!(merged, array![[2.0, 1.0], [1.0, 2.0]]);
}

fn temp_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "tpt_lora_{}_{}.safetensors",
        tag,
        std::process::id()
    ));
    p
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
