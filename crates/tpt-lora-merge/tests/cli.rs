//! Integration test: runs the compiled `tpt-lora-merge` CLI end-to-end.

use std::process::Command;

use tpt_safetensors_io::{SafetensorsBuilder, SafetensorsFile};

fn temp_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "tpt_lora_cli_{}_{}.safetensors",
        tag,
        std::process::id()
    ));
    p
}

#[test]
fn cli_produces_merged_file() {
    let mut base_b = SafetensorsBuilder::new();
    base_b
        .add_f32("w.weight", vec![2, 2], vec![1.0f32, 1.0, 1.0, 1.0])
        .unwrap();
    let base_bytes = base_b.build().unwrap();

    let mut lora_b = SafetensorsBuilder::new();
    lora_b
        .add_f32("w.lora_A.weight", vec![2, 2], vec![1.0f32, 0.0, 0.0, 1.0])
        .unwrap();
    lora_b
        .add_f32("w.lora_B.weight", vec![2, 2], vec![2.0f32, 0.0, 0.0, 2.0])
        .unwrap();
    let lora_bytes = lora_b.build().unwrap();

    let base_path = temp_path("cli_base");
    let lora_path = temp_path("cli_lora");
    let out_path = temp_path("cli_out");
    std::fs::write(&base_path, &base_bytes).unwrap();
    std::fs::write(&lora_path, &lora_bytes).unwrap();

    let exe = std::env::current_exe().expect("current_exe");
    let bin_name = if cfg!(windows) {
        "tpt-lora-merge.exe"
    } else {
        "tpt-lora-merge"
    };
    // Integration tests run from `target/<profile>/deps/`; the binary lives in
    // `target/<profile>/`.
    let bin = exe
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join(bin_name))
        .expect("could not locate tpt-lora-merge binary");
    let status = Command::new(&bin)
        .arg("--base")
        .arg(&base_path)
        .arg("--lora")
        .arg(&lora_path)
        .arg("--output")
        .arg(&out_path)
        .arg("--scale")
        .arg("0.5")
        .status()
        .expect("failed to launch tpt-lora-merge");

    assert!(status.success(), "CLI exited with {status}");

    let out = SafetensorsFile::open(&out_path).unwrap();
    let w = out.get_tensor("w.weight").unwrap();
    assert_eq!(w.to_f32().unwrap(), vec![2.0f32, 1.0, 1.0, 2.0]);

    std::fs::remove_file(&base_path).ok();
    std::fs::remove_file(&lora_path).ok();
    std::fs::remove_file(&out_path).ok();
}
