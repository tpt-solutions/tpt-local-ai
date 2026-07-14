//! CLI entry point for `tpt-lora-merge`.

use std::path::PathBuf;
use std::process::exit;

use clap::Parser;
use tpt_lora_merge::{merge_lora, MergeError};
use tpt_safetensors_io::SafetensorsFile;

/// Merge a LoRA adapter into a base model (CPU only).
#[derive(Parser, Debug)]
#[command(name = "tpt-lora-merge", version, about)]
struct Args {
    /// Path to the base model safetensors file.
    #[arg(long, value_name = "PATH")]
    base: PathBuf,

    /// Path to the LoRA adapter safetensors file.
    #[arg(long, value_name = "PATH")]
    lora: PathBuf,

    /// Path to write the merged safetensors file to.
    #[arg(long, value_name = "PATH")]
    output: PathBuf,

    /// Blend factor applied to the LoRA delta (folds in `alpha / r`).
    #[arg(long, default_value_t = 1.0)]
    scale: f32,
}

fn main() {
    let args = Args::parse();

    let base = match SafetensorsFile::open(&args.base) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error opening base {}: {e}", args.base.display());
            exit(1);
        }
    };
    let lora = match SafetensorsFile::open(&args.lora) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error opening lora {}: {e}", args.lora.display());
            exit(1);
        }
    };

    let merged = match merge_lora(&base, &lora, args.scale) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("merge failed: {e}");
            exit(1);
        }
    };

    if let Err(e) = merged.write_to_file(&args.output) {
        eprintln!("error writing {}: {e}", args.output.display());
        exit(1);
    }

    if let Err(e) = report(&args.output) {
        eprintln!("warning: could not verify output: {e}");
    }
    println!(
        "merged {} base tensors with scale {} -> {}",
        base.len(),
        args.scale,
        args.output.display()
    );
}

fn report(path: &std::path::Path) -> Result<(), MergeError> {
    let f = SafetensorsFile::open(path)?;
    println!("wrote {} tensors:", f.len());
    for name in f.tensor_names() {
        let view = f.get_tensor(name).expect("tensor exists");
        println!(
            "  - {name}: shape={:?} dtype={:?} bytes={}",
            view.shape,
            view.dtype,
            view.data.len()
        );
    }
    Ok(())
}
