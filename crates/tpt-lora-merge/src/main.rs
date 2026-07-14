//! CLI entry point for `tpt-lora-merge`.

use std::path::{Path, PathBuf};
use std::process::exit;

use clap::Parser;
use tpt_lora_merge::{merge_loras, MergeError, MergedWeights};
use tpt_safetensors_io::SafetensorsFile;

/// Merge one or more LoRA adapters into a base model (CPU only).
#[derive(Parser, Debug)]
#[command(name = "tpt-lora-merge", version, about)]
struct Args {
    /// Path to the base model safetensors file.
    #[arg(long, value_name = "PATH")]
    base: PathBuf,

    /// Path to a LoRA adapter safetensors file. May be repeated to merge
    /// several adapters as a weighted sum.
    #[arg(long = "lora", value_name = "PATH", required = true)]
    lora: Vec<PathBuf>,

    /// Path to write the merged safetensors file to.
    #[arg(long, value_name = "PATH")]
    output: PathBuf,

    /// Blend factor applied to a LoRA delta (folds in `alpha / r`). Provide once
    /// to apply to every adapter, or once per `--lora` in matching order. If
    /// omitted, the scale is derived from each adapter's `adapter_config.json`
    /// (`alpha / r`) when available, otherwise defaults to 1.0.
    #[arg(long)]
    scale: Vec<f32>,

    /// Validate base/adapter alignment and report what would be merged without
    /// writing any output file.
    #[arg(long)]
    dry_run: bool,
}

fn main() {
    let args = Args::parse();

    // Refuse to overwrite any input: `write_to_file` truncates the target, so
    // pointing `--output` at `--base`/`--lora` would destroy the source.
    if !args.dry_run && (args.output == args.base || args.lora.contains(&args.output)) {
        eprintln!(
            "error: --output ({}) must not overwrite --base or any --lora",
            args.output.display()
        );
        exit(1);
    }

    // Resolve a scale for each adapter.
    let scales = match resolve_scales(&args) {
        Ok(s) => s,
        Err(msg) => {
            eprintln!("error: {msg}");
            exit(1);
        }
    };

    let base = open_or_exit(&args.base, "base");
    let loras: Vec<SafetensorsFile> = args.lora.iter().map(|p| open_or_exit(p, "lora")).collect();

    let adapters: Vec<(&SafetensorsFile, f32)> = loras.iter().zip(scales.iter().copied()).collect();

    let merged = match merge_loras(&base, &adapters) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("merge failed: {e}");
            exit(1);
        }
    };

    if args.dry_run {
        print_dry_run(&base, &args, &scales, &merged);
        return;
    }

    if let Err(e) = merged.write_to_file(&args.output) {
        eprintln!("error writing {}: {e}", args.output.display());
        exit(1);
    }

    if let Err(e) = report(&args.output) {
        eprintln!("warning: could not verify output: {e}");
    }
    println!(
        "merged {} adapter(s) into {} base tensors ({} modules adapted) -> {}",
        adapters.len(),
        base.len(),
        merged.merged_modules().len(),
        args.output.display()
    );
}

fn resolve_scales(args: &Args) -> Result<Vec<f32>, String> {
    let n = args.lora.len();
    match args.scale.len() {
        // No explicit scale: derive from each adapter_config.json, else 1.0.
        0 => Ok(args
            .lora
            .iter()
            .map(|p| scale_from_adapter_config(p).unwrap_or(1.0))
            .collect()),
        // One scale: apply to every adapter.
        1 => Ok(vec![args.scale[0]; n]),
        // One scale per adapter.
        m if m == n => Ok(args.scale.clone()),
        m => Err(format!(
            "got {m} --scale values for {n} --lora adapters; provide 0, 1, or {n}"
        )),
    }
}

/// Reads `adapter_config.json` next to the adapter file and derives `alpha / r`.
fn scale_from_adapter_config(lora_path: &Path) -> Option<f32> {
    let cfg = lora_path.parent()?.join("adapter_config.json");
    let text = std::fs::read_to_string(cfg).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let alpha = v.get("lora_alpha").and_then(serde_json::Value::as_f64)?;
    let r = v.get("r").and_then(serde_json::Value::as_f64)?;
    if r == 0.0 {
        return None;
    }
    Some((alpha / r) as f32)
}

fn open_or_exit(path: &Path, label: &str) -> SafetensorsFile {
    match SafetensorsFile::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error opening {label} {}: {e}", path.display());
            exit(1);
        }
    }
}

fn print_dry_run(base: &SafetensorsFile, args: &Args, scales: &[f32], merged: &MergedWeights) {
    println!("dry run: no output written");
    println!("base: {} ({} tensors)", args.base.display(), base.len());
    for (path, scale) in args.lora.iter().zip(scales) {
        println!("  adapter: {} (scale {scale})", path.display());
    }
    println!(
        "modules that would be adapted: {}",
        merged.merged_modules().len()
    );
    for m in merged.merged_modules() {
        println!("  - {m}");
    }
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
