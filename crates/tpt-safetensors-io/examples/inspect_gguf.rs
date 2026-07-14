//! Inspects a GGUF file, printing its version, metadata, and tensor list.
//!
//! Requires the `gguf` feature. Run with:
//!
//! ```sh
//! cargo run -p tpt-safetensors-io --features gguf --example inspect_gguf -- model.gguf
//! ```

use std::process::exit;

use tpt_safetensors_io::gguf::{GgufFile, GgufValue};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: inspect_gguf <path.gguf>");
        exit(2);
    }
    let path = &args[1];

    let file = match GgufFile::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error opening {path}: {e}");
            exit(1);
        }
    };

    println!("file:      {path}");
    println!("version:   {}", file.version());
    println!("alignment: {}", file.alignment());
    println!("tensors:   {}", file.len());

    println!("\nmetadata ({}):", file.metadata().len());
    for (key, value) in file.metadata() {
        println!("- {key} = {}", summarize(value));
    }

    println!("\ntensors:");
    for t in file.tensors() {
        println!(
            "- {}: type={:?} dims={:?} offset={}",
            t.name, t.ggml_type, t.dimensions, t.offset
        );
    }
}

/// Renders a value compactly, truncating large arrays.
fn summarize(v: &GgufValue) -> String {
    match v {
        GgufValue::String(s) => format!("{s:?}"),
        GgufValue::Array(items) => {
            let shown = items.len().min(4);
            let preview: Vec<String> = items[..shown].iter().map(summarize).collect();
            if items.len() > shown {
                format!("[{}, … {} total]", preview.join(", "), items.len())
            } else {
                format!("[{}]", preview.join(", "))
            }
        }
        other => format!("{other:?}"),
    }
}
