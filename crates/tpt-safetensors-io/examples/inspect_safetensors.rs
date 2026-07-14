//! Inspects a safetensors file, printing each tensor's name, dtype, shape, and
//! byte length.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p tpt-safetensors-io --example inspect_safetensors -- model.safetensors
//! ```

use std::process::exit;

use tpt_safetensors_io::SafetensorsFile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: inspect_safetensors <path.safetensors>");
        exit(2);
    }
    let path = &args[1];

    let file = match SafetensorsFile::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error opening {path}: {e}");
            exit(1);
        }
    };

    println!("file:    {path}");
    println!("tensors: {}", file.len());
    if !file.metadata().is_empty() {
        println!("metadata: {:?}", file.metadata());
    }
    for name in file.tensor_names() {
        let view = file.get_tensor(name).expect("tensor exists");
        println!(
            "- {name}: dtype={:?} shape={:?} bytes={}",
            view.dtype,
            view.shape,
            view.data.len()
        );
    }
}
