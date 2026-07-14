# tpt-safetensors-io

[![crates.io](https://img.shields.io/crates/v/tpt-safetensors-io.svg)](https://crates.io/crates/tpt-safetensors-io)
[![docs.rs](https://img.shields.io/docsrs/tpt-safetensors-io)](https://docs.rs/tpt-safetensors-io)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Memory-mapped reader and writer for the
[safetensors](https://github.com/huggingface/safetensors) tensor serialisation
format, written in pure Rust.

- **Zero-copy reads** — tensors are borrowed straight out of a `memmap2`
  mapping; no allocation or copy is made unless you ask for one.
- **Spec-compliant writes** — [`SafetensorsBuilder`] produces files with the
  mandatory 8-byte-aligned header and correct `data_offsets`.
- Reads F16 / BF16 / F32 / F64 and the common integer dtypes, with a
  convenience `TensorView::to_f32` for numeric work.

## Usage

```rust
use tpt_safetensors_io::{SafetensorsBuilder, Dtype};

// Write a tiny 2x2 matrix.
let mut builder = SafetensorsBuilder::new();
builder
    .add_f32("w", vec![2, 2], vec![1.0f32, 2.0, 3.0, 4.0])
    .unwrap();
let bytes = builder.build().unwrap();
assert!(!bytes.is_empty());
```

Inspecting an existing file:

```sh
cargo run -p tpt-safetensors-io --example inspect_safetensors -- model.safetensors
```

## GGUF metadata (optional `gguf` feature)

The dominant local-inference format (`llama.cpp`/`ggml`) is GGUF rather than
safetensors. Enable the `gguf` feature for a read-only parser of GGUF headers —
metadata key/value pairs (architecture, hyper-parameters, tokenizer vocab, chat
template, quantization info) and per-tensor descriptors (name, shape, ggml type,
offset). Quantized tensor payloads are not decoded; this is a metadata/inspection
reader.

```toml
[dependencies]
tpt-safetensors-io = { version = "0.1", features = ["gguf"] }
```

```rust,no_run
use tpt_safetensors_io::gguf::GgufFile;

let f = GgufFile::open("model.gguf")?;
println!("arch: {:?}", f.architecture());
for t in f.tensors() {
    println!("{} {:?} {:?}", t.name, t.ggml_type, t.dimensions);
}
# Ok::<(), tpt_safetensors_io::gguf::GgufError>(())
```

```sh
cargo run -p tpt-safetensors-io --features gguf --example inspect_gguf -- model.gguf
```

Only little-endian GGUF versions 2 and 3 are supported (this covers essentially
every GGUF file produced by mainstream tooling).

## License

Licensed under either of MIT or Apache-2.0 at your option.
