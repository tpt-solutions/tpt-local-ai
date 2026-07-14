# tpt-safetensors-io

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

## License

Licensed under either of MIT or Apache-2.0 at your option.
