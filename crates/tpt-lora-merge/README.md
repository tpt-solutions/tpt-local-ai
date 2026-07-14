# tpt-lora-merge

CPU-based merging of [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank
Adaptation) adapters into base-model weights, reading and writing the
[safetensors](https://github.com/huggingface/safetensors) format. No PyTorch,
no GPU required.

For each base weight `<module>.weight` of shape `(out, in)` the tool folds in
the adapter pair `<module>.lora_A.weight` `(r, in)` and
`<module>.lora_B.weight` `(out, r)`:

```text
merged = base + scale * (B @ A)
```

where `scale` folds in the usual `(alpha / r)` factor (and any extra blending
you want). Tensors without a matching adapter are copied through unchanged.

## Library

```rust
use ndarray::array;
use tpt_lora_merge::merge_linear;

let base = array![[1.0_f32, 1.0], [1.0, 1.0]];
let a = array![[1.0_f32, 0.0], [0.0, 1.0]];
let b = array![[2.0_f32, 0.0], [0.0, 2.0]];
let merged = merge_linear(base.view(), a.view(), b.view(), 0.5);
assert_eq!(merged, array![[2.0, 1.0], [1.0, 2.0]]);
```

## CLI

```sh
cargo run -p tpt-lora-merge -- \
  --base model.safetensors \
  --lora adapter.safetensors \
  --output merged.safetensors \
  --scale 1.0
```

## License

Licensed under either of MIT or Apache-2.0 at your option.
