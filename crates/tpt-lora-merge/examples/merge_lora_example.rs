//! Demonstrates merging a small LoRA adapter into a base weight in-memory.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p tpt-lora-merge --example merge_lora_example
//! ```

use ndarray::array;
use tpt_lora_merge::merge_linear;

fn main() {
    // Base linear weight, shape (out=2, in=2).
    let base = array![[1.0_f32, 1.0], [1.0, 1.0]];
    // LoRA down-projection A, shape (r=2, in=2).
    let lora_a = array![[1.0_f32, 0.0], [0.0, 1.0]];
    // LoRA up-projection B, shape (out=2, r=2).
    let lora_b = array![[2.0_f32, 0.0], [0.0, 2.0]];

    let scale = 1.0; // equivalent to alpha / r = 1.0
    let merged = merge_linear(base.view(), lora_a.view(), lora_b.view(), scale);

    println!("base:\n{base}");
    println!(
        "lora delta = scale * (B @ A):\n{}",
        &lora_b.dot(&lora_a) * scale
    );
    println!("merged = base + delta:\n{merged}");
}
