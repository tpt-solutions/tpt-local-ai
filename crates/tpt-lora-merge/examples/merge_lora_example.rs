//! Demonstrates merging a small LoRA adapter into a base weight in-memory.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p tpt-lora-merge --example merge_lora_example
//! ```

use tpt_lora_merge::merge_linear;

fn main() {
    // Base linear weight, shape (out=2, in=2) — flat row-major.
    let base = vec![1.0_f32, 1.0, 1.0, 1.0];
    // LoRA down-projection A, shape (r=2, in=2).
    let lora_a = vec![1.0_f32, 0.0, 0.0, 1.0];
    // LoRA up-projection B, shape (out=2, r=2).
    let lora_b = vec![2.0_f32, 0.0, 0.0, 2.0];

    let scale = 1.0_f32; // equivalent to alpha / r = 1.0
    let merged = merge_linear(&base, &lora_a, &lora_b, 2, 2, 2, scale);

    println!("base:   {:?}", base);
    println!("merged: {:?}", merged);
}
