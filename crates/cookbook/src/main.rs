//! End-to-end "cookbook" that chains all five `tpt-local-ai` crates together.
//!
//! It walks the full local-AI plumbing pipeline:
//!
//! 1. [`tpt_hf_hub`] — resolve a Hub cache and (optionally) download real files.
//! 2. [`tpt_safetensors_io`] — write a tiny base checkpoint + a LoRA adapter.
//! 3. [`tpt_lora_merge`] — merge the adapter into the base weights.
//! 4. [`tpt_jinja_chat`] — render a chat template into a prompt string.
//! 5. [`tpt_tokenizer_core`] — tokenize the rendered prompt (and round-trip it).
//!
//! It is runnable **offline** by default: the download step is skipped unless
//! you point it at a repo/file:
//!
//! ```sh
//! cargo run -p tpt-cookbook
//! # or exercise a real Hub download too:
//! TPT_COOKBOOK_HUB_REPO=gpt2 TPT_COOKBOOK_HUB_FILE=config.json \
//!     cargo run -p tpt-cookbook
//! ```

use std::collections::BTreeMap;
use std::error::Error;

use tpt_hf_hub::{HubClient, NoopProgressReporter};
use tpt_jinja_chat::{ChatTemplate, Context, Value};
use tpt_lora_merge::merge_lora;
use tpt_safetensors_io::{SafetensorsBuilder, SafetensorsFile};
use tpt_tokenizer_core::{BpeTokenizer, TokenId, Tokenizer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let workdir = std::env::temp_dir().join("tpt-cookbook");
    std::fs::create_dir_all(&workdir)?;
    println!("== tpt-local-ai cookbook ==");
    println!("scratch dir: {}\n", workdir.display());

    step_1_hub().await?;
    let merged_path = step_2_and_3_merge(&workdir)?;
    let prompt = step_4_render()?;
    step_5_tokenize(&prompt)?;

    println!(
        "\nDone. Merged checkpoint written to {}",
        merged_path.display()
    );
    Ok(())
}

/// Step 1: set up a Hub client and, if asked, download a real file.
async fn step_1_hub() -> Result<(), Box<dyn Error>> {
    println!("[1/5] tpt-hf-hub");
    let client = HubClient::new()?;
    println!("  cache dir: {}", client.cache_dir().display());

    match (
        std::env::var("TPT_COOKBOOK_HUB_REPO").ok(),
        std::env::var("TPT_COOKBOOK_HUB_FILE").ok(),
    ) {
        (Some(repo), Some(file)) => {
            println!("  downloading {repo}/{file} ...");
            let path = client
                .download_file(&repo, &file, &NoopProgressReporter)
                .await?;
            println!("  -> {}", path.display());
        }
        _ => println!(
            "  (skipping download; set TPT_COOKBOOK_HUB_REPO + TPT_COOKBOOK_HUB_FILE to fetch)"
        ),
    }
    Ok(())
}

/// Steps 2 & 3: write a base checkpoint plus a LoRA adapter, then merge them.
///
/// Returns the path to the merged safetensors file.
fn step_2_and_3_merge(workdir: &std::path::Path) -> Result<std::path::PathBuf, Box<dyn Error>> {
    println!("\n[2/5] tpt-safetensors-io (write base + adapter)");

    // A 4x3 linear layer weight (out=4, in=3).
    let base_weight: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
    let base_path = workdir.join("base.safetensors");
    let mut base = SafetensorsBuilder::new();
    base.add_f32("layer.weight", vec![4, 3], base_weight.clone())?;
    base.write_to_file(&base_path)?;
    println!("  base:  {} (layer.weight [4, 3])", base_path.display());

    // A rank-2 LoRA adapter: A is [r, in] = [2, 3], B is [out, r] = [4, 2].
    let lora_a: Vec<f32> = vec![0.1, 0.0, 0.0, 0.0, 0.1, 0.0];
    let lora_b: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let lora_path = workdir.join("adapter.safetensors");
    let mut lora = SafetensorsBuilder::new();
    lora.add_f32("layer.lora_A.weight", vec![2, 3], lora_a)?;
    lora.add_f32("layer.lora_B.weight", vec![4, 2], lora_b)?;
    lora.write_to_file(&lora_path)?;
    println!(
        "  lora:  {} (lora_A [2, 3], lora_B [4, 2])",
        lora_path.display()
    );

    println!("\n[3/5] tpt-lora-merge (base + scale * B @ A)");
    let base_file = SafetensorsFile::open(&base_path)?;
    let lora_file = SafetensorsFile::open(&lora_path)?;
    let merged = merge_lora(&base_file, &lora_file, 1.0)?;
    println!("  merged modules: {:?}", merged.merged_modules());

    let merged_path = workdir.join("merged.safetensors");
    merged.write_to_file(&merged_path)?;

    // Read the result back to prove the delta was applied.
    let merged_file = SafetensorsFile::open(&merged_path)?;
    let view = merged_file
        .get_tensor("layer.weight")
        .ok_or("merged file missing layer.weight")?;
    let merged_weight = view.to_f32()?;
    println!("  base[0..3]   = {:?}", &base_weight[0..3]);
    println!("  merged[0..3] = {:?}", &merged_weight[0..3]);

    Ok(merged_path)
}

/// Step 4: render a chat template into a prompt string.
fn step_4_render() -> Result<String, Box<dyn Error>> {
    println!("\n[4/5] tpt-jinja-chat (render chat template)");
    let template = ChatTemplate::parse(
        "{%- for message in messages %}\
         {{- '<|' + message['role'] + '|>\n' + message['content'] + '\n' }}\
         {%- endfor %}\
         {%- if add_generation_prompt %}{{- '<|assistant|>\n' }}{%- endif %}",
    )?;

    let mut ctx = Context::new();
    ctx.insert(
        "messages",
        Value::Array(vec![
            message("system", "You are a helpful assistant."),
            message("user", "Explain LoRA in one sentence."),
        ]),
    );
    ctx.insert("add_generation_prompt", Value::Bool(true));

    let prompt = template.render(&ctx)?;
    println!("  rendered prompt:\n---\n{prompt}\n---");
    Ok(prompt)
}

/// Build a `{role, content}` message object [`Value`].
fn message(role: &str, content: &str) -> Value {
    Value::Object(
        [
            ("role".to_string(), Value::String(role.to_string())),
            ("content".to_string(), Value::String(content.to_string())),
        ]
        .into(),
    )
}

/// Step 5: tokenize (and round-trip) the rendered prompt with a byte-level BPE.
fn step_5_tokenize(prompt: &str) -> Result<(), Box<dyn Error>> {
    println!("[5/5] tpt-tokenizer-core (byte-level BPE)");

    // A self-contained byte-level vocab (every byte -> its GPT-2 unicode char),
    // so encoding never fails and decode(encode(s)) == s. No merge rules here,
    // which keeps the demo dependency-free and deterministic.
    let mut vocab: BTreeMap<String, TokenId> = BTreeMap::new();
    for (byte, ch) in byte_to_unicode().into_iter().enumerate() {
        vocab.insert(ch.to_string(), byte as TokenId);
    }
    let tokenizer = BpeTokenizer::from_vocab_merges(vocab, Vec::new()).with_byte_level();

    let ids = tokenizer.encode(prompt)?;
    let roundtrip = tokenizer.decode(&ids)?;
    println!("  token count: {}", ids.len());
    println!("  first ids:   {:?}", &ids[..ids.len().min(12)]);
    println!("  round-trips: {}", roundtrip == prompt);
    Ok(())
}

/// The GPT-2 "byte-to-unicode" table (same mapping `tpt-tokenizer-core` uses
/// internally), reproduced here so the cookbook stays self-contained.
fn byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut used = [false; 256];
    for b in b'!'..=b'~' {
        table[b as usize] = b as char;
        used[b as usize] = true;
    }
    for b in 0xA1u8..=0xAC {
        table[b as usize] = b as char;
        used[b as usize] = true;
    }
    for b in 0xAEu8..=0xFF {
        table[b as usize] = b as char;
        used[b as usize] = true;
    }
    let mut n = 0u32;
    for b in 0..=255usize {
        if !used[b] {
            table[b] = char::from_u32(256 + n).expect("valid code point");
            n += 1;
        }
    }
    table
}
