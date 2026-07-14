# tpt-wasm-demo

A tiny **browser playground** for two of the pure-Rust crates in this workspace:

- [`tpt-jinja-chat`](../tpt-jinja-chat) — render LLM chat templates.
- [`tpt-tokenizer-core`](../tpt-tokenizer-core) — BPE / WordPiece tokenization.

Both crates are pure Rust with zero/minimal dependencies and compile cleanly to
WebAssembly, so the demo doubles as a "zero-dependency" proof point: everything
runs client-side, with no server and no network.

> This crate is `publish = false` and is **excluded** from the main Cargo
> workspace (it targets `wasm32` and pulls in `wasm-bindgen`). Build it on its
> own with the commands below.

## Build

Install the toolchain once:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

Then build the `.wasm` + JS glue into `pkg/`:

```sh
wasm-pack build crates/wasm-demo --target web
```

## Run

Serve the crate directory (the page loads `./pkg/tpt_wasm_demo.js` as an ES
module, so it must be served over HTTP, not opened as a `file://` URL):

```sh
# from crates/wasm-demo, after wasm-pack build
python -m http.server 8080
# then open http://localhost:8080/
```

The page exposes:

- a **chat-template renderer** (template + JSON context → rendered string), and
- a **BPE tokenizer** panel (text + vocab + merges → token ids → decoded text),
  with an optional byte-level toggle.

## Exposed functions

The WASM module (`src/lib.rs`) exports:

| function | purpose |
| --- | --- |
| `render_chat_template(template, context_json)` | render a Jinja chat template |
| `bpe_encode(text, vocab_json, merges_text, byte_level)` | BPE encode → `Uint32Array` |
| `bpe_decode(ids, vocab_json, merges_text, byte_level)` | BPE decode |
| `wordpiece_encode(text, vocab_json, unk_token, lowercase)` | WordPiece encode |
