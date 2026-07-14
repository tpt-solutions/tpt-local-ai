# tpt-jinja-chat

A **pure-Rust, zero-dependency** implementation of the small slice of the
[Jinja2](https://jinja.palletsprojects.com/) templating language used by Hugging
Face `tokenizer_config.json` `chat_template` strings.

It parses and renders LLM chat templates — turning a list of messages into the
exact prompt text that a model's tokenizer expects. No `thiserror`, no
`serde`, no `nom`: a hand-rolled lexer, a recursive-descent parser, and a
manual error type.

## Supported syntax

| Feature | Example |
| --- | --- |
| Variable output | `{{ messages }}` |
| Set | `{% set x = 1 %}` |
| For loop | `{% for m in messages %} … {% endfor %}` |
| Conditionals | `{% if %} / {% elif %} / {% else %} / {% endif %}` |
| Whitespace control | `{{-`, `{%-`, `-}}`, `-%}` |
| Member access | `message['content']`, `message.role` |
| Indexing | `messages[0]` |
| Operators | `+ - * / == != < > <= >= and or not` |
| `loop` variable | `loop.index`, `loop.index0`, `loop.first`, `loop.last`, `loop.length` |

## Usage

```rust
use tpt_jinja_chat::{ChatTemplate, Context, Value};

let tmpl = ChatTemplate::parse(
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}",
)?;

let mut ctx = Context::new();
ctx.insert(
    "messages",
    Value::Array(vec![
        Value::Object(
            [
                ("role".into(), Value::String("user".into())),
                ("content".into(), Value::String("Hello".into())),
            ]
            .into(),
        ),
        Value::Object(
            [
                ("role".into(), Value::String("assistant".into())),
                ("content".into(), Value::String("Hi!".into())),
            ]
            .into(),
        ),
    ]),
);

assert_eq!(tmpl.render(&ctx)?, "user: Hello\nassistant: Hi!\n");
```

### Loading a context from JSON

A tiny, dependency-free JSON parser is bundled so you can build a `Context`
directly from a JSON string (for example a chat-completion request body):

```rust
use tpt_jinja_chat::Context;

let ctx = Context::from_json_str(r#"{ "name": "world" }"#)?;
```

### Rendering a Llama 3 template

See `examples/render_llama3_template.rs` for a complete, runnable example:

```sh
cargo run -p tpt-jinja-chat --example render_llama3_template
```

## Why zero dependencies?

This crate is one of five `tpt-*` crates aimed at clean `crates.io` releases for
local-AI tooling. Keeping `tpt-jinja-chat` dependency-free makes it trivial to
vendor or audit, and guarantees no transitive C/C++ bindings.

## License

Licensed under either of MIT or Apache-2.0 at your option.
