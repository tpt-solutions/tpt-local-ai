//! Renders a Llama 3 chat template against a sample conversation.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p tpt-jinja-chat --example render_llama3_template
//! ```

use tpt_jinja_chat::{ChatTemplate, Context, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A faithful reproduction of the Llama 3 / 3.1 `chat_template`.
    let template = ChatTemplate::parse(
        "{%- set loop_messages = messages %}\
         {%- for message in loop_messages %}\
             {%- set content = message['content'] %}\
             {%- if message['role'] == 'system' %}\
                 {{- '<|start_header_id|>system<|end_header_id|>\n\n' + content + '<|eot_id|>' }}\
             {%- elif message['role'] == 'user' %}\
                 {{- '<|start_header_id|>user<|end_header_id|>\n\n' + content + '<|eot_id|>' }}\
             {%- elif message['role'] == 'assistant' %}\
                 {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' + content + '<|eot_id|>' }}\
             {%- endif %}\
             {%- if loop.last and add_generation_prompt %}\
                 {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}\
             {%- endif %}\
         {%- endfor %}",
    )?;

    // Build the context. `messages` typically comes from a JSON payload such as
    // the `messages` field of a chat completion request.
    let mut ctx = Context::new();
    let messages = Value::Array(vec![
        Value::Object(
            [
                ("role".into(), Value::String("system".into())),
                (
                    "content".into(),
                    Value::String("You are a helpful assistant.".into()),
                ),
            ]
            .into(),
        ),
        Value::Object(
            [
                ("role".into(), Value::String("user".into())),
                ("content".into(), Value::String("What is Rust?".into())),
            ]
            .into(),
        ),
    ]);
    ctx.insert("messages", messages);
    ctx.insert("add_generation_prompt", Value::Bool(true));

    let prompt = template.render(&ctx)?;
    print!("{prompt}");
    Ok(())
}
