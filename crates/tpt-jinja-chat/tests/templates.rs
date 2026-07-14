//! Integration tests covering real-world chat templates (Llama 3, Mistral).

use tpt_jinja_chat::{ChatTemplate, Context, Value};

fn msg(role: &str, content: &str) -> Value {
    Value::Object(
        [
            ("role".into(), Value::String(role.into())),
            ("content".into(), Value::String(content.into())),
        ]
        .into(),
    )
}

#[test]
fn basic_variable_substitution() {
    let tmpl = ChatTemplate::parse("Hello {{ name }}!").unwrap();
    let mut ctx = Context::new();
    ctx.insert("name", Value::String("world".into()));
    assert_eq!(tmpl.render(&ctx).unwrap(), "Hello world!");
}

#[test]
fn if_elif_else() {
    let tmpl =
        ChatTemplate::parse("{% if x > 10 %}big{% elif x > 5 %}mid{% else %}small{% endif %}")
            .unwrap();
    let mut ctx = Context::new();
    ctx.insert("x", Value::Number(7.0));
    assert_eq!(tmpl.render(&ctx).unwrap(), "mid");
}

#[test]
fn for_loop_with_loop_variable() {
    let tmpl =
        ChatTemplate::parse("{% for m in messages %}{{ loop.index }}:{{ m['content'] }};{% endfor %}")
            .unwrap();
    let mut ctx = Context::new();
    ctx.insert(
        "messages",
        Value::Array(vec![
            msg("user", "a"),
            msg("assistant", "b"),
            msg("user", "c"),
        ]),
    );
    assert_eq!(tmpl.render(&ctx).unwrap(), "1:a;2:b;3:c;");
}

#[test]
fn string_concatenation_and_set() {
    let tmpl = ChatTemplate::parse(
        "{% set g = 'Hello, ' %}{{ g + name + '!' }}",
    )
    .unwrap();
    let mut ctx = Context::new();
    ctx.insert("name", Value::String("Ada".into()));
    assert_eq!(tmpl.render(&ctx).unwrap(), "Hello, Ada!");
}

#[test]
fn whitespace_control_trims_surrounding_text() {
    // The newlines/indentation around tags must be stripped.
    let tmpl = ChatTemplate::parse(
        "{%- if true %}\n    hello\n{%- endif %}",
    )
    .unwrap();
    let ctx = Context::new();
    assert_eq!(tmpl.render(&ctx).unwrap(), "hello");
}

#[test]
fn llama3_template() {
    // A faithful reproduction of the Llama 3 / 3.1 chat template (the
    // `{%-` / `{{-` whitespace-control markers keep the output compact).
    let tmpl = ChatTemplate::parse(
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
    )
    .unwrap();

    let mut ctx = Context::new();
    ctx.insert(
        "messages",
        Value::Array(vec![
            msg("system", "You are helpful."),
            msg("user", "Hello?"),
        ]),
    );
    ctx.insert("add_generation_prompt", Value::Bool(true));

    let out = tmpl.render(&ctx).unwrap();
    assert_eq!(
        out,
        "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.\
         <|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello?\
         <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    );
}

#[test]
fn llama3_without_generation_prompt() {
    let tmpl = ChatTemplate::parse(
        "{%- for message in messages %}\
             {%- if message['role'] == 'user' %}\
                 {{- '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}\
             {%- endif %}\
             {%- if loop.last and add_generation_prompt %}\
                 {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}\
             {%- endif %}\
         {%- endfor %}",
    )
    .unwrap();

    let mut ctx = Context::new();
    ctx.insert(
        "messages",
        Value::Array(vec![msg("user", "Hi"), msg("assistant", "Yo")]),
    );
    ctx.insert("add_generation_prompt", Value::Bool(false));

    let out = tmpl.render(&ctx).unwrap();
    assert_eq!(
        out,
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>\
         <|start_header_id|>user<|end_header_id|>\n\nYo<|eot_id|>"
    );
}

#[test]
fn mistral_style_template() {
    // A representative Mistral `chat_template` exercising indexing, `if`/`elif`,
    // and `for`. (The official template also calls `raise_exception`, which is
    // outside this subset; the alternating-roles guard is omitted for parity.)
    let tmpl = ChatTemplate::parse(
        "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + '\n' }}{% endif %}\
         {% for message in messages %}\
             {% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}\
             {% elif message['role'] == 'assistant' %}{{ message['content'] + ' </s>' }}\
             {% endif %}\
         {% endfor %}",
    )
    .unwrap();

    let mut ctx = Context::new();
    ctx.insert(
        "messages",
        Value::Array(vec![
            msg("user", "What is Rust?"),
            msg("assistant", "A systems language."),
        ]),
    );

    let out = tmpl.render(&ctx).unwrap();
    assert_eq!(out, "[INST] What is Rust? [/INST]A systems language. </s>");
}

#[test]
fn mistral_template_with_system_prefix() {
    let tmpl = ChatTemplate::parse(
        "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + '\n' }}{% endif %}\
         {% for message in messages %}\
             {% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}\
             {% elif message['role'] == 'assistant' %}{{ message['content'] + ' </s>' }}\
             {% endif %}\
         {% endfor %}",
    )
    .unwrap();

    let mut ctx = Context::new();
    ctx.insert(
        "messages",
        Value::Array(vec![
            msg("system", "Be concise."),
            msg("user", "Hi"),
            msg("assistant", "Hello!"),
        ]),
    );

    let out = tmpl.render(&ctx).unwrap();
    assert_eq!(out, "Be concise.\n[INST] Hi [/INST]Hello! </s>");
}

#[test]
fn context_from_json_str() {
    let tmpl = ChatTemplate::parse("{{ user['name'] }} is {{ user['age'] }}").unwrap();
    let ctx = Context::from_json_str(r#"{ "user": { "name": "Ada", "age": 36 } }"#).unwrap();
    assert_eq!(tmpl.render(&ctx).unwrap(), "Ada is 36");
}

#[test]
fn parse_error_on_unterminated_tag() {
    let err = ChatTemplate::parse("{{ name ").err().unwrap();
    assert!(matches!(err, tpt_jinja_chat::TemplateError::Parse { .. }));
}

#[test]
fn render_error_on_undefined_variable() {
    let tmpl = ChatTemplate::parse("{{ missing }}").unwrap();
    let ctx = Context::new();
    assert!(matches!(
        tmpl.render(&ctx),
        Err(tpt_jinja_chat::TemplateError::UndefinedVariable(_))
    ));
}
