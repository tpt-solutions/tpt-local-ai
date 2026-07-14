//! Integration tests for `tpt-jinja-chat`.

use tpt_jinja_chat::{ChatTemplate, Context, TemplateError, Value};

fn messages() -> Value {
    Value::Array(vec![
        Value::object(vec![
            ("role".into(), Value::String("system".into())),
            ("content".into(), Value::String("You are helpful.".into())),
        ]),
        Value::object(vec![
            ("role".into(), Value::String("user".into())),
            ("content".into(), Value::String("Hello!".into())),
        ]),
        Value::object(vec![
            ("role".into(), Value::String("assistant".into())),
            ("content".into(), Value::String("Hi there.".into())),
        ]),
    ])
}

#[test]
fn variable_substitution() {
    let t = ChatTemplate::parse("Hello {{ name }}!").unwrap();
    let mut ctx = Context::new();
    ctx.insert("name", Value::String("World".into()));
    assert_eq!(t.render(&ctx).unwrap(), "Hello World!");
}

#[test]
fn for_loop() {
    let t = ChatTemplate::parse("{% for m in messages %}{{ m.role }}|{% endfor %}").unwrap();
    let mut ctx = Context::new();
    ctx.insert("messages", messages());
    assert_eq!(t.render(&ctx).unwrap(), "system|user|assistant|");
}

#[test]
fn for_loop_with_loop_index() {
    let t =
        ChatTemplate::parse("{% for m in messages %}{{ loop.index }}:{{ m.role }} {% endfor %}")
            .unwrap();
    let mut ctx = Context::new();
    ctx.insert("messages", messages());
    assert_eq!(t.render(&ctx).unwrap(), "1:system 2:user 3:assistant ");
}

#[test]
fn if_elif_else() {
    let src = "{% for m in messages %}{% if m.role == 'system' %}S{% elif m.role == 'user' %}U{% else %}A{% endif %}{% endfor %}";
    let t = ChatTemplate::parse(src).unwrap();
    let mut ctx = Context::new();
    ctx.insert("messages", messages());
    assert_eq!(t.render(&ctx).unwrap(), "SUA");
}

#[test]
fn set_and_concat() {
    let src = "{% set sep = ' > ' %}{% for m in messages %}{{ sep }}{{ m.content }}{% endfor %}";
    let t = ChatTemplate::parse(src).unwrap();
    let mut ctx = Context::new();
    ctx.insert("messages", messages());
    assert_eq!(
        t.render(&ctx).unwrap(),
        " > You are helpful. > Hello! > Hi there."
    );
}

#[test]
fn llama3_style_template() {
    let src = "{% for message in messages %}{% if message.role == 'system' %}<|system|> {{ message.content }} {% elif message.role == 'user' %}<|user|> {{ message.content }} {% else %}<|assistant|> {{ message.content }} {% endif %}{% endfor %}";
    let t = ChatTemplate::parse(src).unwrap();
    let mut ctx = Context::new();
    ctx.insert("messages", messages());
    assert_eq!(
        t.render(&ctx).unwrap(),
        "<|system|> You are helpful. <|user|> Hello! <|assistant|> Hi there. "
    );
}

#[test]
fn mistral_style_template_with_indexing() {
    let src = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}{{ eos_token }}";
    let t = ChatTemplate::parse(src).unwrap();
    let mut ctx = Context::new();
    ctx.insert("messages", messages());
    ctx.insert("bos_token", Value::String("<s>".into()));
    ctx.insert("eos_token", Value::String("</s>".into()));
    assert_eq!(
        t.render(&ctx).unwrap(),
        "<s>[INST] Hello! [/INST]Hi there.</s>"
    );
}

#[test]
fn context_from_json() {
    let t = ChatTemplate::parse("{{ a }} + {{ b }} = {{ a + b }}").unwrap();
    let ctx = Context::from_json_str(r#"{"a": 2, "b": 3}"#).unwrap();
    assert_eq!(t.render(&ctx).unwrap(), "2 + 3 = 5");
}

#[test]
fn nested_object_access() {
    let t = ChatTemplate::parse("{{ user.name.first }}").unwrap();
    let mut ctx = Context::new();
    ctx.insert(
        "user",
        Value::object(vec![(
            "name".into(),
            Value::object(vec![("first".into(), Value::String("Ada".into()))]),
        )]),
    );
    assert_eq!(t.render(&ctx).unwrap(), "Ada");
}

#[test]
fn undefined_variable_errors() {
    let t = ChatTemplate::parse("{{ missing }}").unwrap();
    let ctx = Context::new();
    let err = t.render(&ctx).unwrap_err();
    assert!(matches!(err, TemplateError::UndefinedVariable(_)));
}

#[test]
fn parse_error_on_bad_syntax() {
    let err = ChatTemplate::parse("{% if x %}no end").unwrap_err();
    assert!(matches!(err, TemplateError::Parse { .. }));
}

#[test]
fn comments_are_skipped() {
    let t = ChatTemplate::parse("a{# hidden #}b").unwrap();
    let ctx = Context::new();
    assert_eq!(t.render(&ctx).unwrap(), "ab");
}

#[test]
fn whitespace_control_strips_surrounding_text() {
    // `{{-` / `{%-` trim preceding whitespace; `-}}` / `-%}` trim following.
    // The text between the tags is emitted (after the tag trims surrounding
    // whitespace), so only the inner content remains attached to the tag.
    let t = ChatTemplate::parse("a{%- if true %}b{%- endif %}c").unwrap();
    let ctx = Context::new();
    assert_eq!(t.render(&ctx).unwrap(), "abc");
}

#[test]
fn nested_if_chains_in_loop() {
    // Two sequential `if` chains inside one `for` body (as in real Llama 3
    // templates).
    let src = "{% for m in messages %}{% if m.role == 'user' %}U{% else %}X{% endif %}\
               {% if m.role == 'assistant' %}A{% else %}Y{% endif %}{% endfor %}";
    let t = ChatTemplate::parse(src).unwrap();
    let mut ctx = Context::new();
    ctx.insert("messages", messages());
    // system -> "XY", user -> "UY", assistant -> "XA"
    assert_eq!(t.render(&ctx).unwrap(), "XYUYXA");
}
