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

// ---------------------------------------------------------------------------
// UTF-8 correctness (regression for the `bytes[i] as char` mojibake bug)
// ---------------------------------------------------------------------------

#[test]
fn non_ascii_literal_text_is_preserved() {
    let t = ChatTemplate::parse("café — 日本語 — 😀 {{ x }}").unwrap();
    let mut ctx = Context::new();
    ctx.insert("x", Value::String("naïve — 汉字 — 🎉".into()));
    assert_eq!(
        t.render(&ctx).unwrap(),
        "café — 日本語 — 😀 naïve — 汉字 — 🎉"
    );
}

#[test]
fn non_ascii_quoted_string_literal_is_preserved() {
    let t = ChatTemplate::parse("{{ '日本語 — café 😀' }}").unwrap();
    let ctx = Context::new();
    assert_eq!(t.render(&ctx).unwrap(), "日本語 — café 😀");
}

#[test]
fn non_ascii_between_tags_does_not_panic() {
    // Multi-byte text immediately adjacent to tags used to risk a non-char-
    // boundary slice panic in `find_close`.
    let t = ChatTemplate::parse("汉{% if true %}字{% endif %}😀").unwrap();
    let ctx = Context::new();
    assert_eq!(t.render(&ctx).unwrap(), "汉字😀");
}

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

#[test]
fn tojson_filter_is_deterministic() {
    let t = ChatTemplate::parse("{{ obj | tojson }}").unwrap();
    let mut ctx = Context::new();
    ctx.insert(
        "obj",
        Value::object(vec![
            ("b".into(), Value::Number(2.0)),
            ("a".into(), Value::String("x".into())),
        ]),
    );
    // Keys are sorted for reproducible output.
    assert_eq!(t.render(&ctx).unwrap(), r#"{"a":"x","b":2}"#);
}

#[test]
fn tojson_filter_with_indent() {
    let t = ChatTemplate::parse("{{ arr | tojson(indent=2) }}").unwrap();
    let mut ctx = Context::new();
    ctx.insert(
        "arr",
        Value::Array(vec![Value::Number(1.0), Value::Number(2.0)]),
    );
    assert_eq!(t.render(&ctx).unwrap(), "[\n  1,\n  2\n]");
}

#[test]
fn string_filters() {
    let t = ChatTemplate::parse(
        "{{ '  Hi  ' | trim }}|{{ 'abc' | upper }}|{{ 'ABC' | lower }}|{{ 'ab' | length }}",
    )
    .unwrap();
    let ctx = Context::new();
    assert_eq!(t.render(&ctx).unwrap(), "Hi|ABC|abc|2");
}

#[test]
fn join_and_first_last_filters() {
    let t = ChatTemplate::parse("{{ xs | join(', ') }}|{{ xs | first }}|{{ xs | last }}").unwrap();
    let mut ctx = Context::new();
    ctx.insert(
        "xs",
        Value::Array(vec![
            Value::String("a".into()),
            Value::String("b".into()),
            Value::String("c".into()),
        ]),
    );
    assert_eq!(t.render(&ctx).unwrap(), "a, b, c|a|c");
}

#[test]
fn default_filter_handles_undefined() {
    let t = ChatTemplate::parse("{{ missing | default('fallback') }}").unwrap();
    let ctx = Context::new();
    assert_eq!(t.render(&ctx).unwrap(), "fallback");
}

#[test]
fn selectattr_and_map_filters() {
    let t = ChatTemplate::parse(
        "{{ messages | selectattr('role', 'equalto', 'user') | map(attribute='content') | join('|') }}",
    )
    .unwrap();
    let mut ctx = Context::new();
    ctx.insert("messages", messages());
    assert_eq!(t.render(&ctx).unwrap(), "Hello!");
}

// ---------------------------------------------------------------------------
// `is` tests
// ---------------------------------------------------------------------------

#[test]
fn is_defined_and_is_none_tests() {
    let src = "{% if x is defined %}D{% endif %}{% if y is not defined %}U{% endif %}\
               {% if z is none %}N{% endif %}{% if s is string %}S{% endif %}";
    let t = ChatTemplate::parse(src).unwrap();
    let mut ctx = Context::new();
    ctx.insert("x", Value::Number(1.0));
    ctx.insert("z", Value::Null);
    ctx.insert("s", Value::String("hi".into()));
    assert_eq!(t.render(&ctx).unwrap(), "DUNS");
}

// ---------------------------------------------------------------------------
// `~` concat, list literals
// ---------------------------------------------------------------------------

#[test]
fn tilde_concat_operator() {
    let t = ChatTemplate::parse("{{ 'a' ~ 1 ~ 'b' }}").unwrap();
    let ctx = Context::new();
    assert_eq!(t.render(&ctx).unwrap(), "a1b");
}

#[test]
fn list_literal_and_membership() {
    let t = ChatTemplate::parse("{% if role in ['user', 'assistant'] %}yes{% else %}no{% endif %}")
        .unwrap();
    let mut ctx = Context::new();
    ctx.insert("role", Value::String("user".into()));
    assert_eq!(t.render(&ctx).unwrap(), "yes");
}

#[test]
fn iterate_over_list_literal() {
    let t = ChatTemplate::parse("{% for x in [1, 2, 3] %}{{ x }}{% endfor %}").unwrap();
    let ctx = Context::new();
    assert_eq!(t.render(&ctx).unwrap(), "123");
}

// ---------------------------------------------------------------------------
// Function / method calls + tuple-for
// ---------------------------------------------------------------------------

#[test]
fn raise_exception_produces_error() {
    let t = ChatTemplate::parse("{{ raise_exception('boom') }}").unwrap();
    let ctx = Context::new();
    let err = t.render(&ctx).unwrap_err();
    assert!(matches!(err, TemplateError::Exception(m) if m == "boom"));
}

#[test]
fn namespace_and_attribute_set() {
    let src = "{% set ns = namespace(found=false) %}\
               {% for m in messages %}{% if m.role == 'assistant' %}{% set ns.found = true %}{% endif %}{% endfor %}\
               {% if ns.found %}has-assistant{% else %}none{% endif %}";
    let t = ChatTemplate::parse(src).unwrap();
    let mut ctx = Context::new();
    ctx.insert("messages", messages());
    assert_eq!(t.render(&ctx).unwrap(), "has-assistant");
}

#[test]
fn tuple_for_over_items() {
    let src = "{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}";
    let t = ChatTemplate::parse(src).unwrap();
    let mut ctx = Context::new();
    ctx.insert(
        "d",
        Value::object(vec![
            ("a".into(), Value::Number(1.0)),
            ("b".into(), Value::Number(2.0)),
        ]),
    );
    // Keys are iterated in sorted order for determinism.
    assert_eq!(t.render(&ctx).unwrap(), "a=1;b=2;");
}

#[test]
fn string_methods() {
    let t = ChatTemplate::parse(
        "{% if s.startswith('foo') %}Y{% endif %}{% if s.endswith('bar') %}Z{% endif %}",
    )
    .unwrap();
    let mut ctx = Context::new();
    ctx.insert("s", Value::String("foobar".into()));
    assert_eq!(t.render(&ctx).unwrap(), "YZ");
}

#[test]
fn llama31_tool_calling_style_snippet() {
    // Exercises `tojson`, `namespace`, attribute-set, and `~` together as in
    // real Llama 3.1 tool-calling templates.
    let src = "{% set ns = namespace(count=0) %}\
               {% for t in tools %}{% set ns.count = ns.count + 1 %}{{ t.name ~ ':' ~ (t.args | tojson) }}\n{% endfor %}\
               total={{ ns.count }}";
    let t = ChatTemplate::parse(src).unwrap();
    let mut ctx = Context::new();
    ctx.insert(
        "tools",
        Value::Array(vec![Value::object(vec![
            ("name".into(), Value::String("search".into())),
            (
                "args".into(),
                Value::object(vec![("q".into(), Value::String("hi".into()))]),
            ),
        ])]),
    );
    assert_eq!(t.render(&ctx).unwrap(), "search:{\"q\":\"hi\"}\ntotal=1");
}
