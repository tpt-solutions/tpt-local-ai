//! Template rendering (expression evaluation + node walking).

use std::collections::HashMap;

use crate::ast::{Arg, Expr, Node, Op, SetTarget};
use crate::error::TemplateError;
use crate::value::{json_stringify, json_stringify_pretty, Context, Value};

/// Mutable rendering environment holding in-scope variables and loop state.
pub(crate) struct Env {
    vars: HashMap<String, Value>,
    loop_state: Option<LoopState>,
}

#[derive(Debug, Clone, Copy)]
struct LoopState {
    /// 1-based index of the current iteration.
    index: usize,
    /// Total number of iterations.
    length: usize,
}

fn loop_value(ls: LoopState) -> Value {
    let mut m = HashMap::new();
    m.insert("index".to_string(), Value::Number(ls.index as f64));
    m.insert("index0".to_string(), Value::Number((ls.index - 1) as f64));
    m.insert(
        "revindex".to_string(),
        Value::Number((ls.length - ls.index + 1) as f64),
    );
    m.insert(
        "revindex0".to_string(),
        Value::Number((ls.length - ls.index) as f64),
    );
    m.insert("first".to_string(), Value::Bool(ls.index == 1));
    m.insert("last".to_string(), Value::Bool(ls.index == ls.length));
    m.insert("length".to_string(), Value::Number(ls.length as f64));
    Value::Object(m)
}

/// Render a list of nodes into `out`, using `initial` as the starting context.
pub(crate) fn render_nodes(
    nodes: &[Node],
    initial: &Context,
    out: &mut String,
) -> Result<(), TemplateError> {
    let mut env = Env {
        vars: initial.clone().into_vars(),
        loop_state: None,
    };
    walk(nodes, &mut env, out)
}

fn walk(nodes: &[Node], env: &mut Env, out: &mut String) -> Result<(), TemplateError> {
    for node in nodes {
        match node {
            Node::Text(t) => out.push_str(t),
            Node::Output(expr) => {
                let v = eval(expr, env)?;
                out.push_str(&v.to_rendered());
            }
            Node::Set { target, value } => {
                let v = eval(value, env)?;
                match target {
                    SetTarget::Var(name) => {
                        env.vars.insert(name.clone(), v);
                    }
                    SetTarget::Attr { base, attr } => {
                        match env.vars.get_mut(base) {
                            Some(Value::Object(o)) => {
                                o.insert(attr.clone(), v);
                            }
                            Some(other) => return Err(TemplateError::type_error(format!(
                                "cannot set attribute '{attr}' on non-object '{base}' ({other:?})"
                            ))),
                            None => return Err(TemplateError::undefined_variable(base.clone())),
                        }
                    }
                }
            }
            Node::For {
                targets,
                iterable,
                body,
            } => {
                let iterable = eval(iterable, env)?;
                let items: Vec<Value> = match iterable {
                    Value::Array(a) => a,
                    Value::Object(o) => {
                        // Iterating a mapping yields its keys (Jinja semantics).
                        let mut keys: Vec<String> = o.keys().cloned().collect();
                        keys.sort();
                        keys.into_iter().map(Value::String).collect()
                    }
                    Value::String(s) => s.chars().map(|c| Value::String(c.to_string())).collect(),
                    Value::Null => Vec::new(),
                    other => {
                        return Err(TemplateError::type_error(format!(
                            "cannot iterate over {other:?}"
                        )))
                    }
                };
                let length = items.len();
                let saved_loop = env.loop_state.take();
                let saved: Vec<(String, Option<Value>)> = targets
                    .iter()
                    .map(|t| (t.clone(), env.vars.get(t).cloned()))
                    .collect();
                env.loop_state = Some(LoopState { index: 0, length });
                for (i, item) in items.into_iter().enumerate() {
                    bind_targets(targets, item, env)?;
                    if let Some(ls) = env.loop_state.as_mut() {
                        ls.index = i + 1;
                    }
                    walk(body, env, out)?;
                }
                // Restore prior bindings + loop scope.
                for (name, prev) in saved {
                    match prev {
                        Some(v) => {
                            env.vars.insert(name, v);
                        }
                        None => {
                            env.vars.remove(&name);
                        }
                    }
                }
                env.loop_state = saved_loop;
            }
            Node::If {
                branches,
                else_body,
            } => {
                let mut matched = false;
                for (cond, body) in branches {
                    if eval(cond, env)?.is_truthy() {
                        walk(body, env, out)?;
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    walk(else_body, env, out)?;
                }
            }
        }
    }
    Ok(())
}

/// Bind a loop item to one or more targets, unpacking tuples when needed.
fn bind_targets(targets: &[String], item: Value, env: &mut Env) -> Result<(), TemplateError> {
    if targets.len() == 1 {
        env.vars.insert(targets[0].clone(), item);
        return Ok(());
    }
    match item {
        Value::Array(parts) if parts.len() == targets.len() => {
            for (name, v) in targets.iter().zip(parts) {
                env.vars.insert(name.clone(), v);
            }
            Ok(())
        }
        other => Err(TemplateError::type_error(format!(
            "cannot unpack {other:?} into {} loop targets",
            targets.len()
        ))),
    }
}

/// Try to evaluate `expr`, mapping "not present" errors to `Ok(None)` so
/// callers (the `default` filter, the `defined` test) can react to undefined
/// values without aborting the whole render.
fn try_eval(expr: &Expr, env: &Env) -> Result<Option<Value>, TemplateError> {
    match eval(expr, env) {
        Ok(v) => Ok(Some(v)),
        Err(TemplateError::UndefinedVariable(_)) | Err(TemplateError::Render(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Evaluate an expression to a [`Value`].
fn eval(expr: &Expr, env: &Env) -> Result<Value, TemplateError> {
    match expr {
        Expr::Var(name) => {
            if name == "loop" {
                return match env.loop_state {
                    Some(ls) => Ok(loop_value(ls)),
                    None => Err(TemplateError::undefined_variable("loop")),
                };
            }
            env.vars
                .get(name)
                .cloned()
                .ok_or_else(|| TemplateError::undefined_variable(name.clone()))
        }
        Expr::Get(base, prop) => {
            let base = eval(base, env)?;
            match base {
                Value::Object(o) => o
                    .get(prop)
                    .cloned()
                    .ok_or_else(|| TemplateError::render(format!("missing field '{prop}'"))),
                other => Err(TemplateError::type_error(format!(
                    "cannot access field '{prop}' on {other:?}"
                ))),
            }
        }
        Expr::Index(base, idx) => {
            let base = eval(base, env)?;
            let idx = eval(idx, env)?;
            index_value(base, idx)
        }
        Expr::Str(s) => Ok(Value::String(s.clone())),
        Expr::Num(n) => Ok(Value::Number(*n)),
        Expr::Bool(b) => Ok(Value::Bool(*b)),
        Expr::None => Ok(Value::Null),
        Expr::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(eval(item, env)?);
            }
            Ok(Value::Array(out))
        }
        Expr::Not(e) => Ok(Value::Bool(!eval(e, env)?.is_truthy())),
        Expr::Neg(e) => {
            let v = eval(e, env)?;
            match v {
                Value::Number(n) => Ok(Value::Number(-n)),
                other => Err(TemplateError::type_error(format!(
                    "cannot negate {other:?}"
                ))),
            }
        }
        Expr::Bin(op, l, r) => {
            let a = eval(l, env)?;
            let b = eval(r, env)?;
            eval_bin(*op, a, b)
        }
        Expr::Call(func, args) => eval_call(func, args, env),
        Expr::Filter(inner, name, args) => eval_filter(inner, name, args, env),
        Expr::Test(inner, name, args, negated) => {
            let result = eval_test(inner, name, args, env)?;
            Ok(Value::Bool(result ^ negated))
        }
    }
}

fn index_value(base: Value, idx: Value) -> Result<Value, TemplateError> {
    match base {
        Value::Object(o) => {
            let key = match &idx {
                Value::String(s) => s.clone(),
                other => {
                    return Err(TemplateError::type_error(format!(
                        "object index must be a string, got {other:?}"
                    )))
                }
            };
            o.get(&key)
                .cloned()
                .ok_or_else(|| TemplateError::render(format!("missing key '{key}'")))
        }
        Value::Array(a) => {
            let i = match &idx {
                Value::Number(n) if n.fract() == 0.0 => *n as isize,
                other => {
                    return Err(TemplateError::type_error(format!(
                        "array index must be an integer, got {other:?}"
                    )))
                }
            };
            let len = a.len() as isize;
            let ui = if i < 0 { i + len } else { i };
            if ui < 0 || ui >= len {
                return Err(TemplateError::render(format!(
                    "array index {i} out of bounds"
                )));
            }
            Ok(a[ui as usize].clone())
        }
        other => Err(TemplateError::type_error(format!(
            "cannot index into {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Function + method calls
// ---------------------------------------------------------------------------

/// Evaluated call/filter/test arguments, split into positional + keyword.
struct EvalArgs {
    positional: Vec<Value>,
    keyword: HashMap<String, Value>,
}

impl EvalArgs {
    fn pos(&self, i: usize) -> Option<&Value> {
        self.positional.get(i)
    }

    /// Fetch a keyword value, or the positional at `i` as a fallback.
    fn get(&self, name: &str, i: usize) -> Option<&Value> {
        self.keyword.get(name).or_else(|| self.positional.get(i))
    }
}

fn eval_args(args: &[Arg], env: &Env) -> Result<EvalArgs, TemplateError> {
    let mut positional = Vec::new();
    let mut keyword = HashMap::new();
    for a in args {
        let v = eval(&a.value, env)?;
        match &a.name {
            Some(n) => {
                keyword.insert(n.clone(), v);
            }
            None => positional.push(v),
        }
    }
    Ok(EvalArgs {
        positional,
        keyword,
    })
}

fn eval_call(func: &Expr, args: &[Arg], env: &Env) -> Result<Value, TemplateError> {
    match func {
        // Method call: `base.method(...)`.
        Expr::Get(base, method) => {
            let base_val = eval(base, env)?;
            let a = eval_args(args, env)?;
            call_method(base_val, method, &a)
        }
        // Free function: `name(...)`.
        Expr::Var(name) => call_function(name, args, env),
        other => Err(TemplateError::type_error(format!("cannot call {other:?}"))),
    }
}

fn call_function(name: &str, args: &[Arg], env: &Env) -> Result<Value, TemplateError> {
    match name {
        "raise_exception" | "raise" => {
            let a = eval_args(args, env)?;
            let msg = a
                .pos(0)
                .map(|v| v.to_rendered())
                .unwrap_or_else(|| "template raised an exception".to_string());
            Err(TemplateError::exception(msg))
        }
        "namespace" => {
            let a = eval_args(args, env)?;
            Ok(Value::Object(a.keyword))
        }
        "range" => {
            let a = eval_args(args, env)?;
            let nums: Vec<i64> = a
                .positional
                .iter()
                .map(|v| match v {
                    Value::Number(n) => Ok(*n as i64),
                    other => Err(TemplateError::type_error(format!(
                        "range() arguments must be numbers, got {other:?}"
                    ))),
                })
                .collect::<Result<_, _>>()?;
            let (start, stop, step) = match nums.as_slice() {
                [stop] => (0, *stop, 1),
                [start, stop] => (*start, *stop, 1),
                [start, stop, step] => (*start, *stop, *step),
                _ => return Err(TemplateError::render("range() takes 1 to 3 arguments")),
            };
            if step == 0 {
                return Err(TemplateError::render("range() step must not be zero"));
            }
            let mut out = Vec::new();
            let mut i = start;
            if step > 0 {
                while i < stop {
                    out.push(Value::Number(i as f64));
                    i += step;
                }
            } else {
                while i > stop {
                    out.push(Value::Number(i as f64));
                    i += step;
                }
            }
            Ok(Value::Array(out))
        }
        other => Err(TemplateError::render(format!("unknown function '{other}'"))),
    }
}

fn call_method(base: Value, method: &str, a: &EvalArgs) -> Result<Value, TemplateError> {
    match (&base, method) {
        (Value::Object(o), "items") => {
            let mut keys: Vec<&String> = o.keys().collect();
            keys.sort();
            Ok(Value::Array(
                keys.into_iter()
                    .map(|k| Value::Array(vec![Value::String(k.clone()), o[k].clone()]))
                    .collect(),
            ))
        }
        (Value::Object(o), "keys") => {
            let mut keys: Vec<String> = o.keys().cloned().collect();
            keys.sort();
            Ok(Value::Array(keys.into_iter().map(Value::String).collect()))
        }
        (Value::Object(o), "values") => {
            let mut keys: Vec<&String> = o.keys().collect();
            keys.sort();
            Ok(Value::Array(
                keys.into_iter().map(|k| o[k].clone()).collect(),
            ))
        }
        (Value::Object(o), "get") => {
            let key = a
                .pos(0)
                .map(|v| v.to_rendered())
                .ok_or_else(|| TemplateError::render("get() requires a key"))?;
            Ok(o.get(&key)
                .cloned()
                .unwrap_or_else(|| a.pos(1).cloned().unwrap_or(Value::Null)))
        }
        (Value::String(s), "startswith") => {
            let p = a.pos(0).map(|v| v.to_rendered()).unwrap_or_default();
            Ok(Value::Bool(s.starts_with(&p)))
        }
        (Value::String(s), "endswith") => {
            let p = a.pos(0).map(|v| v.to_rendered()).unwrap_or_default();
            Ok(Value::Bool(s.ends_with(&p)))
        }
        (Value::String(s), "strip") => Ok(Value::String(s.trim().to_string())),
        (Value::String(s), "lstrip") => Ok(Value::String(s.trim_start().to_string())),
        (Value::String(s), "rstrip") => Ok(Value::String(s.trim_end().to_string())),
        (Value::String(s), "upper") => Ok(Value::String(s.to_uppercase())),
        (Value::String(s), "lower") => Ok(Value::String(s.to_lowercase())),
        (Value::String(s), "title") => Ok(Value::String(title_case(s))),
        (Value::String(s), "capitalize") => Ok(Value::String(capitalize(s))),
        (Value::String(s), "replace") => {
            let from = a.pos(0).map(|v| v.to_rendered()).unwrap_or_default();
            let to = a.pos(1).map(|v| v.to_rendered()).unwrap_or_default();
            Ok(Value::String(s.replace(&from, &to)))
        }
        (Value::String(s), "split") => {
            let parts: Vec<Value> = match a.pos(0) {
                Some(sep) => {
                    let sep = sep.to_rendered();
                    s.split(&sep)
                        .map(|p| Value::String(p.to_string()))
                        .collect()
                }
                None => s
                    .split_whitespace()
                    .map(|p| Value::String(p.to_string()))
                    .collect(),
            };
            Ok(Value::Array(parts))
        }
        (other, _) => Err(TemplateError::type_error(format!(
            "unknown method '{method}' on {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

fn eval_filter(inner: &Expr, name: &str, args: &[Arg], env: &Env) -> Result<Value, TemplateError> {
    // `default` must tolerate undefined input, so it evaluates the inner
    // expression itself rather than receiving a pre-evaluated value.
    if name == "default" || name == "d" {
        let a = eval_args(args, env)?;
        let fallback = a.pos(0).cloned().unwrap_or(Value::Null);
        let use_falsy = a.get("boolean", 1).map(|v| v.is_truthy()).unwrap_or(false);
        return match try_eval(inner, env)? {
            None => Ok(fallback),
            Some(v) => {
                let missing = matches!(v, Value::Null) || (use_falsy && !v.is_truthy());
                Ok(if missing { fallback } else { v })
            }
        };
    }

    let base = eval(inner, env)?;
    let a = eval_args(args, env)?;
    apply_filter(name, base, &a)
}

fn apply_filter(name: &str, base: Value, a: &EvalArgs) -> Result<Value, TemplateError> {
    match name {
        "tojson" => {
            let indent = a.get("indent", 0).and_then(|v| match v {
                Value::Number(n) => Some(*n as usize),
                _ => None,
            });
            Ok(Value::String(match indent {
                Some(n) => json_stringify_pretty(&base, n),
                None => json_stringify(&base),
            }))
        }
        "trim" => Ok(Value::String(base.to_rendered().trim().to_string())),
        "upper" => Ok(Value::String(base.to_rendered().to_uppercase())),
        "lower" => Ok(Value::String(base.to_rendered().to_lowercase())),
        "capitalize" => Ok(Value::String(capitalize(&base.to_rendered()))),
        "title" => Ok(Value::String(title_case(&base.to_rendered()))),
        "string" => Ok(Value::String(base.to_rendered())),
        "replace" => {
            let from = a.pos(0).map(|v| v.to_rendered()).unwrap_or_default();
            let to = a.pos(1).map(|v| v.to_rendered()).unwrap_or_default();
            Ok(Value::String(base.to_rendered().replace(&from, &to)))
        }
        "join" => {
            let sep = a.get("d", 0).map(|v| v.to_rendered()).unwrap_or_default();
            let items = as_array(&base)?;
            let joined = items
                .iter()
                .map(|v| v.to_rendered())
                .collect::<Vec<_>>()
                .join(&sep);
            Ok(Value::String(joined))
        }
        "length" | "count" => match &base {
            Value::Array(a) => Ok(Value::Number(a.len() as f64)),
            Value::Object(o) => Ok(Value::Number(o.len() as f64)),
            Value::String(s) => Ok(Value::Number(s.chars().count() as f64)),
            Value::Null => Ok(Value::Number(0.0)),
            other => Err(TemplateError::type_error(format!(
                "cannot take length of {other:?}"
            ))),
        },
        "first" => match &base {
            Value::Array(a) => Ok(a.first().cloned().unwrap_or(Value::Null)),
            Value::String(s) => Ok(s
                .chars()
                .next()
                .map(|c| Value::String(c.to_string()))
                .unwrap_or(Value::Null)),
            other => Err(TemplateError::type_error(format!(
                "cannot take first of {other:?}"
            ))),
        },
        "last" => match &base {
            Value::Array(a) => Ok(a.last().cloned().unwrap_or(Value::Null)),
            Value::String(s) => Ok(s
                .chars()
                .next_back()
                .map(|c| Value::String(c.to_string()))
                .unwrap_or(Value::Null)),
            other => Err(TemplateError::type_error(format!(
                "cannot take last of {other:?}"
            ))),
        },
        "list" => match base {
            Value::Array(_) => Ok(base),
            Value::String(s) => Ok(Value::Array(
                s.chars().map(|c| Value::String(c.to_string())).collect(),
            )),
            Value::Object(o) => {
                let mut keys: Vec<String> = o.keys().cloned().collect();
                keys.sort();
                Ok(Value::Array(keys.into_iter().map(Value::String).collect()))
            }
            other => Err(TemplateError::type_error(format!(
                "cannot convert {other:?} to a list"
            ))),
        },
        "int" => {
            let n = match &base {
                Value::Number(n) => *n as i64 as f64,
                Value::String(s) => s
                    .trim()
                    .parse::<f64>()
                    .map(|n| n as i64 as f64)
                    .unwrap_or(0.0),
                Value::Bool(b) => u8::from(*b) as f64,
                _ => 0.0,
            };
            Ok(Value::Number(n))
        }
        "float" => {
            let n = match &base {
                Value::Number(n) => *n,
                Value::String(s) => s.trim().parse::<f64>().unwrap_or(0.0),
                _ => 0.0,
            };
            Ok(Value::Number(n))
        }
        "abs" => match &base {
            Value::Number(n) => Ok(Value::Number(n.abs())),
            other => Err(TemplateError::type_error(format!(
                "cannot take abs of {other:?}"
            ))),
        },
        "selectattr" => select_attr(base, a, true),
        "rejectattr" => select_attr(base, a, false),
        "select" => select_items(base, a, true),
        "reject" => select_items(base, a, false),
        "map" => map_filter(base, a),
        other => Err(TemplateError::render(format!("unknown filter '{other}'"))),
    }
}

fn as_array(v: &Value) -> Result<Vec<Value>, TemplateError> {
    match v {
        Value::Array(a) => Ok(a.clone()),
        Value::Null => Ok(Vec::new()),
        other => Err(TemplateError::type_error(format!(
            "expected a list, got {other:?}"
        ))),
    }
}

fn select_attr(base: Value, a: &EvalArgs, keep_matches: bool) -> Result<Value, TemplateError> {
    let items = as_array(&base)?;
    let attr = a
        .pos(0)
        .map(|v| v.to_rendered())
        .ok_or_else(|| TemplateError::render("selectattr requires an attribute name"))?;
    let test_name = a.pos(1).map(|v| v.to_rendered());
    let test_args: Vec<Value> = a.positional.iter().skip(2).cloned().collect();
    let mut out = Vec::new();
    for item in items {
        let attr_val = match &item {
            Value::Object(o) => o.get(&attr).cloned().unwrap_or(Value::Null),
            _ => Value::Null,
        };
        let passed = match &test_name {
            Some(name) => run_test(name, &attr_val, &test_args)?,
            None => attr_val.is_truthy(),
        };
        if passed == keep_matches {
            out.push(item);
        }
    }
    Ok(Value::Array(out))
}

fn select_items(base: Value, a: &EvalArgs, keep_matches: bool) -> Result<Value, TemplateError> {
    let items = as_array(&base)?;
    let test_name = a.pos(0).map(|v| v.to_rendered());
    let test_args: Vec<Value> = a.positional.iter().skip(1).cloned().collect();
    let mut out = Vec::new();
    for item in items {
        let passed = match &test_name {
            Some(name) => run_test(name, &item, &test_args)?,
            None => item.is_truthy(),
        };
        if passed == keep_matches {
            out.push(item);
        }
    }
    Ok(Value::Array(out))
}

fn map_filter(base: Value, a: &EvalArgs) -> Result<Value, TemplateError> {
    let items = as_array(&base)?;
    if let Some(attr) = a.get("attribute", usize::MAX).map(|v| v.to_rendered()) {
        let out = items
            .into_iter()
            .map(|item| match item {
                Value::Object(o) => o.get(&attr).cloned().unwrap_or(Value::Null),
                _ => Value::Null,
            })
            .collect();
        return Ok(Value::Array(out));
    }
    // `map('filter_name', args...)` applies a filter to each element.
    if let Some(filter_name) = a.pos(0).map(|v| v.to_rendered()) {
        let rest = EvalArgs {
            positional: a.positional.iter().skip(1).cloned().collect(),
            keyword: a.keyword.clone(),
        };
        let out = items
            .into_iter()
            .map(|item| apply_filter(&filter_name, item, &rest))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Value::Array(out));
    }
    Err(TemplateError::render(
        "map requires an 'attribute' keyword or a filter name",
    ))
}

// ---------------------------------------------------------------------------
// Tests (`is`)
// ---------------------------------------------------------------------------

fn eval_test(inner: &Expr, name: &str, args: &[Arg], env: &Env) -> Result<bool, TemplateError> {
    match name {
        "defined" => Ok(try_eval(inner, env)?.is_some()),
        "undefined" => Ok(try_eval(inner, env)?.is_none()),
        _ => {
            let v = eval(inner, env)?;
            let a = eval_args(args, env)?;
            run_test(name, &v, &a.positional)
        }
    }
}

fn run_test(name: &str, value: &Value, args: &[Value]) -> Result<bool, TemplateError> {
    let result = match name {
        "none" => matches!(value, Value::Null),
        "string" => matches!(value, Value::String(_)),
        "number" | "float" => matches!(value, Value::Number(_)),
        "integer" => matches!(value, Value::Number(n) if n.fract() == 0.0),
        "boolean" => matches!(value, Value::Bool(_)),
        "true" => matches!(value, Value::Bool(true)),
        "false" => matches!(value, Value::Bool(false)),
        "mapping" => matches!(value, Value::Object(_)),
        "sequence" => matches!(value, Value::Array(_) | Value::String(_)),
        "iterable" => matches!(value, Value::Array(_) | Value::Object(_) | Value::String(_)),
        "even" => matches!(value, Value::Number(n) if (*n as i64) % 2 == 0),
        "odd" => matches!(value, Value::Number(n) if (*n as i64) % 2 != 0),
        "equalto" | "eq" | "==" => {
            let other = args
                .first()
                .ok_or_else(|| TemplateError::render("equalto requires an argument"))?;
            value == other
        }
        "ne" | "!=" => {
            let other = args
                .first()
                .ok_or_else(|| TemplateError::render("ne requires an argument"))?;
            value != other
        }
        "in" => {
            let container = args
                .first()
                .ok_or_else(|| TemplateError::render("'in' test requires an argument"))?;
            match container {
                Value::Array(a) => a.contains(value),
                Value::Object(o) => matches!(value, Value::String(s) if o.contains_key(s)),
                Value::String(s) => matches!(value, Value::String(sub) if s.contains(sub.as_str())),
                _ => false,
            }
        }
        other => return Err(TemplateError::render(format!("unknown test '{other}'"))),
    };
    Ok(result)
}

// ---------------------------------------------------------------------------
// String helpers + binary ops
// ---------------------------------------------------------------------------

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
    }
}

fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut at_word_start = true;
    for c in s.chars() {
        if c.is_alphanumeric() {
            if at_word_start {
                out.extend(c.to_uppercase());
            } else {
                out.extend(c.to_lowercase());
            }
            at_word_start = false;
        } else {
            out.push(c);
            at_word_start = true;
        }
    }
    out
}

fn eval_bin(op: Op, a: Value, b: Value) -> Result<Value, TemplateError> {
    match op {
        Op::Add => match (a, b) {
            (Value::String(mut s), Value::String(t)) => {
                s.push_str(&t);
                Ok(Value::String(s))
            }
            (Value::Number(x), Value::Number(y)) => Ok(Value::Number(x + y)),
            (Value::Array(mut x), Value::Array(y)) => {
                x.extend(y);
                Ok(Value::Array(x))
            }
            (x, y) => Err(TemplateError::type_error(format!(
                "cannot add {x:?} and {y:?}"
            ))),
        },
        // `~` concatenates the string renderings of both operands.
        Op::Concat => {
            let mut s = a.to_rendered();
            s.push_str(&b.to_rendered());
            Ok(Value::String(s))
        }
        Op::Sub => numeric(op, &a, &b, |x, y| x - y),
        Op::Mul => numeric(op, &a, &b, |x, y| x * y),
        Op::Div => numeric(op, &a, &b, |x, y| x / y),
        Op::Eq => Ok(Value::Bool(a == b)),
        Op::Ne => Ok(Value::Bool(a != b)),
        Op::Lt => compare(op, &a, &b, |o| o == std::cmp::Ordering::Less),
        Op::Gt => compare(op, &a, &b, |o| o == std::cmp::Ordering::Greater),
        Op::Le => compare(op, &a, &b, |o| o != std::cmp::Ordering::Greater),
        Op::Ge => compare(op, &a, &b, |o| o != std::cmp::Ordering::Less),
        Op::And => Ok(Value::Bool(a.is_truthy() && b.is_truthy())),
        Op::Or => Ok(Value::Bool(a.is_truthy() || b.is_truthy())),
        Op::In => Ok(Value::Bool(contains(&b, &a))),
        Op::NotIn => Ok(Value::Bool(!contains(&b, &a))),
    }
}

/// Membership: is `needle` contained in `haystack`?
fn contains(haystack: &Value, needle: &Value) -> bool {
    match haystack {
        Value::Array(a) => a.contains(needle),
        Value::Object(o) => matches!(needle, Value::String(s) if o.contains_key(s)),
        Value::String(s) => matches!(needle, Value::String(sub) if s.contains(sub.as_str())),
        _ => false,
    }
}

fn numeric<F>(op: Op, a: &Value, b: &Value, f: F) -> Result<Value, TemplateError>
where
    F: Fn(f64, f64) -> f64,
{
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => Ok(Value::Number(f(*x, *y))),
        (x, y) => Err(TemplateError::type_error(format!(
            "operator {op:?} requires numbers, got {x:?} and {y:?}"
        ))),
    }
}

fn compare<F>(op: Op, a: &Value, b: &Value, f: F) -> Result<Value, TemplateError>
where
    F: Fn(std::cmp::Ordering) -> bool,
{
    let ord = match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
        }
        (Value::String(x), Value::String(y)) => x.cmp(y),
        (x, y) => {
            return Err(TemplateError::type_error(format!(
                "operator {op:?} requires comparable values, got {x:?} and {y:?}"
            )))
        }
    };
    Ok(Value::Bool(f(ord)))
}
