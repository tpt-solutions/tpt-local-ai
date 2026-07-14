//! Template rendering (expression evaluation + node walking).

use std::collections::HashMap;

use crate::ast::{Expr, Node, Op};
use crate::error::TemplateError;
use crate::value::{Context, Value};

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
            Node::Set { name, value } => {
                let v = eval(value, env)?;
                env.vars.insert(name.clone(), v);
            }
            Node::For {
                var,
                iterable,
                body,
            } => {
                let iterable = eval(iterable, env)?;
                let items: Vec<Value> = match iterable {
                    Value::Array(a) => a,
                    Value::Object(o) => o.into_values().collect(),
                    other => {
                        return Err(TemplateError::type_error(format!(
                            "cannot iterate over {other:?}"
                        )))
                    }
                };
                let length = items.len();
                env.loop_state = Some(LoopState { index: 0, length });
                for (i, item) in items.into_iter().enumerate() {
                    env.vars.insert(var.clone(), item);
                    if let Some(ls) = env.loop_state.as_mut() {
                        ls.index = i + 1;
                    }
                    walk(body, env, out)?;
                }
                env.loop_state = None;
                env.vars.remove(var);
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
        Expr::Str(s) => Ok(Value::String(s.clone())),
        Expr::Num(n) => Ok(Value::Number(*n)),
        Expr::Bool(b) => Ok(Value::Bool(*b)),
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
    }
}

fn eval_bin(op: Op, a: Value, b: Value) -> Result<Value, TemplateError> {
    match op {
        Op::Add => match (a, b) {
            (Value::String(mut s), Value::String(t)) => {
                s.push_str(&t);
                Ok(Value::String(s))
            }
            (Value::Number(x), Value::Number(y)) => Ok(Value::Number(x + y)),
            (x, y) => Err(TemplateError::type_error(format!(
                "cannot add {x:?} and {y:?}"
            ))),
        },
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
