//! Abstract syntax tree for parsed chat templates.
//!
//! These types are internal to the crate; the public entry point is
//! [`crate::ChatTemplate`].

/// A parsed chat template: an ordered list of renderable [`Node`]s.
#[derive(Debug, Clone)]
pub(crate) struct Template {
    pub(crate) nodes: Vec<Node>,
}

#[derive(Debug, Clone)]
pub(crate) enum Node {
    /// Literal text copied verbatim to the output.
    Text(String),
    /// `{{ expr }}` — render the expression's value.
    Output(Expr),
    /// `{% set target = expr %}`.
    Set { target: SetTarget, value: Expr },
    /// `{% for a, b in iterable %} ... {% endfor %}`.
    ///
    /// `targets` holds one name for a simple loop variable, or several for a
    /// tuple-unpacking target such as `{% for k, v in d.items() %}`.
    For {
        targets: Vec<String>,
        iterable: Expr,
        body: Vec<Node>,
    },
    /// `{% if %} / {% elif %} / {% else %}`.
    If {
        branches: Vec<(Expr, Vec<Node>)>,
        else_body: Vec<Node>,
    },
}

/// The left-hand side of a `{% set %}` statement.
#[derive(Debug, Clone)]
pub(crate) enum SetTarget {
    /// A plain variable, e.g. `{% set x = ... %}`.
    Var(String),
    /// An attribute of a namespace object, e.g. `{% set ns.found = ... %}`.
    Attr { base: String, attr: String },
}

/// A single call/filter/test argument, optionally a `name=value` keyword.
#[derive(Debug, Clone)]
pub(crate) struct Arg {
    /// The keyword name, if this was passed as `name=value`.
    pub(crate) name: Option<String>,
    /// The argument expression.
    pub(crate) value: Expr,
}

/// A template expression (right-hand sides of output/set/if/for).
#[derive(Debug, Clone)]
pub(crate) enum Expr {
    /// A variable reference (`messages`).
    Var(String),
    /// Member access (`a.b`).
    Get(Box<Expr>, String),
    /// Index access (`a['k']` or `a[0]`).
    Index(Box<Expr>, Box<Expr>),
    /// A string literal.
    Str(String),
    /// A numeric literal.
    Num(f64),
    /// A boolean literal.
    Bool(bool),
    /// A `none`/`null` literal.
    None,
    /// A list literal, e.g. `['user', 'assistant']`.
    List(Vec<Expr>),
    /// A binary operation.
    Bin(Op, Box<Expr>, Box<Expr>),
    /// A unary `not` operation.
    Not(Box<Expr>),
    /// A unary negation.
    Neg(Box<Expr>),
    /// A function or method call, e.g. `namespace(x=1)` or `d.items()`.
    Call(Box<Expr>, Vec<Arg>),
    /// A filter application, e.g. `messages | tojson` or `x | default('y')`.
    Filter(Box<Expr>, String, Vec<Arg>),
    /// An `is` test, e.g. `x is defined` or `x is not none`. The `bool` marks
    /// negation (`is not`).
    Test(Box<Expr>, String, Vec<Arg>, bool),
}

/// Binary operators supported by template expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
    /// `+` (string concatenation or numeric addition).
    Add,
    /// `~` (always-string concatenation).
    Concat,
    /// `-` (numeric subtraction).
    Sub,
    /// `*` (numeric multiplication).
    Mul,
    /// `/` (numeric division).
    Div,
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `>`
    Gt,
    /// `<=`
    Le,
    /// `>=`
    Ge,
    /// `and`
    And,
    /// `or`
    Or,
    /// `in` (membership test).
    In,
    /// `not in` (negated membership test).
    NotIn,
}
