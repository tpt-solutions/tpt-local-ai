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
    /// `{% set name = expr %}`.
    Set { name: String, value: Expr },
    /// `{% for var in iterable %} ... {% endfor %}`.
    For {
        var: String,
        iterable: Expr,
        body: Vec<Node>,
    },
    /// `{% if %} / {% elif %} / {% else %}`.
    If {
        branches: Vec<(Expr, Vec<Node>)>,
        else_body: Vec<Node>,
    },
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
    /// A binary operation.
    Bin(Op, Box<Expr>, Box<Expr>),
    /// A unary `not` operation.
    Not(Box<Expr>),
    /// A unary negation.
    Neg(Box<Expr>),
}

/// Binary operators supported by template expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
    /// `+` (string concatenation or numeric addition).
    Add,
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
}
