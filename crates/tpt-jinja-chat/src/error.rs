//! Error types for parsing and rendering Jinja chat templates.

use std::fmt;

/// Errors that can occur while parsing or rendering a chat template.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TemplateError {
    /// A syntax error encountered while parsing the template.
    Parse {
        /// Human-readable description of the problem.
        message: String,
        /// Byte offset into the source template where the error occurred.
        position: usize,
    },
    /// A semantic error encountered while rendering (e.g. bad types).
    Render(String),
    /// Reference to a variable that was not present in the context.
    UndefinedVariable(String),
    /// An operation was applied to a value of an incompatible type.
    TypeError(String),
    /// The embedded JSON parser failed to parse a string.
    Json(String),
    /// A template explicitly raised an error via `raise_exception(...)`.
    Exception(String),
}

impl TemplateError {
    /// Construct a [`TemplateError::Parse`] variant.
    pub fn parse(message: impl Into<String>, position: usize) -> Self {
        TemplateError::Parse {
            message: message.into(),
            position,
        }
    }

    /// Construct a [`TemplateError::Render`] variant.
    pub fn render(message: impl Into<String>) -> Self {
        TemplateError::Render(message.into())
    }

    /// Construct a [`TemplateError::UndefinedVariable`] variant.
    pub fn undefined_variable(name: impl Into<String>) -> Self {
        TemplateError::UndefinedVariable(name.into())
    }

    /// Construct a [`TemplateError::TypeError`] variant.
    pub fn type_error(message: impl Into<String>) -> Self {
        TemplateError::TypeError(message.into())
    }

    /// Construct a [`TemplateError::Json`] variant.
    pub fn json(message: impl Into<String>) -> Self {
        TemplateError::Json(message.into())
    }

    /// Construct a [`TemplateError::Exception`] variant (from `raise_exception`).
    pub fn exception(message: impl Into<String>) -> Self {
        TemplateError::Exception(message.into())
    }
}

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemplateError::Parse { message, position } => {
                write!(f, "template parse error at byte {position}: {message}")
            }
            TemplateError::Render(m) => write!(f, "template render error: {m}"),
            TemplateError::UndefinedVariable(n) => write!(f, "undefined variable: {n}"),
            TemplateError::TypeError(m) => write!(f, "type error: {m}"),
            TemplateError::Json(m) => write!(f, "json error: {m}"),
            TemplateError::Exception(m) => write!(f, "template raised: {m}"),
        }
    }
}

impl std::error::Error for TemplateError {}
