//! A script's `param`s: values the script gets from its step's settings (or a
//! group parameter) instead of from its code.
//!
//! ```rhai
//! param nuclei_channel: channel = 0;   // a setting of the step
//! param max_rounds = 10;               // type from the value: a number
//! let step = 100.0;                    // a plain variable, not a setting
//! ```
//!
//! `param` is a statement of the script language (Rhai custom syntax): it
//! declares a constant holding the step's value for it, or its default. The
//! default must be a plain value - a number, a decimal, `true`/`false` or a
//! text - so the declarations can be read without running the script
//! ([`declared_params`]).
//!
//! The script is the only declaration. The step stores just the values that
//! differ from the defaults, by name (`ScriptSettings::values`); the editor
//! shows one setting `values.<name>` per `param` ([`script_parameters`]),
//! which group parameters bind to like any other setting.

use evanalyzer_cfg::core_types::{ImageChannelIdx, ObjectClass};
use evanalyzer_cfg::settings::command_groups::ParamValue;
use evanalyzer_cfg::settings::parameter_def::{ParamType, ParameterDef};
use rhai::{
    Dynamic, Engine, EvalAltResult, ImmutableString, LexError, ParseError, ParseErrorType, Position,
};
use std::cell::RefCell;
use std::collections::BTreeMap;

/// A `param`'s value, typed.
#[derive(Debug, Clone, PartialEq)]
pub enum ScriptValue {
    Number(i64),
    Decimal(f64),
    Bool(bool),
    Text(String),
    Class(ObjectClass),
    Channel(ImageChannelIdx),
}

/// A `param`'s type. Without `: type` it follows from the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    Number,
    Decimal,
    Bool,
    Text,
    /// A class: a number in the script (`"L1"` for a group's own class).
    Class,
    /// An image channel: a number in the script.
    Channel,
}

impl ParamKind {
    fn name(self) -> &'static str {
        match self {
            Self::Number => "number",
            Self::Decimal => "decimal",
            Self::Bool => "bool",
            Self::Text => "text",
            Self::Class => "class",
            Self::Channel => "channel",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "number" => Self::Number,
            "decimal" => Self::Decimal,
            "bool" => Self::Bool,
            "text" => Self::Text,
            "class" => Self::Class,
            "channel" => Self::Channel,
            _ => return None,
        })
    }
}

/// A `param` line of a script.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamDecl {
    pub name: String,
    pub kind: ParamKind,
    /// The value after `=`.
    pub default: ScriptValue,
}

/// The constants every script has - a `param` can't take their names.
const BUILT_INS: [&str; 5] = [
    "image_width",
    "image_height",
    "tile_x",
    "tile_y",
    "image_bits",
];

thread_local! {
    /// Collects the declarations while [`declared_params`] compiles a script.
    static COLLECTED: RefCell<Option<Vec<ParamDecl>>> = const { RefCell::new(None) };
}

/// The `param` lines of `source`, in their order - checked: valid syntax,
/// plain-value defaults fitting their type, no name twice, no built-in name.
///
/// # Errors
/// A message naming the line, as for any script error.
pub fn declared_params(engine: &Engine, source: &str) -> Result<Vec<ParamDecl>, String> {
    COLLECTED.with_borrow_mut(|c| *c = Some(Vec::new()));
    let compiled = engine.compile(source);
    let decls = COLLECTED.with_borrow_mut(Option::take).unwrap_or_default();
    compiled.map_err(|e| e.to_string())?;
    for (index, decl) in decls.iter().enumerate() {
        if BUILT_INS.contains(&decl.name.as_str()) {
            return Err(format!(
                "param `{}`: the name is taken by the tile's own `{}`",
                decl.name, decl.name
            ));
        }
        if decls[..index].iter().any(|d| d.name == decl.name) {
            return Err(format!("param `{}` is declared twice", decl.name));
        }
    }
    Ok(decls)
}

/// A stored value as a value of `kind`.
///
/// # Errors
/// It's no value of `kind` (e.g. a text for a number).
pub fn from_param_value(kind: ParamKind, value: &ParamValue) -> Result<ScriptValue, String> {
    let wrong = || Err(format!("the step's value {value:?} is no {}", kind.name()));
    Ok(match (kind, value) {
        (ParamKind::Number, ParamValue::Int(n)) => ScriptValue::Number(*n),
        (ParamKind::Decimal, ParamValue::Int(n)) => ScriptValue::Decimal(*n as f64),
        (ParamKind::Decimal, ParamValue::Float(x)) => ScriptValue::Decimal(*x),
        (ParamKind::Bool, ParamValue::Bool(b)) => ScriptValue::Bool(*b),
        (ParamKind::Text, ParamValue::Text(t)) => ScriptValue::Text(t.clone()),
        (ParamKind::Class, ParamValue::Int(n)) if *n >= 0 => {
            ScriptValue::Class(ObjectClass::Global(*n as u32))
        }
        (ParamKind::Class, ParamValue::Int(-1)) => ScriptValue::Class(ObjectClass::Unset),
        (ParamKind::Class, ParamValue::Text(t)) => match ObjectClass::from_text(t) {
            Some(class) => ScriptValue::Class(class),
            None => return wrong(),
        },
        (ParamKind::Channel, ParamValue::Int(n)) if *n >= 0 => {
            ScriptValue::Channel(ImageChannelIdx(*n as u32))
        }
        _ => return wrong(),
    })
}

/// `value` as stored in the step.
fn to_param_value(value: &ScriptValue) -> ParamValue {
    match value {
        ScriptValue::Number(n) => ParamValue::Int(*n),
        ScriptValue::Decimal(x) => ParamValue::Float(*x),
        ScriptValue::Bool(b) => ParamValue::Bool(*b),
        ScriptValue::Text(t) => ParamValue::Text(t.clone()),
        ScriptValue::Class(class) => match class {
            ObjectClass::Global(n) => ParamValue::Int(*n as i64),
            ObjectClass::Unset => ParamValue::Int(-1),
            local => ParamValue::Text(local.to_text()),
        },
        ScriptValue::Channel(c) => ParamValue::Int(c.0 as i64),
    }
}

/// `value` as the editor's control shows it.
fn value_text(value: &ScriptValue) -> String {
    match value {
        ScriptValue::Number(n) => n.to_string(),
        ScriptValue::Decimal(x) => x.to_string(),
        ScriptValue::Bool(b) => b.to_string(),
        ScriptValue::Text(t) => t.clone(),
        ScriptValue::Class(class) => class.to_text(),
        ScriptValue::Channel(c) => c.0.to_string(),
    }
}

/// The value `decl` has in a step with `values`: the stored one if it fits,
/// else the default.
fn current(decl: &ParamDecl, values: &BTreeMap<String, ParamValue>) -> ScriptValue {
    values
        .get(&decl.name)
        .and_then(|v| from_param_value(decl.kind, v).ok())
        .unwrap_or_else(|| decl.default.clone())
}

/// The editor's settings for a step's `param`s: one per declaration, named
/// `values.<name>`, with the control of its type.
pub fn script_parameters(
    decls: &[ParamDecl],
    values: &BTreeMap<String, ParamValue>,
) -> Vec<ParameterDef> {
    decls
        .iter()
        .map(|decl| ParameterDef {
            name: format!("values.{}", decl.name),
            display_name: decl.name.clone(),
            description: format!("`param {}` of the script", decl.name),
            value: value_text(&current(decl, values)),
            param_type: match decl.kind {
                ParamKind::Number | ParamKind::Decimal => ParamType::Number,
                ParamKind::Bool => ParamType::Toggle,
                ParamKind::Text => ParamType::Text,
                ParamKind::Class => ParamType::ObjClass,
                ParamKind::Channel => ParamType::ImageChannel,
            },
            options: vec![],
            min: 0.0,
            max: 0.0,
            step: if decl.kind == ParamKind::Decimal {
                0.1
            } else {
                1.0
            },
            groups: vec![],
            default_value: value_text(&decl.default),
            advanced: false,
            option_advanced: vec![],
        })
        .collect()
}

/// Sets `param` `name`'s value in `values` from `text` (as the editor's
/// control reports it); a value equal to the default is removed - only
/// changed values are stored.
///
/// # Errors
/// The script declares no such `param`, or `text` is no value of its type.
pub fn set_script_value(
    decls: &[ParamDecl],
    values: &mut BTreeMap<String, ParamValue>,
    name: &str,
    text: &str,
) -> Result<(), String> {
    let decl = decls
        .iter()
        .find(|d| d.name == name)
        .ok_or_else(|| format!("the script has no `param {name}`"))?;
    let text = text.trim();
    let bad = || format!("`{text}` is no {}", decl.kind.name());
    let value = match decl.kind {
        ParamKind::Number => ScriptValue::Number(
            text.parse::<i64>()
                .or_else(|_| text.parse::<f64>().map(|x| x.round() as i64))
                .map_err(|_| bad())?,
        ),
        ParamKind::Decimal => ScriptValue::Decimal(text.parse().map_err(|_| bad())?),
        ParamKind::Bool => ScriptValue::Bool(match text {
            "true" => true,
            "false" => false,
            _ => return Err(bad()),
        }),
        ParamKind::Text => ScriptValue::Text(text.to_string()),
        ParamKind::Class => ScriptValue::Class(ObjectClass::from_text(text).ok_or_else(bad)?),
        ParamKind::Channel => {
            ScriptValue::Channel(ImageChannelIdx(text.parse().map_err(|_| bad())?))
        }
    };
    if value == decl.default {
        values.remove(name);
    } else {
        values.insert(name.to_string(), to_param_value(&value));
    }
    Ok(())
}

/// `value` as the script sees it.
pub fn to_dynamic(value: &ScriptValue) -> Dynamic {
    match value {
        ScriptValue::Number(n) => Dynamic::from(*n),
        ScriptValue::Decimal(x) => Dynamic::from(*x),
        ScriptValue::Bool(b) => Dynamic::from(*b),
        ScriptValue::Text(t) => Dynamic::from(t.clone()),
        ScriptValue::Class(ObjectClass::Global(n)) => Dynamic::from(*n as i64),
        ScriptValue::Class(ObjectClass::Unset) => Dynamic::from(-1_i64),
        ScriptValue::Class(class @ ObjectClass::Local(_)) => Dynamic::from(class.to_text()),
        ScriptValue::Channel(c) => Dynamic::from(c.0 as i64),
    }
}

fn bad(message: impl Into<String>) -> ParseError {
    ParseError(
        Box::new(ParseErrorType::BadInput(LexError::ImproperSymbol(
            "param".into(),
            message.into(),
        ))),
        Position::NONE,
    )
}

/// The default `text` (as written after `=`, `negative` if a `-` came first)
/// as a value of `kind`.
fn default_value(
    kind: ParamKind,
    marker: &str,
    text: &str,
    negative: bool,
) -> Result<ScriptValue, String> {
    let int = || -> Result<i64, String> {
        let n: i64 = text
            .replace('_', "")
            .parse()
            .map_err(|_| format!("`{text}` is not a whole number"))?;
        Ok(if negative { -n } else { n })
    };
    let wrong = |what: &str| {
        Err(format!(
            "a {what} param needs a {} default",
            match kind {
                ParamKind::Number | ParamKind::Class | ParamKind::Channel => "whole-number",
                ParamKind::Decimal => "number",
                ParamKind::Bool => "`true` or `false`",
                ParamKind::Text => "text",
            }
        ))
    };
    match (kind, marker) {
        (ParamKind::Number, "$int$") => Ok(ScriptValue::Number(int()?)),
        (ParamKind::Decimal, "$int$") => Ok(ScriptValue::Decimal(int()? as f64)),
        (ParamKind::Decimal, "$float$") => {
            let x: f64 = text
                .replace('_', "")
                .parse()
                .map_err(|_| format!("`{text}` is not a number"))?;
            Ok(ScriptValue::Decimal(if negative { -x } else { x }))
        }
        (ParamKind::Bool, "$bool$") => Ok(ScriptValue::Bool(text == "true")),
        (ParamKind::Text, "$string$") => Ok(ScriptValue::Text(text.to_string())),
        (ParamKind::Class, "$int$") => match int()? {
            n if n >= 0 => Ok(ScriptValue::Class(ObjectClass::Global(n as u32))),
            _ => Ok(ScriptValue::Class(ObjectClass::Unset)),
        },
        (ParamKind::Channel, "$int$") => u32::try_from(int()?)
            .map(|c| ScriptValue::Channel(ImageChannelIdx(c)))
            .map_err(|_| "a channel can't be negative".to_string()),
        (ParamKind::Number, _) => wrong("number"),
        (ParamKind::Decimal, _) => wrong("decimal"),
        (ParamKind::Bool, _) => wrong("bool"),
        (ParamKind::Text, _) => wrong("text"),
        (ParamKind::Class, _) => wrong("class"),
        (ParamKind::Channel, _) => wrong("channel"),
    }
}

/// The marker for a default starting with `look_ahead`.
fn literal_marker(look_ahead: &str) -> Option<&'static str> {
    match look_ahead {
        "true" | "false" => Some("$bool$"),
        s if s.starts_with('"') || s.starts_with('`') => Some("$string$"),
        // The lexer reads `-3` as one token.
        s if s
            .trim_start_matches('-')
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_digit()) =>
        {
            if s.contains('.') || s.contains('e') || s.contains('E') {
                Some("$float$")
            } else {
                Some("$int$")
            }
        }
        _ => None,
    }
}

/// Parses `param name [: type] = value;` one symbol at a time (see
/// `Engine::register_custom_syntax_with_state_raw`); the finished
/// declaration goes into `state` (for running) and to [`COLLECTED`].
fn parse(
    symbols: &[ImmutableString],
    look_ahead: &str,
    state: &mut Dynamic,
) -> Result<Option<ImmutableString>, ParseError> {
    let last = symbols.last().map(ImmutableString::as_str).unwrap_or("");
    let next = match symbols.len() {
        1 => "$ident$",
        2 if look_ahead == ":" => ":",
        2 => "=",
        // After ":": the type, then "=".
        3 if last == ":" => "$ident$",
        4 if symbols[2] == ":" => {
            if ParamKind::from_name(last).is_none() {
                return Err(bad(format!(
                    "param `{}`: unknown type `{last}` - number, decimal, bool, text, class or channel",
                    symbols[1]
                )));
            }
            "="
        }
        _ if last == "=" || last == "-" => {
            if last == "=" && look_ahead == "-" {
                "-"
            } else {
                literal_marker(look_ahead).ok_or_else(|| {
                    bad(format!(
                        "param `{}`: the default must be a plain value (a number, `true`/`false` or a text), not `{look_ahead}`",
                        symbols[1]
                    ))
                })?
            }
        }
        // The default was read: a `;` must follow - else `param n = 1 + 1`
        // would quietly go on as an expression.
        _ if last != ";" => {
            if look_ahead != ";" {
                return Err(bad(format!(
                    "param `{}`: the default must be a plain value (a number, `true`/`false` or a text) followed by `;`",
                    symbols[1]
                )));
            }
            ";"
        }
        _ => {
            // `name [: type] = value ;` read: the declaration is complete.
            let literal = symbols[symbols.len() - 2].as_str();
            let name = symbols[1].to_string();
            let negative = symbols.iter().any(|s| s == "-");
            // `symbols` holds the literal's text in place of its marker:
            // the marker asked for is in `state`.
            let marker = state
                .read_lock::<rhai::Map>()
                .and_then(|m| {
                    m.get("marker")
                        .and_then(|v| v.clone().into_immutable_string().ok())
                })
                .map(|s| s.to_string())
                .unwrap_or_default();
            let explicit = (symbols.len() > 2 && symbols[2] == ":")
                .then(|| ParamKind::from_name(&symbols[3]))
                .flatten();
            let kind = explicit.unwrap_or(match marker.as_str() {
                "$float$" => ParamKind::Decimal,
                "$bool$" => ParamKind::Bool,
                "$string$" => ParamKind::Text,
                _ => ParamKind::Number,
            });
            let default = default_value(kind, &marker, literal, negative)
                .map_err(|why| bad(format!("param `{name}`: {why}")))?;
            let decl = ParamDecl {
                name,
                kind,
                default,
            };
            COLLECTED.with_borrow_mut(|c| {
                if let Some(c) = c {
                    c.push(decl.clone());
                }
            });
            let mut map = rhai::Map::new();
            map.insert("name".into(), decl.name.into());
            map.insert("default".into(), Dynamic::from(decl.default.clone()));
            map.insert("kind".into(), decl.kind.name().into());
            *state = map.into();
            return Ok(None);
        }
    };
    if next.starts_with('$') && next != "$ident$" {
        // Remember which literal is read, for the step after.
        let mut map = state.clone().try_cast::<rhai::Map>().unwrap_or_default();
        map.insert("marker".into(), next.into());
        *state = map.into();
    }
    Ok(Some(next.into()))
}

/// Registers `param` with `engine`. `value_of(name)` is the step's value
/// for a param, if it has one; `uses_class` is told the class a class param
/// hands to the script (its commands may use it).
pub fn register(
    engine: &mut Engine,
    value_of: impl Fn(&str) -> Option<ParamValue> + 'static,
    uses_class: impl Fn(ObjectClass) + 'static,
) {
    engine.register_custom_syntax_with_state_raw(
        "param",
        parse,
        true,
        move |context, _inputs, state| {
            let map = state
                .read_lock::<rhai::Map>()
                .map(|m| m.clone())
                .unwrap_or_default();
            let name = map
                .get("name")
                .and_then(|v| v.clone().into_immutable_string().ok())
                .map(|s| s.to_string())
                .unwrap_or_default();
            let kind = map
                .get("kind")
                .and_then(|v| ParamKind::from_name(&v.to_string()))
                .unwrap_or(ParamKind::Number);
            let value = match value_of(&name) {
                Some(stored) => from_param_value(kind, &stored).map_err(|why| {
                    Box::new(EvalAltResult::ErrorRuntime(
                        format!("param `{name}`: {why}").into(),
                        Position::NONE,
                    ))
                })?,
                None => match map
                    .get("default")
                    .and_then(|d| d.clone().try_cast::<ScriptValue>())
                {
                    Some(default) => default,
                    None => return Ok(Dynamic::UNIT),
                },
            };
            if let ScriptValue::Class(class) = &value
                && *class != ObjectClass::Unset
            {
                uses_class(*class);
            }
            context
                .scope_mut()
                .push_constant_dynamic(name, to_dynamic(&value));
            Ok(Dynamic::UNIT)
        },
    );
}
