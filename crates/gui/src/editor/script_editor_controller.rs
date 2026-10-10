//! Backs the script editor dialog (`editor/script_editor_panel.slint`):
//! syntax highlighting, formatting, syntax check and the Tab/Enter/closing
//! bracket editing helpers for Rhai scripts.
//!
//! Everything here is a pure function of the text, so the Slint callbacks are
//! `pure` and re-run whenever the editor text changes. Cursor positions are
//! UTF-8 byte offsets, as Slint's `TextInput` reports them.

use crate::{
    AppWindow, ScriptCommandDoc, ScriptDiagnostic, ScriptEdit, ScriptEditorState, ScriptHighlight,
};
use evanalyzer_cfg::settings::parameter_def::{ParamType, ParameterDef};
use evanalyzer_cfg::settings::pipeline_command::{
    CommandCategory, PipelineCommand, all_command_meta,
};
use slint::{ComponentHandle, ModelRc, SharedString, VecModel};
use std::rc::Rc;

const INDENT: &str = "    ";
const INDENT_WIDTH: usize = INDENT.len();

/// Rhai keywords (incl. reserved ones) shown in the keyword color.
const KEYWORDS: &[&str] = &[
    "as", "break", "catch", "const", "continue", "do", "else", "export", "fn", "for", "global",
    "if", "import", "in", "let", "loop", "private", "return", "switch", "this", "throw", "try",
    "until", "while",
];

/// Literal constants shown in the number color.
const CONSTANTS: &[&str] = &["true", "false"];

pub struct ScriptEditorController {
    ui: slint::Weak<AppWindow>,
}

impl ScriptEditorController {
    pub fn new(ui: slint::Weak<AppWindow>) -> Self {
        Self { ui }
    }

    pub fn attach_callbacks(&self) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let state = ui.global::<ScriptEditorState>();
        state.on_highlight(|text| highlight(&text).into());
        state.on_check(|text| check_syntax(&text).into());
        state.on_format(|text| format_script(&text).into());
        state.on_indent(|text, cursor| indent(&text, to_offset(&text, cursor)).into());
        state.on_newline(|text, cursor| newline(&text, to_offset(&text, cursor)).into());
        state.on_close_bracket(|text, cursor, bracket| {
            let bracket = bracket.chars().next().unwrap_or('}');
            close_bracket(&text, to_offset(&text, cursor), bracket).into()
        });

        let docs = Rc::new(command_docs());
        state.on_filter_commands(move |query| {
            let shown: Vec<ScriptCommandDoc> = docs
                .iter()
                .filter(|d| d.matches(&query))
                .map(ScriptCommandDoc::from)
                .collect();
            ModelRc::new(VecModel::from(shown))
        });
        state.on_insert_command(|text, cursor, key| {
            let cursor = to_offset(&text, cursor);
            match run_snippet(&key, &text[line_start(&text, cursor)..cursor]) {
                Some(snippet) => insert(&text, cursor, &snippet, snippet.len()).into(),
                None => Edit {
                    text: text.to_string(),
                    cursor,
                }
                .into(),
            }
        });
    }
}

/// Clamps a cursor coming from Slint to a valid char boundary of `text`.
fn to_offset(text: &str, cursor: i32) -> usize {
    let mut offset = (cursor.max(0) as usize).min(text.len());
    while !text.is_char_boundary(offset) {
        offset -= 1;
    }
    offset
}

// ----------------------------------------------------------------------------
// Lexer

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenKind {
    Comment,
    String,
    Number,
    Keyword,
    Function,
    Ident,
    /// Any other single character (operators, brackets, `;`, `#`, ...).
    Punct,
    Whitespace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Token {
    kind: TokenKind,
    /// Byte range in the source.
    start: usize,
    end: usize,
}

/// Splits a script into tokens covering every byte of it. Never fails:
/// unterminated strings and comments simply run to the end of the text, so
/// half-typed code still highlights sensibly.
fn tokenize(src: &str) -> Vec<Token> {
    let bytes = src.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < src.len() {
        let c = src[i..].chars().next().unwrap_or(' ');
        let start = i;
        let kind = if c.is_whitespace() {
            i += c.len_utf8();
            while let Some(n) = src[i..].chars().next().filter(|n| n.is_whitespace()) {
                i += n.len_utf8();
            }
            TokenKind::Whitespace
        } else if src[i..].starts_with("//") {
            i = src[i..].find('\n').map_or(src.len(), |n| i + n);
            TokenKind::Comment
        } else if src[i..].starts_with("/*") {
            i = block_comment_end(src, i);
            TokenKind::Comment
        } else if c == '"' || c == '\'' {
            i = quoted_end(src, i, c, false);
            TokenKind::String
        } else if c == '`' {
            i = quoted_end(src, i, '`', true);
            TokenKind::String
        } else if c.is_ascii_digit() {
            i = number_end(src, i);
            TokenKind::Number
        } else if c.is_alphabetic() || c == '_' {
            while let Some(n) = src[i..]
                .chars()
                .next()
                .filter(|n| n.is_alphanumeric() || *n == '_')
            {
                i += n.len_utf8();
            }
            let word = &src[start..i];
            if KEYWORDS.contains(&word) {
                TokenKind::Keyword
            } else if CONSTANTS.contains(&word) {
                TokenKind::Number
            } else if is_call(bytes, i) {
                TokenKind::Function
            } else {
                TokenKind::Ident
            }
        } else {
            i += c.len_utf8();
            TokenKind::Punct
        };
        tokens.push(Token {
            kind,
            start,
            end: i,
        });
    }
    tokens
}

/// End of a (nestable, as in Rhai) block comment starting at `start`.
fn block_comment_end(src: &str, start: usize) -> usize {
    let bytes = src.as_bytes();
    let mut depth = 0usize;
    let mut i = start;
    while i + 1 < bytes.len() {
        match (bytes[i], bytes[i + 1]) {
            (b'/', b'*') => {
                depth += 1;
                i += 2;
            }
            (b'*', b'/') => {
                depth -= 1;
                i += 2;
                if depth == 0 {
                    return i;
                }
            }
            _ => i += 1,
        }
    }
    src.len()
}

/// End of a string/char literal opened by `quote` at `start`. Only
/// backtick strings may span lines.
fn quoted_end(src: &str, start: usize, quote: char, multi_line: bool) -> usize {
    let mut chars = src[start..].char_indices().skip(1);
    while let Some((offset, c)) = chars.next() {
        match c {
            '\\' if quote != '`' => {
                chars.next();
            }
            '\n' if !multi_line => return start + offset,
            c if c == quote => return start + offset + c.len_utf8(),
            _ => {}
        }
    }
    src.len()
}

/// End of a number literal: decimal/hex/binary/octal digits, `_`
/// separators, a fraction (but not the `..` range operator) and an exponent.
fn number_end(src: &str, start: usize) -> usize {
    let bytes = src.as_bytes();
    let is_radix = bytes.get(start) == Some(&b'0')
        && matches!(
            bytes.get(start + 1),
            Some(b'x' | b'X' | b'b' | b'B' | b'o' | b'O')
        );
    let mut i = start;
    while i < bytes.len() {
        let b = bytes[i];
        let next_is_digit = bytes.get(i + 1).is_some_and(u8::is_ascii_digit);
        if b.is_ascii_alphanumeric() || b == b'_' {
            i += 1;
        } else if b == b'.' && !is_radix && next_is_digit {
            i += 1;
        } else if (b == b'+' || b == b'-') && !is_radix && matches!(bytes[i - 1], b'e' | b'E') {
            i += 1;
        } else {
            break;
        }
    }
    i
}

/// A word followed by `(` (or `!(` for macro-like calls) is a function call
/// or definition.
fn is_call(bytes: &[u8], word_end: usize) -> bool {
    match bytes.get(word_end) {
        Some(b'(') => true,
        Some(b'!') => bytes.get(word_end + 1) == Some(&b'('),
        _ => false,
    }
}

// ----------------------------------------------------------------------------
// Highlighting

#[derive(Debug, Default, PartialEq)]
struct Highlight {
    keywords: String,
    functions: String,
    strings: String,
    numbers: String,
    comments: String,
    line_numbers: String,
}

impl From<Highlight> for ScriptHighlight {
    fn from(h: Highlight) -> Self {
        Self {
            keywords: h.keywords.into(),
            functions: h.functions.into(),
            strings: h.strings.into(),
            numbers: h.numbers.into(),
            comments: h.comments.into(),
            line_numbers: h.line_numbers.into(),
        }
    }
}

/// One text layer per colored token kind (see the Slint file for why): the
/// whole script with every character not of that kind blanked to a space.
/// Newlines and tabs are kept everywhere so the layers stay aligned.
fn highlight(src: &str) -> Highlight {
    let mut layers: [String; 5] = Default::default();
    for token in tokenize(src) {
        let layer = match token.kind {
            TokenKind::Keyword => Some(0),
            TokenKind::Function => Some(1),
            TokenKind::String => Some(2),
            TokenKind::Number => Some(3),
            TokenKind::Comment => Some(4),
            _ => None,
        };
        let text = &src[token.start..token.end];
        for (i, out) in layers.iter_mut().enumerate() {
            if layer == Some(i) {
                out.push_str(text);
            } else {
                out.extend(text.chars().map(|c| match c {
                    '\n' | '\t' => c,
                    _ => ' ',
                }));
            }
        }
    }
    let line_count = src.split('\n').count();
    let line_numbers = (1..=line_count)
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    let [keywords, functions, strings, numbers, comments] = layers;
    Highlight {
        keywords,
        functions,
        strings,
        numbers,
        comments,
        line_numbers,
    }
}

// ----------------------------------------------------------------------------
// Syntax check

#[derive(Debug, PartialEq)]
struct Diagnostic {
    ok: bool,
    line: i32,
    message: String,
}

impl From<Diagnostic> for ScriptDiagnostic {
    fn from(d: Diagnostic) -> Self {
        Self {
            ok: d.ok,
            line: d.line,
            message: d.message.into(),
        }
    }
}

/// Parses the script without running it. Unknown functions are not errors
/// here - Rhai resolves calls at run time.
fn check_syntax(src: &str) -> Diagnostic {
    thread_local! {
        static ENGINE: rhai::Engine = rhai::Engine::new();
    }
    ENGINE.with(|engine| match engine.compile(src) {
        Ok(_) => Diagnostic {
            ok: true,
            line: 0,
            message: String::new(),
        },
        Err(err) => Diagnostic {
            ok: false,
            line: err.position().line().unwrap_or(0) as i32,
            message: err.err_type().to_string(),
        },
    })
}

// ----------------------------------------------------------------------------
// Formatting

/// Re-indents the script by bracket depth (four spaces per level), trims
/// trailing whitespace, collapses runs of blank lines into one and ends the
/// text with a single newline. Lines inside block comments and multi-line
/// backtick strings are kept exactly as they are.
fn format_script(src: &str) -> String {
    let tokens = tokenize(src);
    let mut out: Vec<String> = Vec::new();
    let mut depth: usize = 0;
    let mut token_idx = 0;
    let mut line_start = 0;
    let mut previous_blank = true; // drops leading blank lines

    for line in src.split('\n') {
        let line_end = line_start + line.len();

        // Is the start of this line inside a token that began earlier
        // (block comment / backtick string)?
        while token_idx < tokens.len() && tokens[token_idx].end <= line_start {
            token_idx += 1;
        }
        let inside_token = tokens.get(token_idx).is_some_and(|t| {
            t.start < line_start && matches!(t.kind, TokenKind::Comment | TokenKind::String)
        });

        // Bracket depth change on this line, and closers leading it.
        let mut leading_closers = 0;
        let mut seen_other = false;
        let mut delta: isize = 0;
        for t in tokens[token_idx..]
            .iter()
            .take_while(|t| t.start < line_end)
        {
            if t.start < line_start {
                continue;
            }
            match (t.kind, &src[t.start..t.end]) {
                (TokenKind::Whitespace, _) => {}
                (TokenKind::Punct, "(" | "[" | "{") => {
                    delta += 1;
                    seen_other = true;
                }
                (TokenKind::Punct, ")" | "]" | "}") => {
                    delta -= 1;
                    if !seen_other {
                        leading_closers += 1;
                    }
                }
                _ => seen_other = true,
            }
        }

        if inside_token {
            out.push(line.to_string());
            previous_blank = false;
        } else {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                if !previous_blank {
                    out.push(String::new());
                }
                previous_blank = true;
            } else {
                let level = depth.saturating_sub(leading_closers);
                out.push(format!("{}{}", INDENT.repeat(level), trimmed));
                previous_blank = false;
            }
        }

        depth = (depth as isize + delta).max(0) as usize;
        line_start = line_end + 1;
    }

    while out.last().is_some_and(String::is_empty) {
        out.pop();
    }
    if out.is_empty() {
        return String::new();
    }
    let mut formatted = out.join("\n");
    formatted.push('\n');
    formatted
}

// ----------------------------------------------------------------------------
// Command reference

/// One command a script can `run`, for the editor's command list.
#[derive(Debug, Clone, PartialEq)]
struct CommandDoc {
    key: &'static str,
    name: String,
    /// Summary plus one line per parameter.
    doc: String,
}

impl CommandDoc {
    /// Case-insensitive match on display name or key.
    fn matches(&self, query: &str) -> bool {
        let query = query.trim().to_lowercase();
        query.is_empty()
            || self.name.to_lowercase().contains(&query)
            || self.key.contains(&query.replace(' ', "_"))
    }
}

impl From<&CommandDoc> for ScriptCommandDoc {
    fn from(d: &CommandDoc) -> Self {
        Self {
            key: d.key.into(),
            name: d.name.as_str().into(),
            doc: d.doc.as_str().into(),
        }
    }
}

/// Every command a script step can run: all but the script itself and the
/// Object category, whose commands work on the whole image's objects (core
/// tests that this category is exactly those commands).
fn command_docs() -> Vec<CommandDoc> {
    let meta = all_command_meta();
    let mut docs: Vec<CommandDoc> = PipelineCommand::KEYS
        .iter()
        .filter(|key| **key != "script")
        .filter_map(|key| {
            let cmd = PipelineCommand::default_for_key(key)?;
            if matches!(cmd.category(), CommandCategory::Object) {
                return None;
            }
            let summary = meta
                .iter()
                .find(|m| m.name == cmd.name())
                .map_or("", |m| m.summary);
            let mut doc = String::new();
            if !summary.is_empty() {
                doc.push_str(summary);
                doc.push_str("\n\n");
            }
            for (name, def) in snippet_params(&cmd) {
                doc.push_str(&format!("{name}: {}", param_type_doc(&def)));
                if !def.description.is_empty() {
                    let first = def.description.lines().next().unwrap_or("");
                    doc.push_str(&format!("\n    {first}"));
                }
                doc.push('\n');
            }
            Some(CommandDoc {
                key,
                name: cmd.name().to_string(),
                doc: doc.trim_end().to_string(),
            })
        })
        .collect();
    docs.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    docs
}

/// The settable parameters of `cmd` by full name, as a script writes them.
/// An empty list parameter gets one default entry so its fields show up.
fn snippet_params(cmd: &PipelineCommand) -> Vec<(String, ParameterDef)> {
    let mut cmd = cmd.clone();
    for p in cmd.to_parameters() {
        if p.param_type == ParamType::Group && p.groups.is_empty() {
            cmd.add_group_item(&p.name);
        }
    }
    fn walk(params: &[ParameterDef], prefix: &str, out: &mut Vec<(String, ParameterDef)>) {
        for p in params {
            let name = format!("{prefix}{}", p.name);
            match p.param_type {
                ParamType::Group => {
                    for (i, item) in p.groups.iter().enumerate() {
                        walk(item, &format!("{name}.{i}."), out);
                    }
                }
                ParamType::Label | ParamType::Script => {}
                _ => out.push((name, p.clone())),
            }
        }
    }
    let mut out = Vec::new();
    walk(&cmd.to_parameters(), "", &mut out);
    out
}

fn param_type_doc(def: &ParameterDef) -> String {
    let default = &def.default_value;
    match def.param_type {
        ParamType::Number | ParamType::Spinner | ParamType::Slider if def.max > def.min => {
            format!("number {}..={} (default {default})", def.min, def.max)
        }
        ParamType::Number | ParamType::Spinner | ParamType::Slider => {
            format!("number (default {default})")
        }
        ParamType::Toggle => format!("true/false (default {default})"),
        ParamType::Dropdown | ParamType::PixelUnits | ParamType::SizeUnits => {
            format!("one of {} (default {default})", def.options.join(", "))
        }
        ParamType::ObjClass => "object class id, -1 = none".into(),
        ParamType::SegClass => "segmentation class id".into(),
        ParamType::MultiObjClass | ParamType::MultiSegClass => {
            "list of class ids, e.g. [1, 2]".into()
        }
        ParamType::ImageChannel => "channel index".into(),
        ParamType::ImageAddress => "\"channel:N\", \"memory:N\" or \"scratchpad\"".into(),
        ParamType::Text | ParamType::FilePath => "text".into(),
        ParamType::Group | ParamType::Label | ParamType::Script => String::new(),
    }
}

/// `run("key", #{ ... });` with every parameter at its default, indented
/// for a cursor after `line_prefix`. `None` for an unknown key.
fn run_snippet(key: &str, line_prefix: &str) -> Option<String> {
    let cmd = PipelineCommand::default_for_key(key)?;
    let params = snippet_params(&cmd);
    if params.is_empty() {
        return Some(format!("run(\"{key}\");"));
    }
    let base: String = line_prefix
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect();
    let mut out = format!("run(\"{key}\", #{{\n");
    for (name, def) in &params {
        let name = if name.contains('.') {
            format!("\"{name}\"")
        } else {
            name.clone()
        };
        out.push_str(&format!("{base}{INDENT}{name}: {},\n", script_literal(def)));
    }
    out.push_str(&format!("{base}}});"));
    Some(out)
}

/// A parameter's current value as a script literal.
fn script_literal(def: &ParameterDef) -> String {
    let value = def.value.as_str();
    let quoted = || format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""));
    match def.param_type {
        ParamType::Number
        | ParamType::Spinner
        | ParamType::Slider
        | ParamType::ObjClass
        | ParamType::SegClass
        | ParamType::ImageChannel => {
            if value.parse::<f64>().is_ok() {
                value.to_string()
            } else {
                "0".to_string()
            }
        }
        ParamType::Toggle => if value == "true" { "true" } else { "false" }.to_string(),
        ParamType::MultiObjClass | ParamType::MultiSegClass => {
            let ids: Vec<&str> = value
                .split(',')
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .collect();
            format!("[{}]", ids.join(", "))
        }
        _ => quoted(),
    }
}

// ----------------------------------------------------------------------------
// Editing helpers

#[derive(Debug, PartialEq)]
struct Edit {
    text: String,
    cursor: usize,
}

impl From<Edit> for ScriptEdit {
    fn from(e: Edit) -> Self {
        Self {
            text: SharedString::from(e.text),
            cursor: e.cursor as i32,
        }
    }
}

fn line_start(text: &str, cursor: usize) -> usize {
    text[..cursor].rfind('\n').map_or(0, |n| n + 1)
}

fn insert(text: &str, cursor: usize, insertion: &str, cursor_in_insertion: usize) -> Edit {
    let mut out = String::with_capacity(text.len() + insertion.len());
    out.push_str(&text[..cursor]);
    out.push_str(insertion);
    out.push_str(&text[cursor..]);
    Edit {
        text: out,
        cursor: cursor + cursor_in_insertion,
    }
}

/// Tab: spaces up to the next multiple of four columns.
fn indent(text: &str, cursor: usize) -> Edit {
    let column = text[line_start(text, cursor)..cursor].chars().count();
    let spaces = " ".repeat(INDENT_WIDTH - column % INDENT_WIDTH);
    insert(text, cursor, &spaces, spaces.len())
}

/// Enter: keeps the current line's indentation, one level deeper after an
/// opening bracket. Between a pair like `{|}` the closer moves to its own
/// line, back at the outer indentation.
fn newline(text: &str, cursor: usize) -> Edit {
    let start = line_start(text, cursor);
    let before = &text[start..cursor];
    let base: String = before
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect();
    let opener = before.trim_end().chars().last();
    let opens = matches!(opener, Some('{' | '(' | '['));
    if !opens {
        return insert(text, cursor, &format!("\n{base}"), 1 + base.len());
    }
    let inner = format!("\n{base}{INDENT}");
    let closes_pair = matches!(
        (opener, text[cursor..].chars().next()),
        (Some('{'), Some('}')) | (Some('('), Some(')')) | (Some('['), Some(']'))
    );
    if closes_pair {
        insert(text, cursor, &format!("{inner}\n{base}"), inner.len())
    } else {
        insert(text, cursor, &inner, inner.len())
    }
}

/// A closing bracket typed on a line with only whitespace before the
/// cursor first removes one indentation level.
fn close_bracket(text: &str, cursor: usize, bracket: char) -> Edit {
    let start = line_start(text, cursor);
    let before = &text[start..cursor];
    if before.is_empty() || !before.chars().all(|c| c == ' ') {
        let s = bracket.to_string();
        return insert(text, cursor, &s, s.len());
    }
    let remove = before.len().min(INDENT_WIDTH);
    let mut out = String::with_capacity(text.len() + 1);
    out.push_str(&text[..cursor - remove]);
    out.push(bracket);
    out.push_str(&text[cursor..]);
    Edit {
        text: out,
        cursor: cursor - remove + bracket.len_utf8(),
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<(TokenKind, &str)> {
        tokenize(src)
            .into_iter()
            .filter(|t| t.kind != TokenKind::Whitespace)
            .map(|t| (t.kind, &src[t.start..t.end]))
            .collect()
    }

    #[test]
    fn tokens_cover_the_whole_source() {
        let src = "let x = foo(1.5e-3, \"a\\\"b\", `x\n${y}`); // done\n/* a /* nested */ b */ ä";
        let tokens = tokenize(src);
        let mut expected_start = 0;
        for t in &tokens {
            assert_eq!(t.start, expected_start);
            expected_start = t.end;
        }
        assert_eq!(expected_start, src.len());
    }

    #[test]
    fn tokenizer_classifies_rhai_constructs() {
        use TokenKind::*;
        assert_eq!(
            kinds(r#"let n = len("ab") + 0x1F; // c"#),
            vec![
                (Keyword, "let"),
                (Ident, "n"),
                (Punct, "="),
                (Function, "len"),
                (Punct, "("),
                (String, "\"ab\""),
                (Punct, ")"),
                (Punct, "+"),
                (Number, "0x1F"),
                (Punct, ";"),
                (Comment, "// c"),
            ]
        );
    }

    #[test]
    fn range_is_not_part_of_a_number() {
        use TokenKind::*;
        assert_eq!(
            kinds("1..10"),
            vec![(Number, "1"), (Punct, "."), (Punct, "."), (Number, "10")]
        );
    }

    #[test]
    fn nested_block_comment_is_one_token() {
        assert_eq!(
            kinds("/* a /* b */ c */ x"),
            vec![
                (TokenKind::Comment, "/* a /* b */ c */"),
                (TokenKind::Ident, "x")
            ]
        );
    }

    #[test]
    fn unterminated_string_stops_at_line_end() {
        assert_eq!(
            kinds("\"abc\nlet"),
            vec![(TokenKind::String, "\"abc"), (TokenKind::Keyword, "let")]
        );
    }

    #[test]
    fn highlight_layers_keep_the_layout() {
        let src = "let a = \"ä\";\n\tprint(a); // hi";
        let h = highlight(src);
        for layer in [
            &h.keywords,
            &h.functions,
            &h.strings,
            &h.numbers,
            &h.comments,
        ] {
            assert_eq!(layer.chars().count(), src.chars().count());
            assert_eq!(layer.matches('\n').count(), 1);
            assert_eq!(layer.matches('\t').count(), 1);
        }
        assert_eq!(h.keywords.trim_end(), "let");
        assert_eq!(h.strings.trim(), "\"ä\"");
        assert_eq!(h.functions.trim(), "print");
        assert_eq!(h.comments.trim(), "// hi");
        assert_eq!(h.line_numbers, "1\n2");
    }

    #[test]
    fn check_accepts_valid_and_reports_line_of_invalid_scripts() {
        assert!(check_syntax("let a = 1;\nprint(a);").ok);
        let d = check_syntax("let a = 1;\nlet = ;");
        assert!(!d.ok);
        assert_eq!(d.line, 2);
        assert!(!d.message.is_empty());
    }

    #[test]
    fn format_reindents_by_bracket_depth() {
        let src = "\n\nif a {\nlet b = [\n1,\n2\n];\n   }   \n\n\n\nprint(b);";
        assert_eq!(
            format_script(src),
            "if a {\n    let b = [\n        1,\n        2\n    ];\n}\n\nprint(b);\n"
        );
    }

    #[test]
    fn format_handles_else_on_closing_line() {
        assert_eq!(
            format_script("if a {\nx();\n} else {\ny();\n}"),
            "if a {\n    x();\n} else {\n    y();\n}\n"
        );
    }

    #[test]
    fn format_ignores_brackets_in_strings_and_comments() {
        assert_eq!(
            format_script("if a {\nprint(\"{\"); // {\n}"),
            "if a {\n    print(\"{\"); // {\n}\n"
        );
    }

    #[test]
    fn format_keeps_multi_line_strings_and_comments_verbatim() {
        let src = "let s = `a\n   b  \nc`;\n/*\n   x\n*/\n";
        assert_eq!(format_script(src), src);
    }

    #[test]
    fn format_is_idempotent() {
        let once = format_script("fn f(x) {\nif x {\nreturn [1,\n2];\n}\n}\n");
        assert_eq!(format_script(&once), once);
    }

    #[test]
    fn indent_goes_to_next_tab_stop() {
        assert_eq!(
            indent("ab", 2),
            Edit {
                text: "ab  ".into(),
                cursor: 4
            }
        );
        assert_eq!(
            indent("x\n", 2),
            Edit {
                text: "x\n    ".into(),
                cursor: 6
            }
        );
    }

    #[test]
    fn newline_keeps_and_extends_indentation() {
        assert_eq!(
            newline("    a;", 6),
            Edit {
                text: "    a;\n    ".into(),
                cursor: 11
            }
        );
        assert_eq!(
            newline("if a {", 6),
            Edit {
                text: "if a {\n    ".into(),
                cursor: 11
            }
        );
    }

    #[test]
    fn newline_between_brackets_moves_closer_down() {
        let edit = newline("  f({})", 5);
        assert_eq!(edit.text, "  f({\n      \n  })");
        assert_eq!(edit.cursor, 12);
    }

    #[test]
    fn close_bracket_dedents_blank_line() {
        assert_eq!(
            close_bracket("if a {\n        ", 15, '}'),
            Edit {
                text: "if a {\n    }".into(),
                cursor: 12
            }
        );
        assert_eq!(
            close_bracket("f(a", 3, ')'),
            Edit {
                text: "f(a)".into(),
                cursor: 4
            }
        );
    }

    #[test]
    fn command_docs_offer_tile_commands_only() {
        let docs = command_docs();
        let keys: Vec<&str> = docs.iter().map(|d| d.key).collect();
        assert!(keys.contains(&"gaussian_blur"));
        assert!(keys.contains(&"threshold"));
        assert!(!keys.contains(&"voronoi"), "whole-image command offered");
        assert!(!keys.contains(&"script"));
        let blur = docs.iter().find(|d| d.key == "gaussian_blur").unwrap();
        assert!(blur.doc.contains("kernel_size: number"), "{}", blur.doc);
    }

    #[test]
    fn command_search_matches_name_and_key() {
        let docs = command_docs();
        let blur = docs.iter().find(|d| d.key == "gaussian_blur").unwrap();
        assert!(blur.matches("gauss"));
        assert!(blur.matches("GAUSSIAN BLUR"));
        assert!(blur.matches(""));
        assert!(!blur.matches("watershed"));
    }

    #[test]
    fn every_snippet_is_valid_rhai() {
        for doc in command_docs() {
            let snippet = run_snippet(doc.key, "    ").unwrap();
            let d = check_syntax(&snippet);
            assert!(d.ok, "{}: {}\n{snippet}", doc.key, d.message);
        }
    }

    #[test]
    fn snippet_lists_defaults_and_list_entries() {
        let blur = run_snippet("gaussian_blur", "").unwrap();
        assert!(
            blur.starts_with("run(\"gaussian_blur\", #{\n    kernel_size: 3,\n"),
            "{blur}"
        );
        assert!(blur.ends_with("});"), "{blur}");
        let threshold = run_snippet("threshold", "").unwrap();
        assert!(
            threshold.contains("\"thresholds.0.method\": \"Manual\""),
            "{threshold}"
        );
        assert!(run_snippet("no_such_command", "").is_none());
    }

    #[test]
    fn cursor_from_slint_is_clamped_to_a_char_boundary() {
        assert_eq!(to_offset("ä", 1), 0);
        assert_eq!(to_offset("ab", 99), 2);
        assert_eq!(to_offset("ab", -3), 0);
    }
}
