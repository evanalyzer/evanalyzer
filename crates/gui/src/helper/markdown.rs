//! # markdown
//!
//! The help texts' markdown (see `ui/ink/markdown.slint`): the window's
//! `InkMarkdown` converts them here, as Slint's `@markdown` only reads
//! string literals.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::{AppWindow, InkMarkdown};
use slint::{ComponentHandle, StyledText};

/// Lets the window draw markdown.
pub(crate) fn attach(ui: &AppWindow) {
    let markdown = ui.global::<InkMarkdown>();
    markdown.on_render(|text| render(&text));
    markdown.on_plain(|text| plain(&text).into());
    markdown.on_is_web_link(|url| is_web_link(&url));
}

/// Whether a link may be opened: `http://` or `https://` with something
/// after it. Anything else - `file://`, a UNC path (Windows sends the
/// user's login hash to the share), `mailto:`, an app's own scheme - is
/// not opened.
pub(crate) fn is_web_link(url: &str) -> bool {
    let url = url.trim();
    let lower = url.to_ascii_lowercase();
    ["https://", "http://"]
        .iter()
        .any(|scheme| lower.starts_with(scheme) && url.len() > scheme.len())
}

/// `text` styled; as plain text if it isn't markdown Slint can draw.
pub(crate) fn render(text: &str) -> StyledText {
    let text = &reflow(text);
    StyledText::from_markdown(text).unwrap_or_else(|err| {
        log::debug!("Help text drawn as plain text ({err:?}): {text}");
        StyledText::from_plain_text(text)
    })
}

/// A paragraph's lines joined into one, as markdown means them (Slint's
/// `StyledText` breaks the line at each); a list item keeps its own line.
fn reflow(text: &str) -> String {
    let starts_item = |line: &str| {
        let line = line.trim_start();
        line.starts_with("- ")
            || line.starts_with("* ")
            || line
                .split_once(". ")
                .is_some_and(|(n, _)| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
    };
    let mut out = String::with_capacity(text.len());
    let mut previous_blank = true;
    for line in text.lines() {
        let blank = line.trim().is_empty();
        if !out.is_empty() {
            out.push(if previous_blank || blank || starts_item(line) {
                '\n'
            } else {
                ' '
            });
        }
        out.push_str(if blank { "" } else { line.trim() });
        previous_blank = blank;
    }
    out
}

/// `text` without its marks, on one line: `**a**` / `*a*` / `` `a` `` -> a,
/// `[a](url)` -> a.
pub(crate) fn plain(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find('[') {
        let after = &rest[open + 1..];
        match after.find("](").and_then(|close| {
            after[close + 2..]
                .find(')')
                .map(|end| (close, close + 2 + end))
        }) {
            Some((close, end)) => {
                out.push_str(&rest[..open]);
                out.push_str(&after[..close]);
                rest = &after[end + 1..];
            }
            None => {
                out.push_str(&rest[..=open]);
                rest = after;
            }
        }
    }
    out.push_str(rest);
    let out = out.replace('`', "");
    // A `*` touching a word is emphasis; one standing alone (`a * b`) is not.
    let chars: Vec<char> = out.chars().collect();
    let kept: String = chars
        .iter()
        .enumerate()
        .filter(|&(i, &c)| {
            c != '*'
                || (i.checked_sub(1).is_none_or(|p| chars[p].is_whitespace())
                    && chars.get(i + 1).is_none_or(|n| n.is_whitespace()))
        })
        .map(|(_, &c)| c)
        .collect();
    kept.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::settings::parameter_def::ParameterDef;
    use evanalyzer_cfg::settings::pipeline_command::{all_command_meta, default_command};

    /// Web addresses open; a shared module's link to a file, a network
    /// share (Windows would log in there with the user's hash) or an
    /// app's own scheme does not.
    #[test]
    fn only_web_links_are_opened() {
        for url in [
            "https://evanalyzer.org",
            "http://example.org/paper.pdf",
            "HTTPS://Example.org",
            "  https://example.org  ",
        ] {
            assert!(is_web_link(url), "{url}");
        }
        for url in [
            "file:///etc/passwd",
            "file://\\\\attacker\\share\\x",
            "\\\\attacker\\share\\x",
            "//attacker/share",
            "smb://attacker/share",
            "mailto:a@b.c",
            "javascript:alert(1)",
            "ms-settings:",
            "C:\\Windows\\System32\\calc.exe",
            "https://",
            "",
        ] {
            assert!(!is_web_link(url), "{url}");
        }
    }

    /// The window asks Rust before it opens a link.
    #[test]
    fn the_window_opens_web_links_only() {
        let ui = crate::editor::test_support::test_ui_window();
        let markdown = ui.global::<InkMarkdown>();
        assert!(markdown.invoke_is_web_link("https://evanalyzer.org".into()));
        assert!(!markdown.invoke_is_web_link("file://\\\\attacker\\share\\x".into()));
    }

    #[test]
    fn a_paragraphs_lines_are_one_line_a_list_items_its_own() {
        assert_eq!(
            reflow("A long\nsentence.\n\n- one\n  more\n- two\n1. three"),
            "A long sentence.\n\n- one more\n- two\n1. three"
        );
    }

    #[test]
    fn plain_drops_the_marks() {
        assert_eq!(
            plain("Runs a [Rhai](https://rhai.rs) script on **every** `tile`,\n*fast*."),
            "Runs a Rhai script on every tile, fast."
        );
        assert_eq!(
            plain("output = input * contrast"),
            "output = input * contrast"
        );
        assert_eq!(plain("[1, 2] stays"), "[1, 2] stays");
    }

    fn all_parameters(params: Vec<ParameterDef>, out: &mut Vec<ParameterDef>) {
        for p in params {
            for group in &p.groups {
                all_parameters(group.clone(), out);
            }
            out.push(p);
        }
    }

    /// Every step's and setting's help is markdown the window can draw -
    /// no rustdoc left (`[`Type`]` links, code blocks, headings).
    #[test]
    fn every_help_text_is_drawable_markdown() {
        let mut texts = Vec::new();
        for meta in all_command_meta() {
            texts.push((meta.name.to_string(), meta.summary.to_string()));
            texts.push((meta.name.to_string(), meta.description.to_string()));
            let mut params = Vec::new();
            all_parameters(
                default_command(meta.id).unwrap().to_parameters(),
                &mut params,
            );
            for p in params {
                texts.push((format!("{} › {}", meta.name, p.name), p.description));
            }
        }
        assert!(texts.len() > 100);
        for (owner, text) in texts {
            for rustdoc in ["[`", "```", "\n#"] {
                assert!(!text.contains(rustdoc), "{owner}: {rustdoc:?} in {text:?}");
            }
            assert!(!text.starts_with('#'), "{owner}: a heading in {text:?}");
            assert!(
                StyledText::from_markdown(&reflow(&text)).is_ok(),
                "{owner}: not drawable markdown: {text:?}"
            );
        }
    }
}
