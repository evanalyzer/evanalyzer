// # paths
//
// **Author:** Joachim Danmayr
// **Date:** 2026-10-29
//
// ## License
// Copyright 2026 Joachim Danmayr.
// Licensed under the **PolyForm Noncommercial License 1.0.0**.

//! Path parts from a project or pipeline (someone else's, a client's) that
//! are joined onto a folder: checked on the string, the same on every
//! system - a project written on Windows runs on a Linux server, so both
//! `/` and `\` separate and a drive prefix counts everywhere.

/// Whether `path`, joined onto a folder, stays inside it: not empty, no
/// root (`/`, `\`, `\\server`), no drive (`C:`), no `..` part, no NUL.
pub fn is_contained_rel_path(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\0')
        && !path.starts_with(['/', '\\'])
        && !has_drive_prefix(path)
        && !path.split(['/', '\\']).any(|part| part == "..")
}

/// Whether `name` is usable as (a part of) one file's name in a folder:
/// [`is_contained_rel_path`] without any separator. Empty is fine.
pub fn is_file_name_part(name: &str) -> bool {
    name.is_empty() || (is_contained_rel_path(name) && !name.contains(['/', '\\']))
}

/// `C:…` - Windows reads it as on drive C, and joining it replaces the
/// folder.
fn has_drive_prefix(path: &str) -> bool {
    let mut chars = path.chars();
    matches!((chars.next(), chars.next()), (Some(c), Some(':')) if c.is_ascii_alphabetic())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_inside_the_folder_are_contained() {
        for path in [
            "a.tif",
            "plate 1/A01.tif",
            r"plate 1\A01.tif",
            "a..b.tif",
            "..a.tif",
            "./a.tif",
            "12:30.tif",
        ] {
            assert!(is_contained_rel_path(path), "{path}");
        }
    }

    #[test]
    fn paths_leaving_the_folder_are_not() {
        for path in [
            "",
            "/etc/passwd",
            r"\Windows\x.tif",
            r"\\server\share\x.tif",
            r"\\?\C:\x.tif",
            "C:/x.tif",
            r"c:x.tif",
            "..",
            "../x.tif",
            r"..\x.tif",
            "a/../../x.tif",
            r"a\..\..\x.tif",
            "a/..",
            "a\0.tif",
        ] {
            assert!(!is_contained_rel_path(path), "{path:?}");
        }
    }

    #[test]
    fn a_file_name_part_has_no_separator() {
        for name in ["", "overlay", "cells 2", "a..b", "."] {
            assert!(is_file_name_part(name), "{name}");
        }
        for name in ["sub/x", r"sub\x", "/tmp/x", "..", "../x", "C:x", "a\0"] {
            assert!(!is_file_name_part(name), "{name:?}");
        }
    }
}
