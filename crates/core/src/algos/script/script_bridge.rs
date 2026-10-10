//! Lets a script run the regular pipeline commands:
//!
//! ```rhai
//! run("gaussian_blur", #{ kernel_size: 5, sigma: 1.2 });
//! let raw = channel(0);
//! run("threshold", raw, #{ "thresholds.0.method": "otsu" });
//! run("connected_components");
//! if instance_count() < 50 { /* ... */ }
//! ```
//!
//! Images are values ([`ScriptImage`]): `run` returns the image it produced
//! and can take the image to work on. The segmentation and instance maps are
//! not values - they stay with the step, as for any other pipeline step.
//!
//! Commands are named by [`PipelineCommand::key`] and start from their
//! default settings. Parameters use the names and values of the step
//! settings in the GUI (`to_parameters`/`apply_param_change`), so a script
//! and a step behave the same. Every parameter is checked before it is
//! applied: `apply_param_change` silently ignores unknown names and values
//! it can't parse, which in a script would hide typos.

use crate::algos::ExecutionScope;
use crate::image::ImageContainer;
use crate::job::algos_from_config::into_algorithm;
use crate::pipeline::pipeline_cache::{CacheAddress, GlobalPipelineCache};
use crate::pipeline::pipeline_context::PipelineContext;
use evanalyzer_cfg::core_types::{CitationMetadata, ImageAddress, ObjectClass};
use evanalyzer_cfg::settings::parameter_def::{ParamType, ParameterDef};
use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
use rhai::{Dynamic, Map};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

/// An image in a script. Shares the pixel buffer with the pipeline, so
/// passing it around is free; a command writing to the image while a script
/// variable still holds it copies it first (copy-on-write), so the variable
/// keeps its value.
#[derive(Clone)]
pub(crate) struct ScriptImage(pub(crate) Arc<ImageContainer>);

impl ScriptImage {
    pub(crate) fn width(&mut self) -> i64 {
        self.0.size().width as i64
    }

    pub(crate) fn height(&mut self) -> i64 {
        self.0.size().height as i64
    }

    /// Mean over all pixels (and color channels).
    pub(crate) fn mean(&mut self) -> f64 {
        let (sum, count) = self.fold(0.0, |acc, v| acc + v);
        if count == 0 { 0.0 } else { sum / count as f64 }
    }

    pub(crate) fn min(&mut self) -> f64 {
        self.fold(f64::INFINITY, f64::min).0
    }

    pub(crate) fn max(&mut self) -> f64 {
        self.fold(f64::NEG_INFINITY, f64::max).0
    }

    fn fold(&self, init: f64, f: impl Fn(f64, f64) -> f64) -> (f64, usize) {
        match self.0.as_ref() {
            ImageContainer::F32Gray(img) => fold_slice(img.data.as_slice(), init, f),
            ImageContainer::F32Rgb(img) => fold_slice(img.data.as_slice(), init, f),
            ImageContainer::U32(img) => fold_slice(img.data.as_slice(), init, f),
        }
    }
}

fn fold_slice<T: Copy + Into<f64>>(
    data: &[T],
    init: f64,
    f: impl Fn(f64, f64) -> f64,
) -> (f64, usize) {
    (
        data.iter().fold(init, |acc, v| f(acc, (*v).into())),
        data.len(),
    )
}

/// What the script's `run(...)` calls work on: the step's context and the
/// pipeline cache, lent to the script engine for the duration of the script.
pub(crate) struct ScriptHost {
    pub(crate) ctx: PipelineContext,
    pub(crate) cache: GlobalPipelineCache,
    /// The script step's declared classes; commands may use only these.
    pub(crate) classes: Vec<ObjectClass>,
}

impl ScriptHost {
    /// The step's current image.
    pub(crate) fn image(&self) -> ScriptImage {
        ScriptImage(self.ctx.image.clone())
    }

    /// Makes `image` the step's current image, the input of the next command.
    pub(crate) fn set_image(&mut self, image: ScriptImage) -> Result<(), String> {
        let size = self.ctx.image.size();
        let new_size = image.0.size();
        if (size.width, size.height) != (new_size.width, new_size.height) {
            return Err(format!(
                "image is {}x{}, but this tile is {}x{}",
                new_size.width, new_size.height, size.width, size.height
            ));
        }
        self.ctx.image = image.0;
        Ok(())
    }

    /// Channel `index` of the image, for the current tile.
    pub(crate) fn channel(&self, index: i64) -> Result<ScriptImage, String> {
        let index = i32::try_from(index).map_err(|_| format!("no channel {index}"))?;
        let tile = self.ctx.image_meta.image_tile_info;
        self.cache
            .get_image_from_cache(&CacheAddress::Channel((index, tile)), tile)
            .map(ScriptImage)
            .ok_or_else(|| format!("channel {index} is not loaded for this pipeline"))
    }

    /// Number of distinct objects in the instance map of this tile.
    pub(crate) fn instance_count(&self) -> i64 {
        self.ctx.instance_map.as_ref().map_or(0, |map| {
            let ids: HashSet<u32> = map
                .as_slice()
                .iter()
                .copied()
                .filter(|&id| id != 0)
                .collect();
            ids.len() as i64
        })
    }

    /// [`Self::run_command`] on `image`; returns the resulting image.
    pub(crate) fn run_on(
        &mut self,
        key: &str,
        image: ScriptImage,
        params: &Map,
    ) -> Result<ScriptImage, String> {
        self.set_image(image).map_err(|e| format!("{key}: {e}"))?;
        self.run_command(key, params)?;
        Ok(self.image())
    }

    /// Runs the command named `key` with `params` applied on top of its
    /// defaults, exactly like a pipeline step would run it.
    pub(crate) fn run_command(&mut self, key: &str, params: &Map) -> Result<(), String> {
        let cmd = command_from_script(key, params)?;
        if let Some(class) = cmd
            .object_classes()
            .into_iter()
            .find(|c| !self.classes.contains(c))
        {
            return Err(format!(
                "{key} uses class {}, which is not in this script step's classes - \
                 add it to the step's settings",
                class.to_u32().map_or("?".to_string(), |v| v.to_string())
            ));
        }
        let algo = into_algorithm(cmd).map_err(|e| format!("{key}: {e}"))?;
        check_tile_scope(key, algo.execution_scope())?;
        algo.run(&mut self.ctx, &mut self.cache)
            .map_err(|e| format!("{key}: {e}"))
    }
}

/// The command names a script passes as string literals to `run(...)`, in
/// order of first appearance. Comments and other strings are skipped.
/// Names built at run time (`run(name)`) can't be seen here.
pub(crate) fn referenced_commands(source: &str) -> Vec<String> {
    let bytes = source.as_bytes();
    let mut found: Vec<String> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let rest = &source[i..];
        if rest.starts_with("//") {
            i += rest.find('\n').unwrap_or(rest.len());
        } else if rest.starts_with("/*") {
            i += block_comment_len(rest);
        } else if matches!(bytes[i], b'"' | b'`' | b'\'') {
            i += quoted_len(rest);
        } else if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let len = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            if &rest[..len] == "run" {
                if let Some(name) = literal_first_argument(&rest[len..]) {
                    if !found.iter().any(|f| f == name) {
                        found.push(name.to_string());
                    }
                }
            }
            i += len;
        } else {
            i += rest.chars().next().map_or(1, char::len_utf8);
        }
    }
    found
}

/// `("name"` (whitespace allowed) at the start of `rest` -> `name`.
fn literal_first_argument(rest: &str) -> Option<&str> {
    let rest = rest.trim_start().strip_prefix('(')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find(['"', '\\', '\n'])?;
    (rest.as_bytes()[end] == b'"').then(|| &rest[..end])
}

fn block_comment_len(rest: &str) -> usize {
    let bytes = rest.as_bytes();
    let mut depth = 0usize;
    let mut i = 0;
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
    rest.len()
}

fn quoted_len(rest: &str) -> usize {
    let quote = rest.as_bytes()[0];
    let mut chars = rest.char_indices().skip(1);
    while let Some((offset, c)) = chars.next() {
        if c == '\\' && quote != b'`' {
            chars.next();
        } else if c as u32 == quote as u32 {
            return offset + 1;
        } else if c == '\n' && quote != b'`' {
            return offset;
        }
    }
    rest.len()
}

/// Checks every command the script names before it runs, so a typo or a
/// whole-image command fails the step before the script changed anything.
pub(crate) fn check_referenced_commands(source: &str) -> Result<(), String> {
    for key in referenced_commands(source) {
        let cmd = command_from_script(&key, &Map::new())?;
        let algo = into_algorithm(cmd).map_err(|e| format!("{key}: {e}"))?;
        check_tile_scope(&key, algo.execution_scope())?;
    }
    Ok(())
}

/// Citations of the commands the script names, with their default
/// settings - a command whose citation depends on a setting (e.g. the
/// threshold method) is cited for its default.
pub(crate) fn referenced_citations(source: &str) -> Vec<&'static CitationMetadata> {
    let mut cites: Vec<&'static CitationMetadata> = Vec::new();
    for key in referenced_commands(source) {
        let Some(algo) = PipelineCommand::default_for_key(&key)
            .filter(|_| key != "script")
            .and_then(|cmd| into_algorithm(cmd).ok())
        else {
            continue;
        };
        for cite in algo.cite() {
            if !cites.iter().any(|c| std::ptr::eq(*c, cite)) {
                cites.push(cite);
            }
        }
    }
    cites
}

fn check_tile_scope(key: &str, scope: ExecutionScope) -> Result<(), String> {
    if matches!(scope, ExecutionScope::Tile) {
        Ok(())
    } else {
        Err(format!(
            "`{key}` works on the whole image's objects and can't run inside a script step, \
             which runs per tile"
        ))
    }
}

/// Builds the command `key` from its defaults plus the script's parameters.
pub(crate) fn command_from_script(key: &str, params: &Map) -> Result<PipelineCommand, String> {
    if key == "script" {
        return Err("a script can't run another script step".into());
    }
    let mut cmd = PipelineCommand::default_for_key(key).ok_or_else(|| {
        format!(
            "unknown command `{key}`; available: {}",
            PipelineCommand::KEYS
                .iter()
                .filter(|k| **k != "script")
                .copied()
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;

    // Parents before their dependents ("method" before "method.classes"),
    // list items in index order ("thresholds.2" before "thresholds.10").
    let mut entries: Vec<(&str, &Dynamic)> = params.iter().map(|(k, v)| (k.as_str(), v)).collect();
    entries.sort_by(|a, b| compare_param_names(a.0, b.0));

    for (name, value) in entries {
        let value =
            value_to_param_string(value).map_err(|e| format!("{key}: parameter `{name}`: {e}"))?;
        apply_checked(&mut cmd, name, &value).map_err(|e| format!("{key}: {e}"))?;
    }
    Ok(cmd)
}

/// Applies one parameter after checking that the command has it and that
/// the value fits its type. Writing to the next index of a list parameter
/// (`thresholds.1.method` on a one-entry list) adds an entry first.
fn apply_checked(cmd: &mut PipelineCommand, name: &str, value: &str) -> Result<(), String> {
    let mut params = flatten(&cmd.to_parameters());
    if !params.contains_key(name) && grow_list_for(cmd, &params, name)? {
        params = flatten(&cmd.to_parameters());
    }
    let def = params.get(name).ok_or_else(|| {
        let available: Vec<&str> = params
            .iter()
            .filter(|(_, d)| is_settable(d))
            .map(|(n, _)| n.as_str())
            .collect();
        format!(
            "unknown parameter `{name}`; available: {}",
            available.join(", ")
        )
    })?;
    let value = checked_value(def, value).map_err(|e| format!("parameter `{name}`: {e}"))?;
    cmd.apply_param_change(name, &value);
    Ok(())
}

/// `group.index.field` with `index` one past the end of list parameter
/// `group`: adds an entry and returns `true`.
fn grow_list_for(
    cmd: &mut PipelineCommand,
    params: &BTreeMap<String, ParameterDef>,
    name: &str,
) -> Result<bool, String> {
    let mut parts = name.splitn(3, '.');
    let (Some(group), Some(index), Some(_)) = (parts.next(), parts.next(), parts.next()) else {
        return Ok(false);
    };
    let (Some(def), Ok(index)) = (params.get(group), index.parse::<usize>()) else {
        return Ok(false);
    };
    if def.param_type != ParamType::Group {
        return Ok(false);
    }
    let len = def.groups.len();
    if index != len {
        return Err(format!(
            "`{group}` has {len} entries; `{name}` must use index 0..={len}"
        ));
    }
    cmd.add_group_item(group);
    Ok(true)
}

/// Every parameter by its full name; list entries as `group.index.field`
/// (the same paths the GUI sends).
fn flatten(params: &[ParameterDef]) -> BTreeMap<String, ParameterDef> {
    fn walk(params: &[ParameterDef], prefix: &str, out: &mut BTreeMap<String, ParameterDef>) {
        for p in params {
            let name = format!("{prefix}{}", p.name);
            if p.param_type == ParamType::Group {
                for (i, item) in p.groups.iter().enumerate() {
                    walk(item, &format!("{name}.{i}."), out);
                }
            }
            out.insert(name, p.clone());
        }
    }
    let mut out = BTreeMap::new();
    walk(params, "", &mut out);
    out
}

fn is_settable(def: &ParameterDef) -> bool {
    !matches!(
        def.param_type,
        ParamType::Group | ParamType::Label | ParamType::Script
    )
}

/// Compares dotted names segment by segment, numeric segments by value.
fn compare_param_names(a: &str, b: &str) -> Ordering {
    let mut a = a.split('.');
    let mut b = b.split('.');
    loop {
        match (a.next(), b.next()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(x), Some(y)) => {
                let order = match (x.parse::<u64>(), y.parse::<u64>()) {
                    (Ok(x), Ok(y)) => x.cmp(&y),
                    _ => x.cmp(y),
                };
                if order != Ordering::Equal {
                    return order;
                }
            }
        }
    }
}

/// A script value in the string form `apply_param_change` expects.
fn value_to_param_string(value: &Dynamic) -> Result<String, String> {
    if let Ok(b) = value.as_bool() {
        return Ok(b.to_string());
    }
    if let Ok(i) = value.as_int() {
        return Ok(i.to_string());
    }
    if let Ok(f) = value.as_float() {
        if !f.is_finite() {
            return Err(format!("{f} is not a finite number"));
        }
        return Ok(f.to_string());
    }
    if let Ok(c) = value.as_char() {
        return Ok(c.to_string());
    }
    if value.is_string() {
        return Ok(value.clone().into_string().unwrap_or_default());
    }
    if value.is_array() {
        let items: Result<Vec<String>, String> = value
            .clone()
            .into_array()
            .unwrap_or_default()
            .iter()
            .map(|item| match item.as_int() {
                Ok(i) => Ok(i.to_string()),
                Err(_) => Err(format!(
                    "list entries must be integers, got {}",
                    item.type_name()
                )),
            })
            .collect();
        return Ok(items?.join(","));
    }
    Err(format!("unsupported value of type {}", value.type_name()))
}

/// Checks `value` against the parameter's type and range and returns it in
/// the exact form `apply_param_change` expects (dropdown labels are matched
/// ignoring case, spaces, `_` and `-`, so `"iso_data"` selects "Iso Data").
fn checked_value(def: &ParameterDef, value: &str) -> Result<String, String> {
    match def.param_type {
        ParamType::Number | ParamType::Spinner | ParamType::Slider => {
            let v: f64 = value
                .parse()
                .map_err(|_| format!("expected a number, got `{value}`"))?;
            if def.max > def.min && (v < def.min as f64 || v > def.max as f64) {
                return Err(format!(
                    "{value} is outside the allowed range {}..={}",
                    def.min, def.max
                ));
            }
            Ok(value.to_string())
        }
        ParamType::Toggle => match value {
            "true" | "false" => Ok(value.to_string()),
            _ => Err(format!("expected true or false, got `{value}`")),
        },
        ParamType::Dropdown | ParamType::PixelUnits | ParamType::SizeUnits => {
            let wanted = normalize_label(value);
            def.options
                .iter()
                .find(|o| normalize_label(o) == wanted)
                .cloned()
                .ok_or_else(|| {
                    format!(
                        "`{value}` is not one of: {}",
                        def.options.join(", ")
                    )
                })
        }
        ParamType::ObjClass => match value.parse::<i64>() {
            Ok(-1) => Ok(value.to_string()),
            Ok(v) if u32::try_from(v).is_ok() => Ok(value.to_string()),
            _ => Err(format!("expected a class id (or -1 for none), got `{value}`")),
        },
        ParamType::SegClass | ParamType::ImageChannel => value
            .parse::<u32>()
            .map(|_| value.to_string())
            .map_err(|_| format!("expected a non-negative integer, got `{value}`")),
        ParamType::MultiObjClass | ParamType::MultiSegClass => {
            for id in value.split(',').filter(|x| !x.is_empty()) {
                id.trim()
                    .parse::<u32>()
                    .map_err(|_| format!("expected a list of class ids, got `{value}`"))?;
            }
            Ok(value.to_string())
        }
        ParamType::ImageAddress => ImageAddress::from_param_value(value)
            .map(|_| value.to_string())
            .ok_or_else(|| {
                format!("expected an image like \"channel:0\", \"memory:1\" or \"scratchpad\", got `{value}`")
            }),
        ParamType::Text | ParamType::FilePath => Ok(value.to_string()),
        ParamType::Group => Err("is a list; set its entries as `name.index.field`".into()),
        ParamType::Label => Err("is read-only".into()),
        ParamType::Script => Err("can't be set from a script".into()),
    }
}

fn normalize_label(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, ' ' | '_' | '-'))
        .flat_map(char::to_lowercase)
        .collect()
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::{GaussianBlur, ImageAlgorithm};
    use kornia_image::{Image, ImageSize};

    fn map(entries: &[(&str, Dynamic)]) -> Map {
        entries
            .iter()
            .map(|(k, v)| ((*k).into(), v.clone()))
            .collect()
    }

    fn param(cmd: &PipelineCommand, name: &str) -> String {
        flatten(&cmd.to_parameters())
            .get(name)
            .unwrap_or_else(|| panic!("no parameter {name}"))
            .value
            .clone()
    }

    #[test]
    fn named_parameters_are_applied_on_top_of_the_defaults() {
        let cmd = command_from_script(
            "gaussian_blur",
            &map(&[("kernel_size", Dynamic::from(7_i64))]),
        )
        .unwrap();
        assert_eq!(param(&cmd, "kernel_size"), "7");
        let default = PipelineCommand::default_for_key("gaussian_blur").unwrap();
        assert_eq!(param(&cmd, "sigma"), param(&default, "sigma"));
    }

    #[test]
    fn unknown_command_lists_the_available_ones() {
        let err = command_from_script("gausian_blur", &Map::new()).unwrap_err();
        assert!(err.contains("unknown command `gausian_blur`"), "{err}");
        assert!(err.contains("gaussian_blur"), "{err}");
        assert!(!err.contains("script,"), "{err}");
    }

    #[test]
    fn unknown_parameter_lists_the_available_ones() {
        let err = command_from_script(
            "gaussian_blur",
            &map(&[("kernel_sise", Dynamic::from(5_i64))]),
        )
        .unwrap_err();
        assert!(err.contains("unknown parameter `kernel_sise`"), "{err}");
        assert!(err.contains("kernel_size"), "{err}");
    }

    #[test]
    fn values_are_type_and_range_checked() {
        let err = command_from_script(
            "gaussian_blur",
            &map(&[("kernel_size", Dynamic::from("big"))]),
        )
        .unwrap_err();
        assert!(err.contains("expected a number"), "{err}");

        let err = command_from_script(
            "gaussian_blur",
            &map(&[("kernel_size", Dynamic::from(99_i64))]),
        )
        .unwrap_err();
        assert!(err.contains("outside the allowed range"), "{err}");
    }

    #[test]
    fn dropdown_labels_match_loosely_and_list_options_on_error() {
        let cmd = command_from_script(
            "threshold",
            &map(&[("thresholds.0.method", Dynamic::from("iso_data"))]),
        )
        .unwrap();
        assert_eq!(param(&cmd, "thresholds.0.method"), "Iso Data");

        let err = command_from_script(
            "threshold",
            &map(&[("thresholds.0.method", Dynamic::from("magic"))]),
        )
        .unwrap_err();
        assert!(err.contains("Otsu"), "{err}");
    }

    #[test]
    fn dependent_parameters_become_settable_after_their_parent() {
        // "method.classes" only exists once the method is Otsu; the sort
        // applies the parent first even though both come in one map.
        let cmd = command_from_script(
            "threshold",
            &map(&[
                ("thresholds.0.method.classes", Dynamic::from("Three")),
                ("thresholds.0.method", Dynamic::from("Otsu")),
            ]),
        )
        .unwrap();
        assert_eq!(param(&cmd, "thresholds.0.method.classes"), "Three");
    }

    #[test]
    fn writing_one_past_the_end_adds_a_list_entry() {
        // The default threshold has no entries; 0 and then 1 add them.
        let cmd = command_from_script(
            "threshold",
            &map(&[
                ("thresholds.0.min_threshold", Dynamic::from(10_i64)),
                ("thresholds.1.min_threshold", Dynamic::from(100_i64)),
            ]),
        )
        .unwrap();
        assert_eq!(param(&cmd, "thresholds.0.min_threshold"), "10");
        assert_eq!(param(&cmd, "thresholds.1.min_threshold"), "100");

        let err = command_from_script(
            "threshold",
            &map(&[("thresholds.5.min_threshold", Dynamic::from(1_i64))]),
        )
        .unwrap_err();
        assert!(err.contains("index 0..=0"), "{err}");
    }

    #[test]
    fn referenced_commands_skip_comments_strings_and_dynamic_names() {
        let src = r#"
            run("gaussian_blur", #{ kernel_size: 3 });
            let raw = channel(0);
            run ( "threshold" , raw);
            // run("voronoi");
            /* run("voronoi"); /* nested */ run("voronoi"); */
            print("run(\"voronoi\")");
            let name = "blur";
            run(name);
            rerun("voronoi");
            run("gaussian_blur");
        "#;
        assert_eq!(referenced_commands(src), vec!["gaussian_blur", "threshold"]);
    }

    #[test]
    fn referenced_commands_are_checked_before_running() {
        assert!(check_referenced_commands(r#"run("gaussian_blur");"#).is_ok());
        let err = check_referenced_commands(r#"run("gausian_blur");"#).unwrap_err();
        assert!(err.contains("unknown command"), "{err}");
        let err = check_referenced_commands(r#"run("voronoi");"#).unwrap_err();
        assert!(err.contains("per tile"), "{err}");
    }

    #[test]
    fn citations_of_referenced_commands() {
        let direct =
            into_algorithm(PipelineCommand::default_for_key("edge_detection_sobel").unwrap())
                .unwrap()
                .cite();
        let cites =
            referenced_citations(r#"run("edge_detection_sobel"); run("edge_detection_sobel");"#);
        assert_eq!(cites.len(), direct.len());
        assert_eq!(cites[0].cite_key, direct[0].cite_key);
    }

    /// The script editor (which can't see execution scopes) offers every
    /// command outside the Object category; that must be exactly the
    /// commands a script can run.
    #[test]
    fn object_category_is_exactly_the_whole_image_commands() {
        use evanalyzer_cfg::settings::pipeline_command::CommandCategory;
        for key in PipelineCommand::KEYS.iter().filter(|k| **k != "script") {
            let cmd = PipelineCommand::default_for_key(key).unwrap();
            let is_object = matches!(cmd.category(), CommandCategory::Object);
            let Ok(algo) = into_algorithm(cmd) else {
                continue; // feature-gated (ai) command not in this build
            };
            let whole_image = !matches!(algo.execution_scope(), ExecutionScope::Tile);
            assert_eq!(is_object, whole_image, "`{key}`");
        }
    }

    /// The editor inserts every parameter at its current value; those
    /// values must pass the script checks, or an untouched snippet fails.
    #[test]
    fn every_default_value_passes_the_checks() {
        for key in PipelineCommand::KEYS.iter().filter(|k| **k != "script") {
            let mut cmd = PipelineCommand::default_for_key(key).unwrap();
            for p in cmd.to_parameters() {
                if p.param_type == ParamType::Group && p.groups.is_empty() {
                    cmd.add_group_item(&p.name);
                }
            }
            for (name, def) in flatten(&cmd.to_parameters()) {
                if !is_settable(&def) {
                    continue;
                }
                assert!(
                    checked_value(&def, &def.value).is_ok(),
                    "`{key}.{name}` default `{}` is rejected: {:?}",
                    def.value,
                    checked_value(&def, &def.value)
                );
            }
        }
    }

    #[test]
    fn scripts_cannot_nest() {
        assert!(command_from_script("script", &Map::new()).is_err());
    }

    #[test]
    fn param_names_sort_parents_first_and_indexes_numerically() {
        let mut names = vec!["a.10.x", "a.2.x", "m.c", "m"];
        names.sort_by(|a, b| compare_param_names(a, b));
        assert_eq!(names, vec!["a.2.x", "a.10.x", "m", "m.c"]);
    }

    #[test]
    fn script_values_convert_to_param_strings() {
        assert_eq!(value_to_param_string(&Dynamic::from(true)).unwrap(), "true");
        assert_eq!(
            value_to_param_string(&Dynamic::from(1.5_f64)).unwrap(),
            "1.5"
        );
        let list: rhai::Array = vec![Dynamic::from(1_i64), Dynamic::from(3_i64)];
        assert_eq!(value_to_param_string(&Dynamic::from(list)).unwrap(), "1,3");
        assert!(value_to_param_string(&Dynamic::UNIT).is_err());
        assert!(value_to_param_string(&Dynamic::from(f64::NAN)).is_err());
    }

    fn host() -> ScriptHost {
        let mut data = vec![0.0f32; 49];
        data[24] = 1.0;
        let image = Image::<f32, 1>::from_size_slice(
            ImageSize {
                width: 7,
                height: 7,
            },
            &data,
        )
        .unwrap();
        ScriptHost {
            ctx: PipelineContext::new_from_image_test(image).unwrap(),
            cache: GlobalPipelineCache::default(),
            classes: vec![],
        }
    }

    fn pixels(ctx: &PipelineContext) -> Vec<f32> {
        ctx.get_f32_gray_image().unwrap().data.as_slice().to_vec()
    }

    #[test]
    fn running_a_command_matches_the_step_run_directly() {
        let mut scripted = host();
        scripted
            .run_command(
                "gaussian_blur",
                &map(&[
                    ("kernel_size", Dynamic::from(5_i64)),
                    ("sigma", Dynamic::from(1.0_f64)),
                ]),
            )
            .unwrap();

        let mut direct = host();
        GaussianBlur {
            kernel_size: 5,
            sigma: 1.0,
        }
        .run(&mut direct.ctx, &mut direct.cache)
        .unwrap();

        assert_eq!(pixels(&scripted.ctx), pixels(&direct.ctx));
        assert_ne!(pixels(&scripted.ctx), pixels(&host().ctx), "the blur ran");
    }

    #[test]
    fn run_on_an_earlier_image_keeps_that_image_unchanged() {
        let mut host = host();
        let raw = host.image();
        let raw_pixels = match raw.0.as_ref() {
            ImageContainer::F32Gray(img) => img.data.as_slice().to_vec(),
            _ => unreachable!(),
        };
        let blurred = host
            .run_on("gaussian_blur", raw.clone(), &Map::new())
            .unwrap();
        assert!(!Arc::ptr_eq(&raw.0, &blurred.0));
        match raw.0.as_ref() {
            ImageContainer::F32Gray(img) => assert_eq!(img.data.as_slice(), &raw_pixels[..]),
            _ => unreachable!(),
        }
        // Running again from the raw image gives the same result.
        let again = host.run_on("gaussian_blur", raw, &Map::new()).unwrap();
        assert_eq!(pixels(&host.ctx), {
            match blurred.0.as_ref() {
                ImageContainer::F32Gray(img) => img.data.as_slice().to_vec(),
                _ => unreachable!(),
            }
        });
        drop(again);
    }

    #[test]
    fn image_statistics() {
        let mut image = host().image();
        assert_eq!(image.width(), 7);
        assert_eq!(image.max(), 1.0);
        assert_eq!(image.min(), 0.0);
        assert!((image.mean() - 1.0 / 49.0).abs() < 1e-9);
    }

    #[test]
    fn instance_count_counts_distinct_labels() {
        let mut host = host();
        let map = host.ctx.instance_map.as_mut().unwrap();
        let data = map.as_slice_mut();
        data[0] = 3;
        data[1] = 3;
        data[5] = 8;
        assert_eq!(host.instance_count(), 2);
    }

    #[test]
    fn missing_channel_is_an_error() {
        let err = host().channel(2).err().unwrap();
        assert!(err.contains("channel 2"), "{err}");
    }

    #[test]
    fn set_image_rejects_a_different_size() {
        let mut host = host();
        let other = ScriptImage(Arc::new(ImageContainer::F32Gray(
            crate::image::ManagedImage {
                data: Image::new(
                    ImageSize {
                        width: 2,
                        height: 2,
                    },
                    vec![0.0; 4],
                )
                .unwrap(),
                tile_offset: Default::default(),
                plane: None,
            },
        )));
        assert!(host.set_image(other).is_err());
    }

    #[test]
    fn commands_may_only_use_the_declared_classes() {
        let params = map(&[("thresholds.0.object_class_id", Dynamic::from(4_i64))]);
        let mut undeclared = host();
        let err = undeclared.run_command("threshold", &params).unwrap_err();
        assert!(err.contains("class 4"), "{err}");

        let mut declared = host();
        declared.classes = vec![ObjectClass::Valid(4)];
        declared.run_command("threshold", &params).unwrap();
    }

    #[test]
    fn whole_image_commands_are_rejected() {
        let err = host().run_command("voronoi", &Map::new()).unwrap_err();
        assert!(err.contains("per tile"), "{err}");
    }
}
