use std::{collections::HashMap, fs, path::Path};
use syn::{GenericArgument, Item, ItemEnum, ItemStruct, PathArguments, Type, parse_file};

pub fn generate_mappings() -> Result<(), Box<dyn std::error::Error>> {
    let algos_path = Path::new("../core/src/algos");
    let module_features = parse_module_features(Path::new("../core/src/algos.rs"));
    let mut commands = Vec::new();
    let mut enums = Vec::new();

    if let Ok(entries) = fs::read_dir(algos_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            // Top-level module name as declared in algos.rs (`mod <name>;` /
            // `mod <name>.rs`), used to look up an optional `#[cfg(feature = "...")]`
            // gate that must be propagated onto the generated glue code.
            let module_name = path.file_stem().and_then(|n| n.to_str()).unwrap_or("");
            let feature = module_features.get(module_name).and_then(|f| f.clone());

            if path.is_dir() {
                scan_directory(&path, &mut commands, &mut enums, feature.as_deref());
            } else if path.extension().map_or(false, |ext| ext == "rs") {
                extract_command_structs(&path, &mut commands, &mut enums, feature.as_deref());
            }
        }
    }

    commands.sort_by(|a, b| a.struct_name.cmp(&b.struct_name));
    enums.sort_by(|a, b| a.enum_name.cmp(&b.enum_name));

    // --- Generate two separate files ---
    let config_code = generate_config_code(&commands, &enums);
    let from_code = generate_from_impls(&commands, &enums);
    let enum_code = generate_pipeline_command_enum(&commands, &enums);

    write_if_changed(
        Path::new("src/modules/pipeline_command_settings.rs"),
        &config_code,
    );
    write_if_changed(
        Path::new("../core/src/job/algos_from_config.rs"),
        &from_code,
    );
    write_if_changed(Path::new("src/modules/pipeline_command.rs"), &enum_code);

    println!(
        "cargo:warning=Generated {} command settings and {} enum settings",
        commands.len(),
        enums.len()
    );

    Ok(())
}

fn write_if_changed(path: &Path, content: &str) {
    let formatted = format_code(content).unwrap_or_else(|| content.to_string());
    let existing = fs::read_to_string(path).unwrap_or_default();
    if formatted != existing {
        fs::write(path, &formatted).expect("Failed to write file");
    }
}

fn format_code(content: &str) -> Option<String> {
    use std::io::Write as _;
    use std::process::{Command, Stdio};
    let mut child = Command::new("rustfmt")
        .args(["--edition=2021", "--emit=stdout"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.take()?.write_all(content.as_bytes()).ok()?;
    let output = child.wait_with_output().ok()?;
    if output.status.success() {
        String::from_utf8(output.stdout).ok()
    } else {
        None
    }
}

// ============================================================
// FILE 1: Config structs + enums - no From impls, no core imports
// ============================================================

/// Rewrite type names in a default expression from core names to their generated
/// settings equivalents. `quote!` serialises `Foo::Bar` as `"Foo :: Bar"` (with
/// spaces), so we replace both the spaced and the compact forms.
fn remap_default_expr(expr: &str, enums: &[EnumInfo], commands: &[CommandInfo]) -> String {
    let mut result = expr.to_string();
    for enum_info in enums {
        let settings_name = format!(
            "{}{}Settings",
            to_pascal_case(&enum_info.source_file),
            enum_info.enum_name
        );
        result = result.replace(
            &format!("{} ::", enum_info.enum_name),
            &format!("{} ::", settings_name),
        );
        result = result.replace(
            &format!("{}::", enum_info.enum_name),
            &format!("{}::", settings_name),
        );
    }
    for cmd in commands {
        let settings_name = format!("{}Settings", cmd.struct_name);
        result = result.replace(
            &format!("{} ::", cmd.struct_name),
            &format!("{} ::", settings_name),
        );
        result = result.replace(
            &format!("{}::", cmd.struct_name),
            &format!("{}::", settings_name),
        );
    }
    result
}

fn format_default_for_type(ty: &str, val: f64) -> String {
    match ty {
        "f32" => {
            let s = format!("{}", val);
            if s.contains('.') {
                format!("{}f32", s)
            } else {
                format!("{}.0f32", s)
            }
        }
        "f64" => {
            let s = format!("{}", val);
            if s.contains('.') {
                format!("{}f64", s)
            } else {
                format!("{}.0f64", s)
            }
        }
        "usize" => format!("{}usize", val as u64),
        "u32" => format!("{}u32", val as u32),
        "u64" => format!("{}u64", val as u64),
        "i32" => format!("{}i32", val as i32),
        "i64" => format!("{}i64", val as i64),
        "bool" => (if val != 0.0 { "true" } else { "false" }).to_string(),
        _ => "Default::default()".to_string(),
    }
}

/// The default-value expression for one settings field, honouring an explicit
/// `#[cmdsmeta(default = ...)]` if present and falling back to the type's own `Default`
/// otherwise. Shared between a command struct's generated `Default` impl and a rich enum
/// variant's generated `Default` impl (both assemble a field-by-field literal).
fn field_default_expr(field: &FieldInfo, enums: &[EnumInfo], commands: &[CommandInfo]) -> String {
    let field_type = map_to_settings_type(&field.ty, enums, commands);
    if let Some(ref expr) = field.metadata.default_expr {
        remap_default_expr(expr, enums, commands)
    } else if let Some(val) = field.metadata.default {
        format_default_for_type(&field.ty, val)
    } else if field_type.starts_with("Vec<") {
        "vec![]".to_string()
    } else if field_type.starts_with("Option<") {
        "None".to_string()
    } else {
        format!("{}::default()", field_type)
    }
}

/// The value expression converting one settings field into its core-side counterpart for a
/// generated `From<...Settings> for ...` impl: clamps numeric fields to `min`/`max`, recurses
/// into nested user enums/structs via `.into()`, and maps `Vec`/`Option`. Shared between a
/// command struct's `From` impl and a rich enum variant's `From` impl.
fn field_value_expr(
    field: &FieldInfo,
    field_access: String,
    enums: &[EnumInfo],
    commands: &[CommandInfo],
) -> String {
    let meta = &field.metadata;
    if field.ty == "f32" || field.ty == "f64" {
        match (meta.min, meta.max) {
            (Some(min), Some(max)) => format!("{}.clamp({:?}, {:?})", field_access, min, max),
            (Some(min), None) => format!("{}.max({:?})", field_access, min),
            (None, Some(max)) => format!("{}.min({:?})", field_access, max),
            _ => field_access,
        }
    } else if is_user_enum(&field.ty, enums) || is_user_struct(&field.ty, commands) {
        format!("{}::from({})", field.ty, field_access)
    } else if field.ty.starts_with("Vec<") {
        format!("{}.into_iter().map(|v| v.into()).collect()", field_access)
    } else if field.ty.starts_with("Option<") {
        format!("{}.map(|v| v.into())", field_access)
    } else {
        field_access
    }
}

fn generate_config_code(commands: &[CommandInfo], enums: &[EnumInfo]) -> String {
    use std::collections::HashSet;

    let mut used_type_names: HashSet<String> = HashSet::new();
    for cmd in commands {
        for field in &cmd.fields {
            used_type_names.insert(field.ty.clone());
        }
    }

    let filtered_enums: Vec<&EnumInfo> = enums
        .iter()
        .filter(|e| used_type_names.contains(&e.enum_name))
        .collect();

    let filtered_commands: Vec<&CommandInfo> = commands
        .iter()
        // .filter(|c| {
        //     let passes = used_type_names.contains(&c.struct_name)
        //         || !["Other"].contains(&c.category.as_str());
        //     passes
        // })
        .collect();

    let mut out = String::new();

    // Header - only config/serde imports, no core
    out.push_str("// @generated - do not edit by hand\n");
    // out.push_str("use indexmap::IndexMap;\n");
    out.push_str("use crate::{core_types::{ImageAddress,PixelUnits, SizeUnits}, types::classes::{ObjectClass, SegmentationClass}};\n");
    out.push_str("use std::path::PathBuf;\n");
    out.push_str("use schemars::JsonSchema;\n");
    out.push_str("use serde::{Deserialize, Serialize};\n\n");

    // Enums
    out.push_str("// ============ ENUM SETTINGS ============\n\n");
    for enum_info in &filtered_enums {
        let settings_name = enum_info.settings_name();
        let is_rich = enum_info.is_rich();

        for doc in &enum_info.doc_comments {
            out.push_str(&format!("/// {}\n", doc));
        }
        if is_rich {
            // Internally tagged: each variant's own fields sit alongside the discriminant,
            // which schemars renders as `oneOf` + a `const`-valued "type" property — the
            // standard JSON Schema representation for "different fields depending on which
            // option is selected" (mirrors a serde-tagged Rust enum / OpenAPI discriminator).
            out.push_str("#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]\n");
            out.push_str("#[serde(tag = \"type\", rename_all = \"camelCase\")]\n");
        } else {
            out.push_str(
                "#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq, Default)]\n",
            );
            out.push_str("#[serde(rename_all = \"SCREAMING_SNAKE_CASE\")]\n");
        }
        out.push_str(&format!("pub enum {} {{\n", settings_name));
        for (vi, variant) in enum_info.variants.iter().enumerate() {
            for doc in &variant.doc_comments {
                out.push_str(&format!("    /// {}\n", doc));
            }
            if !is_rich && vi == 0 {
                out.push_str("    #[default]\n");
            }
            if variant.is_rich() {
                // The enum-level `rename_all` only renames variant identifiers, not the
                // fields of a struct-like variant — each variant needs its own rename_all
                // to keep field names camelCase, consistent with every other settings struct.
                out.push_str("    #[serde(rename_all = \"camelCase\")]\n");
                out.push_str(&format!("    {} {{\n", variant.name));
                for field in &variant.named_fields {
                    for doc in &field.doc_comments {
                        out.push_str(&format!("        /// {}\n", doc));
                    }
                    let mut range_parts = Vec::new();
                    if let Some(min) = field.metadata.min {
                        range_parts.push(format!("min = {}", min));
                    }
                    if let Some(max) = field.metadata.max {
                        range_parts.push(format!("max = {}", max));
                    }
                    if !range_parts.is_empty() {
                        out.push_str(&format!(
                            "        #[schemars(range({}))]\n",
                            range_parts.join(", ")
                        ));
                    }
                    let field_type = map_to_settings_type(&field.ty, enums, commands);
                    out.push_str(&format!("        {}: {},\n", field.name, field_type));
                }
                out.push_str("    },\n");
            } else if let Some(ref data_type) = variant.data_type {
                out.push_str(&format!("    {}({}),\n", variant.name, data_type));
            } else {
                out.push_str(&format!("    {},\n", variant.name));
            }
        }
        out.push_str("}\n\n");

        // Rich enums opt out of #[derive(Default)] (the field defaults would otherwise be each
        // field type's zero value, ignoring any #[cmdsmeta(default = ...)] override), so they
        // need a hand-rolled Default impl — same idea as a command struct's Default impl below.
        if is_rich {
            if let Some(first) = enum_info.variants.first() {
                out.push_str(&format!("impl Default for {} {{\n", settings_name));
                out.push_str("    fn default() -> Self {\n");
                out.push_str(&format!("        Self::{} {{\n", first.name));
                for field in &first.named_fields {
                    let default_expr = field_default_expr(field, enums, commands);
                    out.push_str(&format!("            {}: {},\n", field.name, default_expr));
                }
                out.push_str("        }\n    }\n}\n\n");
            }
        }
    }

    // Structs by category
    let mut by_category: HashMap<String, Vec<&CommandInfo>> = HashMap::new();
    for cmd in &filtered_commands {
        by_category
            .entry(cmd.category.clone())
            .or_default()
            .push(cmd);
    }

    for category in &[
        "Preprocessing",
        "Segmentation",
        "Object",
        "Measure",
        "Classification",
        "Other",
    ] {
        if let Some(cmds) = by_category.get(*category) {
            out.push_str(&format!(
                "\n// ============ {} ============\n\n",
                category.to_uppercase()
            ));

            for cmd in cmds {
                let settings_name = format!("{}Settings", cmd.struct_name);
                let has_explicit_defaults = cmd
                    .fields
                    .iter()
                    .any(|f| f.metadata.default.is_some() || f.metadata.default_expr.is_some());

                // Generate serde default helper functions for optional fields that
                // carry an explicit default value. A plain `#[serde(default)]` would
                // fall back to the field-type's Default (e.g. 0 for i32), which is
                // wrong. The helper makes serde call the exact cmdsmeta default.
                let prefix = cmd.struct_name.to_ascii_lowercase();
                for field in &cmd.fields {
                    if !field.metadata.optional {
                        continue;
                    }
                    let field_type = map_to_settings_type(&field.ty, enums, commands);
                    let fn_name = format!("_serde_default_{}_{}", prefix, field.name);
                    // Reuse the same default-expression logic as the struct's `Default`
                    // impl so generic types (`Vec<T>`, `Option<T>`) get a valid
                    // expression instead of the invalid `Vec<T>::default()`.
                    let body = field_default_expr(field, enums, commands);
                    out.push_str(&format!(
                        "fn {}() -> {} {{ {} }}\n",
                        fn_name, field_type, body
                    ));
                }

                for doc in &cmd.doc_comments {
                    out.push_str(&format!("/// {}\n", doc));
                }
                if has_explicit_defaults {
                    out.push_str("#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone)]\n");
                    out.push_str("#[schemars(default)]\n");
                } else {
                    out.push_str(
                        "#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]\n",
                    );
                }
                out.push_str("#[serde(rename_all = \"camelCase\")]\n");
                out.push_str(&format!("pub struct {} {{\n", settings_name));

                for field in &cmd.fields {
                    for doc in &field.doc_comments {
                        out.push_str(&format!("    /// {}\n", doc));
                    }

                    // schemars constraints
                    let meta = &field.metadata;
                    let mut range_parts = Vec::new();
                    if let Some(min) = meta.min {
                        range_parts.push(format!("min = {}", min));
                    }
                    if let Some(max) = meta.max {
                        range_parts.push(format!("max = {}", max));
                    }
                    if !range_parts.is_empty() {
                        out.push_str(&format!(
                            "    #[schemars(range({}))]\n",
                            range_parts.join(", ")
                        ));
                    }
                    if let Some(ref unit) = meta.unit {
                        out.push_str(&format!(
                            "    #[schemars(description = \"unit: {}\")]\n",
                            unit
                        ));
                    }

                    // optional field: emit a serde default attribute pointing at
                    // the generated helper so the cmdsmeta default is used, not
                    // the field-type's zero/Default.
                    if meta.optional {
                        let fn_name = format!("_serde_default_{}_{}", prefix, field.name);
                        out.push_str(&format!("    #[serde(default = \"{}\")]\n", fn_name));
                    }

                    let field_type = map_to_settings_type(&field.ty, enums, commands);
                    out.push_str(&format!("    pub {}: {},\n", field.name, field_type));
                }
                out.push_str("}\n\n");

                if has_explicit_defaults {
                    out.push_str(&format!("impl Default for {} {{\n", settings_name));
                    out.push_str("    fn default() -> Self {\n");
                    out.push_str("        Self {\n");
                    for field in &cmd.fields {
                        let default_expr = field_default_expr(field, enums, commands);
                        out.push_str(&format!("            {}: {},\n", field.name, default_expr));
                    }
                    out.push_str("        }\n");
                    out.push_str("    }\n");
                    out.push_str("}\n\n");
                }
            }
        }
    }

    out
}

// ============================================================
// FILE 2: From impls - lives in core, imports both core + config
// ============================================================

fn generate_from_impls(commands: &[CommandInfo], enums: &[EnumInfo]) -> String {
    use std::collections::HashSet;

    let mut used_type_names: HashSet<String> = HashSet::new();
    for cmd in commands {
        for field in &cmd.fields {
            used_type_names.insert(field.ty.clone());
        }
    }

    let filtered_enums: Vec<&EnumInfo> = enums
        .iter()
        .filter(|e| used_type_names.contains(&e.enum_name))
        .collect();

    let filtered_commands: Vec<&CommandInfo> = commands
        .iter()
        .filter(|c| {
            used_type_names.contains(&c.struct_name) || !["Other"].contains(&c.category.as_str())
        })
        .collect();

    let mut out = String::new();

    // Header - imports both core algos and config settings
    out.push_str("// @generated - do not edit by hand\n");
    out.push_str("use evanalyzer_cfg::settings::pipeline_command_settings::*;\n");
    out.push_str("use crate::algos::*;\n\n");

    // Enum From impls
    out.push_str("// ============ ENUM FROM IMPLS ============\n\n");
    for enum_info in &filtered_enums {
        let settings_name = format!(
            "{}{}Settings",
            to_pascal_case(&enum_info.source_file),
            enum_info.enum_name
        );

        if let Some(feature) = &enum_info.feature {
            out.push_str(&format!("#[cfg(feature = {feature:?})]\n"));
        }
        out.push_str(&format!(
            "impl From<{settings_name}> for {} {{\n",
            enum_info.enum_name
        ));
        out.push_str(&format!("    fn from(_s: {settings_name}) -> Self {{\n"));
        out.push_str("        match _s {\n");
        for variant in &enum_info.variants {
            if variant.is_rich() {
                let field_names: Vec<&str> =
                    variant.named_fields.iter().map(|f| f.name.as_str()).collect();
                let pattern = field_names.join(", ");
                out.push_str(&format!(
                    "            {settings_name}::{} {{ {pattern} }} => {}::{} {{\n",
                    variant.name, enum_info.enum_name, variant.name
                ));
                for field in &variant.named_fields {
                    let value = field_value_expr(field, field.name.clone(), enums, commands);
                    out.push_str(&format!("                {}: {},\n", field.name, value));
                }
                out.push_str("            },\n");
            } else if variant.data_type.is_some() {
                out.push_str(&format!(
                    "            {settings_name}::{}(v) => {}::{}(v),\n",
                    variant.name, enum_info.enum_name, variant.name
                ));
            } else {
                out.push_str(&format!(
                    "            {settings_name}::{} => {}::{},\n",
                    variant.name, enum_info.enum_name, variant.name
                ));
            }
        }
        out.push_str("        }\n    }\n}\n\n");
    }

    // Struct From impls
    out.push_str("// ============ STRUCT FROM IMPLS ============\n\n");
    for cmd in &filtered_commands {
        let settings_name = format!("{}Settings", cmd.struct_name);

        if let Some(feature) = &cmd.feature {
            out.push_str(&format!("#[cfg(feature = {feature:?})]\n"));
        }
        out.push_str(&format!(
            "impl From<{settings_name}> for {} {{\n",
            cmd.struct_name
        ));
        out.push_str(&format!("    fn from(_s: {settings_name}) -> Self {{\n"));
        out.push_str(&format!("        {} {{\n", cmd.struct_name));

        for field in &cmd.fields {
            let field_access = format!("_s.{}", field.name);
            let field_value = field_value_expr(field, field_access, enums, commands);
            out.push_str(&format!("            {}: {},\n", field.name, field_value));
        }
        out.push_str("        }\n    }\n}\n\n");
    }

    // into_algorithm standalone function - only structs that implement ImageAlgorithm
    out.push_str("// ============ INTO ALGORITHM ============\n\n");
    out.push_str("use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;\n");
    out.push_str("use evanalyzer_cfg::core_types::InternalErrors;\n\n");
    out.push_str(
        "pub fn into_algorithm(cmd: PipelineCommand) -> Result<Box<dyn ImageAlgorithm>, InternalErrors> {\n",
    );
    out.push_str("    match cmd {\n");
    for cmd in filtered_commands.iter().filter(|c| c.is_algo) {
        match &cmd.feature {
            Some(feature) => {
                out.push_str(&format!("        #[cfg(feature = {feature:?})]\n"));
                out.push_str(&format!(
                    "        PipelineCommand::{}(settings) => Ok(Box::new(crate::algos::{}::from(settings))),\n",
                    cmd.struct_name, cmd.struct_name
                ));
                out.push_str(&format!("        #[cfg(not(feature = {feature:?}))]\n"));
                out.push_str(&format!(
                    "        PipelineCommand::{}(_settings) => Err(InternalErrors::Generic(\"This build was compiled without the {feature} feature; {} is unavailable.\".into())),\n",
                    cmd.struct_name, cmd.struct_name
                ));
            }
            None => {
                out.push_str(&format!(
                    "        PipelineCommand::{}(settings) => Ok(Box::new(crate::algos::{}::from(settings))),\n",
                    cmd.struct_name, cmd.struct_name
                ));
            }
        }
    }
    out.push_str("    }\n}\n");

    out
}

// ============================================================
// All helpers unchanged from your original
// ============================================================

/// Scans `algos.rs` for top-level `#[cfg(feature = "...")] mod <name>;` declarations,
/// returning a map from module name to the gating feature (if any). A module with
/// no `#[cfg(feature = ...)]` attribute maps to `None`.
fn parse_module_features(algos_rs_path: &Path) -> HashMap<String, Option<String>> {
    let mut map = HashMap::new();
    let Ok(content) = fs::read_to_string(algos_rs_path) else {
        return map;
    };
    let Ok(file) = parse_file(&content) else {
        return map;
    };
    for item in &file.items {
        if let Item::Mod(item_mod) = item {
            let name = item_mod.ident.to_string();
            let feature = item_mod.attrs.iter().find_map(|attr| {
                if !attr.path().is_ident("cfg") {
                    return None;
                }
                let tokens = attr.meta.require_list().ok()?.tokens.to_string();
                let start = tokens.find('"')?;
                let end = tokens[start + 1..].find('"')?;
                Some(tokens[start + 1..start + 1 + end].to_string())
            });
            map.insert(name, feature);
        }
    }
    map
}

fn scan_directory(
    dir: &Path,
    commands: &mut Vec<CommandInfo>,
    enums: &mut Vec<EnumInfo>,
    feature: Option<&str>,
) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                scan_directory(&path, commands, enums, feature);
            } else if path.extension().map_or(false, |ext| ext == "rs")
                && path.file_name().map_or(false, |n| n != "mod.rs")
            {
                extract_command_structs(&path, commands, enums, feature);
            }
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct FieldMetadata {
    min: Option<f32>,
    max: Option<f32>,
    default: Option<f64>,
    default_expr: Option<String>,
    step: Option<f32>,
    custom_name: Option<String>,
    unit: Option<String>,
    regex: Option<String>,
    display_name: Option<String>,
    summary: bool,
    optional: bool,
    visible: bool,
    /// Comma-separated list of file extensions (no leading dot, e.g. "pt,pth")
    /// for `PathBuf` fields rendered with a "Browse…" button. Empty/absent
    /// means any file is selectable.
    file_extensions: Option<String>,
}

impl Default for FieldMetadata {
    fn default() -> Self {
        Self {
            min: None,
            max: None,
            default: None,
            default_expr: None,
            step: None,
            custom_name: None,
            unit: None,
            regex: None,
            display_name: None,
            summary: false,
            optional: false,
            visible: true,
            file_extensions: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct StructMetadata {
    /// Explicit category override from #[cmdsmeta(category = "...")]
    category: Option<String>,
    /// Explicit display name override from #[cmdsmeta(display_name = "...")],
    /// shown in the command picker/header instead of the bare struct name.
    display_name: Option<String>,
    /// Explicit list of categories that may follow this command, from
    /// #[cmdsmeta(next = "measure,classify")]. Overrides the category-derived
    /// default successors (see `default_next_for_category`). Used to decouple a
    /// command's display `category` from what is allowed to come after it — e.g.
    /// StarDist/Cellpose live in `segment` for grouping but flow straight to
    /// `measure`, while U-Net (also `segment`) flows to `object`.
    next: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
struct CommandInfo {
    struct_name: String,
    fields: Vec<FieldInfo>,
    category: String,
    _source_file: String,
    doc_comments: Vec<String>,
    is_algo: bool,
    struct_meta: StructMetadata,
    /// Cargo feature (in the `core` crate) that must be enabled for this command's
    /// `crate::algos::<struct_name>` to exist, e.g. `Some("ai")` for U-Net/Cellpose/StarDist.
    feature: Option<String>,
}

#[derive(Debug, Clone)]
struct FieldInfo {
    name: String,
    ty: String,
    doc_comments: Vec<String>,
    metadata: FieldMetadata,
}

#[derive(Debug, Clone)]
struct EnumInfo {
    enum_name: String,
    variants: Vec<EnumVariant>,
    source_file: String,
    doc_comments: Vec<String>,
    feature: Option<String>,
}

impl EnumInfo {
    /// A "rich" enum has at least one struct-like (named-field) variant, e.g.
    /// `Scale { factor: f32 }`. These are generated as an internally-tagged
    /// (`#[serde(tag = "type")]`) settings enum — a `oneOf` + `const` discriminator in the
    /// JSON schema — so each variant only carries the fields it actually needs, instead of
    /// every command sharing one flat struct with fields whose meaning shifts per variant.
    ///
    /// Plain/tuple enums (e.g. `Outliers(f32)`) keep the existing externally-tagged
    /// representation, since internal tagging can't represent a tuple variant and switching
    /// representation would break already-saved project files.
    fn is_rich(&self) -> bool {
        self.variants.iter().any(|v| v.is_rich())
    }

    fn settings_name(&self) -> String {
        format!(
            "{}{}Settings",
            to_pascal_case(&self.source_file),
            self.enum_name
        )
    }
}

#[derive(Debug, Clone)]
struct EnumVariant {
    name: String,
    data_type: Option<String>,
    /// Named (struct-like) variant fields, e.g. `Scale { factor: f32 }`. Empty for
    /// unit variants and for the single-unnamed-field tuple variants covered by
    /// `data_type` (e.g. `Outliers(f32)`). A variant has at most one of `data_type`
    /// / non-empty `named_fields` set.
    named_fields: Vec<FieldInfo>,
    doc_comments: Vec<String>,
    display_name: Option<String>,
}

impl EnumVariant {
    fn is_rich(&self) -> bool {
        !self.named_fields.is_empty()
    }
}

fn extract_command_structs(
    file_path: &Path,
    commands: &mut Vec<CommandInfo>,
    enums: &mut Vec<EnumInfo>,
    feature: Option<&str>,
) {
    let content = match fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(_) => return,
    };

    if !content.contains("impl ImageAlgorithm") && !content.contains("impl ") {
        return;
    }

    let ast = match parse_file(&content) {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("Failed to parse {}: {}", file_path.display(), e);
            return;
        }
    };

    let category = determine_category(file_path);
    let source_file = extract_source_module(file_path);

    // Only structs with `impl ImageAlgorithm for X` are pipeline commands.
    let algo_structs: std::collections::HashSet<String> = ast
        .items
        .iter()
        .filter_map(|item| {
            if let Item::Impl(item_impl) = item {
                let trait_name = item_impl
                    .trait_
                    .as_ref()
                    .and_then(|(_, path, _)| path.segments.last())
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                if trait_name == "ImageAlgorithm" {
                    if let Type::Path(tp) = item_impl.self_ty.as_ref() {
                        return tp.path.segments.last().map(|s| s.ident.to_string());
                    }
                }
            }
            None
        })
        .collect();

    for item in ast.items {
        match item {
            Item::Struct(item_struct) => {
                if !matches!(item_struct.vis, syn::Visibility::Public(_)) {
                    continue;
                }
                let struct_name = item_struct.ident.to_string();
                if struct_name.ends_with("Settings")
                    || struct_name.ends_with("Parameters")
                    || struct_name == "PipelineContext"
                {
                    continue;
                }
                let is_algo = algo_structs.contains(&struct_name);
                let fields = extract_fields(&item_struct);
                let doc_comments = extract_doc_comments(&item_struct.attrs);
                let struct_meta = parse_struct_meta(&item_struct.attrs);
                // Explicit category annotation overrides directory heuristic
                let effective_category = struct_meta
                    .category
                    .as_deref()
                    .map(normalize_category)
                    .unwrap_or_else(|| category.clone());
                eprintln!(
                    "DBG extract struct={} raw_cat={:?} effective_cat={:?}",
                    item_struct.ident, struct_meta.category, effective_category
                );
                commands.push(CommandInfo {
                    struct_name,
                    fields,
                    category: effective_category,
                    _source_file: source_file.clone(),
                    doc_comments,
                    is_algo,
                    struct_meta,
                    feature: feature.map(str::to_string),
                });
            }
            Item::Enum(item_enum) => {
                if !matches!(item_enum.vis, syn::Visibility::Public(_)) {
                    continue;
                }
                let enum_name = item_enum.ident.to_string();
                if enum_name.ends_with("Settings") || enum_name == "Result" || enum_name == "Option"
                {
                    continue;
                }
                let variants = extract_enum_variants(&item_enum);
                if !variants.is_empty() {
                    let doc_comments = extract_doc_comments(&item_enum.attrs);
                    enums.push(EnumInfo {
                        enum_name,
                        variants,
                        source_file: source_file.clone(),
                        doc_comments,
                        feature: feature.map(str::to_string),
                    });
                }
            }
            _ => {}
        }
    }
}

fn determine_category(file_path: &Path) -> String {
    let path_str = file_path.to_string_lossy();
    if path_str.contains("filters")
        || path_str.contains("blur")
        || path_str.contains("morphology")
        || path_str.contains("edge")
        || path_str.contains("contrast")
        || path_str.contains("color")
        || path_str.contains("math")
        || path_str.contains("spartial")
    {
        "Preprocessing".to_string()
    } else if path_str.contains("segmentation") || path_str.contains("threshold") {
        "Segmentation".to_string()
    } else if path_str.contains("classification") || path_str.contains("extract") {
        "Classification".to_string()
    } else {
        "Other".to_string()
    }
}

/// Normalise a user-supplied category string (from #[cmdsmeta(category = "...")]) into
/// the canonical internal name used throughout the generator.
fn normalize_category(raw: &str) -> String {
    match raw.to_ascii_lowercase().as_str() {
        "Preprocessing" | "preprocessing" => "Preprocessing".to_string(),
        "segment" | "segmentation" | "ai_segmentation" => "Segmentation".to_string(),
        "object" | "object_detection" | "detect" => "Object".to_string(),
        "measure" | "measurement" => "Measure".to_string(),
        "classify" | "classification" => "Classification".to_string(),
        _ => "Other".to_string(),
    }
}

/// Read struct-level #[cmdsmeta(category = "...")] attributes.
fn parse_struct_meta(attrs: &[syn::Attribute]) -> StructMetadata {
    let mut meta = StructMetadata::default();
    for attr in attrs {
        if attr.path().is_ident("cmdsmeta") {
            let _ = attr.parse_nested_meta(|m| {
                if m.path.is_ident("category") {
                    let value: syn::LitStr = m.value()?.parse()?;
                    meta.category = Some(value.value());
                } else if m.path.is_ident("display_name") {
                    let value: syn::LitStr = m.value()?.parse()?;
                    meta.display_name = Some(value.value());
                } else if m.path.is_ident("next") {
                    let value: syn::LitStr = m.value()?.parse()?;
                    meta.next = Some(
                        value
                            .value()
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect(),
                    );
                }
                // Consume any value so the parser advances even for unknown keys
                if m.input.peek(syn::Token![=]) {
                    let _: syn::Expr = m.value()?.parse()?;
                }
                Ok(())
            });
        }
    }
    meta
}

fn extract_doc_comments(attrs: &[syn::Attribute]) -> Vec<String> {
    let mut docs = Vec::new();
    for attr in attrs {
        if attr.path().is_ident("doc") {
            if let syn::Meta::NameValue(nv) = &attr.meta {
                if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) = &nv.value
                {
                    docs.push(s.value().trim_end().to_string());
                }
            }
        }
    }
    docs
}

fn extract_source_module(file_path: &Path) -> String {
    let path_str = file_path.to_string_lossy();
    if let Some(pos) = path_str.find("algos/") {
        let after_algos = &path_str[pos + 6..];
        let without_rs = after_algos.strip_suffix(".rs").unwrap_or(after_algos);
        without_rs.replace("/", "_").replace("\\", "_")
    } else {
        "unknown".to_string()
    }
}

fn extract_fields(item_struct: &ItemStruct) -> Vec<FieldInfo> {
    let mut fields = Vec::new();
    if let syn::Fields::Named(named) = &item_struct.fields {
        for field in &named.named {
            if let Some(ident) = &field.ident {
                let doc_comments = extract_doc_comments(&field.attrs);
                let metadata = parse_custom_meta(field);
                let ty = type_to_string(&field.ty);
                fields.push(FieldInfo {
                    name: ident.to_string(),
                    ty,
                    doc_comments,
                    metadata,
                });
            }
        }
    }
    fields
}

fn extract_enum_variants(item_enum: &ItemEnum) -> Vec<EnumVariant> {
    item_enum
        .variants
        .iter()
        .map(|v| {
            let mut doc_comments = Vec::new();
            let mut display_name: Option<String> = None;
            for attr in &v.attrs {
                if attr.path().is_ident("doc") {
                    if let syn::Meta::NameValue(nv) = &attr.meta {
                        if let syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(s),
                            ..
                        }) = &nv.value
                        {
                            doc_comments.push(s.value().trim().to_string());
                        }
                    }
                }
                if attr.path().is_ident("cmdsmeta") {
                    let _ = attr.parse_nested_meta(|meta| {
                        if meta.path.is_ident("display_name") {
                            let value: syn::LitStr = meta.value()?.parse()?;
                            display_name = Some(value.value());
                        } else if meta.input.peek(syn::Token![=]) {
                            let _: syn::Expr = meta.value()?.parse()?;
                        }
                        Ok(())
                    });
                }
            }
            let (data_type, named_fields) = match &v.fields {
                syn::Fields::Unnamed(unnamed) if unnamed.unnamed.len() == 1 => {
                    (Some(type_to_string(&unnamed.unnamed[0].ty)), Vec::new())
                }
                syn::Fields::Named(named) => {
                    let fields = named
                        .named
                        .iter()
                        .filter_map(|f| {
                            let ident = f.ident.as_ref()?;
                            Some(FieldInfo {
                                name: ident.to_string(),
                                ty: type_to_string(&f.ty),
                                doc_comments: extract_doc_comments(&f.attrs),
                                metadata: parse_custom_meta(f),
                            })
                        })
                        .collect();
                    (None, fields)
                }
                _ => (None, Vec::new()),
            };
            EnumVariant {
                name: v.ident.to_string(),
                data_type,
                named_fields,
                doc_comments,
                display_name,
            }
        })
        .collect()
}

fn type_to_string(ty: &Type) -> String {
    match ty {
        Type::Path(type_path) => {
            let mut result = String::new();
            for (i, segment) in type_path.path.segments.iter().enumerate() {
                if i > 0 {
                    result.push_str("::");
                }
                result.push_str(&segment.ident.to_string());
                match &segment.arguments {
                    PathArguments::AngleBracketed(args) => {
                        result.push('<');
                        for (i, arg) in args.args.iter().enumerate() {
                            if i > 0 {
                                result.push_str(", ");
                            }
                            match arg {
                                GenericArgument::Type(inner_ty) => {
                                    result.push_str(&type_to_string(inner_ty));
                                }
                                GenericArgument::Const(expr) => {
                                    result.push_str(&quote::quote!(#expr).to_string());
                                }
                                _ => result.push('_'),
                            }
                        }
                        result.push('>');
                    }
                    _ => {}
                }
            }
            result
        }
        Type::Array(type_array) => {
            format!(
                "[{}; {}]",
                type_to_string(&type_array.elem),
                quote::quote!(&type_array.len).to_string()
            )
        }
        _ => "Unknown".to_string(),
    }
}

fn map_to_settings_type(ty: &str, enums: &[EnumInfo], commands: &[CommandInfo]) -> String {
    if ty.starts_with("Vec<") && ty.ends_with('>') {
        let inner = &ty[4..ty.len() - 1];
        return format!("Vec<{}>", map_to_settings_type(inner, enums, commands));
    }
    if ty.starts_with("Option<") && ty.ends_with('>') {
        let inner = &ty[7..ty.len() - 1];
        return format!("Option<{}>", map_to_settings_type(inner, enums, commands));
    }
    if let Some(e) = enums.iter().find(|e| e.enum_name == ty) {
        return format!("{}{}Settings", to_pascal_case(&e.source_file), ty);
    }
    if commands.iter().any(|c| c.struct_name == ty) {
        return format!("{}Settings", ty);
    }
    ty.to_string()
}

fn parse_custom_meta(field: &syn::Field) -> FieldMetadata {
    let mut metadata = FieldMetadata::default();
    for attr in &field.attrs {
        if attr.path().is_ident("cmdsettings") || attr.path().is_ident("cmdsmeta") {
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("min") {
                    let stream = meta.value()?;
                    metadata.min = Some(if stream.peek(syn::LitFloat) {
                        stream.parse::<syn::LitFloat>()?.base10_parse::<f32>()?
                    } else {
                        stream.parse::<syn::LitInt>()?.base10_parse::<f32>()?
                    });
                } else if meta.path.is_ident("max") {
                    let stream = meta.value()?;
                    metadata.max = Some(if stream.peek(syn::LitFloat) {
                        stream.parse::<syn::LitFloat>()?.base10_parse::<f32>()?
                    } else {
                        stream.parse::<syn::LitInt>()?.base10_parse::<f32>()?
                    });
                } else if meta.path.is_ident("rename") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    let raw_name = value.value();
                    metadata.custom_name = Some(to_camel_case(&raw_name));
                } else if meta.path.is_ident("unit") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    metadata.unit = Some(value.value());
                } else if meta.path.is_ident("display_name") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    metadata.display_name = Some(value.value());
                } else if meta.path.is_ident("default") {
                    let stream = meta.value()?;
                    if stream.peek(syn::LitFloat) {
                        metadata.default =
                            Some(stream.parse::<syn::LitFloat>()?.base10_parse::<f64>()?);
                    } else if stream.peek(syn::LitInt) {
                        metadata.default =
                            Some(stream.parse::<syn::LitInt>()?.base10_parse::<f64>()?);
                    } else if stream.peek(syn::LitBool) {
                        let b = stream.parse::<syn::LitBool>()?;
                        metadata.default = Some(if b.value { 1.0 } else { 0.0 });
                    } else {
                        let expr: syn::Expr = stream.parse()?;
                        metadata.default_expr = Some(quote::quote!(#expr).to_string());
                    }
                } else if meta.path.is_ident("step") {
                    let stream = meta.value()?;
                    metadata.step = Some(if stream.peek(syn::LitFloat) {
                        stream.parse::<syn::LitFloat>()?.base10_parse::<f32>()?
                    } else {
                        stream.parse::<syn::LitInt>()?.base10_parse::<f32>()?
                    });
                } else if meta.path.is_ident("summary") {
                    // Default to true; if "= false" is present, honour it.
                    metadata.summary = true;
                    if let Ok(stream) = meta.value() {
                        if let Ok(b) = stream.parse::<syn::LitBool>() {
                            metadata.summary = b.value;
                        }
                    }
                } else if meta.path.is_ident("optional") {
                    metadata.optional = true;
                    if let Ok(stream) = meta.value() {
                        if let Ok(b) = stream.parse::<syn::LitBool>() {
                            metadata.optional = b.value;
                        }
                    }
                } else if meta.path.is_ident("visible") {
                    metadata.visible = true;
                    if let Ok(stream) = meta.value() {
                        if let Ok(b) = stream.parse::<syn::LitBool>() {
                            metadata.visible = b.value;
                        }
                    }
                } else if meta.path.is_ident("file_extensions") {
                    let value: syn::LitStr = meta.value()?.parse()?;
                    metadata.file_extensions = Some(value.value());
                }
                Ok(())
            });
        }
    }
    metadata
}

fn pascal_to_title_case(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut result = String::new();
    for i in 0..chars.len() {
        let c = chars[i];
        if c.is_uppercase() && i > 0 {
            let prev = chars[i - 1];
            let next = chars.get(i + 1).copied();
            // Standard lower→upper boundary (e.g. "addOutput" → "add Output")
            // OR start of a new capitalized word after an acronym (e.g. "HTMLParser" → "HTML Parser")
            if prev.is_lowercase()
                || (prev.is_uppercase() && next.map_or(false, |n| n.is_lowercase()))
            {
                result.push(' ');
            }
        }
        result.push(c);
    }
    result
}

fn snake_to_title_case(s: &str) -> String {
    s.split('_')
        .filter(|w| !w.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect()
}

fn to_camel_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = false;
    for (i, c) in s.chars().enumerate() {
        if c == '_' || c == '-' || c == ' ' {
            capitalize_next = true;
        } else if i == 0 {
            result.push(c.to_ascii_lowercase());
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(c);
        }
    }
    result
}

fn is_user_enum(ty: &str, all_enums: &[EnumInfo]) -> bool {
    if ty.starts_with("Vec<")
        || ty.starts_with("Option<")
        || ty.starts_with("Result<")
        || ty.starts_with("HashMap<")
        || ty.starts_with("BTreeMap<")
        || ty.starts_with('[')
    {
        return false;
    }
    let stdlib_types = [
        "f32",
        "f64",
        "i32",
        "i64",
        "u32",
        "u64",
        "usize",
        "bool",
        "String",
        "PathBuf",
        "Duration",
        "SystemTime",
    ];
    if stdlib_types.contains(&ty) {
        return false;
    }
    all_enums.iter().any(|e| e.enum_name == ty)
}

fn is_user_struct(ty: &str, all_commands: &[CommandInfo]) -> bool {
    if ty.starts_with("Vec<")
        || ty.starts_with("Option<")
        || ty.starts_with("Result<")
        || ty.starts_with("HashMap<")
        || ty.starts_with("BTreeMap<")
        || ty.starts_with('[')
    {
        return false;
    }
    let stdlib_types = [
        "f32",
        "f64",
        "i32",
        "i64",
        "u32",
        "u64",
        "usize",
        "bool",
        "String",
        "PathBuf",
        "Duration",
        "SystemTime",
    ];
    if stdlib_types.contains(&ty) {
        return false;
    }
    all_commands.iter().any(|c| c.struct_name == ty)
}

/// Escapes a doc-comment block into the single-line, backslash-escaped string literal body
/// used for both a `ParameterDef.description` and a discriminant-variant's own field.
fn escape_doc_comments(doc_comments: &[String]) -> String {
    let lines: Vec<String> = doc_comments
        .iter()
        .map(|s| {
            s.trim_start()
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .trim_end()
                .to_string()
        })
        .collect();
    let last_nonempty = lines
        .iter()
        .rposition(|s| !s.trim().is_empty())
        .map(|i| i + 1)
        .unwrap_or(0);
    lines[..last_nonempty].join("\\n")
}

/// Joins a list of `Vec<ParameterDef>`-typed expressions into one such expression. Used
/// wherever a variable number of sub-expressions (one per field, each already a `Vec`) need to
/// collapse into the single `Vec<ParameterDef>` a command/group-item/variant must produce.
fn concat_param_vecs(parts: &[String]) -> String {
    if parts.is_empty() {
        "vec![]".to_string()
    } else if parts.len() == 1 {
        parts[0].clone()
    } else {
        format!("[{}].concat()", parts.join(", "))
    }
}

/// Builds a single `ParameterDef { ... }` literal for one leaf field (not Vec/Option/HashMap/
/// array/nested-struct/rich-enum — the caller handles those before reaching this point).
///
/// `access` is the Rust expression that reads the field's current value — `"_s.kernel_size"`
/// for an ordinary struct field, or just the bound identifier (e.g. `"factor"`) when called for
/// a field bound out of a rich enum variant's pattern. Returns `None` for unrecognized types
/// (e.g. `ImageAddress`), which are silently skipped, same as today.
fn leaf_param_def_literal(
    ty: &str,
    meta: &FieldMetadata,
    access: &str,
    routing_name: &str,
    display_label: &str,
    description: &str,
    enums: &[EnumInfo],
) -> Option<String> {
    let (param_type, value_expr, options_expr, min, max) = match ty {
        "f32" | "f64" => {
            if meta.step.is_some() {
                // step given → spinner regardless of min/max
                (
                    "ParamType::Spinner",
                    format!("format!(\"{{}}\", {access})"),
                    "vec![]".to_string(),
                    meta.min.unwrap_or(0.0),
                    meta.max.unwrap_or(0.0),
                )
            } else if meta.min.is_some() && meta.max.is_some() {
                // min+max but no step → spinner
                (
                    "ParamType::Spinner",
                    format!("format!(\"{{}}\", {access})"),
                    "vec![]".to_string(),
                    meta.min.unwrap(),
                    meta.max.unwrap(),
                )
            } else {
                (
                    "ParamType::Number",
                    format!("format!(\"{{}}\", {access})"),
                    "vec![]".to_string(),
                    meta.min.unwrap_or(0.0),
                    meta.max.unwrap_or(0.0),
                )
            }
        }
        "usize" | "u32" | "u64" | "i32" | "i64" => {
            // min == max → read-only label; value is shown but not editable.
            if let (Some(min_v), Some(max_v)) = (meta.min, meta.max) {
                if (min_v - max_v).abs() < f32::EPSILON {
                    return Some(format!(
                        "ParameterDef {{ name: \"{routing_name}\".to_string(), display_name: \"{display_label}\".to_string(), description: \"{description}\".to_string(), value: format!(\"{{}}\", {access}), param_type: ParamType::Label, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }}"
                    ));
                }
            }
            if let (Some(step), Some(min_v), Some(max_v)) = (meta.step, meta.min, meta.max) {
                if step > 0.0 && min_v < max_v {
                    let count = ((max_v - min_v) / step).floor() as usize + 1;
                    if count < 10 {
                        let opts: Vec<String> = (0..count)
                            .map(|i| {
                                let v = min_v + i as f32 * step;
                                format!("\"{}\".to_string()", v as i64)
                            })
                            .collect();
                        (
                            "ParamType::Dropdown",
                            format!("format!(\"{{}}\", {access})"),
                            format!("vec![{}]", opts.join(", ")),
                            min_v,
                            max_v,
                        )
                    } else {
                        (
                            "ParamType::Spinner",
                            format!("format!(\"{{}}\", {access})"),
                            "vec![]".to_string(),
                            min_v,
                            max_v,
                        )
                    }
                } else {
                    (
                        "ParamType::Number",
                        format!("format!(\"{{}}\", {access})"),
                        "vec![]".to_string(),
                        meta.min.unwrap_or(0.0),
                        meta.max.unwrap_or(0.0),
                    )
                }
            } else if let Some(_step) = meta.step {
                // step without min/max → spinner with no clamping bounds
                (
                    "ParamType::Spinner",
                    format!("format!(\"{{}}\", {access})"),
                    "vec![]".to_string(),
                    meta.min.unwrap_or(0.0),
                    meta.max.unwrap_or(0.0),
                )
            } else {
                (
                    "ParamType::Number",
                    format!("format!(\"{{}}\", {access})"),
                    "vec![]".to_string(),
                    meta.min.unwrap_or(0.0),
                    meta.max.unwrap_or(0.0),
                )
            }
        }
        "bool" => (
            "ParamType::Toggle",
            format!("format!(\"{{}}\", {access})"),
            "vec![]".to_string(),
            0.0_f32,
            0.0_f32,
        ),
        "String" => (
            "ParamType::Text",
            format!("{access}.clone()"),
            "vec![]".to_string(),
            0.0_f32,
            0.0_f32,
        ),
        "PathBuf" => {
            // The Slint browse button reads a single comma-separated string from
            // `options[0]` (it splits on ',' itself), so emit one CSV element
            // rather than one element per extension — otherwise only the first
            // extension reaches the file dialog filter.
            let extensions_expr = match &meta.file_extensions {
                Some(exts) => {
                    let csv = exts
                        .split(',')
                        .map(|e| e.trim())
                        .filter(|e| !e.is_empty())
                        .collect::<Vec<_>>()
                        .join(",");
                    if csv.is_empty() {
                        "vec![]".to_string()
                    } else {
                        format!("vec![\"{csv}\".to_string()]")
                    }
                }
                None => "vec![]".to_string(),
            };
            (
                "ParamType::FilePath",
                format!("{access}.display().to_string()"),
                extensions_expr,
                0.0_f32,
                0.0_f32,
            )
        }
        "ObjectClass" => (
            "ParamType::ObjClass",
            format!(
                "match {access}.to_u32() {{ Some(v) => format!(\"{{}}\", v), None => \"-1\".to_string() }}"
            ),
            "vec![]".to_string(),
            0.0_f32,
            0.0_f32,
        ),
        "SegmentationClass" => (
            "ParamType::SegClass",
            format!("format!(\"{{}}\", {access}.as_u32())"),
            "vec![]".to_string(),
            0.0_f32,
            0.0_f32,
        ),
        "PixelUnits" => (
            "ParamType::PixelUnits",
            format!(
                "match {access} {{ PixelUnits::Bit => \"bit\".to_string(), PixelUnits::Percent => \"%\".to_string(), PixelUnits::Relative => \"rel\".to_string() }}"
            ),
            "vec![\"bit\".to_string(), \"%\".to_string(), \"rel\".to_string()]".to_string(),
            0.0_f32,
            0.0_f32,
        ),
        "SizeUnits" => (
            "ParamType::SizeUnits",
            format!(
                "match {access} {{ SizeUnits::NanoMeter => \"nm\".to_string(), SizeUnits::Pixels => \"px\".to_string() }}"
            ),
            "vec![\"nm\".to_string(), \"px\".to_string()]".to_string(),
            0.0_f32,
            0.0_f32,
        ),
        _ => {
            if let Some(enum_info) = enums.iter().find(|e| e.enum_name.as_str() == ty) {
                // A rich (named-field) enum's own fields are exposed by the caller as
                // additional sibling ParameterDefs, conditional on the active variant — here
                // we only render the discriminant dropdown itself.
                let settings_name = enum_info.settings_name();
                let display_map: Vec<(String, String)> = enum_info
                    .variants
                    .iter()
                    .map(|v| {
                        let label = v
                            .display_name
                            .as_deref()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| pascal_to_title_case(&v.name));
                        (v.name.clone(), label)
                    })
                    .collect();
                let options: Vec<String> = display_map
                    .iter()
                    .map(|(_, d)| format!("\"{}\".to_string()", d))
                    .collect();
                let match_arms: String = enum_info
                    .variants
                    .iter()
                    .map(|v| {
                        let label = v
                            .display_name
                            .as_deref()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| pascal_to_title_case(&v.name));
                        // Data variants (e.g. Outliers(f32) or a rich Scale { factor: f32 })
                        // need a wildcard inner pattern.
                        let pattern = if v.is_rich() {
                            format!("{settings_name}::{} {{ .. }}", v.name)
                        } else if v.data_type.is_some() {
                            format!("{settings_name}::{}(_)", v.name)
                        } else {
                            format!("{settings_name}::{}", v.name)
                        };
                        format!("{pattern} => \"{label}\".to_string()")
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let value_expr = format!("match {access} {{ {match_arms} }}");
                (
                    "ParamType::Dropdown",
                    value_expr,
                    format!("vec![{}]", options.join(", ")),
                    0.0_f32,
                    0.0_f32,
                )
            } else {
                // Unknown type (ImageAddress, etc.) - skip
                return None;
            }
        }
    };

    let step = if param_type == "ParamType::Spinner" {
        meta.step.unwrap_or(1.0)
    } else {
        1.0_f32
    };

    Some(format!(
        "ParameterDef {{ name: \"{routing_name}\".to_string(), display_name: \"{display_label}\".to_string(), description: \"{description}\".to_string(), value: {value_expr}, param_type: {param_type}, options: {options_expr}, min: {min:.1}f32, max: {max:.1}f32, step: {step:.4}f32, groups: vec![] }}",
    ))
}

/// Builds the additional ParameterDefs contributed by a rich enum field's *currently active*
/// variant — i.e. the part of "conditional fields" that actually varies the field list, not
/// just the value of a fixed one. Returns a single `match &{access} { ... }` expression of type
/// `Vec<ParameterDef>`, one arm per variant, so switching the discriminant naturally switches
/// which sibling fields exist.
fn rich_enum_variant_param_defs(
    enum_info: &EnumInfo,
    access: &str,
    routing_prefix: &str,
    enums: &[EnumInfo],
) -> String {
    let settings_name = enum_info.settings_name();
    let arms: Vec<String> = enum_info
        .variants
        .iter()
        .map(|v| {
            if !v.is_rich() {
                let pattern = if v.data_type.is_some() {
                    format!("{settings_name}::{}(_)", v.name)
                } else {
                    format!("{settings_name}::{}", v.name)
                };
                return format!("{pattern} => vec![]");
            }
            let field_names: Vec<&str> =
                v.named_fields.iter().map(|f| f.name.as_str()).collect();
            let pattern = format!("{settings_name}::{} {{ {} }}", v.name, field_names.join(", "));
            let field_literals: Vec<String> = v
                .named_fields
                .iter()
                .filter(|f| f.metadata.visible)
                .filter_map(|f| {
                    let routing_name = format!("{routing_prefix}.{}", f.name);
                    let display_label = f
                        .metadata
                        .display_name
                        .as_deref()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| snake_to_title_case(&f.name));
                    let description = escape_doc_comments(&f.doc_comments);
                    leaf_param_def_literal(
                        &f.ty,
                        &f.metadata,
                        &f.name,
                        &routing_name,
                        &display_label,
                        &description,
                        enums,
                    )
                })
                .collect();
            format!("{pattern} => vec![{}]", field_literals.join(", "))
        })
        .collect();
    format!("match &{access} {{ {} }}", arms.join(", "))
}

/// Returns a list of `Vec<ParameterDef>`-typed expressions contributing this field's
/// parameter(s); the caller (one field list per command, or per group item) joins them with
/// [`concat_param_vecs`]. A rich-enum field contributes two expressions: the discriminant
/// dropdown, and a `match` over the active variant for its sibling fields — which is what makes
/// the visible field set actually change when the user switches the dropdown, rather than just
/// changing a label's value.
fn field_to_param_def(
    field: &FieldInfo,
    enums: &[EnumInfo],
    commands: &[CommandInfo],
    var: &str,
    name_prefix: &str,
) -> Vec<String> {
    let ty = &field.ty;
    let name = &field.name;
    let meta = &field.metadata;

    if !meta.visible {
        return vec![];
    }

    let routing_name = format!("{}{}", name_prefix, name);
    let display_label = meta
        .display_name
        .as_deref()
        .map(|s| s.to_string())
        .unwrap_or_else(|| snake_to_title_case(name));
    let description = escape_doc_comments(&field.doc_comments);
    let access = format!("{var}.{name}");

    // Vec<UserStruct> → Group param
    if ty.starts_with("Vec<") && ty.ends_with('>') {
        let inner_ty = &ty[4..ty.len() - 1];
        if let Some(inner_cmd) = commands.iter().find(|c| c.struct_name == inner_ty) {
            let inner_parts: Vec<String> = inner_cmd
                .fields
                .iter()
                .flat_map(|inner_field| {
                    field_to_param_def(inner_field, enums, commands, "__item", "")
                })
                .collect();
            let inner_expr = concat_param_vecs(&inner_parts);
            return vec![format!(
                "vec![ParameterDef {{ name: \"{routing_name}\".to_string(), display_name: \"{display_label}\".to_string(), description: \"{description}\".to_string(), value: String::new(), param_type: ParamType::Group, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: {access}.iter().map(|__item| {inner_expr}).collect() }}]"
            )];
        }
        // Vec<ObjectClass> / Vec<SegmentationClass> → multi-select class picker.
        // options holds 33 flag strings ("1"/"0") for classes 0–32; the Slint
        // popup reads options[i] instead of doing a string.contains() check.
        if inner_ty == "ObjectClass" {
            let value_expr = format!(
                "{access}.iter().filter_map(|c| c.to_u32()).map(|v| v.to_string()).collect::<Vec<_>>().join(\",\")"
            );
            let flags_expr = format!(
                "(0u32..33u32).map(|__idx| if {access}.iter().any(|c| c.to_u32().map_or(false, |v| v == __idx)) {{ \"1\".to_string() }} else {{ \"0\".to_string() }}).collect::<Vec<_>>()"
            );
            return vec![format!(
                "vec![ParameterDef {{ name: \"{routing_name}\".to_string(), display_name: \"{display_label}\".to_string(), description: \"{description}\".to_string(), value: {value_expr}, param_type: ParamType::MultiObjClass, options: {flags_expr}, min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }}]"
            )];
        }
        if inner_ty == "SegmentationClass" {
            let value_expr = format!(
                "{access}.iter().map(|c| c.as_u32().to_string()).collect::<Vec<_>>().join(\",\")"
            );
            let flags_expr = format!(
                "(0u32..33u32).map(|__idx| if {access}.iter().any(|c| c.as_u32() == __idx) {{ \"1\".to_string() }} else {{ \"0\".to_string() }}).collect::<Vec<_>>()"
            );
            return vec![format!(
                "vec![ParameterDef {{ name: \"{routing_name}\".to_string(), display_name: \"{display_label}\".to_string(), description: \"{description}\".to_string(), value: {value_expr}, param_type: ParamType::MultiSegClass, options: {flags_expr}, min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }}]"
            )];
        }

        return vec![]; // Vec<primitive> or Vec<unknown> - skip
    }

    // Other non-leaf types - skip
    if ty.starts_with("Option<") || ty.starts_with("HashMap<") || ty.starts_with('[') {
        return vec![];
    }

    // Plain nested UserStruct → flatten its fields inline
    if let Some(nested_cmd) = commands.iter().find(|c| c.struct_name == *ty) {
        let new_prefix = format!("{}{}.", name_prefix, name);
        return nested_cmd
            .fields
            .iter()
            .flat_map(|inner_field| {
                field_to_param_def(inner_field, enums, commands, &access, &new_prefix)
            })
            .collect();
    }

    // Rich enum (e.g. `Scale { factor: f32 }`) → discriminant dropdown, PLUS a second
    // expression that switches in the active variant's own fields. This is what makes the
    // *set* of visible parameters change with the dropdown, not just one field's value.
    if let Some(enum_info) = enums.iter().find(|e| e.enum_name.as_str() == ty.as_str()) {
        if enum_info.is_rich() {
            let dropdown = leaf_param_def_literal(
                ty,
                meta,
                &access,
                &routing_name,
                &display_label,
                &description,
                enums,
            )
            .expect("rich enum dropdown literal is always Some");
            let variant_fields =
                rich_enum_variant_param_defs(enum_info, &access, &routing_name, enums);
            return vec![format!("vec![{dropdown}]"), variant_fields];
        }
    }

    match leaf_param_def_literal(ty, meta, &access, &routing_name, &display_label, &description, enums) {
        Some(literal) => vec![format!("vec![{literal}]")],
        None => vec![],
    }
}

/// Returns (label, value_expr) pairs for fields with `summary = true`.
fn collect_summary_exprs(
    field: &FieldInfo,
    enums: &[EnumInfo],
    commands: &[CommandInfo],
    var: &str,
    name_prefix: &str,
) -> Vec<(String, String)> {
    let ty = &field.ty;
    let name = &field.name;
    let meta = &field.metadata;

    if !meta.visible {
        return vec![];
    }

    if ty.starts_with("Vec<")
        || ty.starts_with("Option<")
        || ty.starts_with("HashMap<")
        || ty.starts_with('[')
    {
        return vec![];
    }
    if let Some(nested_cmd) = commands.iter().find(|c| c.struct_name == *ty) {
        let nested_var = format!("{}.{}", var, name);
        let new_prefix = format!("{}{}", name_prefix, name);
        return nested_cmd
            .fields
            .iter()
            .flat_map(|f| {
                collect_summary_exprs(f, enums, commands, &nested_var, &format!("{}.", new_prefix))
            })
            .collect();
    }
    if !meta.summary {
        return vec![];
    }
    let label = meta
        .display_name
        .as_deref()
        .map(|s| s.to_string())
        .unwrap_or_else(|| snake_to_title_case(name));
    let expr = match ty.as_str() {
        "f32" | "f64" | "usize" | "u32" | "u64" | "i32" | "i64" => {
            format!("format!(\"{{:.3}}\", {var}.{name})")
        }
        "bool" => format!("format!(\"{{}}\", {var}.{name})"),
        "String" => format!("{var}.{name}.clone()"),
        _ => {
            if let Some(enum_info) = enums.iter().find(|e| e.enum_name == *ty) {
                let settings_name = enum_info.settings_name();
                let match_arms: String = enum_info
                    .variants
                    .iter()
                    .map(|v| {
                        let label = v
                            .display_name
                            .as_deref()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| pascal_to_title_case(&v.name));
                        // Rich (named-field) variants only ever show the variant's own name
                        // here — the per-field values would need their own summary slots,
                        // which collect_summary_exprs doesn't support yet.
                        let pattern = if v.is_rich() {
                            format!("{settings_name}::{} {{ .. }}", v.name)
                        } else if v.data_type.is_some() {
                            format!("{settings_name}::{}(_)", v.name)
                        } else {
                            format!("{settings_name}::{}", v.name)
                        };
                        format!("{pattern} => \"{label}\".to_string()")
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("match {var}.{name} {{ {match_arms} }}")
            } else {
                return vec![];
            }
        }
    };
    vec![(label, expr)]
}

fn field_to_apply_change(
    field: &FieldInfo,
    enums: &[EnumInfo],
    commands: &[CommandInfo],
    var: &str,
    name_prefix: &str,
) -> Vec<String> {
    let ty = &field.ty;
    let name = &field.name;
    let display_name = format!("{}{}", name_prefix, name);

    if !field.metadata.visible {
        return vec![];
    }

    // Vec<ObjectClass> / Vec<SegmentationClass> → toggle or full-replace via comma-separated list
    if let Some(inner) = ty.strip_prefix("Vec<").and_then(|s| s.strip_suffix('>')) {
        if inner == "ObjectClass" {
            let branch = format!(
                "if param_name == \"{display_name}\" {{ \
                    if let Some(id) = value.strip_prefix(\"toggle:\").and_then(|x| x.trim().parse::<u32>().ok()) {{ \
                        if {var}.{name}.iter().any(|c| c.to_u32().map_or(false, |v| v == id)) {{ \
                            {var}.{name}.retain(|c| c.to_u32().map_or(true, |v| v != id)); \
                        }} else {{ \
                            {var}.{name}.push(ObjectClass::Valid(id)); \
                        }} \
                    }} else {{ \
                        {var}.{name} = value.split(',').filter(|x| !x.is_empty()).filter_map(|x| x.trim().parse::<u32>().ok()).map(|v| ObjectClass::Valid(v)).collect(); \
                    }} \
                }}"
            );
            return vec![branch];
        }
        if inner == "SegmentationClass" {
            let branch = format!(
                "if param_name == \"{display_name}\" {{ \
                    if let Some(id) = value.strip_prefix(\"toggle:\").and_then(|x| x.trim().parse::<u32>().ok()) {{ \
                        if {var}.{name}.iter().any(|c| c.as_u32() == id) {{ \
                            {var}.{name}.retain(|c| c.as_u32() != id); \
                        }} else {{ \
                            {var}.{name}.push(SegmentationClass(id)); \
                        }} \
                    }} else {{ \
                        {var}.{name} = value.split(',').filter(|x| !x.is_empty()).filter_map(|x| x.trim().parse::<u32>().ok()).map(|v| SegmentationClass(v)).collect(); \
                    }} \
                }}"
            );
            return vec![branch];
        }
    }

    // Vec<UserStruct> → compound key "{field}.{idx}.{nested_field}" for item-level edits
    if let Some(inner) = ty.strip_prefix("Vec<").and_then(|s| s.strip_suffix('>')) {
        if let Some(inner_cmd) = commands.iter().find(|c| c.struct_name == inner) {
            let nested_raw: Vec<String> = inner_cmd
                .fields
                .iter()
                .flat_map(|f| field_to_apply_change(f, enums, commands, "item", ""))
                .collect();
            if !nested_raw.is_empty() {
                // Rename the Rust variable from "param_name" to "nested_name" inside the nested branches
                let nested_branches: Vec<String> = nested_raw
                    .into_iter()
                    .map(|b| {
                        b.replace("param_name ==", "nested_name ==")
                            .replace("param_name.starts_with", "nested_name.starts_with")
                    })
                    .collect();
                let prefix = format!("{display_name}.");
                let prefix_len = prefix.len();
                let mut code = format!(
                    "if param_name.starts_with(\"{prefix}\") {{ \
                     let rest = &param_name[{prefix_len}..]; \
                     let mut _p = rest.splitn(2, '.'); \
                     if let (Some(_i), Some(nested_name)) = (_p.next(), _p.next()) {{ \
                     if let Ok(_idx) = _i.parse::<usize>() {{ \
                     if let Some(item) = {var}.{name}.get_mut(_idx) {{",
                );
                for b in &nested_branches {
                    code.push_str(&format!(" {b}"));
                }
                code.push_str(" } } } }");
                return vec![code];
            }
        }
        return vec![];
    }

    // Option, HashMap, arrays → skip
    if ty.starts_with("Option<") || ty.starts_with("HashMap<") || ty.starts_with('[') {
        return vec![];
    }

    // Plain nested UserStruct → recurse with dotted path
    if let Some(nested_cmd) = commands.iter().find(|c| c.struct_name == *ty) {
        let nested_var = format!("{}.{}", var, name);
        let new_prefix = format!("{}{}.", name_prefix, name);
        let mut results = Vec::new();
        for inner_field in &nested_cmd.fields {
            results.extend(field_to_apply_change(
                inner_field,
                enums,
                commands,
                &nested_var,
                &new_prefix,
            ));
        }
        return results;
    }

    // Rich enum (e.g. `Scale { factor: f32 }`) → setting the field itself picks a variant
    // (constructed with its own field defaults); setting "{field}.{inner}" mutates one field
    // of whichever variant currently happens to be active.
    if let Some(enum_info) = enums.iter().find(|e| e.enum_name.as_str() == ty.as_str()) {
        if enum_info.is_rich() {
            let access = format!("{var}.{name}");
            return vec![rich_enum_apply_change(
                enum_info,
                &access,
                &display_name,
                enums,
                commands,
            )];
        }
    }

    let access = format!("{var}.{name}");
    match leaf_apply_change_branch(ty, &access, &display_name, enums) {
        Some(branch) => vec![branch],
        None => vec![],
    }
}

/// Builds the `if param_name == "{condition}" { ... }` body for a single leaf field, given
/// the Rust expression to assign into (`assign`, e.g. `"_s.kernel_size"` for a struct field, or
/// `"*field1"` for a field bound out of a rich enum variant via `ref mut`). Shared between a
/// command struct's top-level fields and a rich enum variant's own fields.
fn leaf_apply_change_branch(
    ty: &str,
    assign: &str,
    condition: &str,
    enums: &[EnumInfo],
) -> Option<String> {
    let body = match ty {
        "f32" => format!("if let Ok(v) = value.parse::<f32>() {{ {assign} = v; }}"),
        "f64" => format!("if let Ok(v) = value.parse::<f64>() {{ {assign} = v; }}"),
        "usize" => format!("if let Ok(v) = value.parse::<usize>() {{ {assign} = v; }}"),
        "u32" => format!("if let Ok(v) = value.parse::<u32>() {{ {assign} = v; }}"),
        "u64" => format!("if let Ok(v) = value.parse::<u64>() {{ {assign} = v; }}"),
        "i32" => format!("if let Ok(v) = value.parse::<i32>() {{ {assign} = v; }}"),
        "i64" => format!("if let Ok(v) = value.parse::<i64>() {{ {assign} = v; }}"),
        "bool" => format!("{assign} = value == \"true\";"),
        "String" => format!("{assign} = value.to_string();"),
        "PathBuf" => format!("{assign} = std::path::PathBuf::from(value);"),
        "ObjectClass" => format!(
            "if value == \"-1\" {{ {assign} = ObjectClass::Unset; }} \
             else if let Ok(v) = value.parse::<u32>() {{ {assign} = ObjectClass::Valid(v); }}"
        ),
        "SegmentationClass" => {
            format!("if let Ok(v) = value.parse::<u32>() {{ {assign} = SegmentationClass(v); }}")
        }
        "PixelUnits" => format!(
            "{assign} = match value {{ \"bit\" => PixelUnits::Bit, \"%\" => PixelUnits::Percent, _ => PixelUnits::Relative }};"
        ),
        "SizeUnits" => format!(
            "{assign} = match value {{ \"nm\" => SizeUnits::NanoMeter, _ => SizeUnits::Pixels }};"
        ),
        _ => {
            if let Some(enum_info) = enums.iter().find(|e| e.enum_name.as_str() == ty) {
                let settings_name = enum_info.settings_name();
                // Only plain unit variants can be picked by label here; data-carrying
                // variants (tuple or rich) need their own fields supplied, which a flat
                // `param_name == condition` set can't do — see `rich_enum_apply_change` for
                // the rich case.
                let arms: String = enum_info
                    .variants
                    .iter()
                    .filter(|v| v.data_type.is_none() && !v.is_rich())
                    .map(|v| {
                        let label = v
                            .display_name
                            .as_deref()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| pascal_to_title_case(&v.name));
                        format!("\"{}\" => {}::{}, ", label, settings_name, v.name)
                    })
                    .collect();
                format!("{assign} = match value {{ {arms}_ => {assign}.clone() }};")
            } else {
                return None;
            }
        }
    };
    Some(format!("if param_name == \"{condition}\" {{ {body} }}"))
}

/// Builds the apply_param_change branches for a rich enum field: one to switch the active
/// variant (constructing it with its own `#[cmdsmeta(default = ...)]` field values), and one
/// per variant to mutate a single inner field — keyed by `"{field}.{inner_field}"` — without
/// disturbing which variant is active.
fn rich_enum_apply_change(
    enum_info: &EnumInfo,
    access: &str,
    display_name: &str,
    enums: &[EnumInfo],
    commands: &[CommandInfo],
) -> String {
    let settings_name = enum_info.settings_name();

    let switch_arms: String = enum_info
        .variants
        .iter()
        .map(|v| {
            let label = v
                .display_name
                .as_deref()
                .map(|s| s.to_string())
                .unwrap_or_else(|| pascal_to_title_case(&v.name));
            let construct = if v.is_rich() {
                let fields: String = v
                    .named_fields
                    .iter()
                    .map(|f| format!("{}: {}, ", f.name, field_default_expr(f, enums, commands)))
                    .collect();
                format!("{settings_name}::{} {{ {fields} }}", v.name)
            } else if v.data_type.is_some() {
                // Switching onto a tuple variant from the discriminant alone has no value to
                // supply; keep whatever was already there rather than picking an arbitrary one.
                format!("{access}.clone()")
            } else {
                format!("{settings_name}::{}", v.name)
            };
            format!("\"{label}\" => {construct}, ")
        })
        .collect();
    let switch_branch = format!(
        "if param_name == \"{display_name}\" {{ {access} = match value {{ {switch_arms}_ => {access}.clone() }}; }}"
    );

    let mut nested_branches = String::new();
    for v in &enum_info.variants {
        if !v.is_rich() {
            continue;
        }
        let bindings: String = v
            .named_fields
            .iter()
            .map(|f| format!("ref mut {}, ", f.name))
            .collect();
        let inner: Vec<String> = v
            .named_fields
            .iter()
            .filter(|f| f.metadata.visible)
            .filter_map(|f| {
                let condition = format!("{display_name}.{}", f.name);
                let assign = format!("*{}", f.name);
                leaf_apply_change_branch(&f.ty, &assign, &condition, enums)
            })
            .collect();
        if inner.is_empty() {
            continue;
        }
        nested_branches.push_str(&format!(
            "if let {settings_name}::{} {{ {bindings} }} = {access} {{ {} }} ",
            v.name,
            inner.join(" ")
        ));
    }

    format!("{switch_branch} {nested_branches}")
}

/// The name shown to the user for a command: the `#[cmdsmeta(display_name = "...")]`
/// override if present and non-empty, otherwise the bare struct name.
fn command_display_name(cmd: &CommandInfo) -> &str {
    cmd.struct_meta
        .display_name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(&cmd.struct_name)
}

fn category_to_enum_variant(category: &str) -> &str {
    match category {
        "Preprocessing" => "Preprocess",
        "Segmentation" => "Segment",
        "Object" => "Object",
        "Measure" => "Measure",
        "Classification" => "Classify",
        _ => "Preprocess",
    }
}

/// Default successor categories for a command that does not declare an explicit
/// `#[cmdsmeta(next = "...")]`. Mirrors the linear pipeline order
/// (Preprocess → Segment → Object → Measure → Classify) but lets a stage repeat
/// (e.g. Object → Object so Watershed can follow ConnectedComponents).
///
/// Ordering is significant: the **first** entry is the *suggested* next category
/// (the command picker pre-selects its chip), the rest are merely allowed. So the
/// default after an Object step suggests Measure but still permits another Object;
/// ConnectedComponents overrides this to suggest Object (Watershed) first.
fn default_next_for_category(variant: &str) -> &'static [&'static str] {
    match variant {
        "Preprocess" => &["Segment", "Preprocess"],
        "Segment" => &["Object"],
        "Object" => &["Measure", "Object"],
        "Measure" => &["Classify", "Measure"],
        "Classify" => &["Classify"],
        _ => &["Segment"],
    }
}

fn generate_pipeline_command_enum(commands: &[CommandInfo], enums: &[EnumInfo]) -> String {
    let mut out = String::new();
    const GENERATE_ALL_DEFAULT: bool = false;

    // Only true algorithm structs go into the PipelineCommand enum.
    let algo_commands: Vec<&CommandInfo> = commands.iter().filter(|c| c.is_algo).collect();

    out.push_str("// @generated - do not edit by hand\n");
    out.push_str("use crate::modules::pipeline_command_settings::*;\n");
    out.push_str("use crate::modules::parameter_def::{ParamType, ParameterDef};\n");
    out.push_str("use crate::types::classes::{ObjectClass, SegmentationClass};\n");
    out.push_str("use crate::core_types::{PixelUnits, SizeUnits};\n");
    out.push_str("use schemars::JsonSchema;\n");
    out.push_str("use serde::{Deserialize, Serialize};\n\n");

    // --- CommandCategory enum ---
    out.push_str(
        "#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, JsonSchema)]\n",
    );
    out.push_str("#[serde(rename_all = \"camelCase\")]\n");
    out.push_str("pub enum CommandCategory {\n");
    out.push_str("    Preprocess,\n");
    out.push_str("    Segment,\n");
    out.push_str("    Object,\n");
    out.push_str("    Measure,\n");
    out.push_str("    Classify,\n");
    out.push_str("}\n\n");

    // --- CommandCategory methods (ordering rules) ---
    out.push_str("impl CommandCategory {\n");
    out.push_str("    /// Ordered position in the pipeline (0 = first, higher = later).\n");
    out.push_str("#[allow(dead_code)]\n");
    out.push_str("    pub fn display_order(self) -> u8 {\n");
    out.push_str("        match self {\n");
    out.push_str("            Self::Preprocess => 0,\n");
    out.push_str("            Self::Segment    => 1,\n");
    out.push_str("            Self::Object     => 2,\n");
    out.push_str("            Self::Measure    => 3,\n");
    out.push_str("            Self::Classify   => 4,\n");
    out.push_str("        }\n    }\n\n");
    out.push_str("    /// Which categories are valid immediately before this one in a pipeline.\n");
    out.push_str("    /// An empty slice means this category can start a pipeline.\n");
    out.push_str("#[allow(dead_code)]\n");
    out.push_str("    pub fn allowed_after(self) -> &'static [CommandCategory] {\n");
    out.push_str("        match self {\n");
    out.push_str("            Self::Preprocess => &[Self::Preprocess],\n");
    out.push_str("            Self::Segment    => &[Self::Preprocess, Self::Segment],\n");
    out.push_str("            Self::Object     => &[Self::Segment, Self::Object],\n");
    out.push_str("            Self::Measure    => &[Self::Object, Self::Measure],\n");
    out.push_str("            Self::Classify   => &[Self::Measure, Self::Classify],\n");
    out.push_str("        }\n    }\n\n");
    out.push_str("    /// The natural next category after this one, used to pre-filter the command picker.\n");
    out.push_str("#[allow(dead_code)]\n");
    out.push_str("    pub fn suggested_next(self) -> CommandCategory {\n");
    out.push_str("        match self {\n");
    out.push_str("            Self::Preprocess => Self::Segment,\n");
    out.push_str("            Self::Segment    => Self::Object,\n");
    out.push_str("            Self::Object     => Self::Measure,\n");
    out.push_str("            Self::Measure    => Self::Classify,\n");
    out.push_str("            Self::Classify   => Self::Classify,\n");
    out.push_str("        }\n    }\n");
    out.push_str("}\n\n");

    // --- PipelineCommand enum ---
    out.push_str("#[derive(Serialize, Deserialize, Debug, Clone, JsonSchema)]\n");
    out.push_str("#[serde(tag = \"type\", rename_all = \"camelCase\")]\n");
    out.push_str("pub enum PipelineCommand {\n");
    for cmd in &algo_commands {
        let settings_name = format!("{}Settings", cmd.struct_name);
        out.push_str(&format!("    {}({}),\n", cmd.struct_name, settings_name));
    }
    out.push_str("}\n\n");

    // --- CommandMeta + helpers ---

    out.push_str("#[allow(dead_code)]\n");
    out.push_str("pub struct CommandMeta {\n");
    out.push_str("    pub id: i32,\n");
    out.push_str("    pub name: &'static str,\n");
    out.push_str("    pub category: CommandCategory,\n");
    out.push_str("    pub summary: &'static str,\n");
    out.push_str("    pub description: &'static str,\n");
    out.push_str("}\n\n");

    out.push_str("#[allow(dead_code)]\n");
    out.push_str("pub fn all_command_meta() -> Vec<CommandMeta> {\n    vec![\n");
    for (i, cmd) in algo_commands.iter().enumerate() {
        let summary = cmd
            .doc_comments
            .first()
            .map(|s| s.trim().replace('"', "\\\""))
            .unwrap_or_default();
        // Description: everything after the first blank separator line, joined with \n
        let description = {
            let rest = if cmd.doc_comments.len() > 1 {
                &cmd.doc_comments[1..]
            } else {
                &[][..]
            };
            let start = rest
                .iter()
                .position(|s| !s.trim().is_empty())
                .unwrap_or(rest.len());
            rest[start..]
                .iter()
                .map(|s| s.trim_start().replace('\\', "\\\\").replace('"', "\\\""))
                .collect::<Vec<_>>()
                .join("\\n")
        };
        let cat = category_to_enum_variant(&cmd.category);
        let display_name = command_display_name(cmd);
        out.push_str(&format!(
            "        CommandMeta {{ id: {i}, name: \"{display_name}\", category: CommandCategory::{cat}, summary: \"{summary}\", description: \"{description}\" }},\n",
        ));
    }
    out.push_str("    ]\n}\n\n");

    out.push_str("#[allow(dead_code)]\n");
    out.push_str("pub fn default_command(id: i32) -> Option<PipelineCommand> {\n    match id {\n");
    for (i, cmd) in algo_commands.iter().enumerate() {
        out.push_str(&format!(
            "        {i} => Some(PipelineCommand::{}({}Settings::default())),\n",
            cmd.struct_name, cmd.struct_name
        ));
    }
    out.push_str("        _ => None,\n    }\n}\n\n");

    // --- impl PipelineCommand ---
    out.push_str("#[allow(dead_code)]\n");
    out.push_str("impl PipelineCommand {\n");

    // name()
    out.push_str("    pub fn name(&self) -> &str {\n");
    out.push_str("        match self {\n");
    for cmd in &algo_commands {
        out.push_str(&format!(
            "            Self::{}(_) => \"{}\",\n",
            cmd.struct_name,
            command_display_name(cmd)
        ));
    }
    out.push_str("        }\n    }\n\n");

    // category()
    out.push_str("    pub fn category(&self) -> &CommandCategory {\n");
    out.push_str("        match self {\n");
    for cmd in &algo_commands {
        let variant = category_to_enum_variant(&cmd.category);
        out.push_str(&format!(
            "            Self::{}(_) => &CommandCategory::{},\n",
            cmd.struct_name, variant
        ));
    }
    out.push_str("        }\n    }\n\n");

    // allowed_next(): which categories may follow this specific command in a
    // pipeline. Defaults to the category's natural successors, but a command can
    // override this via #[cmdsmeta(next = "...")] to decouple display grouping
    // from flow — e.g. StarDist/Cellpose are shown under `segment` yet flow
    // straight to `measure`, while U-Net (also `segment`) flows to `object`.
    out.push_str(
        "    /// Categories that may be inserted immediately after this command.\n",
    );
    out.push_str("    pub fn allowed_next(&self) -> &'static [CommandCategory] {\n");
    out.push_str("        match self {\n");
    for cmd in &algo_commands {
        let variant = category_to_enum_variant(&cmd.category);
        let next_variants: Vec<String> = match &cmd.struct_meta.next {
            Some(list) => list
                .iter()
                .map(|c| format!("CommandCategory::{}", category_to_enum_variant(&normalize_category(c))))
                .collect(),
            None => default_next_for_category(variant)
                .iter()
                .map(|c| format!("CommandCategory::{c}"))
                .collect(),
        };
        out.push_str(&format!(
            "            Self::{}(_) => &[{}],\n",
            cmd.struct_name,
            next_variants.join(", ")
        ));
    }
    out.push_str("        }\n    }\n\n");

    // to_parameters()
    out.push_str("    pub fn to_parameters(&self) -> Vec<ParameterDef> {\n");
    out.push_str("        match self {\n");
    for cmd in &algo_commands {
        let parts: Vec<String> = cmd
            .fields
            .iter()
            .flat_map(|field| field_to_param_def(field, enums, commands, "_s", ""))
            .collect();
        out.push_str(&format!(
            "            Self::{}(_s) => {},\n",
            cmd.struct_name,
            concat_param_vecs(&parts)
        ));
    }
    out.push_str("        }\n    }\n\n");

    // default_settings() - returns a boxed default for UI "add command" dialogs
    if GENERATE_ALL_DEFAULT {
        out.push_str("    pub fn all_defaults() -> Vec<PipelineCommand> {\n");
        out.push_str("        vec![\n");
        for cmd in &algo_commands {
            out.push_str(&format!(
                "            PipelineCommand::{}({}Settings::default()),\n",
                cmd.struct_name, cmd.struct_name
            ));
        }
        out.push_str("        ]\n    }\n");
    }

    // to_summary() - short human-readable parameter line for the step header
    out.push_str("    pub fn to_summary(&self) -> String {\n");
    out.push_str("        match self {\n");
    for cmd in &algo_commands {
        let parts: Vec<(String, String)> = cmd
            .fields
            .iter()
            .flat_map(|f| collect_summary_exprs(f, enums, commands, "s", ""))
            .collect();
        if parts.is_empty() {
            out.push_str(&format!(
                "            Self::{}(_) => String::new(),\n",
                cmd.struct_name
            ));
        } else {
            let fmt_str: String = parts
                .iter()
                .map(|(lbl, _)| format!("{lbl}: {{}}"))
                .collect::<Vec<_>>()
                .join(" · ");
            let args: String = parts
                .iter()
                .map(|(_, expr)| expr.clone())
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!(
                "            Self::{}(s) => format!(\"{fmt_str}\", {args}),\n",
                cmd.struct_name
            ));
        }
    }
    out.push_str("        }\n    }\n\n");

    // apply_param_change() - write a single leaf parameter value back to settings
    out.push_str("    pub fn apply_param_change(&mut self, param_name: &str, value: &str) {\n");
    out.push_str("        match self {\n");
    for cmd in &algo_commands {
        let branches: Vec<String> = cmd
            .fields
            .iter()
            .flat_map(|f| field_to_apply_change(f, enums, commands, "s", ""))
            .collect();
        if branches.is_empty() {
            out.push_str(&format!(
                "            Self::{}(_) => {{}}\n",
                cmd.struct_name
            ));
        } else {
            out.push_str(&format!("            Self::{}(s) => {{\n", cmd.struct_name));
            for b in &branches {
                out.push_str(&format!("                {b}\n"));
            }
            out.push_str("            }\n");
        }
    }
    out.push_str("        }\n    }\n\n");

    // add_group_item() - clone-last strategy for Vec<UserStruct> fields
    out.push_str("    pub fn add_group_item(&mut self, param_name: &str) {\n");
    out.push_str("        match self {\n");
    for cmd in &algo_commands {
        let vec_fields: Vec<&FieldInfo> = cmd
            .fields
            .iter()
            .filter(|f| {
                if let Some(inner) = f.ty.strip_prefix("Vec<").and_then(|s| s.strip_suffix('>')) {
                    commands.iter().any(|c| c.struct_name == inner)
                } else {
                    false
                }
            })
            .collect();
        if vec_fields.is_empty() {
            out.push_str(&format!(
                "            Self::{}(_) => {{}}\n",
                cmd.struct_name
            ));
        } else {
            out.push_str(&format!("            Self::{}(s) => {{\n", cmd.struct_name));
            for f in &vec_fields {
                let inner_ty = &f.ty[4..f.ty.len() - 1]; // strip Vec< >
                let inner_settings = format!("{}Settings", inner_ty);
                out.push_str(&format!(
                    "                if param_name == \"{}\" {{ if let Some(last) = s.{}.last().cloned() {{ s.{}.push(last); }} else {{ s.{}.push({}::default()); }} }}\n",
                    f.name, f.name, f.name, f.name, inner_settings
                ));
            }
            out.push_str("            }\n");
        }
    }
    out.push_str("        }\n    }\n\n");

    // remove_group_item()
    out.push_str("    pub fn remove_group_item(&mut self, param_name: &str, idx: usize) {\n");
    out.push_str("        match self {\n");
    for cmd in &algo_commands {
        let vec_fields: Vec<&FieldInfo> = cmd
            .fields
            .iter()
            .filter(|f| {
                if let Some(inner) = f.ty.strip_prefix("Vec<").and_then(|s| s.strip_suffix('>')) {
                    commands.iter().any(|c| c.struct_name == inner)
                } else {
                    false
                }
            })
            .collect();
        if vec_fields.is_empty() {
            out.push_str(&format!(
                "            Self::{}(_) => {{}}\n",
                cmd.struct_name
            ));
        } else {
            out.push_str(&format!("            Self::{}(s) => {{\n", cmd.struct_name));
            for f in &vec_fields {
                out.push_str(&format!(
                    "                if param_name == \"{}\" && idx < s.{}.len() {{ s.{}.remove(idx); }}\n",
                    f.name, f.name, f.name
                ));
            }
            out.push_str("            }\n");
        }
    }
    out.push_str("        }\n    }\n\n");

    out.push_str("}\n\n");

    out
}
