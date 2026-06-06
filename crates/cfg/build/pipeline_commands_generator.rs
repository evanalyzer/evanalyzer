use std::{collections::HashMap, fs, path::Path};
use syn::{GenericArgument, Item, ItemEnum, ItemStruct, PathArguments, Type, parse_file};

pub fn generate_mappings() -> Result<(), Box<dyn std::error::Error>> {
    let algos_path = Path::new("../core/src/algos");
    let mut commands = Vec::new();
    let mut enums = Vec::new();

    if let Ok(entries) = fs::read_dir(algos_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                scan_directory(&path, &mut commands, &mut enums);
            } else if path.extension().map_or(false, |ext| ext == "rs") {
                extract_command_structs(&path, &mut commands, &mut enums);
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
        let settings_name = format!(
            "{}{}Settings",
            to_pascal_case(&enum_info.source_file),
            enum_info.enum_name
        );

        for doc in &enum_info.doc_comments {
            out.push_str(&format!("/// {}\n", doc));
        }
        out.push_str(
            "#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq, Default)]\n",
        );
        out.push_str("#[serde(rename_all = \"camelCase\")]\n");
        out.push_str(&format!("pub enum {} {{\n", settings_name));
        for (vi, variant) in enum_info.variants.iter().enumerate() {
            for doc in &variant.doc_comments {
                out.push_str(&format!("    /// {}\n", doc));
            }
            if vi == 0 {
                out.push_str("    #[default]\n");
            }
            if let Some(ref data_type) = variant.data_type {
                out.push_str(&format!("    {}({}),\n", variant.name, data_type));
            } else {
                out.push_str(&format!("    {},\n", variant.name));
            }
        }
        out.push_str("}\n\n");
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
                    let body = if let Some(ref expr) = field.metadata.default_expr {
                        remap_default_expr(expr, enums, commands)
                    } else if let Some(val) = field.metadata.default {
                        format_default_for_type(&field.ty, val)
                    } else {
                        format!("{}::default()", field_type)
                    };
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
                        let field_type = map_to_settings_type(&field.ty, enums, commands);
                        let default_expr = if let Some(ref expr) = field.metadata.default_expr {
                            remap_default_expr(expr, enums, commands)
                        } else if let Some(val) = field.metadata.default {
                            format_default_for_type(&field.ty, val)
                        } else if field_type.starts_with("Vec<") {
                            "vec![]".to_string()
                        } else if field_type.starts_with("Option<") {
                            "None".to_string()
                        } else {
                            format!("{}::default()", field_type)
                        };
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

        out.push_str(&format!(
            "impl From<{settings_name}> for {} {{\n",
            enum_info.enum_name
        ));
        out.push_str(&format!("    fn from(_s: {settings_name}) -> Self {{\n"));
        out.push_str("        match _s {\n");
        for variant in &enum_info.variants {
            if variant.data_type.is_some() {
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

        out.push_str(&format!(
            "impl From<{settings_name}> for {} {{\n",
            cmd.struct_name
        ));
        out.push_str(&format!("    fn from(_s: {settings_name}) -> Self {{\n"));
        out.push_str(&format!("        {} {{\n", cmd.struct_name));

        for field in &cmd.fields {
            let meta = &field.metadata;
            let field_access = format!("_s.{}", field.name);

            let field_value = if field.ty == "f32" || field.ty == "f64" {
                match (meta.min, meta.max) {
                    (Some(min), Some(max)) => {
                        format!("{}.clamp({:?}, {:?})", field_access, min, max)
                    }
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
            };

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
        out.push_str(&format!(
            "        PipelineCommand::{}(settings) => Ok(Box::new(crate::algos::{}::from(settings))),\n",
            cmd.struct_name, cmd.struct_name
        ));
    }
    out.push_str("    }\n}\n");

    out
}

// ============================================================
// All helpers unchanged from your original
// ============================================================

fn scan_directory(dir: &Path, commands: &mut Vec<CommandInfo>, enums: &mut Vec<EnumInfo>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                scan_directory(&path, commands, enums);
            } else if path.extension().map_or(false, |ext| ext == "rs")
                && path.file_name().map_or(false, |n| n != "mod.rs")
            {
                extract_command_structs(&path, commands, enums);
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
        }
    }
}

#[derive(Debug, Clone, Default)]
struct StructMetadata {
    /// Explicit category override from #[cmdsmeta(category = "...")]
    category: Option<String>,
}

#[derive(Debug, Clone)]
struct CommandInfo {
    struct_name: String,
    fields: Vec<FieldInfo>,
    category: String,
    _source_file: String,
    doc_comments: Vec<String>,
    is_algo: bool,
    _struct_meta: StructMetadata,
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
}

#[derive(Debug, Clone)]
struct EnumVariant {
    name: String,
    data_type: Option<String>,
    doc_comments: Vec<String>,
}

fn extract_command_structs(
    file_path: &Path,
    commands: &mut Vec<CommandInfo>,
    enums: &mut Vec<EnumInfo>,
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
                    _struct_meta: struct_meta,
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
        "segment" | "segmentation" => "Segmentation".to_string(),
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
            }
            let data_type = match &v.fields {
                syn::Fields::Unnamed(unnamed) if unnamed.unnamed.len() == 1 => {
                    Some(type_to_string(&unnamed.unnamed[0].ty))
                }
                _ => None,
            };
            EnumVariant {
                name: v.ident.to_string(),
                data_type,
                doc_comments,
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
                }
                Ok(())
            });
        }
    }
    metadata
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

fn field_to_param_def(
    field: &FieldInfo,
    enums: &[EnumInfo],
    commands: &[CommandInfo],
    var: &str,
    indent: &str,
    name_prefix: &str,
) -> Vec<String> {
    let ty = &field.ty;
    let name = &field.name;
    let meta = &field.metadata;

    if !meta.visible {
        return vec![];
    }

    let routing_name = format!("{}{}", name_prefix, name);
    // display_label: from cmdsmeta display_name, else title-case the bare field name
    let display_label = meta
        .display_name
        .as_deref()
        .map(|s| s.to_string())
        .unwrap_or_else(|| snake_to_title_case(name));
    // Full doc comment joined with \n, escaping chars that would break a string literal.
    // Trailing blank lines are stripped. The first non-empty line is the summary;
    // text after the first blank line is the extended description.
    let description = {
        let lines: Vec<String> = field
            .doc_comments
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
    };

    // Vec<UserStruct> → Group param
    if ty.starts_with("Vec<") && ty.ends_with('>') {
        let inner_ty = &ty[4..ty.len() - 1];
        if let Some(inner_cmd) = commands.iter().find(|c| c.struct_name == inner_ty) {
            let inner_indent = format!("{}    ", indent);
            let mut inner_params = String::new();
            for inner_field in &inner_cmd.fields {
                for s in
                    field_to_param_def(inner_field, enums, commands, "__item", &inner_indent, "")
                {
                    inner_params.push_str(&s);
                }
            }
            return vec![format!(
                "{indent}ParameterDef {{ name: \"{routing_name}\".to_string(), display_name: \"{display_label}\".to_string(), description: \"{description}\".to_string(), value: String::new(), param_type: ParamType::Group, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: {var}.{name}.iter().map(|__item| vec![\n{inner_params}{inner_indent}]).collect() }},\n"
            )];
        }
        // Vec<ObjectClass> / Vec<SegmentationClass> → multi-select class picker.
        // options holds 33 flag strings ("1"/"0") for classes 0–32; the Slint
        // popup reads options[i] instead of doing a string.contains() check.
        if inner_ty == "ObjectClass" {
            let value_expr = format!(
                "{var}.{name}.iter().filter_map(|c| c.to_u32()).map(|v| v.to_string()).collect::<Vec<_>>().join(\",\")"
            );
            let flags_expr = format!(
                "(0u32..33u32).map(|__idx| if {var}.{name}.iter().any(|c| c.to_u32().map_or(false, |v| v == __idx)) {{ \"1\".to_string() }} else {{ \"0\".to_string() }}).collect::<Vec<_>>()"
            );
            return vec![format!(
                "{indent}ParameterDef {{ name: \"{routing_name}\".to_string(), display_name: \"{display_label}\".to_string(), description: \"{description}\".to_string(), value: {value_expr}, param_type: ParamType::MultiObjClass, options: {flags_expr}, min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }},\n"
            )];
        }
        if inner_ty == "SegmentationClass" {
            let value_expr = format!(
                "{var}.{name}.iter().map(|c| c.as_u32().to_string()).collect::<Vec<_>>().join(\",\")"
            );
            let flags_expr = format!(
                "(0u32..33u32).map(|__idx| if {var}.{name}.iter().any(|c| c.as_u32() == __idx) {{ \"1\".to_string() }} else {{ \"0\".to_string() }}).collect::<Vec<_>>()"
            );
            return vec![format!(
                "{indent}ParameterDef {{ name: \"{routing_name}\".to_string(), display_name: \"{display_label}\".to_string(), description: \"{description}\".to_string(), value: {value_expr}, param_type: ParamType::MultiSegClass, options: {flags_expr}, min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }},\n"
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
        let nested_var = format!("{}.{}", var, name);
        let new_prefix = format!("{}{}", name_prefix, name);
        let mut results = Vec::new();
        for inner_field in &nested_cmd.fields {
            results.extend(field_to_param_def(
                inner_field,
                enums,
                commands,
                &nested_var,
                indent,
                &format!("{}.", new_prefix),
            ));
        }
        return results;
    }

    let (param_type, value_expr, options_expr, min, max) = match ty.as_str() {
        "f32" | "f64" => {
            if meta.step.is_some() {
                // step given → spinner regardless of min/max
                (
                    "ParamType::Spinner",
                    format!("format!(\"{{}}\", {var}.{name})"),
                    "vec![]".to_string(),
                    meta.min.unwrap_or(0.0),
                    meta.max.unwrap_or(0.0),
                )
            } else if meta.min.is_some() && meta.max.is_some() {
                // min+max but no step → spinner
                (
                    "ParamType::Spinner",
                    format!("format!(\"{{}}\", {var}.{name})"),
                    "vec![]".to_string(),
                    meta.min.unwrap(),
                    meta.max.unwrap(),
                )
            } else {
                (
                    "ParamType::Number",
                    format!("format!(\"{{}}\", {var}.{name})"),
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
                    return vec![format!(
                        "{indent}ParameterDef {{ name: \"{routing_name}\".to_string(), display_name: \"{display_label}\".to_string(), description: \"{description}\".to_string(), value: format!(\"{{}}\", {var}.{name}), param_type: ParamType::Label, options: vec![], min: 0.0f32, max: 0.0f32, step: 1.0000f32, groups: vec![] }},\n"
                    )];
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
                            format!("format!(\"{{}}\", {var}.{name})"),
                            format!("vec![{}]", opts.join(", ")),
                            min_v,
                            max_v,
                        )
                    } else {
                        (
                            "ParamType::Spinner",
                            format!("format!(\"{{}}\", {var}.{name})"),
                            "vec![]".to_string(),
                            min_v,
                            max_v,
                        )
                    }
                } else {
                    (
                        "ParamType::Number",
                        format!("format!(\"{{}}\", {var}.{name})"),
                        "vec![]".to_string(),
                        meta.min.unwrap_or(0.0),
                        meta.max.unwrap_or(0.0),
                    )
                }
            } else if let Some(_step) = meta.step {
                // step without min/max → spinner with no clamping bounds
                (
                    "ParamType::Spinner",
                    format!("format!(\"{{}}\", {var}.{name})"),
                    "vec![]".to_string(),
                    meta.min.unwrap_or(0.0),
                    meta.max.unwrap_or(0.0),
                )
            } else {
                (
                    "ParamType::Number",
                    format!("format!(\"{{}}\", {var}.{name})"),
                    "vec![]".to_string(),
                    meta.min.unwrap_or(0.0),
                    meta.max.unwrap_or(0.0),
                )
            }
        }
        "bool" => (
            "ParamType::Toggle",
            format!("format!(\"{{}}\", {var}.{name})"),
            "vec![]".to_string(),
            0.0_f32,
            0.0_f32,
        ),
        "String" => (
            "ParamType::Text",
            format!("{var}.{name}.clone()"),
            "vec![]".to_string(),
            0.0_f32,
            0.0_f32,
        ),
        "PathBuf" => (
            "ParamType::Text",
            format!("{var}.{name}.display().to_string()"),
            "vec![]".to_string(),
            0.0_f32,
            0.0_f32,
        ),
        "ObjectClass" => (
            "ParamType::ObjClass",
            format!("match {var}.{name}.to_u32() {{ Some(v) => format!(\"{{}}\", v), None => \"-1\".to_string() }}"),
            "vec![]".to_string(),
            0.0_f32,
            0.0_f32,
        ),
        "SegmentationClass" => (
            "ParamType::SegClass",
            format!("format!(\"{{}}\", {var}.{name}.as_u32())"),
            "vec![]".to_string(),
            0.0_f32,
            0.0_f32,
        ),
        "PixelUnits" => (
            "ParamType::PixelUnits",
            format!(
                "match {var}.{name} {{ PixelUnits::Bit => \"bit\".to_string(), PixelUnits::Percent => \"%\".to_string(), PixelUnits::Relative => \"rel\".to_string() }}"
            ),
            "vec![\"bit\".to_string(), \"%\".to_string(), \"rel\".to_string()]".to_string(),
            0.0_f32,
            0.0_f32,
        ),
        "SizeUnits" => (
            "ParamType::SizeUnits",
            format!(
                "match {var}.{name} {{ SizeUnits::NanoMeter => \"nm\".to_string(), SizeUnits::Pixels => \"px\".to_string() }}"
            ),
            "vec![\"nm\".to_string(), \"px\".to_string()]".to_string(),
            0.0_f32,
            0.0_f32,
        ),
        _ => {
            if let Some(enum_info) = enums.iter().find(|e| e.enum_name.as_str() == ty.as_str()) {
                let variants: Vec<String> = enum_info
                    .variants
                    .iter()
                    .map(|v| format!("\"{}\".to_string()", v.name))
                    .collect();
                (
                    "ParamType::Dropdown",
                    format!("format!(\"{{:?}}\", {var}.{name})"),
                    format!("vec![{}]", variants.join(", ")),
                    0.0_f32,
                    0.0_f32,
                )
            } else {
                // Unknown type (ImageAddress, etc.) - skip
                return vec![];
            }
        }
    };

    let step = if param_type == "ParamType::Spinner" {
        meta.step.unwrap_or(1.0)
    } else {
        1.0_f32
    };

    vec![format!(
        "{indent}ParameterDef {{ name: \"{routing_name}\".to_string(), display_name: \"{display_label}\".to_string(), description: \"{description}\".to_string(), value: {value_expr}, param_type: {param_type}, options: {options_expr}, min: {min:.1}f32, max: {max:.1}f32, step: {step:.4}f32, groups: vec![] }},\n",
    )]
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
            if enums.iter().any(|e| e.enum_name == *ty) {
                format!("format!(\"{{:?}}\", {var}.{name})")
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

    let branch = match ty.as_str() {
        "f32" => format!(
            "if param_name == \"{display_name}\" {{ if let Ok(v) = value.parse::<f32>() {{ {var}.{name} = v; }} }}"
        ),
        "f64" => format!(
            "if param_name == \"{display_name}\" {{ if let Ok(v) = value.parse::<f64>() {{ {var}.{name} = v; }} }}"
        ),
        "usize" => format!(
            "if param_name == \"{display_name}\" {{ if let Ok(v) = value.parse::<usize>() {{ {var}.{name} = v; }} }}"
        ),
        "u32" => format!(
            "if param_name == \"{display_name}\" {{ if let Ok(v) = value.parse::<u32>() {{ {var}.{name} = v; }} }}"
        ),
        "u64" => format!(
            "if param_name == \"{display_name}\" {{ if let Ok(v) = value.parse::<u64>() {{ {var}.{name} = v; }} }}"
        ),
        "i32" => format!(
            "if param_name == \"{display_name}\" {{ if let Ok(v) = value.parse::<i32>() {{ {var}.{name} = v; }} }}"
        ),
        "i64" => format!(
            "if param_name == \"{display_name}\" {{ if let Ok(v) = value.parse::<i64>() {{ {var}.{name} = v; }} }}"
        ),
        "bool" => {
            format!("if param_name == \"{display_name}\" {{ {var}.{name} = value == \"true\"; }}")
        }
        "String" => {
            format!("if param_name == \"{display_name}\" {{ {var}.{name} = value.to_string(); }}")
        }
        "PathBuf" => format!(
            "if param_name == \"{display_name}\" {{ {var}.{name} = std::path::PathBuf::from(value); }}"
        ),
        "ObjectClass" => format!(
            "if param_name == \"{display_name}\" {{ \
                if value == \"-1\" {{ {var}.{name} = ObjectClass::Unset; }} \
                else if let Ok(v) = value.parse::<u32>() {{ {var}.{name} = ObjectClass::Valid(v); }} \
            }}"
        ),
        "SegmentationClass" => format!(
            "if param_name == \"{display_name}\" {{ if let Ok(v) = value.parse::<u32>() {{ {var}.{name} = SegmentationClass(v); }} }}"
        ),
        "PixelUnits" => format!(
            "if param_name == \"{display_name}\" {{ {var}.{name} = match value {{ \"bit\" => PixelUnits::Bit, \"%\" => PixelUnits::Percent, _ => PixelUnits::Relative }}; }}"
        ),
        "SizeUnits" => format!(
            "if param_name == \"{display_name}\" {{ {var}.{name} = match value {{ \"nm\" => SizeUnits::NanoMeter, _ => SizeUnits::Pixels }}; }}"
        ),
        _ => {
            if let Some(enum_info) = enums.iter().find(|e| e.enum_name.as_str() == ty.as_str()) {
                let settings_name =
                    format!("{}{}Settings", to_pascal_case(&enum_info.source_file), ty);
                let arms: String = enum_info
                    .variants
                    .iter()
                    .filter(|v| v.data_type.is_none())
                    .map(|v| format!("\"{}\" => {}::{}, ", v.name, settings_name, v.name))
                    .collect();
                format!(
                    "if param_name == \"{display_name}\" {{ {var}.{name} = match value {{ {arms}_ => {var}.{name}.clone() }}; }}"
                )
            } else {
                return vec![];
            }
        }
    };

    vec![branch]
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
        out.push_str(&format!(
            "        CommandMeta {{ id: {i}, name: \"{}\", category: CommandCategory::{cat}, summary: \"{summary}\", description: \"{description}\" }},\n",
            cmd.struct_name
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
            cmd.struct_name, cmd.struct_name
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

    // to_parameters()
    out.push_str("    pub fn to_parameters(&self) -> Vec<ParameterDef> {\n");
    out.push_str("        match self {\n");
    for cmd in &algo_commands {
        out.push_str(&format!(
            "            Self::{}(_s) => vec![\n",
            cmd.struct_name
        ));
        for field in &cmd.fields {
            for param_str in
                field_to_param_def(field, enums, commands, "_s", "                ", "")
            {
                out.push_str(&param_str);
            }
        }
        out.push_str("            ],\n");
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
