use proc_macro::TokenStream;
use syn::{Data, DeriveInput, parse_macro_input};

/// This derive itself only validates `#[cmdsmeta(...)]` attribute syntax at
/// compile time (it emits no code of its own) - the actual `PipelineCommand`/
/// `...Settings`/`to_parameters()`/`apply_param_change()` code is generated
/// by `crates/cfg/build/pipeline_commands_generator.rs`, which re-parses
/// every `#[derive(CommandsMeta)]` struct across the workspace at build time.
/// See `README.md`'s "Adding a new pipeline command" section for the full
/// picture of what declaring a field a certain way gets you for free.
///
/// A `pub field: PathBuf` is one such case worth calling out here: the build
/// script maps `PathBuf` to `ParamType::FilePath` purely from the Rust type,
/// no attribute required - `file_extensions` (below) only narrows the native
/// file-picker's filter for that field, it doesn't opt the field into
/// `FilePath` handling. Every `ParamType::FilePath` field is, in turn,
/// automatically stored relative to the project directory on save and
/// resolved back to absolute on load (see
/// `evanalyzer_app::extensions::utils::relativize_file_paths`/
/// `resolve_file_paths`, which drive this off `to_parameters()`'s
/// `ParamType` rather than any hardcoded field/command name) - so a new
/// command with a `PathBuf` field gets project-relative storage for free too,
/// with nothing to add or remember here.
#[proc_macro_derive(CommandsMeta, attributes(cmdsmeta))]
pub fn commands_meta_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // Keys allowed on struct-level #[cmdsmeta(...)]
    let struct_allowed_keys = ["category", "display_name", "next"];

    // Keys allowed on field-level #[cmdsmeta(...)]
    let field_allowed_keys = [
        "default",
        "min",
        "max",
        "step",
        "rename",
        "unit",
        "regex",
        "display_name",
        "summary",
        "optional",
        "visible",
        // Comma-separated file extensions (no dots, e.g. "pt,pth") for a
        // `PathBuf` field's native file-picker filter - see this module's
        // doc comment above for what a `PathBuf` field gets automatically
        // *without* this attribute (ParamType::FilePath, project-relative
        // storage). Optional; an empty/absent filter just allows any file.
        "file_extensions",
    ];

    // Validate struct-level #[cmdsmeta(...)]
    for attr in &input.attrs {
        if attr.path().is_ident("cmdsmeta") {
            let result = attr.parse_nested_meta(|meta| {
                if struct_allowed_keys.iter().any(|&k| meta.path.is_ident(k)) {
                    if meta.input.peek(syn::Token![=]) {
                        let _value: syn::Expr = meta.value()?.parse()?;
                    }
                    Ok(())
                } else {
                    Err(meta.error(format!(
                        "Unsupported struct-level key: '{}'. Allowed: {:?}",
                        meta.path.get_ident().unwrap(),
                        struct_allowed_keys
                    )))
                }
            });
            if let Err(e) = result {
                return e.to_compile_error().into();
            }
        }
    }

    // Validate field-level #[cmdsmeta(...)]
    if let Data::Struct(data) = input.data {
        for field in data.fields {
            for attr in field.attrs {
                if attr.path().is_ident("cmdsmeta") {
                    let result = attr.parse_nested_meta(|meta| {
                        if field_allowed_keys.iter().any(|&k| meta.path.is_ident(k)) {
                            if meta.input.peek(syn::Token![=]) {
                                let _value: syn::Expr = meta.value()?.parse()?;
                            }
                            Ok(())
                        } else {
                            Err(meta.error(format!(
                                "Unsupported field-level key: '{}'. Allowed: {:?}",
                                meta.path.get_ident().unwrap(),
                                field_allowed_keys
                            )))
                        }
                    });
                    if let Err(e) = result {
                        return e.to_compile_error().into();
                    }
                }
            }
        }
    }

    // Return empty: satisfies the compiler without generating code
    TokenStream::new()
}
