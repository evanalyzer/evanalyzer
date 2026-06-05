use proc_macro::TokenStream;
use syn::{parse_macro_input, Data, DeriveInput};

#[proc_macro_derive(CommandsMeta, attributes(cmdsmeta))]
pub fn commands_meta_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // Keys allowed on struct-level #[cmdsmeta(...)]
    let struct_allowed_keys = ["category"];

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
