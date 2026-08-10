use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::templates::PipelineTemplate;
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, SimpleVob, TokEnv, TokRxInfo, TokTrie};
use llguidance::{Constraint, ParserFactory};
use log::{info, warn};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

const MAX_NEW_TOKENS: usize = 1000;

/// Caps how many existing templates get folded into the prompt when
/// [`GenerateOptions::include_existing_templates`] is set, so a large template
/// library doesn't blow the model's (small) context budget.
const MAX_EXAMPLE_TEMPLATES: usize = 100;

/// Optional, off-by-default behaviour for [`generate_pipeline`].
#[derive(Debug, Clone, Copy, Default)]
pub struct GenerateOptions {
    /// Prepend a few of the project's existing saved `PipelineTemplate`s to
    /// the prompt as few-shot examples, so the model has real content to
    /// imitate instead of confabulating field values.
    pub include_existing_templates: bool,
}

/// Gets a user prompt and the project's schema as input and the predicted pipeline as output.
///
/// Generation is grammar-constrained: at every step the sampler is restricted to
/// tokens that keep the output on a valid path through `PipelineTemplate`'s JSON
/// schema (see [`mask_logits`]), so the model can only ever produce
/// schema-conformant JSON. That's what makes the final `serde_json::from_str`
/// below safe to trust rather than just a hopeful parse of free-form output.
pub fn generate_pipeline(
    prompt: &str,
    options: GenerateOptions,
) -> Result<PipelineTemplate, InternalErrors> {
    let schema = schemars::schema_for!(PipelineTemplate);
    let schema_value = serde_json::to_value(&schema)
        .map_err(|e| InternalErrors::Generic(format!("failed to serialize schema: {e}")))?;
    let prompt = build_chat_prompt(prompt, &options, &schema_value);

    let dir = models_dir();
    let model_path = dir.join("qwen2.5-coder-1.5b-instruct-q5_k_m.gguf");
    let tokenizer_path = dir.join("tokenizer.json");
    if !model_path.exists() || !tokenizer_path.exists() {
        return Err(InternalErrors::Generic(format!(
            "expected {} and {} - place a Qwen2-architecture GGUF model (e.g. \
             Qwen2.5-0.5B-Instruct) quantized with a supported format (Q4_K/Q5_K/Q6_K/Q8_0, \
             not IQ*) and its matching tokenizer.json there",
            model_path.display(),
            tokenizer_path.display()
        )));
    }

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| InternalErrors::Generic(format!("failed to load tokenizer: {e}")))?;

    let device = Device::Cpu;
    let mut file = std::fs::File::open(&model_path)?;
    let content =
        gguf_file::Content::read(&mut file).map_err(|e| InternalErrors::Generic(format!("{e}")))?;
    let eos_token_id = content
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0);

    // llguidance masks tokens using its own byte-level view of the vocabulary,
    // built straight from tokenizer.json - separate from the `Tokenizer` above,
    // which only handles text <-> token-id encoding/decoding.
    let tokenizer_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&tokenizer_path)?)
            .map_err(|e| InternalErrors::Generic(format!("failed to parse tokenizer.json: {e}")))?;
    let vocab_bytes = llguidance::token_bytes_from_tokenizer_json(&tokenizer_json)
        .map_err(|e| InternalErrors::Generic(format!("failed to read tokenizer vocab: {e}")))?;
    let tok_info = TokRxInfo::new(vocab_bytes.len() as u32, eos_token_id);
    let tok_trie = TokTrie::from(&tok_info, &vocab_bytes);
    let tok_env: TokEnv = Arc::new(ApproximateTokEnv::new(tok_trie));

    let parser_factory = ParserFactory::new_simple(&tok_env)
        .map_err(|e| InternalErrors::Generic(format!("failed to init grammar engine: {e}")))?;
    let grammar = TopLevelGrammar::from_json_schema(schema_value);
    let parser = parser_factory.create_parser(grammar).map_err(|e| {
        InternalErrors::Generic(format!("failed to compile schema to grammar: {e}"))
    })?;
    let mut constraint = Constraint::new(parser);

    let mut model = ModelWeights::from_gguf(content, &mut file, &device)
        .map_err(|e| InternalErrors::Generic(format!("{e}")))?;

    let prompt_tokens = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| InternalErrors::Generic(format!("failed to tokenize prompt: {e}")))?
        .get_ids()
        .to_vec();
    if prompt_tokens.is_empty() {
        return Err(InternalErrors::Generic(
            "tokenizer produced no tokens for the prompt".into(),
        ));
    }
    let prompt_tokens = constraint.process_prompt(prompt_tokens);

    let mut logits_processor = LogitsProcessor::new(299792458, Some(0.8), Some(0.95));

    print!("{prompt}");
    std::io::stdout().flush()?;

    let input = Tensor::new(prompt_tokens.as_slice(), &device)
        .and_then(|t| t.unsqueeze(0))
        .map_err(|e| InternalErrors::Generic(format!("{e}")))?;
    let mut logits = model
        .forward(&input, 0)
        .and_then(|t| t.squeeze(0))
        .map_err(|e| InternalErrors::Generic(format!("{e}")))?;
    let mut index_pos = prompt_tokens.len();

    let mut generated_json = String::new();
    for _ in 0..MAX_NEW_TOKENS {
        let mask = {
            let step = constraint
                .compute_mask()
                .map_err(|e| InternalErrors::Generic(format!("grammar error: {e}")))?;
            step.sample_mask.clone()
        };
        let Some(mask) = mask else {
            // No sample_mask means the grammar has nothing left to accept -
            // only reachable here if fast-forward tokens were enabled, which
            // they aren't for this (non-canonical) tokenizer environment.
            break;
        };
        let masked_logits =
            mask_logits(&logits, &mask).map_err(|e| InternalErrors::Generic(format!("{e}")))?;
        let next_token = logits_processor
            .sample(&masked_logits)
            .map_err(|e| InternalErrors::Generic(format!("{e}")))?;

        let commit = constraint
            .commit_token(Some(next_token))
            .map_err(|e| InternalErrors::Generic(format!("grammar error: {e}")))?;

        let piece = tokenizer
            .decode(&[next_token], true)
            .map_err(|e| InternalErrors::Generic(format!("failed to decode token: {e}")))?;
        print!("{piece}");
        std::io::stdout().flush()?;
        generated_json.push_str(&piece);

        if commit.stop {
            break;
        }

        let input = Tensor::new(&[next_token], &device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| InternalErrors::Generic(format!("{e}")))?;
        logits = model
            .forward(&input, index_pos)
            .and_then(|t| t.squeeze(0))
            .map_err(|e| InternalErrors::Generic(format!("{e}")))?;
        index_pos += 1;
    }
    println!();

    serde_json::from_str::<PipelineTemplate>(&generated_json)
        .map_err(|e| InternalErrors::Generic(format!("model output did not match schema: {e}")))
}

/// Sets every grammar-disallowed token's logit to `-inf` so sampling can only
/// ever pick a token that keeps the output on a valid path through the schema.
fn mask_logits(logits: &Tensor, mask: &SimpleVob) -> candle_core::Result<Tensor> {
    let vocab_size = logits.dims1()?;
    let mut bias = vec![f32::NEG_INFINITY; vocab_size];
    for i in 0..vocab_size.min(mask.len()) {
        if mask.is_allowed(i as u32) {
            bias[i] = 0.0;
        }
    }
    let bias = Tensor::new(bias.as_slice(), logits.device())?;
    logits.to_dtype(DType::F32)? + bias
}

/// Wraps the prompt in Qwen2.5-Instruct's ChatML format (`<|im_start|>role ...
/// <|im_end|>`) so the model actually reads the request as an instruction to
/// act on, instead of raw text to continue - confirmed against the real
/// `tokenizer.json` on disk, where `<|im_start|>`/`<|im_end|>` are registered
/// special tokens (ids 151644/151645), not just plain text.
fn build_chat_prompt(
    user_prompt: &str,
    options: &GenerateOptions,
    schema_value: &serde_json::Value,
) -> String {
    format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        system_message(schema_value),
        user_message(user_prompt, options)
    )
}

/// Task framing plus the list of command types the model is allowed to pick
/// from - grammar-constrained decoding already guarantees the `"type"` field
/// will be *one of* these, but without being told what each one means and
/// which literal string to use, the model has no way to pick the *right*
/// one for the request.
fn system_message(schema_value: &serde_json::Value) -> String {
    format!(
        "You design image-analysis pipelines for the evanalyzer app. Given a user's request, \
         respond with ONLY a single JSON object matching the required pipeline schema - no \
         explanation, no markdown code fences, just the JSON.\n\n{}\n{}",
        command_reference(schema_value),
        category_order_reference()
    )
}

/// Renders `CommandCategory::allowed_after`'s actual ordering rule as prompt
/// text (generated from that method itself, not hand-written prose, so it
/// can't silently drift out of sync with it) - schemars/JSON Schema has no
/// way to express this as a schema constraint, since it's a cross-element
/// sequencing rule over `pipelineSteps`, not a shape of any single object.
fn category_order_reference() -> String {
    use evanalyzer_cfg::settings::pipeline_command::CommandCategory;
    let categories = [
        CommandCategory::Preprocess,
        CommandCategory::Segment,
        CommandCategory::Object,
        CommandCategory::Measure,
        CommandCategory::Classify,
    ];
    let mut out = String::from(
        "Pipeline step ordering: each command belongs to one of these categories, and a \
         step's category may only come after one of the categories listed for it (or start \
         the pipeline, if none are listed):\n",
    );
    for category in categories {
        let allowed = category.allowed_after();
        let rule = if allowed.is_empty() {
            "can start the pipeline".to_string()
        } else {
            let names: Vec<String> = allowed.iter().map(|c| format!("{c:?}")).collect();
            format!("must come after {}", names.join(" or "))
        };
        out.push_str(&format!("- {category:?}: {rule}\n"));
    }
    out
}

/// Human-readable `"type" value (display name): summary` list, built by
/// pairing each `PipelineCommand` variant's generated metadata
/// (`evanalyzer_cfg::settings::pipeline_command::all_command_meta`) with the
/// literal tag string schemars emitted for it in the schema's `PipelineCommand`
/// definition (`$defs.PipelineCommand.oneOf[].properties.type.const`) - both
/// lists are generated together from the same source in the same order, so
/// pairing them positionally is reliable.
fn command_reference(schema_value: &serde_json::Value) -> String {
    let tags = command_type_tags(schema_value);
    let metas = evanalyzer_cfg::settings::pipeline_command::all_command_meta();
    let mut out = String::from(
        "Available command types - use the exact \"type\" value shown, chosen for what it does:\n",
    );
    for (tag, meta) in tags.iter().zip(metas.iter()) {
        out.push_str(&format!("- \"{tag}\" ({}): {}\n", meta.name, meta.summary));
    }
    out
}

/// Extracts every `PipelineCommand` variant's serialized `"type"` tag literal
/// (e.g. `"gaussianBlur"`) from the generated JSON schema, in declaration order.
fn command_type_tags(schema_value: &serde_json::Value) -> Vec<String> {
    schema_value["$defs"]["PipelineCommand"]["oneOf"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|branch| {
            branch["properties"]["type"]["const"]
                .as_str()
                .map(str::to_string)
        })
        .collect()
}

/// The user's actual request, or - when
/// [`GenerateOptions::include_existing_templates`] is set - that request
/// preceded by a few of the project's existing saved pipelines as style
/// reference, so the model has real content to draw on rather than
/// confabulating field values with no grounding in the project's data.
fn user_message(user_prompt: &str, options: &GenerateOptions) -> String {
    if !options.include_existing_templates {
        return user_prompt.to_string();
    }

    let examples: Vec<String> = evanalyzer_app::templates::load_pipeline_templates()
        .into_iter()
        .take(MAX_EXAMPLE_TEMPLATES)
        .filter_map(|(_, template)| serde_json::to_string(&template).ok())
        .collect();
    if examples.is_empty() {
        warn!("No templates found!");
        return user_prompt.to_string();
    }

    let mut message = String::from(
        "For style reference only - these are unrelated existing pipelines, not the answer, \
         do not copy them:\n",
    );
    let mut nr: i32 = 0;
    for example in &examples {
        message.push_str(example);
        message.push('\n');
        nr = nr + 1;
    }
    info!("Included {} templates!", nr);
    message.push_str("\nNow create a NEW pipeline for this different request: ");
    message.push_str(user_prompt);
    message
}

/// Next to the running executable in a release build (the "ideal world"
/// layout); falls back to the crate's own `models/` directory so the demo
/// also works via `cargo run --example` during development.
fn models_dir() -> PathBuf {
    "/workspaces/evanalyzer/libs/llm".into()

    //    std::env::current_exe()
    //        .ok()
    //        .and_then(|exe| exe.parent().map(|dir| dir.join("models")))
    //        .filter(|dir| dir.join("model.gguf").exists())
    //        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt() {
        let options = GenerateOptions {
            include_existing_templates: true,
        };
        match generate_pipeline(
            "Create a pipeline which extracts spots from an image.",
            options,
        ) {
            Ok(ok) => print!("{:?}", ok),
            Err(err) => print!("{:?}", err),
        }
    }
}

// https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/main
