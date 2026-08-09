use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use evanalyzer_cfg::core_types::InternalErrors;
use evanalyzer_cfg::settings::templates::PipelineTemplate;
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;

const MAX_NEW_TOKENS: usize = 100;

/// Gets a user prompt and the project's schema as input and the predicted pipeline as output.
pub fn generate_pipeline(prompt: &str) -> Result<PipelineTemplate, InternalErrors> {
    let schema = schemars::schema_for!(PipelineTemplate);
    let schema_string = serde_json::to_string_pretty(&schema).unwrap();
    // println!("Generated JSON Schema for Local LLM:\n{}", schema_string);

    let dir = models_dir();
    let model_path = dir.join("qwen2.5-0.5b-instruct-q6_k.gguf");
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
        .and_then(|v| v.to_u32().ok());
    let mut model = ModelWeights::from_gguf(content, &mut file, &device)
        .map_err(|e| InternalErrors::Generic(format!("{e}")))?;

    let prompt_tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| InternalErrors::Generic(format!("failed to tokenize prompt: {e}")))?
        .get_ids()
        .to_vec();
    if prompt_tokens.is_empty() {
        return Err(InternalErrors::Generic(
            "tokenizer produced no tokens for the prompt".into(),
        ));
    }

    let mut logits_processor = LogitsProcessor::new(299792458, Some(0.8), Some(0.95));

    print!("{prompt}");
    std::io::stdout().flush()?;

    let input = Tensor::new(prompt_tokens.as_slice(), &device)
        .and_then(|t| t.unsqueeze(0))
        .map_err(|e| InternalErrors::Generic(format!("{e}")))?;
    let logits = model
        .forward(&input, 0)
        .and_then(|t| t.squeeze(0))
        .map_err(|e| InternalErrors::Generic(format!("{e}")))?;
    let mut next_token = logits_processor
        .sample(&logits)
        .map_err(|e| InternalErrors::Generic(format!("{e}")))?;
    let mut index_pos = prompt_tokens.len();

    for _ in 0..MAX_NEW_TOKENS {
        if Some(next_token) == eos_token_id {
            break;
        }
        let piece = tokenizer
            .decode(&[next_token], true)
            .map_err(|e| InternalErrors::Generic(format!("failed to decode token: {e}")))?;
        print!("{piece}");
        std::io::stdout().flush()?;

        let input = Tensor::new(&[next_token], &device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| InternalErrors::Generic(format!("{e}")))?;
        let logits = model
            .forward(&input, index_pos)
            .and_then(|t| t.squeeze(0))
            .map_err(|e| InternalErrors::Generic(format!("{e}")))?;
        next_token = logits_processor
            .sample(&logits)
            .map_err(|e| InternalErrors::Generic(format!("{e}")))?;
        index_pos += 1;
    }
    println!();

    Err(InternalErrors::Generic(
        "Not implemented yet: raw model output is not parsed into a PipelineTemplate".into(),
    ))
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
        if let Err(err) = generate_pipeline("Hello how are you!") {
            print!("{:?}", err);
        }
    }
}

// https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/main
