use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};
use log::{debug, info};
use macros::CommandsMeta;
use rhai::{Engine, EvalAltResult, Position, Scope};

/// Upper bound on the operations one script run may execute, so an endless
/// `loop {}` fails the step instead of hanging the analysis (or a server).
const MAX_OPERATIONS: u64 = 1_000_000;
const MAX_CALL_LEVELS: usize = 32;
const MAX_STRING_SIZE: usize = 1 << 20;
const MAX_ARRAY_SIZE: usize = 100_000;
const MAX_MAP_SIZE: usize = 10_000;

/// Runs a user-written [Rhai](https://rhai.rs) script as a pipeline step.
///
/// The script sees the current tile as read-only constants:
/// `image_width`, `image_height`, `tile_x`, `tile_y` and `image_bits`.
/// `print(...)` writes to the log.
///
/// # Examples
///
/// ```rhai
/// print(`Hello world from a ${image_width}x${image_height} tile`);
/// ```
// TODO: pick the real category/successors once scripts can sit at any stage.
#[derive(CommandsMeta)]
#[cmdsmeta(
    category = "Preprocessing",
    next = "preprocessing,segment,instance_segmentation,measure,object"
)]
pub struct Script {
    /// Script source code
    #[cmdsmeta(
        script,
        default = String::from("print(`Hello world from a ${image_width}x${image_height} tile`);\n")
    )]
    pub source: String,
}

impl ImageAlgorithm for Script {
    /// Runs the script once for the current tile.
    ///
    /// # Errors
    /// Returns [`InternalErrors::Generic`] with line and column when the
    /// script does not parse, fails at runtime or exceeds the engine limits.
    fn execute(
        &self,
        ctx: &mut PipelineContext,
        _cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let engine = new_engine();
        let tile = &ctx.image_meta.image_tile_info;
        let mut scope = Scope::new();
        scope.push_constant("image_width", ctx.image.size().width as i64);
        scope.push_constant("image_height", ctx.image.size().height as i64);
        scope.push_constant("tile_x", tile.offset_x as i64);
        scope.push_constant("tile_y", tile.offset_y as i64);
        scope.push_constant("image_bits", ctx.image_meta.nr_of_bits as i64);

        engine
            .run_with_scope(&mut scope, &self.source)
            .map_err(|e| script_error(&e))
    }

    fn name(&self) -> &'static str {
        "Script"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![]
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }

    /// The script does not touch the image yet, so there is no workspace to
    /// prepare.
    fn scratch_is_workspace(&self) -> bool {
        false
    }
}

/// A sandboxed engine: no file or network access (Rhai has none unless
/// registered), bounded run time, recursion and memory.
fn new_engine() -> Engine {
    let mut engine = Engine::new();
    engine
        .set_max_operations(MAX_OPERATIONS)
        .set_max_call_levels(MAX_CALL_LEVELS)
        .set_max_string_size(MAX_STRING_SIZE)
        .set_max_array_size(MAX_ARRAY_SIZE)
        .set_max_map_size(MAX_MAP_SIZE);
    engine.on_print(|text| info!("[script] {text}"));
    engine.on_debug(|text, _source, pos| debug!("[script] {}{text}", position_prefix(pos)));
    engine
}

fn script_error(err: &EvalAltResult) -> InternalErrors {
    // Rhai's own message already ends with "(line x, position y)".
    InternalErrors::Generic(format!("Script error: {err}"))
}

fn position_prefix(pos: Position) -> String {
    match (pos.line(), pos.position()) {
        (Some(line), Some(col)) => format!("line {line}, column {col}: "),
        (Some(line), None) => format!("line {line}: "),
        _ => String::new(),
    }
}

// --- Test ------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};

    fn run(source: &str) -> Result<(), InternalErrors> {
        let image = Image::<f32, 1>::from_size_slice(
            ImageSize {
                width: 4,
                height: 3,
            },
            &[0.0; 12],
        )
        .unwrap();
        let mut ctx = PipelineContext::new_from_image_test(image).unwrap();
        let mut cache = GlobalPipelineCache::default();
        Script {
            source: source.into(),
        }
        .execute(&mut ctx, &mut cache)
    }

    #[test]
    fn hello_world_runs() {
        run(r#"print("Hello world");"#).expect("hello world must run");
    }

    #[test]
    fn script_sees_the_tile_size() {
        run(r#"if image_width != 4 || image_height != 3 { throw "wrong size"; }"#)
            .expect("the script must see the 4x3 test tile");
    }

    #[test]
    fn syntax_error_reports_the_line() {
        let err = run("let a = 1;\nlet = ;").unwrap_err().to_string();
        assert!(err.contains("line 2"), "{err}");
    }

    #[test]
    fn runtime_error_fails_the_step() {
        let err = run(r#"throw "boom";"#).unwrap_err().to_string();
        assert!(err.contains("boom"), "{err}");
    }

    #[test]
    fn endless_loop_is_stopped_by_the_operation_limit() {
        assert!(run("loop {}").is_err());
    }

    #[test]
    fn tile_info_is_read_only() {
        assert!(run("image_width = 1;").is_err());
    }
}
