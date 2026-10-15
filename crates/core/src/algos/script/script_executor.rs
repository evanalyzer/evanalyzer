use crate::algos::script::script_bridge::{
    ScriptHost, ScriptImage, check_referenced_commands, referenced_citations,
};
use crate::algos::script::script_params;
use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use evanalyzer_cfg::core_types::{CitationMetadata, ImageChannelIdx, InternalErrors, ObjectClass};
use log::{debug, info};
use macros::CommandsMeta;
use rhai::{AST, Engine, EvalAltResult, Map, Position, Scope};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

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
/// `print(...)` writes to the log. `run(command, #{ params })` runs a
/// pipeline command on the current image, with the parameter names of the
/// step settings.
///
/// # Examples
///
/// ```rhai
/// print(`Hello world from a ${image_width}x${image_height} tile`);
/// run("gaussian_blur", #{ kernel_size: 5 });
/// ```
// TODO: pick the real category/successors once scripts can sit at any stage.
#[derive(CommandsMeta)]
#[cmdsmeta(
    category = "Preprocessing",
    next = "preprocessing,segment,instance_segmentation,measure,object"
)]
pub struct Script {
    /// Object classes the script creates or uses. Commands the script runs
    /// may only use these classes, so the pipeline knows them without
    /// running the script.
    #[cmdsmeta(optional)]
    pub classes: Vec<ObjectClass>,

    /// Script source code
    #[cmdsmeta(
        script,
        default = String::from("print(`Hello world from a ${image_width}x${image_height} tile`);\n")
    )]
    pub source: String,

    /// The values of the script's `param`s. The script declares them
    /// (`param max_rounds = 10;`); this list follows it when the script is
    /// edited or the project loaded - a `param` without an entry runs with
    /// its default.
    #[cmdsmeta(optional, display_name = "Parameters")]
    pub params: Vec<ScriptParam>,
}

/// A `param` of a script: a value the script gets from its step's settings
/// (or a group parameter) instead of from its code.
#[derive(CommandsMeta)]
pub struct ScriptParam {
    /// The name the script declares it with.
    #[cmdsmeta(read_only)]
    pub name: String,

    /// Its value. The type comes from the script's `param` line.
    #[cmdsmeta(default = ScriptValue::Number(0), read_only)]
    pub value: ScriptValue,
}

/// A script `param`'s value and type.
#[derive(CommandsMeta, Debug, Clone, PartialEq)]
pub enum ScriptValue {
    Number(i64),
    Decimal(f64),
    #[cmdsmeta(display_name = "On/off")]
    Bool(bool),
    Text(String),
    Class(ObjectClass),
    Channel(ImageChannelIdx),
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
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        let ast = compiled(&self.source)?;

        let tile = &ctx.image_meta.image_tile_info;
        let mut scope = Scope::new();
        scope.push_constant("image_width", ctx.image.size().width as i64);
        scope.push_constant("image_height", ctx.image.size().height as i64);
        scope.push_constant("tile_x", tile.offset_x as i64);
        scope.push_constant("tile_y", tile.offset_y as i64);
        scope.push_constant("image_bits", ctx.image_meta.nr_of_bits as i64);

        // A class param's class is added when its `param` line runs.
        let classes = self.classes.clone();
        let params = self
            .params
            .iter()
            .map(|p| (p.name.clone(), p.value.clone()))
            .collect();
        let _lent = LentHost::lend(ctx, cache, &classes, params)?;
        ENGINE
            .with(|engine| engine.run_ast_with_scope(&mut scope, &ast))
            .map_err(|e| script_error(&e))
    }

    fn name(&self) -> &'static str {
        "Script"
    }

    /// The commands the script runs by name (see
    /// [`referenced_citations`](crate::algos::script::script_bridge::referenced_citations)).
    fn cite(&self) -> Vec<&'static CitationMetadata> {
        let mut cites = vec![&CitationMetadata::DANMAYR];
        for cite in referenced_citations(&self.source) {
            if !cites.iter().any(|c| std::ptr::eq(*c, cite)) {
                cites.push(cite);
            }
        }
        cites
    }

    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::Tile
    }

    /// Each command run by the script prepares its own workspace.
    fn scratch_is_workspace(&self) -> bool {
        false
    }
}

/// The `param` lines of `source` (see [`script_params`](crate::algos::script::script_params)).
///
/// # Errors
/// The script doesn't parse, or a `param` line is wrong - with its line.
pub fn declared_script_params(source: &str) -> Result<Vec<script_params::ParamDecl>, String> {
    ENGINE.with(|engine| script_params::declared_params(engine, source))
}

/// Compiled scripts kept per thread; a pipeline rarely has more than a few.
const COMPILED_CACHE_SIZE: usize = 16;

thread_local! {
    /// One engine per worker thread: creating one costs ~0.2 ms (release),
    /// which would otherwise be paid on every tile.
    static ENGINE: Engine = new_engine();
    /// Scripts compiled on this thread, by source text.
    static COMPILED: RefCell<HashMap<String, Rc<AST>>> = RefCell::new(HashMap::new());
    /// The step state the engine's functions work on while a script runs on
    /// this thread (see [`LentHost`]).
    static HOST: RefCell<Option<ScriptHost>> = const { RefCell::new(None) };
}

fn compiled(source: &str) -> Result<Rc<AST>, InternalErrors> {
    if let Some(ast) = COMPILED.with_borrow(|c| c.get(source).cloned()) {
        return Ok(ast);
    }
    // Checks the `param` lines too (names, defaults).
    declared_script_params(source)
        .map_err(|e| InternalErrors::Generic(format!("Script error: {e}")))?;
    let ast = ENGINE
        .with(|engine| engine.compile(source))
        .map_err(|e| script_error(&e.into()))?;
    check_referenced_commands(source)
        .map_err(|e| InternalErrors::Generic(format!("Script error: {e}")))?;
    let ast = Rc::new(ast);
    COMPILED.with_borrow_mut(|c| {
        if c.len() >= COMPILED_CACHE_SIZE {
            c.clear();
        }
        c.insert(source.to_string(), ast.clone());
    });
    Ok(ast)
}

/// Lends the step's context and cache to [`HOST`] for one script run and
/// puts them back when dropped - after errors and panics too.
struct LentHost<'a> {
    ctx: &'a mut PipelineContext,
    cache: &'a mut GlobalPipelineCache,
}

impl<'a> LentHost<'a> {
    fn lend(
        ctx: &'a mut PipelineContext,
        cache: &'a mut GlobalPipelineCache,
        classes: &[ObjectClass],
        params: HashMap<String, ScriptValue>,
    ) -> Result<Self, InternalErrors> {
        HOST.with_borrow_mut(|host| {
            if host.is_some() {
                return Err(InternalErrors::Internal(
                    "a script step is already running on this thread".into(),
                ));
            }
            *host = Some(ScriptHost {
                ctx: ctx.take(),
                cache: std::mem::take(cache),
                classes: classes.to_vec(),
                params,
            });
            Ok(())
        })?;
        Ok(Self { ctx, cache })
    }
}

impl Drop for LentHost<'_> {
    fn drop(&mut self) {
        if let Some(host) = HOST.with_borrow_mut(Option::take) {
            *self.ctx = host.ctx;
            *self.cache = host.cache;
        }
    }
}

type ScriptResult<T> = Result<T, Box<EvalAltResult>>;

/// Calls `f` with the running step's state.
fn with_host<T>(f: impl FnOnce(&mut ScriptHost) -> Result<T, String>) -> ScriptResult<T> {
    HOST.with_borrow_mut(|host| match host.as_mut() {
        Some(host) => f(host).map_err(Into::into),
        None => Err("no pipeline step is running".into()),
    })
}

/// A sandboxed engine: no file or network access (Rhai has none unless
/// registered), bounded run time, recursion and memory. The script API:
///
/// - `run(command [, image] [, #{ params }]) -> Image`
/// - `image()`, `set_image(image)`, `channel(index)`, `instance_count()`
/// - on `Image`: `width`, `height`, `mean()`, `min()`, `max()`
fn new_engine() -> Engine {
    let mut engine = Engine::new();
    script_params::register(
        &mut engine,
        |name| HOST.with_borrow(|host| host.as_ref().and_then(|h| h.params.get(name).cloned())),
        |class| {
            HOST.with_borrow_mut(|host| {
                if let Some(host) = host
                    && !host.classes.contains(&class)
                {
                    host.classes.push(class);
                }
            })
        },
    );
    engine
        .set_max_operations(MAX_OPERATIONS)
        .set_max_call_levels(MAX_CALL_LEVELS)
        .set_max_string_size(MAX_STRING_SIZE)
        .set_max_array_size(MAX_ARRAY_SIZE)
        .set_max_map_size(MAX_MAP_SIZE);
    engine.on_print(|text| info!("[script] {text}"));
    engine.on_debug(|text, _source, pos| debug!("[script] {}{text}", position_prefix(pos)));

    engine
        .register_type_with_name::<ScriptImage>("Image")
        .register_get("width", ScriptImage::width)
        .register_get("height", ScriptImage::height)
        .register_fn("mean", ScriptImage::mean)
        .register_fn("min", ScriptImage::min)
        .register_fn("max", ScriptImage::max);

    fn run(key: &str, params: Map) -> ScriptResult<ScriptImage> {
        with_host(|host| {
            host.run_command(key, &params)?;
            Ok(host.image())
        })
    }
    fn run_on(key: &str, image: ScriptImage, params: Map) -> ScriptResult<ScriptImage> {
        with_host(|host| host.run_on(key, image, &params))
    }
    engine
        .register_fn("run", |key: &str| run(key, Map::new()))
        .register_fn("run", run)
        .register_fn("run", |key: &str, image: ScriptImage| {
            run_on(key, image, Map::new())
        })
        .register_fn("run", run_on)
        .register_fn("image", || with_host(|host| Ok(host.image())))
        .register_fn("set_image", |image: ScriptImage| {
            with_host(|host| host.set_image(image))
        })
        .register_fn("channel", |index: i64| {
            with_host(|host| host.channel(index))
        })
        .register_fn("instance_count", || {
            with_host(|host| Ok(host.instance_count()))
        });
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
        run_with(source, vec![])
    }

    fn param(name: &str, value: ScriptValue) -> ScriptParam {
        ScriptParam {
            name: name.into(),
            value,
        }
    }

    fn run_with(source: &str, params: Vec<ScriptParam>) -> Result<(), InternalErrors> {
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
            classes: vec![],
            source: source.into(),
            params,
        }
        .execute(&mut ctx, &mut cache)
    }

    #[test]
    fn param_lines_are_read_with_their_types_and_defaults() {
        use crate::algos::{ParamDecl, ParamKind};
        let decls = declared_script_params(
            r#"
            // param in_a_comment = 1;
            param rounds = 10;
            param start = 800.0;
            param offset: decimal = -3;
            param verbose = false;
            param label = "Nuclei";
            param nuclei: class = 4;
            param input: channel = 2;
            let plain = 5;
            let text = "param not_one = 1;";
        "#,
        )
        .unwrap();
        let decl = |name: &str, kind, default| ParamDecl {
            name: name.into(),
            kind,
            default,
        };
        assert_eq!(
            decls,
            [
                decl("rounds", ParamKind::Number, ScriptValue::Number(10)),
                decl("start", ParamKind::Decimal, ScriptValue::Decimal(800.0)),
                decl("offset", ParamKind::Decimal, ScriptValue::Decimal(-3.0)),
                decl("verbose", ParamKind::Bool, ScriptValue::Bool(false)),
                decl("label", ParamKind::Text, ScriptValue::Text("Nuclei".into())),
                decl(
                    "nuclei",
                    ParamKind::Class,
                    ScriptValue::Class(ObjectClass::Global(4))
                ),
                decl(
                    "input",
                    ParamKind::Channel,
                    ScriptValue::Channel(ImageChannelIdx(2))
                ),
            ]
        );
    }

    #[test]
    fn wrong_param_lines_are_reported_with_their_line() {
        let error = |source: &str| declared_script_params(source).unwrap_err();
        let e = error("let a = 1;\nparam n = a * 2;");
        assert!(e.contains("plain value") && e.contains("line 2"), "{e}");
        let e = error("param n = 1 + 1;");
        assert!(e.contains("followed by `;`"), "{e}");
        let e = error("param n: size = 1;");
        assert!(e.contains("unknown type `size`"), "{e}");
        let e = error("param n: bool = 1;");
        assert!(e.contains("`true` or `false`"), "{e}");
        let e = error("param n = 1;\nparam n = 2;");
        assert!(e.contains("declared twice"), "{e}");
        let e = error("param image_width = 1;");
        assert!(e.contains("taken"), "{e}");
    }

    #[test]
    fn a_param_has_its_default_or_the_steps_value() {
        let source = r#"
            param rounds = 3;
            let count = 0;
            for i in 0..rounds { count += 1; }
            if count != expected { throw `ran ${count} rounds`; }
        "#;
        run_with(&format!("const expected = 3;{source}"), vec![]).expect("the default");
        run_with(
            &format!("const expected = 5;{source}"),
            vec![param("rounds", ScriptValue::Number(5))],
        )
        .expect("the step's value");
    }

    #[test]
    fn params_are_constants() {
        let err = run("param n = 1;\nn = 2;").unwrap_err().to_string();
        assert!(err.contains("constant"), "{err}");
    }

    #[test]
    fn a_class_param_is_a_number_or_a_groups_class_and_may_be_used_by_commands() {
        let check = r#"
            param cell: class = 4;
            if cell != expected { throw `got ${cell}`; }
        "#;
        run_with(&format!("const expected = 4;{check}"), vec![]).unwrap();
        run_with(
            &format!("const expected = \"L1\";{check}"),
            vec![param("cell", ScriptValue::Class(ObjectClass::Local(1)))],
        )
        .unwrap();
        // Not in `classes`, yet allowed: it's a param.
        run(r#"
            param seg: class = 4;
            run("threshold", #{ "thresholds.0.segmentation_class": seg });
        "#)
        .expect("a class param counts as declared");
    }

    #[test]
    fn a_groups_own_class_from_a_param_reaches_a_command() {
        // Save Image needs an output folder the test doesn't have - what
        // matters is that the class got past the checks.
        let result = run_with(
            r#"
            param cell: class = 1;
            run("save_image", #{ "display_classes": [cell] });
        "#,
            vec![param("cell", ScriptValue::Class(ObjectClass::Local(1)))],
        );
        if let Err(e) = result {
            let e = e.to_string();
            assert!(
                e.contains("No output path"),
                "\"L1\" must be a class a command takes: {e}"
            );
        }
    }

    #[test]
    fn a_value_of_another_type_than_the_script_says_is_refused() {
        let err = run_with(
            "param n = 1;",
            vec![param("n", ScriptValue::Text("x".into()))],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("edit the script"), "{err}");
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

    #[test]
    fn script_runs_commands() {
        run(r#"run("gaussian_blur", #{ kernel_size: 3 }); run("gaussian_blur");"#)
            .expect("commands must run from a script");
    }

    #[test]
    fn conditional_rerun_from_a_kept_image() {
        run(r#"
            let raw = image();
            let blurred = run("gaussian_blur", raw, #{ kernel_size: 3 });
            if blurred.max() < 100.0 {
                blurred = run("gaussian_blur", raw, #{ kernel_size: 5 });
            }
            if blurred.width != image_width { throw "size changed"; }
            set_image(raw);
            run("connected_components");
            print(`objects: ${instance_count()}, mean: ${raw.mean()}`);
        "#)
        .expect("image values must work in scripts");
    }

    #[test]
    fn command_errors_report_the_script_line() {
        let err = run("let a = 1;\nrun(\"gaussian_blur\", #{ kernel_sise: 3 });")
            .unwrap_err()
            .to_string();
        assert!(err.contains("kernel_sise"), "{err}");
        assert!(err.contains("line 2"), "{err}");
    }

    #[test]
    fn bad_command_fails_before_the_script_runs() {
        // The first `run` would change the image; the typo in the second
        // must stop the step before that.
        let image = Image::<f32, 1>::from_size_slice(
            ImageSize {
                width: 3,
                height: 3,
            },
            &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let mut ctx = PipelineContext::new_from_image_test(image).unwrap();
        let before = ctx.get_f32_gray_image().unwrap().data.as_slice().to_vec();
        let err = Script {
            classes: vec![],
            source: r#"run("gaussian_blur"); run("gausian_blur");"#.into(),
            params: vec![],
        }
        .execute(&mut ctx, &mut GlobalPipelineCache::default())
        .unwrap_err()
        .to_string();
        assert!(err.contains("gausian_blur"), "{err}");
        assert_eq!(
            ctx.get_f32_gray_image().unwrap().data.as_slice(),
            &before[..]
        );
    }

    #[test]
    fn cites_itself_and_the_commands_it_runs() {
        let script = Script {
            classes: vec![],
            source: r#"run("edge_detection_sobel");"#.into(),
            params: vec![],
        };
        let keys: Vec<&str> = script.cite().iter().map(|c| c.cite_key).collect();
        assert!(keys.contains(&"danmayr2026"), "{keys:?}");
        assert!(keys.contains(&"sobel1968isotropic"), "{keys:?}");
    }

    #[test]
    fn compiled_scripts_are_reused() {
        let source = "let reused_marker = 1;";
        run(source).unwrap();
        let first = COMPILED.with_borrow(|c| c.get(source).cloned()).unwrap();
        run(source).unwrap();
        let second = COMPILED.with_borrow(|c| c.get(source).cloned()).unwrap();
        assert!(Rc::ptr_eq(&first, &second));
    }

    #[test]
    fn host_is_released_after_each_run() {
        run("let a = 1;").unwrap();
        let _ = run("throw 1;");
        assert!(HOST.with_borrow(Option::is_none));
    }

    #[test]
    fn context_is_restored_after_a_failing_script() {
        let image = Image::<f32, 1>::from_size_slice(
            ImageSize {
                width: 4,
                height: 3,
            },
            &[0.5; 12],
        )
        .unwrap();
        let mut ctx = PipelineContext::new_from_image_test(image).unwrap();
        let mut cache = GlobalPipelineCache::default();
        let result = Script {
            classes: vec![],
            source: r#"run("gaussian_blur"); throw "stop";"#.into(),
            params: vec![],
        }
        .execute(&mut ctx, &mut cache);
        assert!(result.is_err());
        assert_eq!(ctx.image.size().width, 4, "the real context is back");
        assert!(ctx.instance_map.is_some());
    }
}
