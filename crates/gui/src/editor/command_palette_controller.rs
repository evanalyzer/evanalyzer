//! The command palette (Ctrl+K): finds what was typed among the app's
//! actions, the steps of the shown pipeline, the commands, the images and
//! the classes, and runs the chosen entry - through the same callbacks the
//! buttons and menus use.

use crate::AppWindow;
use crate::{
    AiLearningState, Appearance, ClassificationState, CommandPaletteState, CommandPickerState,
    DialogType, GlobalAppState, ImagesListState, InkScheme, LayoutState, PaletteEntry, PaletteKind,
    PipelinesPanelState, ToolbarState, WorkspaceMode,
};
use slint::{ComponentHandle, Model, ModelRc, VecModel};

/// At most this many entries are shown.
const MAX_ENTRIES: usize = 40;

/// The app's actions: id, title, shortcut.
const ACTIONS: &[(&str, &str, &str)] = &[
    ("save", "Save project", "Ctrl S"),
    ("open", "Open project…", "Ctrl O"),
    ("new", "New project", ""),
    ("settings", "Project settings…", ""),
    ("run", "Run analysis…", ""),
    ("preview", "Refresh preview", "F5"),
    ("preview-whole", "Preview the whole image", "Shift F5"),
    ("add-step", "Add step…", ""),
    ("new-pipeline", "New pipeline", ""),
    ("mode-images", "Show images", "Ctrl 1"),
    ("mode-pipelines", "Show pipelines", "Ctrl 2"),
    ("mode-results", "Show results", "Ctrl 3"),
    ("annotate", "Annotate mode on / off", ""),
    ("debug", "Debug mode on / off", ""),
    ("ai", "AI training…", ""),
    ("scheme-light", "Colors: light", ""),
    ("scheme-dark", "Colors: dark", ""),
    ("scheme-dim", "Colors: dim", ""),
    ("shortcuts", "Keyboard shortcuts", "F1"),
];

pub struct CommandPaletteController {
    ui: slint::Weak<AppWindow>,
}

impl CommandPaletteController {
    pub fn new(ui: slint::Weak<AppWindow>) -> Self {
        Self { ui }
    }

    pub fn attach_callbacks(&self) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let weak = self.ui.clone();
        ui.global::<CommandPaletteState>()
            .on_query_changed(move |query| {
                if let Some(ui) = weak.upgrade() {
                    let entries = entries_for(&ui, &query);
                    let state = ui.global::<CommandPaletteState>();
                    state.set_entries(ModelRc::new(VecModel::from(entries)));
                    state.set_selected(0);
                }
            });
        let weak = self.ui.clone();
        ui.global::<CommandPaletteState>().on_run(move |id| {
            if let Some(ui) = weak.upgrade() {
                ui.global::<CommandPaletteState>().set_open(false);
                run(&ui, &id);
            }
        });
    }
}

/// Every word of `query` is somewhere in `text` (case ignored).
fn matches(text: &str, query: &str) -> bool {
    let text = text.to_lowercase();
    query
        .to_lowercase()
        .split_whitespace()
        .all(|word| text.contains(word))
}

/// The entries for `query`: matching actions, steps of the shown pipeline,
/// and - once something is typed - commands, classes and images.
fn entries_for(ui: &AppWindow, query: &str) -> Vec<PaletteEntry> {
    let mut entries: Vec<PaletteEntry> = ACTIONS
        .iter()
        .filter(|(_, title, _)| matches(title, query))
        .map(|(id, title, shortcut)| PaletteEntry {
            id: format!("action:{id}").into(),
            title: (*title).into(),
            detail: "".into(),
            shortcut: (*shortcut).into(),
            kind: PaletteKind::Action,
        })
        .collect();

    let steps = ui.global::<PipelinesPanelState>().get_active_commands();
    entries.extend(
        steps
            .iter()
            .filter(|step| {
                matches(
                    &format!("{} {} {}", step.step_id, step.name, step.summary),
                    query,
                )
            })
            .map(|step| PaletteEntry {
                id: format!("step:{}", step.id).into(),
                title: format!("#{} {}", step.step_id, step.name).into(),
                detail: step.summary.clone(),
                shortcut: "".into(),
                kind: PaletteKind::Step,
            }),
    );

    // The settings of the shown pipeline's steps - once something is typed.
    if !query.trim().is_empty() {
        for (idx, step) in steps.iter().enumerate() {
            let settings = step
                .parameters
                .iter()
                .chain(step.advanced_parameters.iter())
                .filter(|p| !matches!(p.param_type, crate::ParamType::Group))
                .filter(|p| {
                    matches(
                        &format!("{} {} {}", p.display_name, p.name, step.name),
                        query,
                    )
                })
                .map(|p| PaletteEntry {
                    id: format!("setting:{idx}").into(),
                    title: p.display_name.clone(),
                    detail: format!("#{} {} · {}", step.step_id, step.name, p.value).into(),
                    shortcut: "".into(),
                    kind: PaletteKind::Setting,
                })
                .collect::<Vec<_>>();
            entries.extend(settings);
        }
    }

    if !query.trim().is_empty() {
        let commands = ui.global::<CommandPickerState>().get_all_commands();
        entries.extend(
            commands
                .iter()
                .filter(|command| {
                    matches(
                        &format!("{} {} {}", command.name, command.summary, command.keywords),
                        query,
                    )
                })
                .map(|command| PaletteEntry {
                    id: format!("command:{}", command.id).into(),
                    title: format!("Add “{}”", command.name).into(),
                    detail: command.summary.clone(),
                    shortcut: "".into(),
                    kind: PaletteKind::Command,
                }),
        );
    }
    if !query.trim().is_empty() {
        let classes = ui.global::<ClassificationState>().get_classes_list();
        entries.extend(
            classes
                .iter()
                .filter(|class| matches(&format!("class {} {}", class.name, class.id), query))
                .map(|class| PaletteEntry {
                    id: format!("class:{}", class.id).into(),
                    title: format!("Class “{}”", class.name).into(),
                    detail: format!("{} objects", class.count).into(),
                    shortcut: "".into(),
                    kind: PaletteKind::Class,
                }),
        );
        // A project can have thousands: stop at what's shown.
        let images = ui.global::<ImagesListState>().get_images_list();
        entries.extend(
            images
                .iter()
                .filter(|image| matches(&image.path, query))
                .take(MAX_ENTRIES)
                .map(|image| PaletteEntry {
                    id: format!("image:{}", image.path).into(),
                    title: image.name.clone(),
                    detail: image.path.clone(),
                    shortcut: "".into(),
                    kind: PaletteKind::Image,
                }),
        );
    }
    entries.truncate(MAX_ENTRIES);
    entries
}

/// Runs the entry `id` (see [`PaletteEntry::id`]).
fn run(ui: &AppWindow, id: &str) {
    let Some((kind, what)) = id.split_once(':') else {
        return;
    };
    let panel = ui.global::<PipelinesPanelState>();
    let layout = ui.global::<LayoutState>();
    match kind {
        "action" => match what {
            "save" => ui.global::<ToolbarState>().invoke_save_file_clicked(),
            "open" => ui.global::<ToolbarState>().invoke_open_file_clicked(),
            "new" => ui.global::<ToolbarState>().invoke_new_file_clicked(),
            "settings" => ui
                .global::<GlobalAppState>()
                .set_active_dialog(DialogType::ProjectSettings),
            "run" => panel.invoke_run_all(),
            "preview" => panel.invoke_dry_run(),
            "preview-whole" => panel.invoke_preview_whole_image(),
            "add-step" => {
                layout.set_mode(WorkspaceMode::Pipelines);
                panel.invoke_insert_step(
                    panel.get_active_pipeline_id(),
                    panel.get_active_commands().row_count() as i32,
                );
            }
            "new-pipeline" => panel.invoke_new_pipeline(),
            "mode-images" => layout.set_mode(WorkspaceMode::Images),
            "mode-pipelines" => layout.set_mode(WorkspaceMode::Pipelines),
            "mode-results" => layout.set_mode(WorkspaceMode::Results),
            "annotate" => layout.set_annotate(!layout.get_annotate()),
            "debug" => panel.invoke_debug_mode_toggled(!panel.get_debug_mode()),
            "ai" => ui.global::<AiLearningState>().invoke_open_requested(),
            "scheme-light" => ui.global::<Appearance>().invoke_pick(InkScheme::Light),
            "scheme-dark" => ui.global::<Appearance>().invoke_pick(InkScheme::Dark),
            "scheme-dim" => ui.global::<Appearance>().invoke_pick(InkScheme::Dim),
            "shortcuts" => ui
                .global::<GlobalAppState>()
                .set_active_dialog(DialogType::ShortcutsHelp),
            other => log::warn!("Unknown palette action {other}"),
        },
        // Opens the step on its card.
        // A step, or a setting of it: the step opened on its card (and
        // selected).
        "step" | "setting" => {
            let Ok(idx) = what.parse::<usize>() else {
                return;
            };
            layout.set_mode(WorkspaceMode::Pipelines);
            panel.set_selected_step_id(idx as i32);
            let steps = panel.get_active_commands();
            if let Some(mut step) = steps.row_data(idx)
                && !step.expanded
            {
                step.expanded = true;
                steps.set_row_data(idx, step);
                panel.invoke_expand_step(idx as i32);
            }
        }
        // Added at the end of the shown pipeline, as from "Add step".
        "command" => {
            let Ok(command) = what.parse::<i32>() else {
                return;
            };
            layout.set_mode(WorkspaceMode::Pipelines);
            panel.invoke_insert_step(
                panel.get_active_pipeline_id(),
                panel.get_active_commands().row_count() as i32,
            );
            ui.global::<CommandPickerState>().invoke_confirm(command);
            // Added at once: the picker, which stays open for the next step
            // otherwise, isn't wanted here.
            ui.global::<GlobalAppState>()
                .set_active_dialog(DialogType::None);
        }
        // Opened, as from the images list.
        "image" => ui
            .global::<ImagesListState>()
            .invoke_image_selected(what.into()),
        // Selected in the classes panel, which opens.
        "class" => {
            let Ok(class) = what.parse::<i32>() else {
                return;
            };
            layout.set_classes_open(true);
            let classes = ui.global::<ClassificationState>();
            classes.set_selected_class_id(class);
            classes.invoke_class_selected(class);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::test_support::test_ui_window;
    use crate::{PipelineCommand, StepCategory};

    fn palette(ui: &AppWindow) -> CommandPaletteController {
        let controller = CommandPaletteController::new(ui.as_weak());
        controller.attach_callbacks();
        controller
    }

    fn titles(ui: &AppWindow) -> Vec<String> {
        ui.global::<CommandPaletteState>()
            .get_entries()
            .iter()
            .map(|entry| entry.title.to_string())
            .collect()
    }

    fn with_steps(ui: &AppWindow) {
        let step = |id: i32, name: &str, summary: &str| PipelineCommand {
            id,
            step_id: (id + 1).to_string().into(),
            name: name.into(),
            summary: summary.into(),
            category: StepCategory::Preprocess,
            ..Default::default()
        };
        ui.global::<PipelinesPanelState>()
            .set_active_commands(ModelRc::new(VecModel::from(vec![
                step(0, "Load channel", "CY5"),
                step(1, "Gaussian blur", "σ = 2"),
            ])));
    }

    #[test]
    fn every_typed_word_must_be_found_ignoring_case() {
        assert!(matches("Save project", "save"));
        assert!(matches("Save project", "PRO sa"));
        assert!(!matches("Save project", "save image"));
        assert!(matches("anything", ""));
    }

    #[test]
    fn opening_shows_the_actions_and_the_steps_but_no_commands() {
        let ui = test_ui_window();
        let _palette = palette(&ui);
        with_steps(&ui);
        ui.global::<CommandPickerState>()
            .set_all_commands(ModelRc::new(VecModel::from(vec![crate::CommandDef {
                id: 7,
                name: "Threshold".into(),
                ..Default::default()
            }])));

        ui.global::<CommandPaletteState>()
            .invoke_query_changed("".into());

        let shown = titles(&ui);
        assert!(shown.contains(&"Save project".to_string()));
        assert!(shown.contains(&"#2 Gaussian blur".to_string()));
        assert!(!shown.iter().any(|t| t.contains("Threshold")));
    }

    #[test]
    fn typing_finds_steps_by_summary_and_commands_by_name() {
        let ui = test_ui_window();
        let _palette = palette(&ui);
        with_steps(&ui);
        ui.global::<CommandPickerState>()
            .set_all_commands(ModelRc::new(VecModel::from(vec![crate::CommandDef {
                id: 7,
                name: "Threshold".into(),
                summary: "Separates objects from the background".into(),
                ..Default::default()
            }])));
        let state = ui.global::<CommandPaletteState>();

        state.invoke_query_changed("cy5".into());
        assert_eq!(titles(&ui), ["#1 Load channel"]);

        state.invoke_query_changed("thresh".into());
        assert_eq!(titles(&ui), ["Add “Threshold”"]);
        assert_eq!(state.get_entries().row_data(0).unwrap().id, "command:7");
        assert_eq!(state.get_selected(), 0);
    }

    #[test]
    fn typing_finds_images_by_path_and_classes_by_name() {
        let ui = test_ui_window();
        let _palette = palette(&ui);
        ui.global::<ImagesListState>()
            .set_images_list(ModelRc::new(VecModel::from(vec![
                crate::ImageItemData {
                    name: "B03_01.tif".into(),
                    path: "plate1/B03_01.tif".into(),
                    ..Default::default()
                },
                crate::ImageItemData {
                    name: "C04_01.tif".into(),
                    path: "plate1/C04_01.tif".into(),
                    ..Default::default()
                },
            ])));
        ui.global::<ClassificationState>()
            .set_classes_list(ModelRc::new(VecModel::from(vec![crate::ClassItemData {
                id: 3,
                name: "Nuclei".into(),
                count: 12,
                ..Default::default()
            }])));
        let state = ui.global::<CommandPaletteState>();

        state.invoke_query_changed("b03".into());
        assert_eq!(titles(&ui), ["B03_01.tif"]);
        assert_eq!(
            state.get_entries().row_data(0).unwrap().id,
            "image:plate1/B03_01.tif"
        );

        state.invoke_query_changed("nucl".into());
        assert_eq!(titles(&ui), ["Class “Nuclei”"]);
        assert_eq!(
            state.get_entries().row_data(0).unwrap().detail,
            "12 objects"
        );

        // Not on opening: a project can have thousands of images.
        state.invoke_query_changed("".into());
        assert!(!titles(&ui).iter().any(|t| t.ends_with(".tif")));
    }

    #[test]
    fn running_a_class_entry_selects_it_in_the_opened_classes_panel() {
        let ui = test_ui_window();
        let _palette = palette(&ui);
        let layout = ui.global::<LayoutState>();
        layout.set_classes_open(false);
        let picked = std::rc::Rc::new(std::cell::Cell::new(-1));
        let seen = picked.clone();
        ui.global::<ClassificationState>()
            .on_class_selected(move |id| seen.set(id));

        ui.global::<CommandPaletteState>()
            .invoke_run("class:3".into());

        assert!(layout.get_classes_open());
        assert_eq!(
            ui.global::<ClassificationState>().get_selected_class_id(),
            3
        );
        assert_eq!(picked.get(), 3);
    }

    #[test]
    fn typing_finds_a_setting_of_a_step_and_running_it_opens_the_step() {
        let ui = test_ui_window();
        let _palette = palette(&ui);
        let steps = vec![PipelineCommand {
            id: 0,
            step_id: "3".into(),
            name: "Threshold".into(),
            category: StepCategory::Segment,
            parameters: ModelRc::new(VecModel::from(vec![crate::CommandParameter {
                name: "min".into(),
                display_name: "Min threshold".into(),
                value: "240".into(),
                ..Default::default()
            }])),
            ..Default::default()
        }];
        ui.global::<PipelinesPanelState>()
            .set_active_commands(ModelRc::new(VecModel::from(steps)));
        let state = ui.global::<CommandPaletteState>();

        state.invoke_query_changed("min thr".into());
        let entry = state.get_entries().row_data(0).unwrap();
        assert_eq!(entry.title, "Min threshold");
        assert_eq!(entry.detail, "#3 Threshold · 240");
        assert_eq!(entry.id, "setting:0");

        state.invoke_run(entry.id);
        let panel = ui.global::<PipelinesPanelState>();
        assert_eq!(panel.get_selected_step_id(), 0);
        assert!(panel.get_active_commands().row_data(0).unwrap().expanded);
    }

    #[test]
    fn running_an_entry_closes_the_palette_and_does_it() {
        let ui = test_ui_window();
        let _palette = palette(&ui);
        with_steps(&ui);
        let state = ui.global::<CommandPaletteState>();
        let layout = ui.global::<LayoutState>();

        state.set_open(true);
        state.invoke_run("action:mode-results".into());
        assert!(!state.get_open());
        assert_eq!(layout.get_mode(), WorkspaceMode::Results);

        state.invoke_run("action:annotate".into());
        assert!(layout.get_annotate());

        // A step: back to the pipeline, the step opened.
        state.invoke_run("step:1".into());
        assert_eq!(layout.get_mode(), WorkspaceMode::Pipelines);
        let steps = ui.global::<PipelinesPanelState>().get_active_commands();
        assert!(steps.row_data(1).unwrap().expanded);
        assert!(!steps.row_data(0).unwrap().expanded);

        state.invoke_run("action:scheme-dim".into());
        assert_eq!(ui.global::<crate::Ink>().get_scheme(), InkScheme::Dim);
    }
}
