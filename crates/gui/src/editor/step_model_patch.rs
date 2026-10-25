//! # step_model_patch
//!
//! The steps panel's models, updated in place.
//!
//! A sync after each change and each preview builds the steps' rows anew.
//! Handing Slint new models makes it recreate every card and field - a
//! field being typed into or scrubbed loses the keyboard focus. Here the
//! new rows are merged into the shown models instead: a row whose content
//! is unchanged stays as it is, a changed one is updated in place (its
//! field keeps the focus), and only rows added, removed or moved replace a
//! model.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-17
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::{
    CardTable, CardTableRow, CommandParameter, GroupItem, LeafParam,
    PipelineCommand as SlintPipelineCommand, SectionView, TunedSetting,
};
use slint::{Model, ModelRc, SharedString, VecModel};

/// `new` merged into the shown `old`: when both have the same rows (by
/// `key`, in order) and `old` is a `VecModel`, each row is merged with
/// `merge` and set only if it changed - and `old` is kept. Otherwise `new`.
fn patch<T, K>(
    old: &ModelRc<T>,
    new: ModelRc<T>,
    key: impl Fn(&T) -> K,
    merge: impl Fn(&T, T) -> T,
) -> ModelRc<T>
where
    T: Clone + PartialEq + 'static,
    K: PartialEq,
{
    let Some(shown) = old.as_any().downcast_ref::<VecModel<T>>() else {
        return new;
    };
    let count = shown.row_count();
    if count != new.row_count() {
        return new;
    }
    let pairs: Vec<(T, T)> = (0..count)
        .filter_map(|i| Some((shown.row_data(i)?, new.row_data(i)?)))
        .collect();
    if pairs.len() != count || pairs.iter().any(|(a, b)| key(a) != key(b)) {
        return new;
    }
    for (i, (before, row)) in pairs.into_iter().enumerate() {
        let merged = merge(&before, row);
        if merged != before {
            shown.set_row_data(i, merged);
        }
    }
    old.clone()
}

/// The same strings: the shown model.
fn strings(old: &ModelRc<SharedString>, new: ModelRc<SharedString>) -> ModelRc<SharedString> {
    patch(old, new, |_| (), |_, s| s)
}

fn leaf(old: &LeafParam, mut new: LeafParam) -> LeafParam {
    new.options = strings(&old.options, new.options);
    new
}

fn group(old: &GroupItem, mut new: GroupItem) -> GroupItem {
    new.fields = patch(&old.fields, new.fields, |l| l.name.clone(), leaf);
    new
}

/// A setting: its options and list entries kept where unchanged.
pub(crate) fn param(old: &CommandParameter, mut new: CommandParameter) -> CommandParameter {
    new.options = strings(&old.options, new.options);
    new.group_items = patch(&old.group_items, new.group_items, |_| (), group);
    new
}

fn params(
    old: &ModelRc<CommandParameter>,
    new: ModelRc<CommandParameter>,
) -> ModelRc<CommandParameter> {
    patch(old, new, |p| p.name.clone(), param)
}

fn section_view(old: &SectionView, mut new: SectionView) -> SectionView {
    new.params = params(&old.params, new.params);
    new
}

fn table_row(old: &CardTableRow, mut new: CardTableRow) -> CardTableRow {
    new.cells = params(&old.cells, new.cells);
    new
}

fn table(old: &CardTable, mut new: CardTable) -> CardTable {
    new.columns = strings(&old.columns, new.columns);
    new.rows = patch(&old.rows, new.rows, |r| r.name.clone(), table_row);
    new.problems = strings(&old.problems, new.problems);
    new
}

fn command(old: &SlintPipelineCommand, mut new: SlintPipelineCommand) -> SlintPipelineCommand {
    new.parameters = params(&old.parameters, new.parameters);
    new.advanced_parameters = params(&old.advanced_parameters, new.advanced_parameters);
    new.sections = strings(&old.sections, new.sections);
    new.section_views = patch(&old.section_views, new.section_views, |_| (), section_view);
    new.card_tables = patch(&old.card_tables, new.card_tables, |_| (), table);
    new.problems = patch(&old.problems, new.problems, |_| (), |_, p| p);
    new
}

/// The steps (`new`) merged into the shown ones - the same steps in the
/// same order are updated in place.
pub(crate) fn commands(
    old: &ModelRc<SlintPipelineCommand>,
    new: Vec<SlintPipelineCommand>,
) -> ModelRc<SlintPipelineCommand> {
    patch(
        old,
        ModelRc::new(VecModel::from(new)),
        |c| c.step_id.clone(),
        command,
    )
}

/// The Tuning panel's settings, the same way.
pub(crate) fn tuned(old: &ModelRc<TunedSetting>, new: Vec<TunedSetting>) -> ModelRc<TunedSetting> {
    patch(
        old,
        ModelRc::new(VecModel::from(new)),
        |t| (t.step_id.clone(), t.param.name.clone()),
        |before, mut t| {
            t.param = param(&before.param, t.param);
            t
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setting(name: &str, value: &str) -> CommandParameter {
        CommandParameter {
            name: name.into(),
            value: value.into(),
            options: ModelRc::new(VecModel::from(vec![SharedString::from("a")])),
            ..Default::default()
        }
    }

    fn step(id: &str, values: &[(&str, &str)]) -> SlintPipelineCommand {
        SlintPipelineCommand {
            step_id: id.into(),
            parameters: ModelRc::new(VecModel::from(
                values
                    .iter()
                    .map(|(n, v)| setting(n, v))
                    .collect::<Vec<_>>(),
            )),
            ..Default::default()
        }
    }

    /// A changed value updates the shown models: the steps' and the
    /// settings' models stay the same ones (Slint keeps their fields - and
    /// the focus), only the changed row is new.
    #[test]
    fn the_same_steps_are_updated_in_place() {
        let shown = commands(
            &ModelRc::default(),
            vec![step("1", &[("min", "1")]), step("2", &[("max", "9")])],
        );
        let settings = shown.row_data(0).unwrap().parameters;

        let again = commands(
            &shown,
            vec![step("1", &[("min", "5")]), step("2", &[("max", "9")])],
        );

        assert_eq!(again, shown, "the same model");
        let first = again.row_data(0).unwrap();
        assert_eq!(first.parameters, settings, "the same settings model");
        assert_eq!(first.parameters.row_data(0).unwrap().value, "5");
        assert_eq!(
            again
                .row_data(1)
                .unwrap()
                .parameters
                .row_data(0)
                .unwrap()
                .value,
            "9"
        );
    }

    /// A step added, removed or moved: new models.
    #[test]
    fn other_steps_replace_the_model() {
        let shown = commands(&ModelRc::default(), vec![step("1", &[]), step("2", &[])]);
        for steps in [
            vec![step("1", &[])],
            vec![step("2", &[]), step("1", &[])],
            vec![step("1", &[]), step("2", &[]), step("3", &[])],
        ] {
            let ids: Vec<_> = steps.iter().map(|s| s.step_id.clone()).collect();
            let again = commands(&shown, steps);
            assert_ne!(again, shown, "{ids:?}");
            assert_eq!(again.row_count(), ids.len());
        }
    }
}
