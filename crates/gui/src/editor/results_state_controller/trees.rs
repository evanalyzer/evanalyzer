// Copyright 2026 Joachim Danmayr.
// Licensed under the PolyForm Noncommercial License 1.0.0.

//! # trees
//!
//! Results › Trees: the traced neurons (Trace Filaments) with their Measure
//! Filament Tree values, and the selected neuron's dendrogram - laid out
//! here, drawn by `results_trees.slint`. Only stored values: the segments'
//! parents and lengths (path distances); the coloring is how it's drawn.
//! A neuron or segment selected selects it in the split view (its image
//! shown); a click on the image selects back.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-09

use super::*;
use crate::{ChartCard, ChartTimeLine, TreeRow};
use evanalyzer_app::results::{TREE_NEURON_VALUES, TreeNeuron, TreeSegment};

/// Branch colors: by level / order 1, 2, 3, … (then again).
const TREE_PALETTE: [u32; 6] = [0x5B8DEF, 0xF97316, 0x2DD4BF, 0xC084FC, 0xFACC15, 0xF472B6];

#[derive(Default)]
pub(super) struct TreesState {
    neurons: Vec<TreeNeuron>,
    /// A Measure Filament Tree step ran.
    measured: bool,
    /// The list's rows: indices into `neurons`, sorted.
    order: Vec<usize>,
    /// Index into `neurons`.
    selected: Option<usize>,
    segments: Vec<TreeSegment>,
    layout: Vec<Placed>,
    /// Index into `segments`.
    selected_segment: Option<usize>,
    /// A segment to select once its neuron's segments are read.
    pending_segment: Option<String>,
    /// The view's setting (how it's drawn).
    pub(super) by_strahler: bool,
    sort: i32,
    generation: u64,
}

/// A segment in the dendrogram's 1000 x 1000 box: from x0 to x1 on row y,
/// and the rows of its children (its branch point's line).
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Placed {
    pub(super) x0: f64,
    pub(super) x1: f64,
    pub(super) y: f64,
    pub(super) children: Vec<usize>,
}

/// The dendrogram's layout (and the longest path distance): the tips a row
/// each in depth-first order (roots and children in `segments` order), a
/// branch point between its children's rows; x from where the parent ends
/// to where it ends - its path distance if measured, else its parent's end
/// plus its length.
pub(super) fn layout(segments: &[TreeSegment]) -> (Vec<Placed>, f64) {
    let n = segments.len();
    let index: HashMap<&str, usize> = segments
        .iter()
        .enumerate()
        .map(|(i, s)| (s.object_id.as_str(), i))
        .collect();
    let parent: Vec<Option<usize>> = segments
        .iter()
        .map(|s| s.parent.as_deref().and_then(|p| index.get(p).copied()))
        .collect();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, p) in parent.iter().enumerate() {
        if let Some(p) = p {
            children[*p].push(i);
        }
    }
    // Ends: from the roots down (a loop of parents: never reached).
    let mut start = vec![0.0f64; n];
    let mut end = vec![f64::NAN; n];
    let mut stack: Vec<usize> = (0..n).filter(|&i| parent[i].is_none()).rev().collect();
    while let Some(i) = stack.pop() {
        if end[i].is_finite() {
            continue;
        }
        start[i] = parent[i].map_or(0.0, |p| end[p]);
        end[i] = segments[i]
            .path_distance
            .filter(|d| *d >= start[i])
            .unwrap_or(start[i] + segments[i].length.max(0.0));
        stack.extend(children[i].iter().rev());
    }
    // Rows: tips in depth-first order.
    let mut row = vec![f64::NAN; n];
    let mut next_row = 0.0;
    fn place(i: usize, children: &[Vec<usize>], row: &mut [f64], next: &mut f64) -> f64 {
        if children[i].is_empty() {
            row[i] = *next;
            *next += 1.0;
        } else {
            let rows: Vec<f64> = children[i]
                .iter()
                .map(|&c| place(c, children, row, next))
                .collect();
            row[i] = rows.iter().sum::<f64>() / rows.len() as f64;
        }
        row[i]
    }
    for i in (0..n).filter(|&i| parent[i].is_none()) {
        place(i, &children, &mut row, &mut next_row);
    }
    let longest = end
        .iter()
        .copied()
        .filter(|e| e.is_finite())
        .fold(0.0f64, f64::max);
    let rows = next_row.max(1.0);
    let x = |d: f64| 20.0 + d / longest.max(1e-9) * 960.0;
    let y = |r: f64| (r + 0.5) / rows * 1000.0;
    let placed = (0..n)
        .map(|i| Placed {
            x0: if end[i].is_finite() { x(start[i]) } else { 0.0 },
            x1: if end[i].is_finite() { x(end[i]) } else { 0.0 },
            y: if row[i].is_finite() { y(row[i]) } else { 0.0 },
            children: children[i].clone(),
        })
        .collect();
    (placed, longest)
}

/// The segment nearest the point (`x`, `y`) of the box (`w` x `h`): within
/// a few pixels of its line.
pub(super) fn nearest_segment(placed: &[Placed], x: f64, y: f64, w: f64, h: f64) -> Option<usize> {
    let (bx, by) = (x / w.max(1.0) * 1000.0, y / h.max(1.0) * 1000.0);
    let tolerance_y = 8.0 / h.max(1.0) * 1000.0;
    placed
        .iter()
        .enumerate()
        .filter(|(_, p)| bx >= p.x0 - 4.0 && bx <= p.x1 + 4.0)
        .map(|(i, p)| (i, (p.y - by).abs()))
        .filter(|(_, d)| *d <= tolerance_y)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

fn format_value(v: Option<f64>) -> String {
    v.map_or("–".to_string(), format_chart_value)
}

impl ResultsStateController {
    pub(super) fn attach_trees_callbacks(self: &Arc<Self>, ui: &AppWindow) {
        let state = ui.global::<ResultsState>();
        let manager = self.clone();
        state.on_trees_row_clicked(move |row| {
            let neuron = {
                let trees = manager.trees.lock().expect("Poisoned");
                trees.order.get(row.max(0) as usize).copied()
            };
            if let Some(neuron) = neuron {
                manager.trees_select_neuron(neuron, None);
            }
        });
        let manager = self.clone();
        state.on_trees_sort_changed(move || {
            if let Some(ui) = manager.ui.upgrade() {
                manager.trees.lock().expect("Poisoned").sort =
                    ui.global::<ResultsState>().get_trees_sort();
            }
            manager.trees_push_rows();
        });
        let manager = self.clone();
        state.on_trees_color_by_changed(move || {
            if let Some(ui) = manager.ui.upgrade() {
                manager.trees.lock().expect("Poisoned").by_strahler =
                    ui.global::<ResultsState>().get_trees_color_by() == 1;
            }
            manager.remember_view();
            manager.trees_push_tree();
        });
        let manager = self.clone();
        state.on_trees_dendrogram_clicked(move |x, y, w, h| {
            let hit = {
                let trees = manager.trees.lock().expect("Poisoned");
                nearest_segment(
                    &trees.layout,
                    f64::from(x),
                    f64::from(y),
                    f64::from(w),
                    f64::from(h),
                )
            };
            if let Some(i) = hit {
                manager.trees_select_segment(i);
            }
        });
    }

    /// Trees shown: its view's setting, then the neurons.
    pub(super) fn trees_show(self: &Arc<Self>) {
        let by_strahler = self.trees.lock().expect("Poisoned").by_strahler;
        if let Some(ui) = self.ui.upgrade() {
            ui.global::<ResultsState>()
                .set_trees_color_by(i32::from(by_strahler));
        }
        let generation = {
            let mut trees = self.trees.lock().expect("Poisoned");
            trees.generation += 1;
            trees.generation
        };
        self.run_db_job(move |manager| {
            let answer = {
                let guard = manager.result_generator.lock().expect("Poisoned");
                match guard.as_ref() {
                    None => return,
                    Some(db) => db.tree_neurons(None),
                }
            };
            let neurons = answer.unwrap_or_else(|err| {
                error!("Could not read the traced neurons: {err}");
                Default::default()
            });
            {
                let mut trees = manager.trees.lock().expect("Poisoned");
                if trees.generation != generation {
                    return;
                }
                // The neuron selected before, if it's still there.
                let kept = trees
                    .selected
                    .and_then(|i| trees.neurons.get(i))
                    .map(|n| n.soma_id.clone())
                    .and_then(|id| neurons.neurons.iter().position(|n| n.soma_id == id));
                trees.neurons = neurons.neurons;
                trees.measured = neurons.measured;
                trees.selected = kept;
            }
            manager.trees_push_rows();
            manager.trees_push_tree();
        });
    }

    /// The list, sorted, into the window.
    fn trees_push_rows(&self) {
        let (rows, selected_row, note) = {
            let mut trees = self.trees.lock().expect("Poisoned");
            let mut order: Vec<usize> = (0..trees.neurons.len()).collect();
            let sort = trees.sort;
            let neurons = &trees.neurons;
            let value = |i: usize| -> f64 {
                let column = match sort {
                    1 => 0,
                    2 => 1,
                    3 => 2,
                    _ => 3,
                };
                neurons[i]
                    .values
                    .get(column)
                    .copied()
                    .flatten()
                    .unwrap_or(f64::NEG_INFINITY)
            };
            match sort {
                0 => order.sort_by(|&a, &b| {
                    (&neurons[a].image_name, &neurons[a].soma_id)
                        .cmp(&(&neurons[b].image_name, &neurons[b].soma_id))
                }),
                _ => order.sort_by(|&a, &b| value(b).total_cmp(&value(a)).then(a.cmp(&b))),
            }
            let rows: Vec<TreeRow> = order
                .iter()
                .map(|&i| {
                    let n = &trees.neurons[i];
                    let v = |k: usize| n.values.get(k).copied().flatten();
                    TreeRow {
                        label: format!("{} · {}", n.image_name, i + 1).into(),
                        length: v(0).map_or("–".into(), |l| {
                            format!("{} px", format_chart_value(l)).into()
                        }),
                        primary: format_value(v(1)).into(),
                        branch_points: format_value(v(2)).into(),
                        tips: format_value(v(3)).into(),
                    }
                })
                .collect();
            let selected_row = trees
                .selected
                .and_then(|s| order.iter().position(|&i| i == s))
                .map_or(-1, |r| r as i32);
            trees.order = order;
            let note = match (trees.neurons.is_empty(), trees.measured) {
                (true, _) => "No traced neurons in these results.".to_string(),
                (false, false) => {
                    "Add Measure Filament Tree for branch levels and lengths.".to_string()
                }
                _ => String::new(),
            };
            (rows, selected_row, note)
        };
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let state = ui.global::<ResultsState>();
                state.set_trees_rows(ModelRc::from(Rc::new(VecModel::from(rows))));
                state.set_trees_selected(selected_row);
                state.set_trees_note(note.into());
            }
        })
        .ok();
    }

    /// A neuron selected: its segments read, its image shown with its soma
    /// selected - then `segment` (an object id) selected, if given.
    pub(super) fn trees_select_neuron(self: &Arc<Self>, index: usize, segment: Option<String>) {
        let (soma, rel_path, bbox, generation) = {
            let mut trees = self.trees.lock().expect("Poisoned");
            let Some(neuron) = trees.neurons.get(index).cloned() else {
                return;
            };
            trees.selected = Some(index);
            trees.selected_segment = None;
            trees.pending_segment = segment.clone();
            trees.generation += 1;
            (
                neuron.soma_id,
                neuron.image_rel_path,
                neuron.soma_bbox_px,
                trees.generation,
            )
        };
        self.trees_push_rows();
        if segment.is_none() {
            self.split_show_image(rel_path, Some(bbox), None);
        }
        self.run_db_job(move |manager| {
            let answer = {
                let guard = manager.result_generator.lock().expect("Poisoned");
                match guard.as_ref() {
                    None => return,
                    Some(db) => db.tree_segments(&soma),
                }
            };
            let segments = answer.unwrap_or_else(|err| {
                error!("Could not read the neuron's segments: {err}");
                Vec::new()
            });
            {
                let mut trees = manager.trees.lock().expect("Poisoned");
                if trees.generation != generation {
                    return;
                }
                let (placed, _) = layout(&segments);
                trees.layout = placed;
                trees.selected_segment = trees
                    .pending_segment
                    .take()
                    .and_then(|id| segments.iter().position(|s| s.object_id == id));
                trees.segments = segments;
            }
            manager.trees_push_tree();
        });
    }

    /// A segment selected: drawn so, selected in the split view.
    fn trees_select_segment(self: &Arc<Self>, index: usize) {
        let target = {
            let mut trees = self.trees.lock().expect("Poisoned");
            trees.selected_segment = Some(index);
            let rel = trees
                .selected
                .and_then(|i| trees.neurons.get(i))
                .map(|n| n.image_rel_path.clone());
            rel.zip(trees.segments.get(index).map(|s| s.bbox_px))
        };
        self.trees_push_tree();
        if let Some((rel, bbox)) = target {
            self.split_show_image(rel, Some(bbox), None);
        }
    }

    /// A click on the split image picked `object` (its id and its parents'):
    /// its segment, else its neuron (its soma, or a segment's soma).
    pub(super) fn trees_object_clicked(self: &Arc<Self>, object: &str, parents: &[String]) {
        let found = {
            let trees = self.trees.lock().expect("Poisoned");
            if let Some(i) = trees.segments.iter().position(|s| s.object_id == object) {
                Ok(i)
            } else {
                let neuron = trees
                    .neurons
                    .iter()
                    .position(|n| n.soma_id == object || parents.contains(&n.soma_id));
                let is_segment = neuron.is_some_and(|i| trees.neurons[i].soma_id != object);
                Err(neuron.map(|i| (i, is_segment.then(|| object.to_string()))))
            }
        };
        match found {
            Ok(i) => {
                self.trees.lock().expect("Poisoned").selected_segment = Some(i);
                self.trees_push_tree();
            }
            Err(Some((neuron, segment))) => self.trees_select_neuron(neuron, segment),
            Err(None) => {}
        }
    }

    /// The selected neuron's title and dendrogram into the window.
    fn trees_push_tree(&self) {
        let trees = self.trees.lock().expect("Poisoned");
        let neuron = trees.selected.and_then(|i| trees.neurons.get(i));
        let title = neuron
            .map(|n| {
                format!(
                    "{} · neuron {}",
                    n.image_name,
                    trees.selected.unwrap_or(0) + 1
                )
            })
            .unwrap_or_default();
        let summary = neuron
            .map(|n| {
                let parts: Vec<String> = TREE_NEURON_VALUES
                    .iter()
                    .zip(&n.values)
                    .filter_map(|((_, name), v)| {
                        v.map(|v| format!("{name} {}", format_chart_value(v)))
                    })
                    .collect();
                match parts.is_empty() {
                    true => format!("{} segments", n.segments),
                    false => parts.join(" · "),
                }
            })
            .unwrap_or_default();
        let (_, longest) = layout(&trees.segments);
        // A path per color: the branch and its branch point's line.
        let class_of = |s: &TreeSegment| {
            let v = if trees.by_strahler {
                s.strahler_order
            } else {
                s.branch_level
            };
            v.map_or(0, |v| (v.max(1.0) as usize - 1) % TREE_PALETTE.len())
        };
        let mut by_color: Vec<String> = vec![String::new(); TREE_PALETTE.len()];
        let mut used = vec![false; TREE_PALETTE.len()];
        let mut tips = String::new();
        for (s, p) in trees.segments.iter().zip(&trees.layout) {
            let c = class_of(s);
            used[c] = true;
            let path = &mut by_color[c];
            path.push_str(&format!(
                "M {:.1} {:.1} L {:.1} {:.1} ",
                p.x0, p.y, p.x1, p.y
            ));
            if !p.children.is_empty() {
                let ys: Vec<f64> = p.children.iter().map(|&c| trees.layout[c].y).collect();
                let (lo, hi) = ys
                    .iter()
                    .fold((f64::MAX, f64::MIN), |(a, b), &y| (a.min(y), b.max(y)));
                path.push_str(&format!(
                    "M {:.1} {:.1} L {:.1} {:.1} ",
                    p.x1,
                    lo.min(p.y),
                    p.x1,
                    hi.max(p.y)
                ));
            } else {
                tips.push_str(&format!(
                    "M {:.1} {:.1} L {:.1} {:.1} ",
                    p.x1,
                    p.y,
                    p.x1 + 0.1,
                    p.y
                ));
            }
        }
        let label = if trees.by_strahler { "order" } else { "level" };
        let lines: Vec<ChartTimeLine> = (0..TREE_PALETTE.len())
            .filter(|&c| used[c])
            .map(|c| ChartTimeLine {
                label: format!("{label} {}", c + 1).into(),
                color: bg_color_to_slint(TREE_PALETTE[c]),
                line: by_color[c].trim_end().into(),
                band: "".into(),
            })
            .collect();
        // The selected segment and its way back to the soma.
        let index: HashMap<&str, usize> = trees
            .segments
            .iter()
            .enumerate()
            .map(|(i, s)| (s.object_id.as_str(), i))
            .collect();
        let (mut way, mut selected_line, mut cards) = (String::new(), String::new(), Vec::new());
        if let Some(i) = trees.selected_segment.filter(|&i| i < trees.layout.len()) {
            let p = &trees.layout[i];
            selected_line = format!("M {:.1} {:.1} L {:.1} {:.1}", p.x0, p.y, p.x1, p.y);
            let mut at = trees.segments[i]
                .parent
                .as_deref()
                .and_then(|id| index.get(id).copied());
            let mut child_y = p.y;
            let mut guard = 0;
            while let Some(j) = at {
                let q = &trees.layout[j];
                way.push_str(&format!(
                    "M {:.1} {:.1} L {:.1} {:.1} M {:.1} {:.1} L {:.1} {:.1} ",
                    q.x1, child_y, q.x1, q.y, q.x0, q.y, q.x1, q.y
                ));
                child_y = q.y;
                at = trees.segments[j]
                    .parent
                    .as_deref()
                    .and_then(|id| index.get(id).copied());
                guard += 1;
                if guard > trees.segments.len() {
                    break;
                }
            }
            let s = &trees.segments[i];
            let px = |v: Option<f64>| {
                v.map_or("–".to_string(), |v| {
                    format!("{} px", format_chart_value(v))
                })
            };
            cards = vec![
                ("BRANCH LEVEL".to_string(), format_value(s.branch_level)),
                ("STRAHLER ORDER".to_string(), format_value(s.strahler_order)),
                ("PATH DISTANCE".to_string(), px(s.path_distance)),
                ("LENGTH".to_string(), px(Some(s.length))),
                ("STRAIGHTNESS".to_string(), format_value(s.straightness)),
            ];
        }
        let x_max = format!("{} px", format_chart_value(longest));
        drop(trees);
        let ui_weak = self.ui.clone();
        crate::helper::ui_thread::invoke_from_event_loop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                let state = ui.global::<ResultsState>();
                state.set_trees_title(title.into());
                state.set_trees_summary(summary.into());
                state.set_trees_lines(ModelRc::from(Rc::new(VecModel::from(lines))));
                state.set_trees_tips(tips.trim_end().into());
                state.set_trees_path(way.trim_end().into());
                state.set_trees_selected_segment(selected_line.into());
                state.set_trees_x_max(x_max.into());
                state.set_trees_cards(ModelRc::from(Rc::new(VecModel::from(
                    cards
                        .into_iter()
                        .map(|(label, value)| ChartCard {
                            label: label.into(),
                            value: value.into(),
                        })
                        .collect::<Vec<_>>(),
                ))));
            }
        })
        .ok();
    }

    pub(super) fn in_trees(&self) -> bool {
        *self.rail_mode.lock().expect("Poisoned") == ResultsRailMode::Trees
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(id: &str, parent: Option<&str>, length: f64) -> TreeSegment {
        TreeSegment {
            object_id: id.into(),
            parent: parent.map(Into::into),
            length,
            ..Default::default()
        }
    }

    /// A root (10) splitting into two (5, 7): the tips a row each, the
    /// branch point between them; x by the path distance.
    #[test]
    fn the_layout_puts_tips_on_rows_and_branch_points_between() {
        let segments = [
            segment("a", None, 10.0),
            segment("b", Some("a"), 5.0),
            segment("c", Some("a"), 7.0),
        ];
        let (placed, longest) = layout(&segments);
        assert_eq!(longest, 17.0);
        assert_eq!(placed[1].y, 250.0);
        assert_eq!(placed[2].y, 750.0);
        assert_eq!(placed[0].y, 500.0);
        assert_eq!(placed[0].children, [1, 2]);
        assert_eq!(placed[0].x0, 20.0);
        assert_eq!(placed[2].x1, 980.0);
        assert!(
            (placed[1].x0 - placed[0].x1).abs() < 1e-9,
            "a child starts where its parent ends"
        );
    }

    #[test]
    fn a_measured_path_distance_places_the_end() {
        let mut b = segment("b", Some("a"), 5.0);
        b.path_distance = Some(20.0);
        let (placed, longest) = layout(&[segment("a", None, 10.0), b]);
        assert_eq!(longest, 20.0);
        assert_eq!(placed[1].x1, 980.0);
    }

    #[test]
    fn a_click_finds_the_branch_under_it() {
        let segments = [
            segment("a", None, 10.0),
            segment("b", Some("a"), 5.0),
            segment("c", Some("a"), 7.0),
        ];
        let (placed, _) = layout(&segments);
        // The box 1000 x 400: c's row at y 300, its x from ~585 to 980.
        assert_eq!(
            nearest_segment(&placed, 800.0, 301.0, 1000.0, 400.0),
            Some(2)
        );
        assert_eq!(nearest_segment(&placed, 800.0, 200.0, 1000.0, 400.0), None);
    }

    #[test]
    fn a_loop_of_parents_does_not_hang_the_layout() {
        let (placed, _) = layout(&[segment("a", Some("b"), 1.0), segment("b", Some("a"), 1.0)]);
        assert_eq!(placed.len(), 2);
    }
}
