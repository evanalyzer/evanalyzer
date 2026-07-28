//! Bounds the results table's memory usage while scrolling through a very
//! large file: [`RowWindow`] tracks which contiguous run of pages of
//! [`ObjectRow`]s is currently loaded and evicts the oldest page from the
//! opposite scroll edge whenever a new page pushes the window past its size
//! limit, instead of letting the loaded set grow forever as the user scrolls.
//!
//! Deliberately free of any Slint/GUI dependency so it's usable/testable on
//! its own — the GUI layer (`crates/gui/src/editor/results_table_controller.rs`)
//! owns translating [`EvictEdge`] into an actual `VecModel` edit and scroll-
//! position compensation.

use super::results_loader::ObjectRow;
use std::collections::{HashMap, VecDeque};

/// Default number of pages kept resident before evicting the oldest page from
/// the opposite edge. At the GUI's `PAGE_SIZE = 500`, this bounds the table to
/// ~10,000 resident rows (a few tens of MB, negligible against the multi-GB
/// baseline this is meant to fix) while still being large enough that
/// continuous fast scrolling only triggers an evict+backfill cycle roughly
/// every 20 page-loads in a given direction.
pub const DEFAULT_WINDOW_PAGES: usize = 20;

/// One loaded page's bookkeeping: which page index it is, and exactly which
/// (lowercased) object ids it contributed to `RowWindow::rows` — needed so
/// eviction knows precisely which map entries to drop.
struct PageEntry {
    page: usize,
    object_ids: Vec<String>,
}

/// Instructs the caller to remove `row_count` rows from the front or back of
/// its own row model/list widget, to mirror an eviction `RowWindow` just made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvictEdge {
    pub from_front: bool,
    pub row_count: usize,
}

/// Tracks the contiguous run of pages of [`ObjectRow`]s currently loaded, bounding
/// memory by evicting the oldest page from the opposite edge whenever a new
/// page pushes the window past `window_pages`. Rows are looked up by their
/// stable, case-insensitive `object_id` rather than by array position, so an
/// eviction never invalidates an in-flight selection lookup — unlike indexing
/// a plain `Vec` by a positional row number, which breaks the moment anything
/// is removed from the front.
pub struct RowWindow {
    window_pages: usize,
    /// Always contiguous: `{p, p+1, ..., p+len-1}` for some `p`.
    pages: VecDeque<PageEntry>,
    rows: HashMap<String, ObjectRow>,
}

impl RowWindow {
    pub fn new(window_pages: usize) -> Self {
        Self {
            window_pages,
            pages: VecDeque::new(),
            rows: HashMap::new(),
        }
    }

    /// Clears all loaded pages/rows. Call on every full reload (a filter,
    /// sort, or group-config change) — those already fully replace the
    /// caller's row model, so this just keeps `RowWindow`'s bookkeeping in
    /// sync with that reset.
    pub fn reset(&mut self) {
        self.pages.clear();
        self.rows.clear();
    }

    /// Records a newly-appended page (the user scrolled down and a new page
    /// loaded past the bottom). Returns an eviction instruction if the window
    /// now exceeds `window_pages` (evicts from the front — the oldest,
    /// topmost page — since the window grew from the bottom).
    pub fn note_appended(&mut self, page: usize, objects: &[ObjectRow]) -> Option<EvictEdge> {
        self.insert_rows(objects);
        self.pages.push_back(PageEntry {
            page,
            object_ids: lowercased_ids(objects),
        });
        self.evict_if_needed(true)
    }

    /// Records a newly-prepended page (the user scrolled up past the top and
    /// an earlier, previously-evicted page was re-fetched). Returns an
    /// eviction instruction (from the tail this time) if the window now
    /// exceeds `window_pages`.
    pub fn note_prepended(&mut self, page: usize, objects: &[ObjectRow]) -> Option<EvictEdge> {
        self.insert_rows(objects);
        self.pages.push_front(PageEntry {
            page,
            object_ids: lowercased_ids(objects),
        });
        self.evict_if_needed(false)
    }

    fn insert_rows(&mut self, objects: &[ObjectRow]) {
        for object in objects {
            self.rows
                .insert(object.object_id.to_lowercase(), object.clone());
        }
    }

    fn evict_if_needed(&mut self, evict_from_front: bool) -> Option<EvictEdge> {
        if self.pages.len() <= self.window_pages {
            return None;
        }
        let evicted = if evict_from_front {
            self.pages.pop_front()
        } else {
            self.pages.pop_back()
        }?;
        let row_count = evicted.object_ids.len();
        for id in &evicted.object_ids {
            self.rows.remove(id);
        }
        Some(EvictEdge {
            from_front: evict_from_front,
            row_count,
        })
    }

    /// Looks up a currently-loaded row by its (case-insensitive) `object_id`.
    /// Returns `None` for an id that was evicted, or was never loaded.
    pub fn get(&self, object_id: &str) -> Option<&ObjectRow> {
        self.rows.get(&object_id.to_lowercase())
    }

    /// The lowest currently-loaded page index, or `None` if nothing is loaded.
    pub fn oldest_loaded_page(&self) -> Option<usize> {
        self.pages.front().map(|p| p.page)
    }

    /// The highest currently-loaded page index, or `None` if nothing is loaded.
    pub fn newest_loaded_page(&self) -> Option<usize> {
        self.pages.back().map(|p| p.page)
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}

fn lowercased_ids(objects: &[ObjectRow]) -> Vec<String> {
    objects.iter().map(|r| r.object_id.to_lowercase()).collect()
}

/// Lighter sibling of [`RowWindow`] for views with no row selection (the
/// colocalization detail flat table): tracks the same contiguous run of
/// loaded *source* pages and evicts the same way, but only remembers how many
/// *displayed* rows each source page produced (its flattening can fan one
/// source object out into zero or more rows), not the rows' content — nothing in
/// that view ever looks a row up by id, so there's no map to keep in sync.
pub struct PageRowCounts {
    window_pages: usize,
    /// Always contiguous: `{p, p+1, ..., p+len-1}`.
    pages: VecDeque<(usize, usize)>, // (page, displayed row count)
}

impl PageRowCounts {
    pub fn new(window_pages: usize) -> Self {
        Self {
            window_pages,
            pages: VecDeque::new(),
        }
    }

    pub fn reset(&mut self) {
        self.pages.clear();
    }

    /// Records a newly-appended source page that produced `row_count`
    /// displayed rows. Returns an eviction instruction (front) if the window
    /// now exceeds `window_pages`.
    pub fn note_appended(&mut self, page: usize, row_count: usize) -> Option<EvictEdge> {
        self.pages.push_back((page, row_count));
        self.evict_if_needed(true)
    }

    fn evict_if_needed(&mut self, evict_from_front: bool) -> Option<EvictEdge> {
        if self.pages.len() <= self.window_pages {
            return None;
        }
        let (_, row_count) = if evict_from_front {
            self.pages.pop_front()
        } else {
            self.pages.pop_back()
        }?;
        Some(EvictEdge {
            from_front: evict_from_front,
            row_count,
        })
    }

    pub fn oldest_loaded_page(&self) -> Option<usize> {
        self.pages.front().map(|(p, _)| *p)
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(id: &str) -> ObjectRow {
        ObjectRow {
            image_name: String::new(),
            image_rel_path: String::new(),
            c_stack: None,
            z_stack: None,
            t_stack: None,
            object_id: id.to_string(),
            seg_class_name: None,
            seg_class_id: None,
            object_class_name: vec![],
            object_class_id: vec![],
            parent_id: None,
            children: vec![],
            track_id: 0,
            centroid_x_px: 0.0,
            centroid_y_px: 0.0,
            centroid_x_nm: 0.0,
            centroid_y_nm: 0.0,
            area_px: 0,
            area_nm2: 0.0,
            perimeter_px: 0.0,
            perimeter_nm: 0.0,
            circularity: 0.0,
            solidity: 0.0,
            aspect_ratio: 0.0,
            roundness: 0.0,
            compactness: 0.0,
            major_axis_px: 0.0,
            minor_axis_px: 0.0,
            touches_edge: false,
            intensities_json: String::new(),
            coloc_json: String::new(),
            bbox_px: [0, 0, 0, 0],
        }
    }

    fn page_of(ids: &[&str]) -> Vec<ObjectRow> {
        ids.iter().map(|id| object(id)).collect()
    }

    #[test]
    fn append_within_budget_never_evicts() {
        let mut w = RowWindow::new(3);
        assert_eq!(w.note_appended(0, &page_of(&["a"])), None);
        assert_eq!(w.note_appended(1, &page_of(&["b"])), None);
        assert_eq!(w.note_appended(2, &page_of(&["c"])), None);
        assert_eq!(w.oldest_loaded_page(), Some(0));
        assert_eq!(w.newest_loaded_page(), Some(2));
        assert!(w.get("a").is_some());
        assert!(w.get("b").is_some());
        assert!(w.get("c").is_some());
    }

    #[test]
    fn append_past_budget_evicts_oldest_from_front() {
        let mut w = RowWindow::new(2);
        assert_eq!(w.note_appended(0, &page_of(&["a"])), None);
        assert_eq!(w.note_appended(1, &page_of(&["b"])), None);
        // Third page pushes the window (3 pages) past window_pages (2).
        let evict = w.note_appended(2, &page_of(&["c"])).expect("must evict");
        assert!(evict.from_front);
        assert_eq!(evict.row_count, 1);

        assert_eq!(w.oldest_loaded_page(), Some(1));
        assert_eq!(w.newest_loaded_page(), Some(2));
        assert!(w.get("a").is_none(), "evicted row must no longer be found");
        assert!(w.get("b").is_some());
        assert!(w.get("c").is_some());
    }

    #[test]
    fn prepend_past_budget_evicts_from_tail() {
        let mut w = RowWindow::new(2);
        w.note_appended(5, &page_of(&["e"]));
        w.note_appended(6, &page_of(&["f"]));
        // Scrolled back up and backfilled an earlier page — now 3 pages loaded.
        let evict = w.note_prepended(4, &page_of(&["d"])).expect("must evict");
        assert!(
            !evict.from_front,
            "prepend evicts from the tail, not the front"
        );
        assert_eq!(evict.row_count, 1);

        assert_eq!(w.oldest_loaded_page(), Some(4));
        assert_eq!(w.newest_loaded_page(), Some(5));
        assert!(w.get("d").is_some());
        assert!(w.get("e").is_some());
        assert!(w.get("f").is_none(), "tail page must have been evicted");
    }

    #[test]
    fn reset_clears_everything() {
        let mut w = RowWindow::new(2);
        w.note_appended(0, &page_of(&["a"]));
        w.reset();
        assert!(w.is_empty());
        assert_eq!(w.oldest_loaded_page(), None);
        assert!(w.get("a").is_none());
    }

    #[test]
    fn lookup_is_case_insensitive() {
        let mut w = RowWindow::new(5);
        w.note_appended(0, &page_of(&["ABC-123"]));
        assert!(w.get("abc-123").is_some());
        assert!(w.get("ABC-123").is_some());
        assert!(w.get("AbC-123").is_some());
    }

    #[test]
    fn get_returns_none_for_never_loaded_id() {
        let mut w = RowWindow::new(5);
        w.note_appended(0, &page_of(&["a"]));
        assert!(w.get("z").is_none());
    }

    #[test]
    fn page_row_counts_evicts_by_displayed_row_count_not_page_count() {
        let mut w = PageRowCounts::new(2);
        assert_eq!(w.note_appended(0, 3), None); // one source object -> 3 flattened rows
        assert_eq!(w.note_appended(1, 1), None);
        let evict = w.note_appended(2, 5).expect("must evict");
        assert!(evict.from_front);
        assert_eq!(
            evict.row_count, 3,
            "evicted page's displayed row count, not 1"
        );
        assert_eq!(w.oldest_loaded_page(), Some(1));
    }

    #[test]
    fn page_row_counts_reset_clears_everything() {
        let mut w = PageRowCounts::new(2);
        w.note_appended(0, 4);
        w.reset();
        assert!(w.is_empty());
        assert_eq!(w.oldest_loaded_page(), None);
    }
}
