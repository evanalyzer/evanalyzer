#[derive(PartialEq, Eq, Debug)]
pub enum TaskDispatch {
    HighRes,
    LowRes,
    HighResAndLowRes,
    Rois,
}

#[derive(Debug, Clone)]
pub struct DrawingTask {
    pub(crate) auto_adjust_selected: bool,
    pub(crate) auto_adjust_if_not_set: bool, // Apply auto adjustment only not yet set
    pub(crate) is_new_image: bool,
    pub(crate) fit_to_screen: bool,
    pub(crate) is_new_series: bool,
}

impl Default for DrawingTask {
    fn default() -> Self {
        Self {
            auto_adjust_selected: false,
            auto_adjust_if_not_set: false,
            is_new_image: false,
            fit_to_screen: false,
            is_new_series: false,
        }
    }
}

impl DrawingTask {
    pub fn reset_job(&mut self) {
        self.auto_adjust_selected = false;
        self.auto_adjust_if_not_set = false;
        self.is_new_image = false;
        self.fit_to_screen = false;
        self.is_new_series = false;
    }

    /// Combines two tasks so dispatching a new one never silently drops a
    /// task already sitting in the worker's single-slot queue (a plain
    /// overwrite there previously let a debounced pan/zoom redraw clobber a
    /// still-pending "new image" task, skipping histogram computation and
    /// rendering the image black). Every field here is a "this must still
    /// happen" flag, not a current-state snapshot (that's `ViewportState`,
    /// dispatched separately) - so a plain OR is exactly correct: if either
    /// task asked for it, the merged task must still ask for it.
    pub fn merge(self, other: DrawingTask) -> DrawingTask {
        DrawingTask {
            auto_adjust_selected: self.auto_adjust_selected || other.auto_adjust_selected,
            auto_adjust_if_not_set: self.auto_adjust_if_not_set || other.auto_adjust_if_not_set,
            is_new_image: self.is_new_image || other.is_new_image,
            fit_to_screen: self.fit_to_screen || other.fit_to_screen,
            is_new_series: self.is_new_series || other.is_new_series,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_is_a_field_wise_or_in_either_order() {
        let new_image_task = DrawingTask {
            auto_adjust_selected: false,
            auto_adjust_if_not_set: true,
            is_new_image: true,
            fit_to_screen: true,
            is_new_series: true,
        };
        let plain_default_task = DrawingTask::default();

        let a = new_image_task.clone().merge(plain_default_task.clone());
        let b = plain_default_task.merge(new_image_task.clone());

        for merged in [a, b] {
            assert!(
                merged.is_new_image,
                "a still-pending new-image request must survive being merged with a plain redraw"
            );
            assert!(merged.auto_adjust_if_not_set);
            assert!(merged.fit_to_screen);
            assert!(merged.is_new_series);
            assert!(!merged.auto_adjust_selected);
        }
    }

    #[test]
    fn merge_of_two_defaults_is_a_default() {
        let merged = DrawingTask::default().merge(DrawingTask::default());
        assert!(!merged.auto_adjust_selected);
        assert!(!merged.auto_adjust_if_not_set);
        assert!(!merged.is_new_image);
        assert!(!merged.fit_to_screen);
        assert!(!merged.is_new_series);
    }
}
