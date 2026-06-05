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
}
