use evanalyzer_app::{Frontend, ProjectOwner};

pub struct CliFrontend;

impl Frontend for CliFrontend {
    fn start(self: Box<Self>, _owner: ProjectOwner) {
        // run CLI loop
    }
}

pub fn create() -> CliFrontend {
    CliFrontend
}
