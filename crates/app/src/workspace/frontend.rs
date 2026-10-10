// app/src/frontend.rs - app defines the trait, knows nothing about gui

use crate::workspace::ProjectOwner;

pub trait Frontend: Send + Sync {
    fn start(self: Box<Self>, owner: ProjectOwner);
}
