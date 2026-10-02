// app/src/frontend.rs - app defines the trait, knows nothing about gui

use crate::ProjectOwner;

pub trait Frontend: Send + Sync {
    fn start(self: Box<Self>, owner: ProjectOwner);
}
