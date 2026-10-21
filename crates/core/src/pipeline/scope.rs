//! Scopes: which objects a command sees.
//!
//! Like variables in a programming language, objects live in a scope: the
//! project's (global) or that of one module instance. A module's
//! local classes (`ObjectClass::Local`) only name objects of the instance
//! the command runs in, so two instances of one module never see each
//! other's objects. Scopes exist only while a job runs - nothing is saved.

use evanalyzer_cfg::core_types::ObjectClass;
use serde::{Deserialize, Serialize};

/// One module instance's scope, numbered per job at expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ScopeId(pub u32);

/// Where an object lives, and where a command runs.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum Scope {
    /// The project's: commands outside modules, objects of project classes.
    #[default]
    Global,
    /// One module instance's.
    Scoped(ScopeId),
}

impl Scope {
    /// The scope of the objects `class` names, for a command running in
    /// `self`: a local class names objects of this scope, every other class
    /// the project's.
    pub fn of_class(self, class: &ObjectClass) -> Scope {
        match class {
            ObjectClass::Local(_) => self,
            ObjectClass::Unset | ObjectClass::Global(_) => Scope::Global,
        }
    }
}
