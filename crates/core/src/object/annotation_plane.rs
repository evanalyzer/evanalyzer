//! # annotation_plane
//!
//! Which plane an annotation (a stored object) belongs to.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use evanalyzer_cfg::core_types::ZProjection;
use evanalyzer_cfg::settings::object_settings::ObjectMetricSettings;

/// The plane of a stored object.
pub trait ObjectMetricSettingsExt {
    /// Whether this object (of one plane, or drawn on a z-projection - not a
    /// volume's) belongs to plane `z` of frame `t`, or (`z: None`) to the
    /// z-projection of frame `t`: what an analysis of that image uses, and
    /// what the viewer shows on it (see [`ObjectMetricSettings::z_projection`]).
    fn belongs_to(&self, z: Option<i32>, t: i32) -> bool;
}

impl ObjectMetricSettingsExt for ObjectMetricSettings {
    fn belongs_to(&self, z: Option<i32>, t: i32) -> bool {
        self.t_stack == t
            && match z {
                None => true,
                Some(z) => self.z_projection == ZProjection::None && self.z_stack == z,
            }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_planes_object_belongs_to_its_plane_and_the_projections_of_its_frame() {
        let object = ObjectMetricSettings {
            z_stack: 3,
            t_stack: 1,
            ..Default::default()
        };
        assert!(object.belongs_to(Some(3), 1));
        assert!(object.belongs_to(None, 1));
        assert!(!object.belongs_to(Some(2), 1), "another plane");
        assert!(!object.belongs_to(Some(3), 0), "another frame");
        assert!(!object.belongs_to(None, 0), "a projection of another frame");
    }

    #[test]
    fn an_object_drawn_on_a_projection_belongs_to_the_projections_of_its_frame_only() {
        let object = ObjectMetricSettings {
            t_stack: 1,
            z_projection: ZProjection::MaxIntensity,
            ..Default::default()
        };
        assert!(object.belongs_to(None, 1));
        assert!(!object.belongs_to(Some(0), 1), "not plane 0, its unused z");
        assert!(!object.belongs_to(None, 2));
    }
}
