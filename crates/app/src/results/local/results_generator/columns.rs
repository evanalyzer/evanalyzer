//! # columns
//!
//! What the results hold: their classes, the coloc partners and groups, the
//! columns to pick, the relations and the measurements.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use super::*;

impl ResultsGenerator {
    /// Snapshot of the `classes` table. Cached after the first call for this
    /// opened database — see `classes_cache` — since it's read on essentially
    /// every `get_list` call (both to translate the class filter and to
    /// resolve display colors) but the registry itself only ever changes by
    /// opening a different database.
    pub fn get_object_classes(&self) -> Result<Vec<Class>, InternalErrors> {
        if let Some(cached) = self.classes_cache.borrow().as_ref() {
            return Ok(cached.clone());
        }

        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut stmt = self
            .database
            .prepare("SELECT class_id, name, color FROM classes ORDER BY class_id")
            .map_err(err)?;
        let classes = stmt
            .query_map([], |row| {
                let class_id: u32 = row.get(0)?;
                let color: Option<u32> = row.get(2)?;
                Ok(Class {
                    id: class_id,
                    name: row.get(1)?,
                    color: color.unwrap_or(0),
                    notes: String::new(),
                    results: Default::default(),
                })
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;

        *self.classes_cache.borrow_mut() = Some(classes.clone());
        Ok(classes)
    }

    /// Every class id that appears as a colocalization partner in at least
    /// one object's `coloc_partner_ids` — i.e. the candidates for a
    /// `Column::ColocCount(class)` column, mirroring how
    /// `get_available_columns` enumerates one Avg/Sum/Min/Max intensity
    /// column per `get_nr_of_c_stacks()` channel. The union of
    /// [`Self::get_coloc_class_sets`].
    pub fn get_object_classes_with_at_least_coloc(
        &self,
    ) -> Result<Vec<ObjectClass>, InternalErrors> {
        let mut classes: Vec<ObjectClass> =
            self.get_coloc_class_sets()?.into_iter().flatten().collect();
        classes.sort();
        classes.dedup();
        Ok(classes)
    }

    /// The combinations of partner classes objects colocalize with - each
    /// distinct set of `coloc_partner_ids`' keys (sorted), e.g. `[2, 3]` for
    /// objects with partners of class 2 and 3 and of no other. The map is
    /// keyed by class id directly (see evanalyzer_core's duckdb.rs) and
    /// only holds classes with partners.
    ///
    /// Cached after the first call for this opened database (see
    /// `coloc_classes_cache`), same reasoning as `get_object_classes`: this
    /// scans every non-empty `coloc_partner_ids` in the table, and what was
    /// recorded can't change without re-exporting (i.e. opening a
    /// different database).
    pub fn get_coloc_class_sets(&self) -> Result<Vec<Vec<ObjectClass>>, InternalErrors> {
        Ok(self.coloc_sets_and_groups()?.0)
    }

    /// The coloc groups: the classes one Colocalize step looked at, each
    /// group sorted, the groups by their first class. Read from the objects:
    /// an object's partners are only ever of its own step's classes, so
    /// joining every partner class with the object's classes - those that
    /// are partners somewhere, not e.g. a marker class it also has - splits
    /// them into the groups.
    pub fn get_coloc_groups(&self) -> Result<Vec<Vec<ObjectClass>>, InternalErrors> {
        Ok(self.coloc_sets_and_groups()?.1)
    }

    /// `get_coloc_class_sets` and `get_coloc_groups` from one scan of the
    /// objects with partners. Cached after the first call for this opened
    /// database (see `coloc_classes_cache`), same reasoning as
    /// `get_object_classes`: what was recorded can't change without opening
    /// a different database.
    #[allow(clippy::type_complexity)]
    pub(super) fn coloc_sets_and_groups(
        &self,
    ) -> Result<(Vec<Vec<ObjectClass>>, Vec<Vec<ObjectClass>>), InternalErrors> {
        if let Some(cached) = self.coloc_classes_cache.borrow().as_ref() {
            return Ok(cached.clone());
        }

        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        // The objects' own classes (only partners join the groups without).
        let own = if self.has_object_column("object_class_id")? {
            "CAST(list_sort(object_class_id) AS VARCHAR)"
        } else {
            "'[]'"
        };
        let mut stmt = self
            .database
            .prepare(&format!(
                "SELECT DISTINCT {own}, \
                        CAST(list_sort(map_keys(coloc_partner_ids)) AS VARCHAR) \
                 FROM objects \
                 WHERE cardinality(coloc_partner_ids) > 0"
            ))
            .map_err(err)?;
        let rows: Vec<(String, String)> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;

        // "[2, 3]" - a class key per entry (-1, no class, isn't a column).
        let parse = |list: &str| -> Vec<ObjectClass> {
            list.trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .filter_map(|key| key.trim().parse::<u32>().ok())
                .map(ObjectClass::Global)
                .collect()
        };
        let pairs: Vec<(Vec<ObjectClass>, Vec<ObjectClass>)> = rows
            .iter()
            .map(|(own, partners)| (parse(own), parse(partners)))
            .filter(|(_, partners)| !partners.is_empty())
            .collect();
        let mut sets: Vec<Vec<ObjectClass>> = pairs.iter().map(|(_, set)| set.clone()).collect();
        sets.sort();
        sets.dedup();
        let groups = coloc_groups_of(&pairs);

        *self.coloc_classes_cache.borrow_mut() = Some((sets.clone(), groups.clone()));
        Ok((sets, groups))
    }

    pub fn get_available_columns(&self) -> Result<Vec<ColumnEntry>, InternalErrors> {
        let classes = self.get_object_classes()?;
        let entry = |key: Column, group: &str| ColumnEntry {
            display_name: key.display_label(&classes),
            key,
            group: group.into(),
            coloc_group: 0,
        };
        let mut ret = vec![
            entry(Column::ObjectId, "General"),
            entry(Column::TrackId, "General"),
            entry(Column::ImageName, "General"),
            entry(Column::ObjectClass, "General"),
            entry(Column::Count, "General"),
            entry(Column::AreaSizePx, "Geometry"),
            entry(Column::AreaSizeNm, "Geometry"),
            entry(Column::PerimeterPx, "Geometry"),
            entry(Column::PerimeterNm, "Geometry"),
            entry(Column::Circularity, "Shape"),
            entry(Column::Solidity, "Shape"),
            entry(Column::Eccentricity, "Shape"),
            entry(Column::MaxFeret, "Shape"),
            entry(Column::MinFeret, "Shape"),
            entry(Column::MajorAxis, "Shape"),
            entry(Column::MinorAxis, "Shape"),
            entry(Column::EquivalentDiameter, "Shape"),
            entry(Column::Extent, "Shape"),
        ];
        // A volume's objects' size - only offered for results with any.
        if self.has_volumes()? {
            ret.push(entry(Column::VolumePx, "Geometry"));
            ret.push(entry(Column::VolumeNm, "Geometry"));
            ret.push(entry(Column::SurfaceAreaNm, "Geometry"));
        }

        // Coloc count is measured per candidate partner class (like
        // intensity is measured per channel below), so there's one column
        // per class that actually shows up as a colocalization partner
        // somewhere in this database, rather than a single shared "total
        // across every class" column.
        let coloc_classes = self.get_object_classes_with_at_least_coloc()?;
        let groups = self.get_coloc_groups()?;
        for class_id in &coloc_classes {
            let mut column = entry(Column::ColocCount(*class_id), "Coloc");
            column.coloc_group = groups
                .iter()
                .position(|group| group.contains(class_id))
                .map_or(0, |i| i as u32 + 1);
            ret.push(column);
        }
        // The combinations that occur: "Coloc with exactly" each set of
        // partner classes found, among all of them - the GUI makes its
        // combination columns from them (`coloc_columns`).
        for set in self.get_coloc_class_sets()? {
            ret.push(entry(
                Column::coloc(ColocMatch::Exactly, set, coloc_classes.clone(), false),
                "Coloc",
            ));
        }

        // Intensity is measured per image channel, so there's one Avg/Sum/
        // Min/Max column per channel rather than a single shared one.
        for channel in 0..self.get_nr_of_c_stacks() {
            ret.push(entry(Column::IntensityAvg(channel), "intensity"));
            ret.push(entry(Column::IntensitySum(channel), "intensity"));
            ret.push(entry(Column::IntensityMin(channel), "intensity"));
            ret.push(entry(Column::IntensityMax(channel), "intensity"));
            ret.push(entry(Column::IntensityStd(channel), "intensity"));
            ret.push(entry(Column::IntensityMedian(channel), "intensity"));
            ret.push(entry(Column::IntensityLowerQuartile(channel), "intensity"));
            ret.push(entry(Column::IntensityUpperQuartile(channel), "intensity"));
            ret.push(entry(Column::IntensityMad(channel), "intensity"));
        }

        // The measurement steps' measurements, by step - and the objects
        // those that name one are about.
        let naming = self.get_measurement_keys_naming_objects()?;
        for column in self.get_measurement_columns()? {
            let names_objects = naming.contains(&column.key);
            ret.push(entry(Column::Measurement(column.clone()), "Measurements"));
            if names_objects {
                ret.push(entry(Column::MeasurementObject(column), "Measurements"));
            }
        }

        // The links Relate Objects made: per (parent class, child class) the
        // children's count, per parent class the parent's id - the List's
        // Children and Linked objects groups are built from these.
        let relations = self.get_relations()?;
        for (parent_class, child_class) in &relations {
            ret.push(entry(
                Column::Children(ChildColumn {
                    parent_class: *parent_class,
                    child_class: *child_class,
                    value: ChildValue::Count,
                }),
                "Children",
            ));
        }
        let mut parents: Vec<ObjectClass> = relations.iter().map(|(p, _)| *p).collect();
        parents.dedup();
        for parent_class in parents {
            ret.push(entry(
                Column::Parent(ParentColumn {
                    parent_class,
                    value: ParentValue::Id,
                }),
                "Parent",
            ));
        }

        Ok(ret)
    }

    /// The (parent class, child class) pairs Relate Objects linked in these
    /// results, sorted - none in results without `parent_ids`.
    pub fn get_relations(&self) -> Result<Vec<(ObjectClass, ObjectClass)>, InternalErrors> {
        if let Some(cached) = self.relations_cache.borrow().as_ref() {
            return Ok(cached.clone());
        }
        let relations = self.read_relations()?;
        *self.relations_cache.borrow_mut() = Some(relations.clone());
        Ok(relations)
    }

    pub(super) fn read_relations(&self) -> Result<Vec<(ObjectClass, ObjectClass)>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        if !self.has_parent_ids()? {
            return Ok(Vec::new());
        }
        let mut stmt = self
            .database
            .prepare(
                "SELECT DISTINCT parent_key, UNNEST(object_class_id) AS child_key FROM \
                 (SELECT UNNEST(map_keys(parent_ids)) AS parent_key, object_class_id \
                 FROM objects WHERE cardinality(parent_ids) > 0) ORDER BY 1, 2",
            )
            .map_err(err)?;
        let pairs = stmt
            .query_map([], |row| Ok((row.get::<_, i32>(0)?, row.get::<_, i32>(1)?)))
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;
        Ok(pairs
            .into_iter()
            .filter_map(|(parent, child)| {
                Some((
                    ObjectClass::Global(u32::try_from(parent).ok()?),
                    ObjectClass::Global(u32::try_from(child).ok()?),
                ))
            })
            .collect())
    }

    /// The measurements of this analysis (`measurement_info`), by step and
    /// name - none in results from before measurement steps existed.
    pub fn get_measurement_columns(&self) -> Result<Vec<MeasurementColumn>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let exists: bool = self
            .database
            .query_row(
                "SELECT count(*) > 0 FROM information_schema.tables \
                 WHERE table_name = 'measurement_info'",
                [],
                |row| row.get(0),
            )
            .map_err(err)?;
        if !exists {
            return Ok(Vec::new());
        }
        // Profiles (a curve per object, since format 2) are no single value,
        // track values (Measure Track Motion) no object's or image's - the
        // tracks' views show those: not offered here.
        let not_profile = if self.has_profiles()? {
            "AND NOT coalesce(profile, false)"
        } else {
            ""
        };
        let mut stmt = self
            .database
            .prepare(&format!(
                "SELECT key, step, measurement, target = 'Object' FROM measurement_info \
                 WHERE target <> 'Track' {not_profile} ORDER BY step, measurement, key"
            ))
            .map_err(err)?;
        let columns = stmt
            .query_map([], |row| {
                Ok(MeasurementColumn {
                    key: row.get(0)?,
                    step: row.get(1)?,
                    name: row.get(2)?,
                    per_object: row.get(3)?,
                })
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;
        Ok(columns)
    }

    /// The keys of the measurements whose values name another object
    /// (`measurement_info.related_objects`).
    pub(super) fn get_measurement_keys_naming_objects(
        &self,
    ) -> Result<Vec<String>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let has_column: bool = self
            .database
            .query_row(
                "SELECT count(*) > 0 FROM information_schema.columns \
                 WHERE table_name = 'measurement_info' AND column_name = 'related_objects'",
                [],
                |row| row.get(0),
            )
            .map_err(err)?;
        if !has_column {
            return Ok(Vec::new());
        }
        let mut stmt = self
            .database
            .prepare("SELECT key FROM measurement_info WHERE related_objects")
            .map_err(err)?;
        let keys = stmt
            .query_map([], |row| row.get(0))
            .map_err(err)?
            .collect::<Result<Vec<String>, _>>()
            .map_err(err)?;
        Ok(keys)
    }
}

/// The coloc groups of objects' `(own classes, partner classes)`: every
/// partner class joined with the others of its set and with the object's
/// own classes that are partners somewhere. Each group sorted, the groups
/// by their first class.
pub(super) fn coloc_groups_of(
    pairs: &[(Vec<ObjectClass>, Vec<ObjectClass>)],
) -> Vec<Vec<ObjectClass>> {
    let mut partner_classes: Vec<ObjectClass> = pairs
        .iter()
        .flat_map(|(_, set)| set.iter().copied())
        .collect();
    partner_classes.sort();
    partner_classes.dedup();
    // A small union-find over the partner classes.
    let mut parent: Vec<usize> = (0..partner_classes.len()).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let index = |class: &ObjectClass| partner_classes.binary_search(class).ok();
    for (own, set) in pairs {
        let members: Vec<usize> = set.iter().chain(own.iter()).filter_map(&index).collect();
        if let Some(first) = members.first() {
            for other in &members[1..] {
                let (a, b) = (root(&mut parent, *first), root(&mut parent, *other));
                parent[a.max(b)] = a.min(b);
            }
        }
    }
    let mut groups: Vec<Vec<ObjectClass>> = Vec::new();
    let mut group_of_root: std::collections::HashMap<usize, usize> = Default::default();
    for (i, class) in partner_classes.iter().enumerate() {
        let r = root(&mut parent, i);
        let g = *group_of_root.entry(r).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[g].push(*class);
    }
    groups
}
