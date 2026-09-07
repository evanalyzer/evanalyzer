use std::path::PathBuf;

use evanalyzer_cfg::core_types::{InternalErrors, ObjectClass};

pub struct ResultsGenerator {
    database_file: PathBuf,
}

#[derive(Clone)]
pub enum View {
    List,
    Heatmap,
}

#[derive(Clone)]
pub enum Aggregation {
    Avg,
    Min,
    Max,
    Stddev,
    Count,
}

#[derive(Clone)]
pub enum Column {
    ObjectId,
    ObjectClass,
    AreaSizePx,
    AreaSizeNm,
    Intensity(i32),
    PerimeterPx,
    PerimeterNm,
    Circularity,
    Solidity,
    Eccentricity,
}

#[derive(Clone)]
pub struct PlaneFilter {
    pub z_stack: u32,
    pub t_stack: u32,
}

#[derive(Clone)]
pub struct Pagination {
    pub limit: i32,
    pub offset: i32,
}

#[derive(Clone)]
pub struct GroupFilter {
    pub plane: PlaneFilter,
    pub grouping_regex: String,
    pub aggregation: Aggregation,
    pub object_class: ObjectClass,
    pub column: Column,
}

#[derive(Clone)]
pub struct ListFilter {
    pub plane: PlaneFilter,
    pub images: Option<Vec<String>>,
    pub object_classes: Option<Vec<ObjectClass>>,
    pub columns: Vec<Column>,
    pub page: Pagination,
}

pub enum Cell {
    String(String),
    Float(f32),
    Integer(i32),
    /// Object class with color
    Class((String, u32)),
}

pub struct DatabaseResult {
    pub column_names: Vec<String>,
    pub row_names: Vec<String>,
    /// One row with its colums
    pub rows: Vec<Vec<Cell>>,
}

impl ResultsGenerator {
    pub fn open_database(&self, database_file: PathBuf) {}

    pub fn get_list(
        &self,
        filter: &ListFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        Err(("not implemented").into())
    }
    pub fn get_group_by_well(
        &self,
        filter: &GroupFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        Err(("not implemented").into())
    }
    pub fn get_group_by_plate(
        &self,
        filter: &GroupFilter,
        view: &View,
    ) -> Result<DatabaseResult, InternalErrors> {
        Err(("not implemented").into())
    }
    pub fn get_coloc_objects(&self) {}

    pub fn get_histogram(&self) {}
    pub fn get_scatter(&self) {}
    pub fn get_heatmap(&self) {}
}
