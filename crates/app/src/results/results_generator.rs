use std::path::PathBuf;

use duckdb::Connection;
use evanalyzer_cfg::{
    core_types::{InternalErrors, ObjectClass},
    settings::classification_settings::Class,
};

pub struct ResultsGenerator {
    database: Connection,
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

pub struct ColumnEntry {
    pub display_name: String,
    pub name: String,
    pub group: String,
}

pub struct ImageEntry {
    pub name: String,
    pub rel_path: PathBuf,
    pub disabled: bool,
}

pub struct DatabaseResult {
    pub column_names: Vec<String>,
    pub row_names: Vec<String>,
    /// One row with its colums
    pub rows: Vec<Vec<Cell>>,
}

impl ResultsGenerator {
    pub fn open_database(path: PathBuf) -> Result<Self, InternalErrors> {
        let to_io_err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let database = Connection::open(path).map_err(to_io_err)?;
        Ok(Self { database })
    }

    pub fn get_object_classes(&self) -> Result<Vec<Class>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut stmt = self
            .database
            .prepare("SELECT class_id, name, color FROM classes ORDER BY class_id")
            .map_err(err)?;
        stmt.query_map([], |row| {
            let class_id: u32 = row.get(0)?;
            let color: Option<u32> = row.get(2)?;
            Ok(Class {
                id: ObjectClass::Valid(class_id),
                name: row.get(1)?,
                color: color.unwrap_or(0),
                notes: String::new(),
            })
        })
        .map_err(err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)
    }

    pub fn get_available_columns(&self) -> Result<Vec<ColumnEntry>, InternalErrors> {
        let ret = vec![
            ColumnEntry {
                display_name: "Object ID".into(),
                name: "object_id".into(),
                group: "General".into(),
            },
            ColumnEntry {
                display_name: "Class".into(),
                name: "object_class_name".into(),
                group: "General".into(),
            },
            ColumnEntry {
                display_name: "Area [px]".into(),
                name: "area_px".into(),
                group: "Geometry".into(),
            },
            ColumnEntry {
                display_name: "Area [nm²]".into(),
                name: "area_nm2".into(),
                group: "Geometry".into(),
            },
            ColumnEntry {
                display_name: "Perimeter [px]".into(),
                name: "perimeter_px".into(),
                group: "Geometry".into(),
            },
            ColumnEntry {
                display_name: "Perimeter [nm]".into(),
                name: "perimeter_nm".into(),
                group: "Geometry".into(),
            },
            ColumnEntry {
                display_name: "Circularity".into(),
                name: "circularity".into(),
                group: "Shape".into(),
            },
            ColumnEntry {
                display_name: "Solidity".into(),
                name: "solidity".into(),
                group: "Shape".into(),
            },
            ColumnEntry {
                display_name: "Eccentricity".into(),
                name: "eccentricity".into(),
                group: "Shape".into(),
            },
            ColumnEntry {
                display_name: "Coloc count".into(),
                name: "n_colocalized".into(),
                group: "Coloc".into(),
            },
            ColumnEntry {
                display_name: "Avg Intensity".into(),
                name: "mean_raw".into(),
                group: "intensity".into(),
            },
            ColumnEntry {
                display_name: "Sum Intensity".into(),
                name: "sum_raw".into(),
                group: "intensity".into(),
            },
            ColumnEntry {
                display_name: "Min Intensity".into(),
                name: "min_raw".into(),
                group: "intensity".into(),
            },
            ColumnEntry {
                display_name: "Max Intensity".into(),
                name: "max_raw".into(),
                group: "intensity".into(),
            },
        ];

        Ok(ret)
    }

    pub fn get_images(&self) -> Result<Vec<ImageEntry>, InternalErrors> {
        let err = |e: duckdb::Error| InternalErrors::Io(e.to_string());
        let mut stmt = self
            .database
            .prepare("SELECT image_name, image_rel_path, disabled FROM images ORDER BY image_name")
            .map_err(err)?;
        let map = stmt
            .query_map([], |row| {
                Ok(ImageEntry {
                    name: row.get(0)?,
                    rel_path: PathBuf::from(row.get::<_, String>(1)?),
                    disabled: row.get(2)?,
                })
            })
            .map_err(err)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(err);
        map
    }

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
