//! # model_file
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-26
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! What an AI step's model file is, read from the start of the file (its
//! header, never the weights): for the card below the file field, and a
//! warning when the file doesn't fit the step - before anything runs.

use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;
use evanalyzer_nn::model::{cellpose_sam, stardist, unet};
use evanalyzer_nn::weights::{ModelMetadata, parse_header};
use std::path::Path;

/// Bytes read from a model file's start for its card: enough for the
/// headers of the converted models (Cellpose-SAM's is ~60 KB); a larger one
/// is read again in full ([`header_size`]).
pub const FIRST_READ: usize = 256 * 1024;

pub use evanalyzer_nn::weights::header_size;

/// The kinds of model the AI segmentation steps run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelKind {
    UNet,
    StarDist,
    CellposeSam,
}

impl ModelKind {
    const ALL: [Self; 3] = [Self::UNet, Self::StarDist, Self::CellposeSam];

    /// The kind of a model file's architecture (its `evanalyzer_arch`).
    pub fn of_arch(arch: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.arch() == arch)
    }

    fn arch(self) -> &'static str {
        match self {
            Self::UNet => unet::ARCH,
            Self::StarDist => stardist::ARCH,
            Self::CellposeSam => cellpose_sam::ARCH,
        }
    }

    /// As the card names it.
    pub fn label(self) -> &'static str {
        match self {
            Self::UNet => "U-Net",
            Self::StarDist => "StarDist 2D",
            Self::CellposeSam => "Cellpose-SAM",
        }
    }

    /// The step running it, as the step list names it.
    pub fn step(self) -> String {
        let command = match self {
            Self::UNet => PipelineCommand::UNet(Default::default()),
            Self::StarDist => PipelineCommand::Stardist(Default::default()),
            Self::CellposeSam => PipelineCommand::Cellpose(Default::default()),
        };
        command.name().to_string()
    }

    /// The kind the step called `step` runs; `None` for any other step.
    pub fn of_step(step: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.step() == step)
    }
}

/// What the card below the file field shows.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelCard {
    /// "U-Net".
    pub kind: &'static str,
    /// The model's name, else the file's name.
    pub name: String,
    /// What it is for; empty when the file doesn't say.
    pub description: String,
    /// "CC-BY-4.0 · https://…": licence and source, what the file has of them.
    pub details: String,
}

/// Why a step's model file doesn't fit: the box below the file field, and
/// the run error.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelProblem {
    /// The box's heading, in bold: "U-Net", "Not an EVAnalyzer model".
    pub heading: String,
    /// After the heading, dimmed: "· not for this step"; may be empty.
    pub aside: String,
    /// The box's text below the heading.
    pub text: String,
    /// All of it in one sentence, for a run error.
    pub message: String,
}

/// What a step's model file is, for the step.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelFileNote {
    /// A model this step runs.
    Fits(ModelCard),
    /// A model of another kind (a warning).
    WrongKind(ModelProblem),
    /// No model, or none this version runs, or the file can't be read (an
    /// error).
    Unusable(ModelProblem),
}

impl ModelFileNote {
    /// Why the file doesn't fit; `None` when it does.
    pub fn problem(&self) -> Option<&ModelProblem> {
        match self {
            Self::Fits(_) => None,
            Self::WrongKind(problem) | Self::Unusable(problem) => Some(problem),
        }
    }
}

/// The start of the local file `path`: [`FIRST_READ`] bytes, or the whole
/// header if it is larger - or why it can't be read.
pub fn read_start(path: &Path) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let read = |len: usize| -> Result<Vec<u8>, String> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut data = Vec::new();
        file.take(len as u64)
            .read_to_end(&mut data)
            .map_err(|e| e.to_string())?;
        Ok(data)
    };
    let start = read(FIRST_READ)?;
    match header_size(&start) {
        Ok(size) if size > start.len() && start.len() == FIRST_READ => read(size),
        _ => Ok(start),
    }
}

/// What the file `path` of a step that runs `kind` models is, from the
/// start of the file (`start`: at least [`header_size`] bytes, or why it
/// couldn't be read).
pub fn model_file_note(
    kind: ModelKind,
    path: &Path,
    start: Result<&[u8], String>,
) -> ModelFileNote {
    let file = path
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let start = match start {
        Ok(start) => start,
        Err(why) => {
            return ModelFileNote::Unusable(ModelProblem {
                heading: "The model file can't be read".into(),
                aside: String::new(),
                message: format!("The model file can't be read: {why}"),
                text: why,
            });
        }
    };
    let Ok(metadata) = parse_header(start) else {
        const CONVERT: &str = "EVAnalyzer runs its own .evanet files - a model from elsewhere \
                               (e.g. a Cellpose .pt) has to be converted once.";
        return ModelFileNote::Unusable(ModelProblem {
            heading: "Not an EVAnalyzer model".into(),
            aside: String::new(),
            text: CONVERT.into(),
            message: format!("{file} is no EVAnalyzer model. {CONVERT}"),
        });
    };
    let Some(found) = ModelKind::of_arch(&metadata.arch) else {
        return ModelFileNote::Unusable(ModelProblem {
            heading: "A model this version doesn't run".into(),
            aside: String::new(),
            text: format!(
                "{file} is a “{}” model - this version of EVAnalyzer doesn't run it.",
                metadata.arch
            ),
            message: format!(
                "{file} is a model this version of EVAnalyzer doesn't run (“{}”).",
                metadata.arch
            ),
        });
    };
    let card = card(found, &file, &metadata);
    if found != kind {
        return ModelFileNote::WrongKind(ModelProblem {
            heading: found.label().into(),
            aside: "· not for this step".into(),
            text: format!(
                "This step runs {} models. {} is a {} - it runs in {}.",
                kind.label(),
                card.name,
                found.label(),
                found.step()
            ),
            message: format!(
                "{} is a {} model - this step runs {} models. Use the step {} for it.",
                card.name,
                found.label(),
                kind.label(),
                found.step()
            ),
        });
    }
    ModelFileNote::Fits(card)
}

fn card(kind: ModelKind, file: &str, metadata: &ModelMetadata) -> ModelCard {
    let info = |key: &str| {
        metadata
            .info
            .get(key)
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };
    let details: Vec<String> = ["license", "source"].into_iter().filter_map(info).collect();
    ModelCard {
        kind: kind.label(),
        name: info("name").unwrap_or_else(|| file.to_string()),
        description: info("description").unwrap_or_default(),
        details: details.join(" · "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_nn::model::unet::Preprocessing;
    use evanalyzer_nn::test_support::{
        stardist_file, tiny_stardist_config, tiny_unet_config, unet_file,
    };
    use evanalyzer_nn::weights::{NamedTensor, to_bytes};

    fn path(name: &str) -> &Path {
        Path::new(name)
    }

    /// A file of `arch` with `info` in its metadata (no tensors needed).
    fn file_with(arch: &str, info: &[(&str, &str)]) -> Vec<u8> {
        let metadata = ModelMetadata {
            arch: arch.into(),
            arch_version: 1,
            config: "{}".into(),
            preprocessing: None,
            info: info
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        };
        to_bytes(&Vec::<NamedTensor>::new(), &metadata).unwrap()
    }

    #[test]
    fn every_ai_step_has_its_kind() {
        for kind in ModelKind::ALL {
            assert_eq!(ModelKind::of_step(&kind.step()), Some(kind));
        }
        assert_eq!(
            ModelKind::of_step("AI UNet Segmentation"),
            Some(ModelKind::UNet)
        );
        assert_eq!(ModelKind::of_step("Threshold"), None);
    }

    #[test]
    fn a_fitting_model_gets_its_card() {
        let file = file_with(
            unet::ARCH,
            &[
                ("name", "NucleiSegmentationBoundaryModel"),
                ("description", "Nuclei in fluorescence"),
                ("license", "CC-BY-4.0"),
                ("source", "https://bioimage.io/#/?id=affable-shark"),
                ("citation", "not shown"),
            ],
        );
        assert_eq!(
            model_file_note(ModelKind::UNet, path("a/affable-shark.evanet"), Ok(&file)),
            ModelFileNote::Fits(ModelCard {
                kind: "U-Net",
                name: "NucleiSegmentationBoundaryModel".into(),
                description: "Nuclei in fluorescence".into(),
                details: "CC-BY-4.0 · https://bioimage.io/#/?id=affable-shark".into(),
            })
        );
    }

    /// Converted before name and description were written: the file's name.
    #[test]
    fn a_model_without_a_name_is_called_by_its_file() {
        let file = stardist_file(&tiny_stardist_config([2, 2]));
        let ModelFileNote::Fits(card) = model_file_note(
            ModelKind::StarDist,
            path("2D_versatile_fluo.evanet"),
            Ok(&file),
        ) else {
            panic!("fits")
        };
        assert_eq!(card.kind, "StarDist 2D");
        assert_eq!(card.name, "2D_versatile_fluo.evanet");
        assert_eq!(card.description, "");
    }

    #[test]
    fn a_model_of_another_kind_names_its_step() {
        let file = unet_file(tiny_unet_config(), Preprocessing::None);
        let ModelFileNote::WrongKind(problem) =
            model_file_note(ModelKind::StarDist, path("shark.evanet"), Ok(&file))
        else {
            panic!("a problem")
        };
        assert_eq!(
            problem,
            ModelProblem {
                heading: "U-Net".into(),
                aside: "· not for this step".into(),
                text: "This step runs StarDist 2D models. shark.evanet is a U-Net - \
                       it runs in AI UNet Segmentation."
                    .into(),
                message: "shark.evanet is a U-Net model - this step runs StarDist 2D models. \
                          Use the step AI UNet Segmentation for it."
                    .into(),
            }
        );
    }

    #[test]
    fn no_model_an_unknown_one_and_an_unreadable_file_are_problems() {
        let problem = |start: Result<&[u8], String>| match model_file_note(
            ModelKind::CellposeSam,
            path("cpsam.pt"),
            start,
        ) {
            ModelFileNote::Unusable(problem) => problem,
            fits => panic!("{fits:?}"),
        };
        let zip = problem(Ok(b"PK\x03\x04 a zip, not a model"));
        assert_eq!(zip.heading, "Not an EVAnalyzer model");
        assert!(
            zip.text
                .starts_with("EVAnalyzer runs its own .evanet files")
        );
        assert!(zip.message.starts_with("cpsam.pt is no EVAnalyzer model"));
        let yolo = problem(Ok(&file_with("yolo", &[])));
        assert!(yolo.text.contains("“yolo” model"), "{}", yolo.text);
        assert!(yolo.message.contains("doesn't run (“yolo”)"));
        let missing = problem(Err("not found".into()));
        assert_eq!(missing.heading, "The model file can't be read");
        assert_eq!(missing.text, "not found");
        assert!(missing.message.contains("can't be read: not found"));
    }

    /// A header larger than the first read is told by its first bytes.
    #[test]
    fn the_header_size_comes_from_the_first_bytes() {
        let file = file_with(unet::ARCH, &[("name", "x")]);
        let size = header_size(&file[..8]).unwrap();
        assert!(size <= file.len());
        assert!(parse_header(&file[..size]).is_ok());
        assert!(header_size(&file[..3]).is_err());
    }
}
