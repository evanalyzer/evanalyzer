//! # command_ram
//!
//! What one command of a pipeline takes in RAM on a tile, for the user to
//! see before adding it (the step picker): per analysis thread (its
//! temporary planes, [`ImageAlgorithm::ram_per_tile_pixel`]) and once for
//! the job (an AI network and its weights,
//! [`ImageAlgorithm::shared_ram_bytes`]) - the numbers the RAM plan uses
//! ([`super::analysis`]).
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-04
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::algos::{ExecutionScope, ImageAlgorithm};
use crate::job::algos_from_config::into_algorithm;
use evanalyzer_cfg::settings::pipeline_command::PipelineCommand;

/// A command's RAM on a tile beyond the analysis' own buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CommandRam {
    /// Taken by each analysis thread running it at once.
    pub per_thread_bytes: u64,
    /// Taken once for the whole job (its runs take turns: an AI inference).
    pub once_bytes: u64,
    /// Runs on the whole image, bounded per window or object - not per
    /// tile pixel (`tests::whole_image_ram`).
    pub whole_image: bool,
}

/// What `command` takes on a `tile_width x tile_height` tile on this
/// computer (its RAM decides an AI network's pieces). `None`: its settings
/// don't make a command.
pub fn command_ram(
    command: PipelineCommand,
    tile_width: usize,
    tile_height: usize,
) -> Option<CommandRam> {
    let algorithm = into_algorithm(command).ok()?;
    Some(ram_of(
        algorithm.as_ref(),
        tile_width,
        tile_height,
        super::SystemRam::now().total,
    ))
}

fn ram_of(
    algorithm: &dyn ImageAlgorithm,
    tile_width: usize,
    tile_height: usize,
    ram_total: u64,
) -> CommandRam {
    let pixels = (tile_width * tile_height) as f64;
    CommandRam {
        per_thread_bytes: (algorithm.ram_per_tile_pixel() * pixels).ceil() as u64,
        once_bytes: algorithm.shared_ram_bytes(tile_width, tile_height, ram_total),
        whole_image: algorithm.execution_scope() != ExecutionScope::Tile,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evanalyzer_cfg::settings::pipeline_command::default_command;

    /// Every command answers - each with what it declares to the planner.
    #[test]
    fn every_command_has_its_ram() {
        for command in (0..).map_while(default_command) {
            let Some(ram) = command_ram(command.clone(), 4096, 4096) else {
                continue;
            };
            let algorithm = into_algorithm(command).unwrap();
            assert_eq!(
                ram.per_thread_bytes,
                (algorithm.ram_per_tile_pixel() * 4096.0 * 4096.0).ceil() as u64,
                "{}",
                algorithm.name()
            );
        }
    }

    /// Hessian: its declared 28 bytes per tile pixel; a U-Net: its network
    /// once, nothing per thread beyond what it declares.
    #[test]
    fn a_filter_takes_ram_per_thread_an_ai_network_once() {
        let hessian =
            command_ram(PipelineCommand::Hessian(Default::default()), 4096, 4096).unwrap();
        assert_eq!(hessian.per_thread_bytes, 28 * 4096 * 4096);
        assert_eq!(hessian.once_bytes, 0);
        let unet = command_ram(PipelineCommand::UNet(Default::default()), 4096, 4096).unwrap();
        assert!(unet.once_bytes > 1 << 30, "{unet:?}");
        assert!(!unet.whole_image);
    }
}
