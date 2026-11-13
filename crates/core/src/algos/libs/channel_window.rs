//! # Channel windows
//!
//! Reads a rectangle of a channel out of its tiles in the cache - for
//! whole-image commands that work window by window around objects (Grow
//! Regions) and never hold the whole image. Trace Filaments and Propagation
//! still have their own copies.
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-13
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.

use crate::pipeline::pipeline_cache::{
    CacheAddress, GlobalPipelineCache, channel_pixel_slice, sample_channel_pixel,
};
use evanalyzer_cfg::core_types::{ImageTile, InternalErrors};
use std::collections::HashMap;

/// `[x0, y0, x1, y1)` in image pixels.
pub(crate) type Rect = [usize; 4];

/// A channel's tiles in the cache and the part of the image they cover (the
/// whole image in a run, the visible part in a preview) - no pixels: a
/// window reads its own, one tile at a time.
pub(crate) struct ChannelWindow {
    command: &'static str,
    channel: i32,
    tiles: Vec<ImageTile>,
    /// The tiles by the grid cells they cover (cell: the largest tile's
    /// size), so a small window doesn't look at every tile of a slide.
    cells: HashMap<(usize, usize), Vec<u32>>,
    cell: (usize, usize),
    pub extent: Rect,
}

impl ChannelWindow {
    /// The tiles of `channel`; `command` names the command in the error
    /// when the image has no such channel.
    pub fn of(
        cache: &GlobalPipelineCache,
        channel: i32,
        command: &'static str,
    ) -> Result<Self, InternalErrors> {
        let tiles: Vec<ImageTile> = cache
            .channel_keys()
            .filter(|(c, _)| *c == channel)
            .map(|(_, tile)| tile)
            .collect();
        if tiles.is_empty() {
            return Err(no_channel(command, channel));
        }
        let extent = [
            tiles.iter().map(|t| t.offset_x).min().unwrap_or(0),
            tiles.iter().map(|t| t.offset_y).min().unwrap_or(0),
            tiles
                .iter()
                .map(|t| t.offset_x + t.width)
                .max()
                .unwrap_or(0),
            tiles
                .iter()
                .map(|t| t.offset_y + t.height)
                .max()
                .unwrap_or(0),
        ];
        let cell = (
            tiles.iter().map(|t| t.width).max().unwrap_or(1).max(1),
            tiles.iter().map(|t| t.height).max().unwrap_or(1).max(1),
        );
        let mut cells: HashMap<(usize, usize), Vec<u32>> = HashMap::new();
        for (i, t) in tiles.iter().enumerate() {
            if t.width == 0 || t.height == 0 {
                continue;
            }
            for cy in t.offset_y / cell.1..=(t.offset_y + t.height - 1) / cell.1 {
                for cx in t.offset_x / cell.0..=(t.offset_x + t.width - 1) / cell.0 {
                    cells.entry((cx, cy)).or_default().push(i as u32);
                }
            }
        }
        Ok(Self {
            command,
            channel,
            tiles,
            cells,
            cell,
            extent,
        })
    }

    /// The indices of the tiles meeting `rect`, in order.
    fn tiles_in(&self, [x0, y0, x1, y1]: Rect) -> Vec<usize> {
        let mut found = Vec::new();
        if x0 >= x1 || y0 >= y1 {
            return found;
        }
        for cy in y0 / self.cell.1..=(y1 - 1) / self.cell.1 {
            for cx in x0 / self.cell.0..=(x1 - 1) / self.cell.0 {
                if let Some(ids) = self.cells.get(&(cx, cy)) {
                    found.extend(ids.iter().map(|&i| i as usize));
                }
            }
        }
        found.sort_unstable();
        found.dedup();
        found
    }

    /// The normalized (0..1) values of `rect`, row by row (NaN where no
    /// tile was read or the pixel isn't a number).
    pub fn read(
        &self,
        cache: &GlobalPipelineCache,
        rect: Rect,
    ) -> Result<Vec<f32>, InternalErrors> {
        let [x0, y0, x1, y1] = rect;
        let width = x1.saturating_sub(x0);
        let mut values = vec![f32::NAN; width * y1.saturating_sub(y0)];
        for i in self.tiles_in(rect) {
            let tile = &self.tiles[i];
            let (tx0, ty0) = (tile.offset_x.max(x0), tile.offset_y.max(y0));
            let (tx1, ty1) = (
                (tile.offset_x + tile.width).min(x1),
                (tile.offset_y + tile.height).min(y1),
            );
            if tx0 >= tx1 || ty0 >= ty1 {
                continue;
            }
            let container = cache
                .image_cache
                .get(&CacheAddress::Channel((self.channel, *tile)))
                .ok_or_else(|| no_channel(self.command, self.channel))?;
            let rgb = matches!(*container, crate::image::ImageContainer::F32Rgb(_));
            let slice = channel_pixel_slice(&container).unwrap_or_default();
            let len = if rgb { slice.len() / 3 } else { slice.len() };
            for y in ty0..ty1 {
                for x in tx0..tx1 {
                    let sample = (y - tile.offset_y) * tile.width + (x - tile.offset_x);
                    if sample >= len {
                        break;
                    }
                    values[(y - y0) * width + (x - x0)] = sample_channel_pixel(rgb, slice, sample);
                }
            }
        }
        Ok(values)
    }
}

fn no_channel(command: &str, channel: i32) -> InternalErrors {
    InternalErrors::InvalidArgument(format!("{command}: the image has no channel {channel}"))
}
