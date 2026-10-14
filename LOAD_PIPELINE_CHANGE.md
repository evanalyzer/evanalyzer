# Pipelines without an initial load

Status 2026-10-14: implemented (TODO_FOR_GROUP_FINISH.md, step 9) - the
source "None", the command Load Image Channel, the focus and the pre-start
check from load steps (also inside groups), clear error messages.
Before: concept (2026-10-13). Background: command groups
(VISION_GROUP.md) need to load several image channels, e.g. "Cell
reconstruction" detects cells in channel 0 and the cell area in channel 1.

## Today: Scratchpad + "Image Cache: Load channel" already does it

A pipeline with **Scratchpad** as its image source and **Image Cache (mode
Load, channel X)** as its first step behaves like a pipeline without an
initial load: it starts with an empty scratchpad image, and the Load step
replaces it with the channel. The result is identical.

This works because every tile loads **all** channels into the cache
(`read_image_tile_combined(..., None, ...)` in `job_executor.rs`), so any
channel is available at any step.

## What differs - and what to fix before this becomes the official way

| Difference | Today | Needed |
|---|---|---|
| **Wasted memory** | The scratchpad start allocates a zeroed image **for every tile** (`get_image_from_cache`, `CacheAddress::Scratchpad`): with 4096×4096 tiles, 64 MB (192 MB for RGB) per tile and worker, only to throw it away right after | A source "None": start without an image; the first Load sets it |
| **Viewer focus** | Selecting a pipeline shows only its channel, but for Scratchpad all channels (`focus_pipeline` / `refresh_pipeline_focus` in `project_ext.rs`) | Derive the focus channels from the pipeline's Load steps |
| **Pre-start check** | `check_pipeline_channels` (`job_generator.rs`) only checks `image_source`; a Load from channel 9 slips through | Check Load steps too, including those inside groups |
| **Error message** | A missing channel ends with `CacheMiss("")` (`image_cache.rs`), no text | The same clear message as at the pipeline start ("reads image channel 9, the image has 0-2") |

## Consequence for groups

With these four points fixed, a group can contain its own "Load channel"
steps, and their channels become **group parameters** (channel roles),
picked when the group is inserted. The engine needs no other change.
