# Tracking – suggested order

Status 2026-10-13. Background (use cases, what belongs in the viewer and
what to export): VISION.md, sections 6 and 7.

Rule of thumb: **in the app, what's needed to check the tracking and to
compare conditions. Model fitting and statistics (MSD, diffusion, survival
curves, time-series clustering) go to export.**

1. **SimpleTracker writes track IDs, plus the frame interval**
   - `SimpleTracker`: links objects frame to frame per image and z-plane
     (global greedy nearest neighbour within a maximum distance, nm or px);
     settings for class, maximum distance and unit. *(done: 400,000
     objects in 0.7 s end to end, about +100 MB while running)*
   - `DatasetStore::set_track_ids` / `max_track_id_except`, writing the
     `track_id` column. *(done)*
   - Frame interval [s]: read from the image metadata; shown and editable
     for the project in the image info panel (TIME, below CALIBRATION), like
     a channel colour. Calculations fall back to 1 s. *(in progress: not yet
     used by the analysis)*
2. **Track overlay in the viewer, plus highlighting a whole track**
   - Trajectories with a tail of the last N frames, coloured by track.
   - Clicking an object highlights its whole track; step through its frames.
   - Users trust the tracking only once they can see it.
3. **Track table** (one row per track)
   - Image, class, first frame, duration, path length, net displacement,
     mean speed, straightness, mean intensity.
   - Computed in SQL with `GROUP BY track_id`; with CSV/XLSX export.
4. **Histograms and plate heatmap on track measurements**
   - Reuse the existing charts and the matrix view, e.g. mean speed per well.
5. **Curves over time**
   - A measurement over time, one line per track, plus the mean.
   - Capped (say 200 tracks) to stay fast.
6. **Later**
   - Spider plot (all tracks shifted to start at 0/0).
   - Lineage tree, once divisions are detected.
   - TrackMate XML / Cell Tracking Challenge export, later GEFF.
