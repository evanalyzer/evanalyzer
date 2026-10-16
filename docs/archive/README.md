# Archive

Code no longer part of the app, kept for reference.

- `bio_runner.slint` — "BioRunner", the little game the old analysis
  dialog showed while an analysis ran. Removed with the redesign (the
  analysis now runs while the window stays usable). It uses `Ink`'s
  colors and its own `GameColors`; its imports point at
  `crates/gui/ui/ink/`, so move it back there to use it again.
