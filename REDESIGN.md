# UI redesign - overview

The new UI is built on **Ink**, the design system in `crates/gui/ui/ink/`
(colors per scheme, icons, components; its `README.md` has the rules that
keep the build small and lists the files). Design boards: the "Workspace
redesign" canvas (layout C, light and dark Ink), boards B to T.

## Done

- **Ink library:** buttons, fields (text, search, number, select), toggles,
  chips, badges, panels and heads, dialogs, menus, rows, progress, banner,
  settings layout (`settings.slint`), pickers (`picker.slint`), multi-select
  (`multi_select.slint`), form controls with the std-widgets API
  (`std_compat.slint`). `crates/gui/ui/components/` is gone.
- **Colors:** light, dark, dim (setting `color_scheme`, picked in the EVA
  menu - one segmented row).
- **Fonts:** IBM Plex Sans and Mono, bundled.
- **Icons:** one Phosphor set (white, tinted in code), licence in `icons/`.
- **Shell:** top bar, bottom bar, floating panels over the image, annotate
  bar.
- **Menus fit the window:** `InkMenuFrame` scrolls when the window is too
  short (EVA menu, pipeline switcher, every ⋯ menu); `Ink.window-height`.
- **One window:** Results is a mode of the main window.
- **Panels:** steps, pipeline switcher, problems, classes, objects (list or
  table), image, images (list and plate), image info, analyses.
- **Images › Plate:** wired - the images grouped into wells by the
  project's plate settings (`evanalyzer_app::images::plate_layout`, the same
  patterns as the results queries); a well's images below the plate.
- **Results:** List with images / classes / columns as check lists in
  place (All / None, search) and "N rows · page" (L); Charts with "A box
  per" class / image / well / plate row / column (`BoxplotFilter::group_by`)
  and a click on a box that shows its objects in the List (M); Plate's well
  card lists the well's images - a click opens one, its eye leaves it out
  of the results (H).
- **Objects as a table** (design K): at the bottom between the panels,
  columns ID, classes, area, circularity, the mean of each channel; the
  selected object's buttons in its footer. ⋯ › "Show as a table / list".
- **Tuning panel** with pins (settings, list entries, table rows).
- **Command palette** (Ctrl+K): actions, steps, settings ("Min threshold ·
  #3 Threshold · 240" opens the step), commands, images, classes.
- **Pipelines (J):** reorder by dragging ⋮⋮ (`move-pipeline-by`); with none
  shown the steps panel lists them to pick one; the shown one is deselected
  by a second click or "Show no pipeline".
- **Objects (K):** a class filter chip and a search (an id, `area > 50`,
  `circ < 0.6`, a class name; terms by commas) for list and table;
  "Columns" in the table's ⋯ (area, circularity, each channel's mean).
- **Histogram:** "Auto-scale new images" (a project setting; off: the
  project's scaling or the full range) and "Use for all images".
- **Kept with the project:** the Tuning panel's pins and which images were
  previewed / analysed (`ProjectSettings::editor`).
- **Keyboard:** Tab reaches the buttons, icon buttons, checkboxes and
  switches; Space/Enter press them; a focus ring shows only for the
  keyboard.
- **Brush and eraser:** a dragged stroke, as wide as the brush; the brush
  paints a new object of the chosen class, the eraser cuts the hand-drawn
  objects (an emptied one is deleted, the rest measured again).
- **Analysis while editing:** progress card; setting
  `edit_during_analysis` (default on); asks before switching project.
- **Navigator:** flat, clickable mini-map.
- **Dialogs, inside too:** AI training (label column, fixed field widths,
  caption sections), export (grouped File / Images & objects / Plate &
  wells / Filter), project and component settings (Ink settings layout),
  create component / update / expose, Add step (description pane), step
  settings (design Q: tabs, search, the card's rows), new from template,
  save as template, about, connect, shortcuts. Titles in sentence case.
- **Step lines:** a click selects one; Enter settings, Ctrl / disable, F9
  breakpoint, Ctrl D duplicate, Alt ↑/↓ move (new: `move-step`), Del
  remove - also in its ⋯ menu.
- **Number fields:** drag sideways over the number (a step per 4 px),
  Ctrl + wheel, ↑↓. Number settings use it too (no range when min = max).
- **Class pickers:** "+ New class…" in the steps' object class lists and in
  the annotate bar ("Paint as" is a real picker now) - the new class is
  picked at once (`ClassificationSettingsState.last-created-id`).
- **Brush:** paints into the selected hand-drawn object (else a new one);
  every stroke and drawn shape is an undo step of its own.
- **Images (G):** head with count · folder, Plate / List, folder, refresh;
  wells that fill the width with a state bar (analysed / preview / new),
  rows with size, channels and state; image info in short form plus "All
  metadata ›". Previewed images (this session) and the images of the
  analysis open in Results are tracked (`UiState::image_states`).
- **Object card (F):** the selected object next to it on the image -
  classes, area, circularity, mean per channel, change class, delete.
- **Before / after (C):** a divider to drag (top bar button), hold Space
  for the image without the preview's objects. Can be switched off
  (Settings › Viewport › Before / after); costs nothing while off.
- **Tuning (E):** tabs "All pinned · N" and one per step; the ⌖ in a step's
  line pins all its settings (again: unpins).
- **Dialog heads:** title and ✕ centred; one line under the head (no
  double line); Settings' search in its head. AI training as on board R:
  Pixel / Object, the classes with their counts, the training in short,
  "Uses the annotations of N images", "More settings ›" for the rest.
- **Dialogs where the boards put them (N, O, P, Q, R), none modal:** no
  backdrop - the image, panels and shortcuts stay usable; all movable by
  their head; Esc closes the open one also after a click into the window
  (`DialogEscape`). Settings: a 440 px sheet on the right, one column,
  sections opened and closed by their head, search on top (O). AI training:
  440 px sheet (R). Export: a 380 px card below its button - what, format,
  classes, folder; "More options ›" for the rest (R). Step settings, Add
  step, component settings (tabs General / Parameters), create / update
  component, pipeline edit: next to the steps panel (P, Q). Class settings
  (narrow, 7 colors a row), delete / auto-add classes, set class, delete
  objects: left of the right-hand panels (N, P). Run analysis below Run
  with "N pipelines · M images", connect at the bottom right (above "This computer"), unsaved
  changes centred a third down (N). Places from `Places` (api/layout.slint).
  Only the analysis' waiting dialog ("Edit while an analysis runs" off)
  still blocks.
- **Build:** generated code 34.9 MB → 32.78 MB → 34.0 MB → 34.98 MB
  (Results views, objects filter, dialogs as on the boards; before that:
  dialogs movable/non-modal ~0.33 MB - `InkDialog` is copied into each of
  its 31 uses -, object card ~0.2 MB, images panel ~0.15 MB, the rest
  ~0.5 MB).

## Open

### From the design boards, not built yet

- [ ] **Own title bar:** no system frame - window move, minimise, maximise,
      close and resize at the edges via winit (differs per platform).
- [ ] **Before / after (C), the full version:** compare two previews (a
      setting before and after a change, "Before · 200 / After · 240") -
      today it compares the image without and with the objects.
- [ ] **Object card (F):** "Coloc with" (not in the object data yet).
- [ ] **Images (G):** "+ Add images" (images are the folder's content
      today; there is no action for single images).
- [ ] **Dialogs, not yet:** more than one open at a time (`active_dialog`
      is one value); class settings exactly at the clicked class row (they
      open at the top of the classes panel).
- [ ] **Charts (M):** a click on a scatter point (points carry no object
      yet) - boxes do.
- [ ] **List (L):** "Rows 1–200 of N" (the total isn't known; it shows the
      page's rows).
- [ ] **About:** a redesign (not in this round's scope).
- [ ] **Export (R):** "Images with objects drawn".

### Follow-ups of this round

- [ ] **Images › Plate:** a custom grouping regex is compiled by the Rust
      `regex` crate here and by DuckDB (RE2) in the results - the same for
      the usual patterns, but not for every one.
- [ ] **Keyboard:** list rows, segmented controls, selects and the step
      lines don't take Tab focus yet.

### Findings

- **One dialog at a time:** dialogs are non-modal now, but there is still
  one `active_dialog` - opening another replaces the open one.
- **Slint 1.18:** a `let` string local used twice in a callback generates
  Rust that doesn't compile (use a property); a PopupWindow sizes itself
  from its child's min/max constraints, not from a fixed width/height on
  the child (that made the EVA menu deaf to clicks).
- **Size of the generated code:** the objects table cost ~0.4 MB (its
  columns exist once: list and table are one component), the keyboard
  focus ~0.1 MB (the face's border is the ring - a separate ring element
  cost 0.2 MB). A component used once is inlined and its unused branches
  dropped; a second use brings them back.
- **Fixed on the way:** `InkChip` cut its label (its width left out the
  layout's gaps); the "No … match" empty states of the Add step and
  template dialogs were pulled apart by a stretching layout; "Analazer" in
  About.

### Checks after each step

- Measure the generated code (it is what the build time follows).
- Screenshot the changed screens in light and dark.
- Run the GUI tests.

### New feature

- [x] Allow to disable Histogram Auto Scale of the image when a new one is opend should be stored in project settings
- [x] Allow to apply the histogram scale settings to all images in the project
- [x] If no pipeline is selectd show the list of pipelines obove the "New Pipeline" button
- [x] Allow to deselect a pipeline