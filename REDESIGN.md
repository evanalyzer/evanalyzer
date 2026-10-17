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
- **Compare (C, changed):** the breakpoint image and the original side
  by side - a divider to drag (top bar button, only with a breakpoint
  image), a click on its handle swaps the sides, near an edge it snaps
  there (only one side), hold Space for the whole original. The original
  is the image, the breakpoint image a second render over it (own
  buffers, only while comparing); dragging the divider only moves the
  cut. It replaces the "Breakpoint image" switch (and its setting).
  Histogram: "Original | Breakpoint" while comparing - the breakpoint
  image has its own window (auto-adjusted for a new breakpoint, not
  saved); the footer shows its value next to the channels'
  ("Breakpoint: 128", "label 3").
- **Breakpoints in the gutter:** the step's number at the far left of its
  line is the breakpoint - a click: red stop, orange snapshot, off (like
  a source editor's line numbers); the ● on the right is gone.
- **Histogram:** values in the image's own units (2^bits, like the
  footer); only the selected channel's curve (each channel counts on its
  own axis - the others' curves didn't line up with the black / white
  lines).
- **Histogram rework:** log counts (default; "Log" switches), the black
  and white values at their lines, the axis' ends below, 120 px; the four
  fields behind "Values ›"; double click: auto-scale; "Auto-scale" (the
  channel) and "All channels".
- **User settings (Settings › Viewport):** "Auto-scale new images" (on;
  every channel of an image opened the first time - replaces the project
  setting) and "Color channels without a wavelength" (off: white; on: a
  wavelength is assigned and kept in the project, like the scaling).
- **Results frame (H, L, M):** List / Plate / Charts, the analysis (a
  menu of all) and the plate's breadcrumb in the top bar, Export and Z/T
  on its right; the views as panels - settings left (280 px), content
  beside, Plate's cell card right (300 px, always there so the grid
  doesn't move); no analysis open: the analyses to pick one; List's
  "Coloc details" and "Transpose" as switches.
- **Results, this round:** a clicked List row stays highlighted; a click
  on an object (List row, heatmap tile, scatter point - points carry
  their object now, protocol 15) opens it on its image with a bar there
  ("RESULTS · Row 12 of 200 ‹ › Back to results ✕") instead of the
  results covering the image; the breadcrumb only inside a well
  ("Plate › Well C3 › image", names elided); the chart kinds as tiles;
  the status bar "81 images · 1 disabled"; `InkSegmented` only shows
  the caller's state (a click no longer broke its binding - the tabs
  didn't follow "All results").
- **Images › Plate:** a plate bigger than 8 x 12 in sections of 8 x 12
  (a picker above it: "A1–H12 · A13–H24 / …").
- **Class settings** open at the clicked class row.
- **Histogram curve aligned:** the curve's Path was fitted with Slint's
  default `contain` - squeezed into a square in the middle of the box,
  off the black / white lines. `fit: fill` (also the disabled tile's
  cross). The cause of the "lines don't match the histogram" report.
- **Keyboard:** Tab reaches segmented controls (← →), selects (Space /
  Enter / ↓ open), list rows (Space / Enter, menu key) and step lines
  (focus selects, Space / Enter opens); the ring only for the keyboard.
- **Object card:** "Coloc with" - each class and how many of its objects.
- **Export:** "Images with objects drawn (PNG)" - each image at the
  pyramid level up to 4096 px, the channels in their colors (auto-scaled),
  every object's box in its class color (the database keeps boxes, not
  outlines). Protocol 17.
- **EVA menu:** "Open images folder…".
- **Number fields:** a setting with only a lower bound (Min Size >= 1) or
  no range stepped nothing (clamped into an empty range) - fixed in the
  field.
- **About (board Y):** dark hero (version, platform, stable / pre-
  release), tabs About (the people turning around the mark, the
  creators, AGPL-3.0), the contributors, System (where the work runs:
  cores, RAM, CUDA, OS) and Licenses (searchable, EVAnalyzer's own first);
  only facts the app knows (no live load charts, update check, build id).
- **Series:** the top bar's image chip shows "Series 2 of 5 ▾" (a menu with
  the sizes); an image with several series and none chosen for the
  project asks once which (over the image, "Decide later").
- **Results, round 3:** a click selects (List row, heatmap tile); "Show on
  image ↗" or a double click shows it on its image; Image is the first
  column (`Column` order), defaults Image · Object ID · Class · Area px ·
  Area nm · Circularity; object ids by their last block; space between
  the columns; the class's color as a dot; Charts › Display: histogram
  bins and log counts, the scatter's points drawn.
- **Histogram axis:** auto-scale's axis reaches the image's real values
  (it stopped 1 % past the 99.5 % white point - the brightest pixels,
  often the signal, were cut off the histogram).
- **Series:** a project series the image lacks shows its first series
  (an analysis still refuses it); an image that can't be read empties
  the viewer instead of keeping the last one.
- **Fixed:** auto preview didn't run for another image or series; a
  preview closed the open dialog (its pill used the dialog slot) and was
  skipped while one was open; a new / opened project kept the last one's
  steps and image; the steps panel showed an empty "pipeline 0" instead
  of the list on first load.
- **Add step stays open:** each added step goes in after the one added
  before, the footer says "✓ Added Threshold as step 3", Cancel turns
  Done. Adding from the command palette closes it.
- **User settings (Viewport):** "Objects as a table" (what an image opens
  with, default list - ⋯ still switches) and "Object card" (default off).
- **Objects table:** the class chip and the search in its head (K).
- **Component settings (P):** General / Parameters in the head; the
  parameters as a list by section with the chosen one's form beside it
  (name, section, help, default, "Sets" as chips, "Also set…", ↑ ↓,
  remove); Cancel (✕, Esc) undoes everything changed while it was open,
  Apply keeps it - "Changes apply to every use of the component".
- **Step settings (Q):** the step in the head ("Threshold · step 3 ·
  settings"), 430 px, as high as its settings.
- **Fields keep the focus:** the steps' models are updated in place after
  a change or a preview (`step_model_patch`), not rebuilt - a field being
  typed into or scrubbed isn't recreated.
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
- **Results, split with the image (Z):** "Full | Split with image" in the
  top bar (List and Plate; Charts greyed out). The results left, the one
  image viewer in a pane right, a divider between (dragged; width, on/off
  and Follow selection kept as user settings, `AppSettings::results_split`).
  The analysis' objects are drawn on the image as ellipses in their boxes
  (`editor/results_overlay.rs`, instead of the project's objects), the
  others dimmed while one is selected; a ring, faint lines and a card (class,
  ID, area, mean of the channel picked in the head, position) mark it.
  A List row (click, ↑ ↓, ‹ ›) flies the view there (0.4 s, eased; jumps
  with Settings › Viewport › Reduce motion; a new image opens at the object
  - `ViewportCache::focus`); a click on an object selects its row. Esc lets
  go. Plate: a well shows its images, a filmstrip under the image (thumbnails
  from the smallest pyramid level, kept at 184 × 112), "Color by Class |
  Value" with the heatmap's scale (Value only for a per-object property).
  In the split, the list's and the plate's settings are fields in the
  table's / grid's head, the well card is under the plate. Status bar:
  "Split view · list and image". Code: `results/results_split.slint`,
  `results_state_controller/split.rs`. Generated code +1.0 MB
  (36.81 → 37.83 MB).
- **Build:** generated code 34.9 MB → 32.78 MB → 34.0 MB → 34.98 MB
  (Results views, objects filter, dialogs as on the boards; before that:
  dialogs movable/non-modal ~0.33 MB - `InkDialog` is copied into each of
  its 31 uses -, object card ~0.2 MB, images panel ~0.15 MB, the rest
  ~0.5 MB).

## Open

### From the design boards, not built yet

- [ ] **Own title bar:** no system frame - window move, minimise, maximise,
      close and resize at the edges via winit (differs per platform).
- [ ] **Charts (M):** the small "same values" previews (histogram,
      scatter) under the chart - left out on purpose: two more queries on
      every change.
- [ ] **List (L):** "Rows 1–200 of N" - not now (TODO: a COUNT per
      change costs speed for little).

### Follow-ups of this round

- Split view: the channel picker sets only the card's mean, not which
  channels the image shows (that stays the project's channel settings);
  List paging still pages the table only - ↑ ↓ stop at the page's ends.

- Later (TODO): one regex engine for the plate grouping, "+ Add images",
  compare two previews.

### Findings

- **One dialog at a time:** dialogs are non-modal now, but there is still
  one `active_dialog` - opening another replaces the open one. Kept on
  purpose (no use case for more).
- **Slint 1.18, FocusScope:** one without focus takes a mouse press to get
  focus - a FocusScope over the window (`DialogShortcuts`) ate the press
  that should pan the image. `focus-on-click: false` on such layers.
- **Hover-shown buttons** go inside the row's TouchArea: as its siblings,
  hovering a button un-hovers the row, the button hides, and it flickers.
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