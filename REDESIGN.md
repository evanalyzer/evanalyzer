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
- **Results:** List, Plate, Charts with their settings columns; the Plate's
  well card lists the well's images (a click opens one).
- **Objects as a table** (design K): at the bottom between the panels,
  columns ID, classes, area, circularity, the mean of each channel; the
  selected object's buttons in its footer. ⋯ › "Show as a table / list".
- **Tuning panel** with pins (settings, list entries, table rows).
- **Command palette** (Ctrl+K): actions, steps, commands, images, classes.
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
- **No modal dialogs:** dialogs have no backdrop - the image, the panels
  and the shortcuts stay usable. Large ones are sheets on the right
  (settings, AI training, export, step / component settings, Add step,
  templates, script editor), small ones float under the top bar; all can be
  moved by their head. Only the analysis' waiting dialog ("Edit while an
  analysis runs" off) still blocks.
- **Build:** generated code 34.9 MB → 32.78 MB → 34.0 MB (this round:
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
- [ ] **Settings sheet (O):** one column with collapsible sections - today
      the settings sheet keeps its two-column layout (920 px wide).
- [ ] **Class / component settings in place (P):** next to the class or
      step they belong to - today they float under the top bar.
- [ ] **Pipelines (J):** reorder by dragging ⋮⋮.
- [ ] **Charts (M):** group by well row / column; a click on a box or
      point shows its objects in the List.
- [ ] **Results › Plate (H):** leave an image out right from the well's
      list ("Disable D3_04.vsi in the results").
- [ ] **Export (R):** "Images with objects drawn".
- [ ] **AI training (R):** say how many images' annotations are used
      ("Uses the annotations of 6 images").
- [ ] **Command palette (D):** settings too ("Min threshold · EV Detection
      · 240").
- [ ] **Objects table (K):** a class filter chip, a search ("id, area >
      50"), "Columns…".

### Follow-ups of this round

- [ ] **Image states:** "previewed" lasts the session and "analysed" means
      "in the analysis open in Results" - to keep them, store them with the
      project.
- [ ] **Images › Plate:** a custom grouping regex is compiled by the Rust
      `regex` crate here and by DuckDB (RE2) in the results - the same for
      the usual patterns, but not for every one.
- [ ] **Keyboard:** list rows, segmented controls, selects and the step
      lines don't take Tab focus yet.
- [ ] **Pins:** save with the project (see `TODO`).

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

- [ ] Allow to disable Histogram Auto Scale of the image when a new one is opend should be stored in project settings
- [ ] Allow to apply the histogram scale settings to all images in the project
- [ ] If no pipeline is selectd show the list of pipelines obove the "New Pipeline" button
- [ ] Allow to deselect a pipeline