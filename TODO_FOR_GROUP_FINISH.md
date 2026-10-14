# Command groups – still to do

Status 2026-10-14. Done so far: the data model (`command_groups.rs`,
`StepContent`, typed `ParamValue`), stable step ids, and running groups:
the job generator expands each group into its commands
(`core/src/job/command_groups.rs`, `expand_step`). Concept: VISION_GROUP.md.

- [x] **Group card with parameters** – done (step 11).
- [ ] **Scoping for classes, objects and memory slots** – decided, see
      "Scoping" below. Replaces the earlier plans "internal classes per
      instance" (created at insertion) and "renumber memory slots".
- [x] **Groups in the command picker** – step 12 (library part with step 16).
- [x] **Creating groups in the editor** – steps 13 and 14.
- [ ] **Focus classes for group steps** – `PipelineSettings::object_classes`
      ignores group steps (marked TODO); they should contribute the classes
      their commands use.
- [ ] **Validation in the editor** – today mistakes surface when a run starts.
      Check earlier, when loading/editing: unique group keys, every instance's
      key is defined, bindings point to existing steps/parameters, all
      bindings of a parameter have the same type, values fit the bound
      parameters.

---

## Implementation order

Each step builds and is tested on its own; the order follows the
dependencies (A before B: the runtime needs `Local`; B is the core; C can
run in parallel to B; D before E: the editor needs the validation; 13
before 14: most groups are made from existing steps).

**Phase A – types and config (no runtime change yet)**

1. [x] **Rename `ObjectClass::Valid` → `Global`** – `#[serde(alias =
       "VALID")]`, protocol version bump (postcard). Mechanical: 503 places
       in 56 files. *Done when:* old projects load, all suites pass.
2. [x] **Add `ObjectClass::Local(u32)`** – last in the enum (postcard), text
       form in the generated code (`"L1"`), handled in every `match`; `Local`
       outside a group is a validation error. *Done when:* `{"LOCAL": 1}`
       round-trips, a local class in a pipeline is refused.
       Done: text form via `ObjectClass::to_text`/`from_text`; generated
       `local_classes()`; `expand_step` refuses `Local` outside a group and
       undeclared local classes inside one. Protocol version 13.
       Open: a nested group's `Local` value set from the outer group is read
       in the inner group's scope - decide with scopes (step 4).
3. [x] **`classes` in `GroupDefinition`** (reuses `Class`: id, colour,
       name, notes); `ParamKind` removed altogether (only `Exposed` was left).
       `Class.id` is a plain `u32` - the kind follows from where the class is
       defined (`Class::global()` / `Class::local()`). Old files converted on
       open: project schema v4 (`migration_v3_to_v4`). Templates aren't
       migrated (breaking changes allowed) - project template schema v3.

**Phase B – scopes at runtime (the core)**

4. [x] **`Scope` / `ScopeId` and `Object.scope`** – expansion assigns a
       `ScopeId` per instance, commands carry their scope; new objects get
       their class's scope; reading by class filters by scope through a
       central cache API (not direct `object_cache` access). The biggest
       step: every object algorithm has to be checked. *Done when:* two
       instances with `Local(1)` don't see each other's objects.
       Done (2026-10-14): `pipeline::scope`; `expand_step` returns
       `ExpandedCommand { step_id, scope, command }` (a `ScopeCounter` per
       job), the pipeline sets `ctx.scope` per command. `Object`'s classes
       and scope are private: `is_of` / `is_of_any` / `is_visible_in`
       (empty filter = the project's objects plus the own scope's),
       `add_object_class(class, scope)` refuses mixing, removing classes
       keeps the scope (an internal object stays internal). Passing a
       `Local` value into a nested group is refused; a dataset step in a
       group with a local class too. Until step 5: scoped objects are
       dropped right before the export (`job_executor`, TODO there).
       Benchmark (8 × 4096², 90k objects, 4 classify steps): 1.04–1.08 s vs
       1.06–1.11 s before, peak RSS 2.15–2.24 GB vs 2.20–2.22 GB - no
       measurable change.
       **ExtractObjects class mapping** (optional `class_mapping`:
       segmentation class → object class, local or global; without an entry
       the project class with the same number, as before). Lets a group
       extract objects for internal use. Benchmark unchanged (1.02–1.11 s,
       2.20–2.23 GB). For step 10: a mapping to `Global(n)` needs class n in
       the project; `object_classes()` still also lists the mapped
       segmentation class as `Global(n)`.
5. [x] **`EndScope`** after an instance's last step across both phases:
       removes the scope's objects and the references to them. **Benchmark
       speed and RAM before/after.** *Done when:* no local object reaches the
       export, RAM drops measurably at `EndScope`.
       Done (2026-10-14): the job generator adds `EndScope(scope)` after an
       instance's last command - in the whole-image list if it has a command
       there, else in the tile list (frees the objects per tile, before the
       merge). `EndScope` (`remove_scope`) removes the objects and the
       references to them (parent, children, coloc partners, track members);
       the filter before the export stays as a safety net for runs stopped at
       a breakpoint. Micro-benchmark (`end_scope::tests::bench_end_scope`,
       release, `--ignored`): 1M objects, half removed: 0.22 s (0.07 s of it
       dropping the objects), RSS 717 → 467 MB; 5M: 1.0 s, 3062 → 2079 MB.
       The map is rebuilt when ≥ 1/4 of the objects go - `retain` alone left
       the `BTreeMap`'s nodes (objects inline, 480 bytes each) half empty and
       freed nothing. End to end (8 × 4096², group making 70k of 90k objects
       internal): 0.86 s vs 0.89–0.93 s, peak RSS 2.21–2.22 vs 2.23–2.24 GB.

6. [x] **The mixing rule** – `ReclassifyIfMatch` moves an object to
       `Global`; `AddOutputClassIfMatch` local → global is an error (checked
       at expansion, plus a runtime guard).
       Done (2026-10-14): `mixed_classes` in `command_groups.rs`, checked per
       command of a group after its values are applied: Classify Objects
       ("add output class if (not) match") and the AI object classifier ("add
       output class", per output class) may not add a class of the other kind
       than their input classes - nor a local class with no input classes
       (that's every object, the project's too). Reclassifying is fine.
7. [x] **Tile merge by (scope, class, plane)** – key done with step 4
       (merged object keeps the scope; `classes_to_not_merge` only for
       global objects). Seam test: two instances' seed fragments of `L1` at
       the same spot of one seam merge per instance (the test fails without
       the scope in the key).

8. [x] **`MemoryId::GroupContext(slot)`** – scoped like classes, freed at
       `EndScope`.
       Done (2026-10-14): `MemoryId::GroupContext` - `{"groupContext": n}` in
       the file, `"group:n"` as text.
       Resolved in one place, `CacheAddress::memory(id, scope)` →
       `CacheAddress::GroupMemory(scope_id, slot)` (Image Cache, Image Math,
       Threshold's value source, a pipeline's start image - there an error).
       `EndScope` frees the instance's slots (also spilled ones). Expansion
       refuses a group slot outside a group and passing `"group:n"` into a
       nested group. GUI: the slot picker keeps a group slot a group slot;
       choosing one belongs to phase E.

**Phase C – loading channels inside groups**

9. [x] **The four fixes from LOAD_PIPELINE_CHANGE.md** (source "None",
       viewer focus from Load steps, pre-start check incl. groups, error
       message), then **"Load image channel" as a visible command**.
       Done (2026-10-14): `ImageAddress::None` (pipeline dialog: "None");
       such a pipeline starts with a 1×1 placeholder
       (`PipelineContext::without_image`) and its first tile step must load
       an image (checked when the job is built). New command **Load Image
       Channel** (`LoadChannel`, Preprocessing). `pipeline_channels` (core):
       the source's channel plus the load steps' (Load Image Channel, Image
       Cache Load, Image Math's second image), also inside groups - used by
       the pre-start check and the viewer focus. Missing channel: "reads
       image channel 9, which this image doesn't have (it has channels 0,
       2)"; an empty memory slot names the slot. Found on the way: the
       scratch pad was only re-made on a type change, not a size change -
       a loaded image of another size broke the next step; fixed in
       `prepare_*_scratch`. Benchmark (4 × 8192², 182k objects): channel
       source / Scratchpad + Load / None + Load all 2.25–2.41 s, peak RSS
       3.96–4.17 GB - the scratch pad was only allocated briefly, so the peak
       doesn't change.

**Phase D – validation** (postponed 2026-10-14 - Phase E first)

10. [ ] **One validation function in the app layer** (load and editor):
        unique keys, every instance's key exists, bindings exist and fit
        (types, values), local classes only inside groups and not in
        dataset steps, optional coloc warning. Also from step 4: a mapping
        to `Global(n)` (ExtractObjects) needs class n in the project.

        **Open question - memory slots across pipelines** (found in step 8):
        `GlobalPipelineCache::clear_pipeline_context` runs at the end of
        *every* pipeline and deletes every memory slot, `ProjectCache` ones
        included (only image channels stay) - the per-tile cache passes
        through the pipelines in order (`job_executor`, tile loop). So a
        pipeline can't read a slot an earlier pipeline stored: the pipeline
        dialog's image source "Memory slot" can never find its image, and
        `Pipeline::missing_start_image_message` ("add an Image Cache step that
        stores it to a pipeline running before") describes something that
        can't work. The test
        `clear_pipeline_context_also_drops_scratchpad_and_project_cache_via_catch_all`
        pins the behaviour as "possibly surprising"; the function's doc
        comment only mentions `PipelineContext` slots. Suggestion: clear only
        `PipelineContext` (and group) slots per pipeline, keep `ProjectCache`
        slots until the image (plane) is done. Not decided - ask first.

**Phase E – editor and picker** (started 2026-10-14; decisions under
"Command picker" and "Creating groups")

11. [x] Group card with editable parameters.
        Done (2026-10-14): `group_parameters` / `set_group_parameter` (core,
        re-exported as `evanalyzer_app::groups`). Each group parameter shows
        as the inner setting of its first binding - same control, options and
        range - with its label, description and default; a change is applied
        to a copy of the bound command and read back, so only values the
        command takes are stored. Works through nested groups. Icon/colour of
        the card: with step 12.
12. [x] Picker: "Groups" chip (in the project / library), "Update
        available" - see "Command picker".
        Done (2026-10-14): `StepCategory::group` with `Theme.cat-group` - the
        group colour and "Group" label on picker rows and group cards. Chip
        "◆ Groups" lists the project's groups ("In this project"); without
        it, a search lists matching groups in a "Groups" section. Row: "used
        N×"; detail: the group's parameters with their defaults, source line
        "this project · Segment → Object" (first/last command). Picker ids
        from `GROUP_PICKER_ID_BASE` (1 000 000); confirming inserts a group
        step at its defaults. Also fixed: the Templates section was in the
        list twice (each template showed twice).
        Still open: "Library" section and "Update available" (need step 16),
        "Remove unused groups". Look not checked on screen (no headless
        screenshot in the tests).
13. [x] "Group selected steps" with automatic detection of input, output
        and local classes.
        Done (2026-10-14): "Group…" on each step card opens "Group steps"
        (name, range from/to - instead of Shift-click selection, which the
        step list doesn't have; can come later). Logic in the app layer
        (`workspace/grouping.rs`, `evanalyzer_app::groups`): `plan_grouping`
        lists the project classes the steps use with a suggested role -
        used elsewhere too (other steps/pipelines, also inside groups; hand
        annotations; tile merge) → group parameter, only here → internal
        (local), only a segmentation class without Extract Objects among the
        steps → fixed. No read/write distinction needed: a class parameter
        binds the same either way. `group_steps` builds the definition
        (steps keep their ids), local classes rewritten to `L1`, `L2`, ...
        (an Extract Objects among the steps maps the segmentation class when
        a step there makes it), parameters bound to every single-class
        setting with the class (also a list with only it, and the matching
        segmentation class); a list with other classes too stays as it is.
        Load Image Channel steps' channels become parameters. The dialog
        says internal classes don't reach the results. Tested: with no
        internal classes the group runs exactly the commands the steps ran.
14. [x] "Open group" with breadcrumb, "Expose as group parameter",
        "Ungroup".
        Done (2026-10-14): a group card has "Open" and "Ungroup". Open shows
        the definition's steps in the panel with a bar "‹ Back · Pipeline ›
        ◆ Group · changes apply to every use"; the step handlers (toggle,
        remove, duplicate, insert, values, list entries, file browse) work on
        `shown_steps(_mut)` - the opened group's or the pipeline's. Used by
        several steps: "Edit for all" / "Only this step"
        (`clone_group_for_step`: "Name (2)", own key, origin cleared).
        ◇/◆ on each setting of an opened group exposes it as a group
        parameter (its current value the default) or hides it again
        (`toggle_exposed`; a parameter goes with its last binding).
        Removing a step of a group removes the bindings to it
        (`remove_group_step`). Ungroup (`ungroup`): the steps with the
        instance's values back in the pipeline; the group's classes become
        new project classes, its memory slots pipeline slots (rewritten in
        the steps' file form); the definition stays. Inside an opened group:
        no breakpoints, no "Group…" (groups in groups only from the
        pipeline, for now), a group can't be inserted into itself. A group
        that's gone (undo) closes.
        Also done: settings of list entries ("thresholds.1.min_threshold",
        label "Thresholds 2 · Min Threshold") get ◇ too; exposing a setting
        that existing parameters with the same control (type, a dropdown's
        options) could set asks "New parameter" / "Add to “Cell class”"
        (`compatible_parameters`, `add_binding`). Removing a list entry inside a
        group: bindings to it go, bindings to later entries move up
        (`list_entry_removed`, called from `modify_group_item`).
15. [ ] Focus classes for group steps.
16. [ ] Group library (`.evagroup`): save, insert, update while keeping the
        user's values.
17. [ ] Allow to remove (unused) groups from project

## Scoping

Decided 2026-10-14. Like variables in a programming language: classes,
objects and memory slots live in a **scope** – global (project) or the scope
of one group instance. Nothing is renumbered in the file and nothing
internal is saved: a scope's objects are thrown away when the group is done.

### Decisions

1. **Named scopes, one per group instance** (`ScopeId`, assigned at
   expansion, never saved). Not a `StartScope … EndScope` stack: the
   executor runs a pipeline's tile steps first and its whole-image steps
   afterwards, so scopes of different instances overlap. A scope is thrown
   away only after the instance's **last step across all phases**.
2. **No mixed objects.** An object never carries local and global classes
   at the same time. Its local classes must be gone before it gets a global
   one – a local class may have a name the user never sees.
3. **`EndScope(id)` removes** the scope's objects and memory slots, and the
   references to them from the remaining objects (`parent_id`, `children`,
   coloc lists) – RAM is freed right away, and nothing internal reaches the
   export.
4. **Variant B: one cache, the scope explicit** (not separate caches per
   scope – every object algorithm accesses `cache.object_cache` directly).

### Where the scope lives

**Runtime – one field per object.** Because of decision 2, an object always
belongs to exactly one scope:

```rust
pub enum Scope {
    Global,
    Scoped(ScopeId), // one per group instance, never saved
}

pub struct Object {
    // ...
    pub scope: Scope,
}
```

- A new object gets the scope of the class it's created with.
- Reading by class within a group means: objects of that class in this scope.
- Once an object has no local class left, it becomes `Global`.

**Config – the class reference says whether it's local.** A list of local
classes alone isn't enough: a group's local class 1 and the user's project
class 1, passed in through a group parameter, would end up as the same
number in an inner command. So:

```rust
pub enum ObjectClass {
    Unset,
    #[serde(alias = "VALID")] // projects written before
    Global(u32),               // a project class (today's `Valid`)
    Local(u32),                // a class of the enclosing group's scope
}
```

- Readable in the file: `{"LOCAL": 1}` versus `{"GLOBAL": 5}`.
- No scope id in the file: `Local` always means "the scope of the group this
  command sits in"; the concrete `ScopeId` is assigned per instance at
  expansion – two pipelines using the same group look identical.
- The definition declares its local classes with names
  (`local_classes: [{id: 1, name: "Seed"}]`); "Open group" shows them
  separately from the project classes.
- `ParamKind::InternalClass` goes away.
- **Memory slots** the same way: `MemoryId::Group(slot)` next to
  `PipelineContext` and `ProjectCache`.

### Renaming `Valid` → `Global`

- Old project files keep loading: `#[serde(alias = "VALID")]`; written as
  `"GLOBAL"` from then on.
- In the code: 503 uses of `ObjectClass::Valid` in 56 files – a mechanical
  rename, the compiler finds the rest.
- `ObjectClass` is in the remote binary format (results types): postcard
  encodes variants by position, so `Global` must stay second and `Local` be
  added last; the protocol version needs a bump.
- The generated code: the text form in `apply_param_change` / `to_parameters`
  needs one for local classes (e.g. `"L1"`).

### When could an object become mixed?

Commands that **create new objects** are never a problem: their result is
in the scope of the class it's written to. In the RECONSTRUCT_CELLS example
(nuclei → local CENTERS, cell area → local AREA, Voronoi of both → global
CELLS) the cells are new global objects; their references to the local
centres are removed at `EndScope`. If the user also wants the nuclei, the
class becomes a group parameter – a global class – and nothing is mixed.

Mixing can only happen through commands that **change the classes of
existing objects**:

| Command (mode) | What happens with local input, global output | Rule |
|---|---|---|
| Classify Objects – `ReclassifyIfMatch` (also AI Object Classifier) | The matching objects' local class is **replaced** by the global one | Allowed – the object **moves to `Global`** and survives `EndScope`. A "filter" group: local candidates in, global hits out |
| Classify Objects – `AddOutputClassIfMatch` (also AI Object Classifier) | The global class is **added**, the local one stays → mixed | **Forbidden** – error, ideally already when the group is expanded (input class local + output class global + mode "add" can be checked from the settings) |
| Colocalization (links) | Global objects get links to local ones | Not mixing – links are references, removed at `EndScope` |
| Tile merge | Merges fragments of one class across tiles | Must merge within one scope only – see below |
| Dataset steps inside a group (e.g. a tracker) | They run after all images, on stored objects – only global ones exist | A dataset step may only use global classes; a local one is a validation error |

So mixing has exactly one source – adding a global class to a local object –
and that is forbidden. The legitimate way from local to global is to replace
(reclassify), or to create new objects in a global class.

### Colocalization, tile merge and dataset steps (decided)

**Colocalization links to local objects are removed at `EndScope`.** A
reference to an object the user never sees and can't name has no meaning.
Consequence: every number computed from those links disappears too – the
results view's "Coloc with …" columns come from the stored partner lists. A
"count spots per cell" group with *local* spots has no spot count at the end.
That's consistent: if the number is wanted, the spot class is a result and
must be an output class (a group parameter, global).
- Optional: a validation warning, e.g. "Colocalization in group X refers to
  the local class 'Spots' – its links are dropped at the end of the group".

**Tile merge must group fragments by (scope, class, plane)** – today it
groups by class and plane. Large images are analysed in 4096 px tiles; an
object on a tile boundary is cut into fragments, and `TileMerge` (the first
whole-image step) puts touching fragments of the same class back together.
Two instances of one group (e.g. "Cell reconstruction" in pipeline 1 and 2)
both write fragments of `Local(1)` ("Seed"), just in different scopes:

```
Tile 1 | Tile 2
  A ●──┼──● B      A = seed fragment of instance 1 (scope 1)
       |           B = seed fragment of instance 2 (scope 2), same spot
```

Grouped by class only, A and B are merged into one object belonging to two
scopes – exactly the mixed state ruled out above; it would vanish or survive
arbitrarily at `EndScope`, and the second instance's Voronoi would get a
wrong seed.
- Timing fits: tile merge runs at the start of the whole-image phase, a
  scope ends only after its last step – local fragments a whole-image step
  needs (Voronoi uses CENTERS) are merged before. A group with only tile
  steps ends its scope earlier; its local fragments are never merged and
  needn't be (they're discarded).
- `classes_to_not_merge` (project setting) knows only global classes, so
  local objects are always merged – the right default. "Don't merge this
  local class" would need a setting in the group definition; left out until
  someone needs it.

**Dataset steps only work on global classes; a local class is a validation
error.** They run after all images, on stored objects, and local objects are
never stored. No use case needs it otherwise: tracking internal objects to
report per-cell values, or normalising a local measurement per plate, both
mean the class actually is a result – make it global (an output class). A
dataset step inside a group with global classes (bound via parameters) is
fine; it runs after all images, long after the group's scope has ended.

## Command picker

Decided 2026-10-14 (replaces the earlier "groups in the category lists"):

- **Two sources of groups.** Groups made in the project live only there
  (`ProjectSettings::command_groups`). Groups from a library file
  (`.evagroup`) are **copied** into the project when first used and live
  there from then on: changes affect only the project, never the file, and
  the project stays complete without it - like project templates.
  `GroupDefinition::origin` remembers the file, key and version ("Update
  available" later; an update must not silently overwrite the project's own
  changes - needs a "modified since inserted" marker or hash).
- **Own chip "Groups"**, not mixed into the category lists - a project easily
  has several variants of one group, which would crowd out the commands. Two
  sections: "In this project" first, then "Library" (not yet in the
  project; inserting copies it).
- **Search finds groups too** (e.g. "cell" finds commands and groups), marked
  as groups.
- **Recognisable as a group**: own icon and an accent colour - the same on
  the picker row and on the group card in the pipeline. The category derived
  from the first and last command is shown as a small tag on the row.
- **Against variant clutter**: "used N×" on each row; "Remove unused groups".
- **Pipeline templates** (`.evapipe`) hold only steps - a pipeline with a
  group step saved as template misses the definition elsewhere. Carry the
  used definitions along (with step 16).

## Creating groups

1. **"Group selected steps"** (main path): select consecutive steps
   (Shift-click) → context menu → dialog (name; key generated from it;
   optional description). The steps move into a new definition, the
   pipeline gets an instance in their place – what runs doesn't change.
2. **Detect parameters automatically**: the app knows which classes each
   step reads and writes, so it suggests:
   - a class **written and only used inside** → local (hidden);
   - a class **read but not written** inside → input class (exposed);
   - a class **written and used after the group** → output class (exposed);
   - **image channels** of Load steps → exposed.

   The user confirms or deselects.
3. **"Open group"**: the inner pipeline with a breadcrumb
   (`Pipeline 1 › Cell reconstruction`), same step editor. Every parameter
   gets "Expose as group parameter" (new, or added to an existing one of the
   same type).
4. **On the group card**: the exposed parameters, "Open", "Ungroup", "Save
   to library" (`.evagroup`).
5. **"New empty group"** – rarely needed once 1 and 2 exist.

Order: first 1 and 2, then 3; the library actions after.

**Editing a group** (decided 2026-10-14): one entry point, "Edit group". If
several steps use the definition, the dialog says so ("Used by 3 steps in 2
pipelines") and offers **"Edit for all"** (default - the definition changes,
every instance follows) or **"Only this step"** (clones the definition as
"Name (2)" with a new key, points just this step at the clone, opens the
clone). Used only once: edited directly.

