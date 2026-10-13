# Command groups and ready-to-use templates

Status 2026-10-13: concept, not implemented. Open questions at the end.
Related: LOAD_PIPELINE_CHANGE.md (pipelines without an initial load).

## Goal

Make EVAnalyzer usable as a **ready-to-use application** for users who don't
want to build pipelines. A user loads a template, adjusts a few parameters
(mostly image channels and object classes) and runs it. Everything else stays
as the template author set it, yet experts can still look inside and change
anything.

One concept on two levels:

- **Command group:** a reusable command made of other commands (e.g. "Cell
  reconstruction": blur, threshold, connected components, watershed,
  Voronoi ...). It sits in a pipeline like any other command - insert it
  anywhere, move it, duplicate it - and exposes only its own parameters
  (e.g. image channel and class).
- **Project template ("application"):** a whole project whose selected
  parameters are offered in a setup wizard after loading.

## How others do it

| Tool | Concept | What's good |
|---|---|---|
| **Simulink** masked subsystems | A subsystem with its own "mask" parameters; inner block parameters refer to them; library blocks are linked | The model for groups here: own parameters, one parameter used by several inner blocks |
| **Blender** node groups | Nodes collapsed into one; chosen inputs are "exposed" on the group | Expose, open, ungroup |
| **KNIME** components | Sub-workflow with its own settings dialog, built from configuration nodes; shareable, optionally *linked* to the original | One parameter can drive several inner nodes |
| **Galaxy** workflows | Whole workflow; only marked "runtime parameters" appear in the run form | Simple run form for non-experts |
| **Harmony** (screening software) | Building blocks plus analysis-sequence templates | Normal user experience in screening |
| **Figma** components | Instances with **overrides**; when the original is updated, overrides are kept | The best model for updating templates |

Rare on the market: a **project template with a setup wizard** (map channels
and classes, check a few values, run), combined with reproducibility and the
option to look inside. That would be a real selling point.

## Architecture

### 1. A group is a command with its own pipeline inside

| Simulink | EVAnalyzer |
|---|---|
| Subsystem | **Group step:** one step in the pipeline that contains its own list of steps |
| Mask parameter | **Group parameter:** "Image channel", "Cell class" |
| Inner block parameter set to a mask variable | Inner parameter **bound** to a group parameter, shown in the group as locked ("← Cell class") |
| Inport / Outport | **Input/output type** from the first and last inner step: image in, objects out |
| Library block with link | **Group library** (`.evagroup`); the project holds a copy plus its origin |

A pipeline step is either a command or a group:

```rust
pub enum StepContent {
    Command(PipelineCommand),
    Group(Box<CommandGroup>),   // name, steps: Vec<PipelineStepSettings>, parameters, origin
}
```

- **Old project files must still load** (`{"enabled", "command"}`): schema
  version bump with a migration test.
- **Execution stays unchanged: flatten before running.** The job generator
  expands each group into its inner steps, writes the group parameter values
  into the bound inner parameters, and then only plain commands exist.
  Executor, tiles and dataset steps see no difference.
- **Groups inside groups come almost for free:** flatten recursively, refuse
  a group that contains itself.
- **The picker gets a "Groups" area,** fed from the bundled and user group
  libraries. A group's category comes from its first and last step, so the
  ordering rules keep working ("Cell reconstruction": Preprocess in →
  Object out).

### 2. Group parameters with bindings (the central part)

One group parameter sets **several** inner parameters - e.g. the class in
Threshold and in the classifier, always the same value.

```rust
pub struct GroupParameter {
    pub id: ParamId,
    pub label: String,              // "Cell class"
    pub description: String,
    pub value: ParamValue,          // the single source of truth
    pub bindings: Vec<ParamBinding>, // Threshold.class, Classify.class, Voronoi.output_class ...
}

pub struct ParamBinding {
    pub step: StepId,               // inner step - stable ID, not an index
    pub param: String,              // parameter path, as scripts already use it
}
```

- **The value lives on the group, not in the steps.** Bound inner parameters
  are locked and always set from the group, so they can't drift apart.
- **Parameter paths already exist:** the generated `to_parameters` and the
  script bridge address parameters by name. Reuse that addressing.
- **Steps need a stable ID.** Today `PipelineStepSettings` is only
  `{enabled, command}`; an index breaks as soon as someone inserts a step.
  Add a `StepId` field (`serde(default)` generates one when loading old
  files).
- **Validation:** all bindings of a group parameter must have the same type
  and must exist; no recursion. Checked on saving and loading, with a clear
  message for a broken binding.

### 3. Ports: what a group reads

- **Input = the incoming image.** "Cell reconstruction" works on whatever
  image flows in: inserted after a pipeline on channel 0, it works on
  channel 0.
- **Additional channels** come from "Load image channel" steps **inside**
  the group (e.g. cells in channel 0, cell area in channel 1). Their channels
  are **group parameters** (channel roles), picked when inserting.
- **A pipeline's `image_source` stays,** as the pipeline's input (like a
  Simulink inport): no migration; the UI and the checks keep their anchor.
- **"Load image channel" becomes a visible command** of its own. Under the
  hood it's the existing Image Cache Load with a channel address. Before
  that, the four points in LOAD_PIPELINE_CHANGE.md need fixing (memory per
  tile, viewer focus, pre-start check, error message).
- **Memory slots / intermediate images:** if the group uses memory slot 1
  and the surrounding pipeline does too, they overwrite each other. At
  flattening, the group's slots are renumbered into a range of their own;
  the user never sees it.
- **Later, optionally:** if most pipelines end up consisting of groups,
  `image_source` can become optional ("no input").

### 4. Classes in groups

Three kinds:

| Kind | Example | Visible to the user? | When inserting |
|---|---|---|---|
| **Input class** | The group uses objects from another pipeline | Yes, as a group parameter | User picks an existing class |
| **Output class** | "Nucleus", "Cell" - what the user wants in the results | Yes, as a group parameter | Create new or map to an existing one |
| **Internal class** | "Seed", "Rejected" - intermediate results | No | Created automatically, hidden |

Internal classes:

- **Declared by the template author** in the group: a class not exposed as a
  parameter is internal.
- **Each group instance gets its own project classes when inserted.** The
  group's class IDs inside the copy are replaced with fresh IDs once, at
  insertion - inserted twice, the two copies must not share the "Seed"
  class.
- **Stored in the project with the instance,** not re-numbered at each run,
  so class IDs in the results database stay stable.
- **The class gets an owner:** `Class { …, owner: Option<GroupInstanceId> }`.
  From that:
  - hidden in class lists, in pickers outside the group and in the results
    view (with a "show internal classes" toggle for debugging);
  - a readable name if it ever does show: "Cell reconstruction · Seed";
  - deleting the group instance deletes its internal classes; duplicating
    creates new ones; ungrouping turns them into normal, visible classes.
- **Not usable from outside:** a step outside the group can't select an
  internal class; only exposed output classes are visible outside.
- **Results database:** store internal objects, but hide them (reproducible
  and debuggable, simple to build). Later, optionally, "discard at the end
  of the group" per internal class, for a smaller database.

### 5. Editing and the library

- **The group card** shows only the group parameters.
- **"Open group"** shows the inner pipeline with a breadcrumb (`Pipeline 1 ›
  Cell reconstruction`). Any inner parameter offers "Expose as group
  parameter" - new, or added to an existing one.
- **"Group selected steps"** and **"Ungroup".**
- **Editing inside a group is allowed;** an instance changed this way is
  marked "changed from library". Locking by the template author can follow
  later if needed.
- **Copy, never a live link:** the project stays self-contained and
  reproducible. Through `origin` (library name + version) the app can later
  say "a newer version of *Cell reconstruction* exists" and update while
  keeping the user's parameter values (Figma / Simulink library style).
- **Every place that reads `step.command`** (GUI, citation export, scripts,
  LLM prompt, focus, breakpoints) must also handle groups - the largest
  effort, mostly mechanical. A helper like `steps_flattened()` keeps most
  places small. Breakpoints and preview focus address a step as a path
  ("step 3, inside it step 2") instead of an index.

### 6. Project templates: the same thing one level up

`ProjectTemplate` gets its own parameters with bindings. Its bindings point
to steps in the pipelines (including group parameters) **and** to project
settings. Two special kinds matter most:

- **Channel roles:** "Nuclear channel", "Spot channel" - the user picks which
  image channel that is.
- **Class roles:** "Nuclei", "Spots" - keep, rename, or map to existing
  classes.

When the template is loaded, a **setup wizard** appears: name, images,
channel roles, class roles, two or three values - done. Afterwards the same
parameters stay available as a "Template settings" panel, so changing
something later doesn't mean reopening the pipelines.

### 7. Headless and CLI for free

Because everything is a binding on normal parameters:

```
evanalyzer analyze --template spots.evapt --images ./plate1 --set "Spot channel=2" --set "Spot size=3"
```

Useful for servers and batch runs, at almost no extra cost.

## Suggested order

1. **Stable step IDs** (`StepId` with `serde(default)`) - the foundation.
2. **`StepContent` (command | group) in cfg,** with migration, and
   **flattening in the job generator** including slot renumbering. From here
   on, groups run.
3. **Group parameters with bindings,** applied during flattening, and
   validated (types, existing steps, no recursion).
4. **"Load image channel" as a visible command,** with the four fixes from
   LOAD_PIPELINE_CHANGE.md.
5. **Classes:** input/output classes as group parameters, internal classes
   with an owner, created per instance at insertion.
6. **Editor:** group card, open group with breadcrumb, "Expose as group
   parameter", "Group selected steps", ungroup.
7. **Group library (`.evagroup`) in the picker,** with the class/channel
   dialog when inserting.
8. **Project templates** with a setup wizard, reusing the same parameter
   mechanism; CLI `--set`.
9. Later: library update while keeping the user's values; locking; "discard
   internal objects".

## Open questions

- Internal objects in the results database: store and hide (suggested), or
  discard at the end of the group from the start?
- Editing inside a group: always allowed (suggested), or lockable by the
  template author from the start?
