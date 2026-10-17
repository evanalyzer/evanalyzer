# Ink - the EVAnalyzer design system

Everything the UI is built from: `palette.slint` (colors per scheme, sizes,
type), `icons.slint` (every icon by meaning) and the components. Views use
these and nothing else - no color literals, no hand-made buttons.

`import { … } from "../ink/ink.slint";` gives all of it.

## Rules that keep the build fast

The Slint compiler generates a component **once** and reuses it - unless it
has to copy ("inline") it into every place it's used. It copies a component
when its **root element**

- sets `opacity`, `visible`, `drop-shadow-*` or `cache-rendering-hint`,
- animates a public property (`animate background` on the root),
- has a percentage `width`/`height`,

or when, where it's used, it

- is given children (`@children`),
- gets `clip`,
- is a layout and gets `padding`/`spacing`/`alignment`.

(A component that *contains* a `PopupWindow` is fine; the popup itself is
always copied.) A copied component costs its full
generated code at every use (a 65-times-used button: ~1 MB of Rust).

So in Ink:

- the root is a plain `Rectangle` (or `TouchArea`) with sizes only; the look
  (fill, border, opacity, animation, clip) is on an inner element;
- leaf controls take no children; only a few containers (`InkPanel`,
  `InkSheet`, `InkFormRow` …) use `@children`, and they are small;
- every `if` and `for` body becomes a struct of its own: for something
  small that doesn't change the layout, prefer `visible`.

## Schemes

`Ink.scheme` is `light`, `dark` or `dim`. Every color is `Ink.<token>`
(`Ink.text`, `Ink.panel` …), defined in `palette.slint` with its value per
scheme: `root.pick(light, dark, dim)`. A new scheme is one more value per
token.

Keep each token a property bound to plain values - not a field of a
palette struct: the compiler inlines a global's bindings into every read,
and with a struct that meant building all 37 colors at each of the ~1300
reads (14 MB more generated code, and the work at runtime).

## Files

| File | What |
| --- | --- |
| `palette.slint` | colors per scheme, radii, heights, gaps, type; `Ink.window-height` |
| `icons.slint` | every icon by meaning (`InkIcons.save`) |
| `button.slint` | `InkButton`, `InkIconButton`, `InkToolButton` |
| `field.slint` | `InkTextField`, `InkSearchField`, `InkNumberField`, `InkSelect` |
| `toggle.slint` | `InkCheckbox`, `InkSwitch`, `InkSegmented`, `InkModeTab` |
| `marker.slint` | `InkChip`, `InkBadge`, `InkNewTag`, `InkSwatch`, `InkKbd`, `InkDot` |
| `panel.slint` | `InkPanel`, `InkPanelHead`, `InkDialog`, `InkDialogFooter`, labels, captions, hints, dividers |
| `menu.slint` | `InkMenuFrame` (scrolls when the window is too short - give it `open-y`), items, captions, separators |
| `feedback.slint` | `InkProgressBar`, `InkBusy`, `InkBanner`, `InkTooltip`, `InkStatusItem` |
| `row.slint` | `InkListRow` |
| `settings.slint` | settings dialogs: `InkNavItem`, `InkSettingsHead`, `InkSettingsCard`, `InkSettingRow`, `InkSettingBlock` |
| `picker.slint` | the closed "pick one" field and its popup rows: `InkPickerField`, `InkPickerRow`, `InkSwatchName` |
| `multi_select.slint` | `InkMultiSelect` (one or many of a list, with search) and its `MultiSelectItem` |
| `std_compat.slint` | `LineEdit`, `ComboBox`, `CheckBox`, `Switch`, `SpinBox` with the std-widgets API, drawn in Ink |
| `gallery.slint` | every component on one page, for screenshots |
