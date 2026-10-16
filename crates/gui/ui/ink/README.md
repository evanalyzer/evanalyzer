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

`Ink.scheme` is `light`, `dark` or `dim`. Every color is `Ink.<token>`;
a new scheme is a new `InkPalette` value in `palette.slint`.
