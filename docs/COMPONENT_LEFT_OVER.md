Small leftovers
- Breakpoints inside components: not possible. The vision wanted step paths ("step 3, inside it step 2").
- Components inside components: only via inserting from the pipeline. There's no "Component…" inside an opened component.
- Step selection: a from/to range dialog instead of Shift-click.
- "New empty component": not built. The TODO itself calls it rarely needed.
- Checks run on load and in the editor, not on save. Runs still refuse bad components at expansion, so this is cosmetic.
- The LLM prompt doesn't handle components. That crate is unfinished anyway.
- Not checked on screen: step 12's picker look ("no headless screenshot").

Vision not started
- Project templates with parameters and a setup wizard (channel roles, class roles, a "Template settings" panel). VISION_COMPONENT.md calls this the real selling point.
- CLI analyze --template … --set "Spot channel=2".
- Locking by the template author. Deliberately postponed.

Stale documentation

- VISION_COMPONENT.md still says "concept, not implemented", and parts are superseded:
  - classes use scopes and Local instead of Class.owner;
  - internal objects are dropped at EndScope rather than stored and hidden;
  - both open questions are decided (discard; always editable).
- TODO step 14 still describes "Open" on the card. Step 18 explains the rename to Edit/View, but step 14 wasn't updated.