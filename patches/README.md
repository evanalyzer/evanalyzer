# patches

Temporary copies of third-party crates, wired in through `[patch.crates-io]` in the
workspace `Cargo.toml`. Each keeps its own licence (not ours).

## burn-dispatch 0.22.1

Unchanged burn-dispatch 0.22.1 (MIT OR Apache-2.0, © Nathaniel Simard & Burn Framework
Contributors) except its `Cargo.toml`: the optional `burn-tch` dependency, the `tch`
feature's content and the three `burn-tch?/…` feature entries are removed.

Why: burn's `std` feature (needed by `burn-optim`, i.e. MLP training in `crates/nn`)
enables `burn-dispatch/std`, which names `burn-tch?/std`. Cargo resolves such optional
dependencies into the lock even when they are never built, and then refuses two crates
linking the native `tch` library - burn-tch's `torch-sys` 0.22 and our `tch` 0.26
(`crates/ai`).

Remove this patch (the directory and the `[patch.crates-io]` entry) together with
`crates/ai` and `tch` (burn migration plan, P2.3).
