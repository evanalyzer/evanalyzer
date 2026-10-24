"""Writes relate_objects_cellprofiler.json: which parent CellProfiler's
RelateObjects gives each child (cellprofiler_core Objects.relate_labels -
the largest overlap, ties to the lower label, no overlap: no parent), for a
few label images, to check EVA's Relate Objects ("It overlaps the parent
most") against.

Run with numpy, scipy and cellprofiler-core installed; cellprofiler-core's
other dependencies aren't needed (pip install --no-deps cellprofiler-core):
    python relate_objects_cellprofiler.py
"""

import importlib.util
import json
import os
import sys
import types

import numpy as np


def cellprofiler_objects():
    """CellProfiler's `Objects` class from its own file
    (cellprofiler_core/object/_objects.py), unchanged. Its imports for
    other methods (centrosome, the package's preferences with matplotlib,
    psutil, ...) are stubbed - relate_labels needs only numpy and scipy."""
    root = importlib.util.find_spec("cellprofiler_core").submodule_search_locations[0]
    stubs = {
        "centrosome": {},
        "centrosome.index": {},
        "centrosome.outline": {},
        "cellprofiler_core": {"__path__": [root]},
        "cellprofiler_core.object": {"__path__": [os.path.join(root, "object")]},
        "cellprofiler_core.object._segmentation": {"Segmentation": object},
        "cellprofiler_core.utilities": {"__path__": []},
        "cellprofiler_core.utilities.core": {"__path__": []},
        "cellprofiler_core.utilities.core.object": {"downsample_labels": None},
    }
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        sys.modules[name] = module
    name = "cellprofiler_core.object._objects"
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(root, "object", "_objects.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Objects


Objects = cellprofiler_objects()


def paint(labels, rng, count, size):
    """`count` random blobs (unions of 1-3 rectangles) of about `size` px;
    later ones paint over earlier ones, so shapes get irregular."""
    height, width = labels.shape
    for label in range(1, count + 1):
        for _ in range(rng.integers(1, 4)):
            h, w = rng.integers(1, size + 1, 2)
            y, x = rng.integers(0, height - h + 1), rng.integers(0, width - w + 1)
            labels[y : y + h, x : x + w] = label
    return labels


def in_reading_order(labels):
    """The labels renumbered by their first pixel in reading order - as
    scipy/scikit-image number a segmentation's labels (and EVA breaks ties:
    the object whose first pixel comes first)."""
    values, first = np.unique(labels.ravel(), return_index=True)
    order = [v for _, v in sorted(zip(first, values)) if v != 0]
    lookup = np.zeros(labels.max() + 1, dtype=int)
    lookup[order] = np.arange(1, len(order) + 1)
    return lookup[labels]


def case(name, parents, children):
    # relate_labels only calls static methods on `self`.
    _, parents_of_children = Objects.relate_labels(Objects, parents, children)
    parents_of_children = np.asarray(parents_of_children).ravel()
    return {
        "name": name,
        "width": int(parents.shape[1]),
        "height": int(parents.shape[0]),
        "parents": parents.flatten().tolist(),
        "children": children.flatten().tolist(),
        # Child label i + 1 -> its parent's label (0: none).
        "parent_of_child": [int(p) for p in parents_of_children],
    }


def main():
    cases = []

    # Two cells side by side, a spot on their border with equal overlap.
    parents = np.zeros((6, 8), dtype=int)
    parents[:, :4] = 1
    parents[:, 4:] = 2
    children = np.zeros((6, 8), dtype=int)
    children[2:4, 3:5] = 1
    children[0, 0] = 2
    cases.append(case("a tie goes to the lower label", parents, children))

    rng = np.random.default_rng(20261023)
    for i in range(6):
        size = (64, 64)
        parents = in_reading_order(paint(np.zeros(size, dtype=int), rng, 12, 18))
        children = in_reading_order(paint(np.zeros(size, dtype=int), rng, 40, 4))
        cases.append(case(f"random {i}", parents, children))

    with open("relate_objects_cellprofiler.json", "w") as out:
        json.dump(cases, out)


if __name__ == "__main__":
    main()
