#!/usr/bin/env python3
"""Convert an EVAnalyzer results database (.evadb) to the current format.

Older results files store per-object intensities and colocalization partners
as JSON text and the object classes as a JSON string:

    intensities_json   {"<channel>": {"sum_raw": .., "sum_scaled": .., "mean_raw": ..,
                                      "mean_scaled": .., "min_raw": .., ...}, ...}
    coloc_json         {"<partner class id>": ["<partner object id>", ...], ...}
    object_class_id    "[1,3]"

The current format stores them as typed columns (see `CREATE_TABLES` in
crates/core/src/storage/duckdb.rs):

    intensity_{sum,mean,min,max}_normalized   DOUBLE[]   (was <stat>_raw)
    intensity_{sum,mean,min,max}_gray         DOUBLE[]   (was <stat>_scaled)
        position c + 1 holds channel c, NULL for a channel that wasn't measured
    coloc_partner_ids  MAP(INTEGER, UUID[])  (key -1: partners without a class)
    object_class_id    INTEGER[]

Every other column, and the `images` and `classes` tables, are copied as they
are. Values are copied exactly as stored in the JSON (no recomputation).

Requires the DuckDB Python package:  pip install duckdb

Usage:
    python tests/convert_evadb.py results.evadb                  # -> results_converted.evadb
    python tests/convert_evadb.py results.evadb new.evadb
    python tests/convert_evadb.py results.evadb --in-place       # keeps results.evadb.bak
    python tests/convert_evadb.py results.evadb --verify         # also compare every value

Keep `NEW_SCHEMA` below in sync with `CREATE_TABLES` if the format changes again.
"""

import argparse
import os
import sys
import time

try:
    import duckdb
except ImportError:
    sys.exit("The DuckDB Python package is required: pip install duckdb")

# Copy of `CREATE_TABLES` in crates/core/src/storage/duckdb.rs.
NEW_SCHEMA = """
CREATE TABLE objects (
    image_name           VARCHAR NOT NULL,
    image_rel_path       VARCHAR NOT NULL,
    c_stack              INTEGER,
    z_stack              INTEGER,
    t_stack              INTEGER,
    object_id            UUID NOT NULL,
    seg_class_name       VARCHAR,
    seg_class_id         INTEGER,
    object_class_name    VARCHAR,
    object_class_id      INTEGER[],
    parent_id            VARCHAR,
    children             VARCHAR,
    track_id             UBIGINT,
    centroid_x_px        DOUBLE,
    centroid_y_px        DOUBLE,
    centroid_x_nm        DOUBLE,
    centroid_y_nm        DOUBLE,
    bbox_xmin_px         UINTEGER,
    bbox_ymin_px         UINTEGER,
    bbox_xmax_px         UINTEGER,
    bbox_ymax_px         UINTEGER,
    bbox_xmin_nm         DOUBLE,
    bbox_ymin_nm         DOUBLE,
    bbox_xmax_nm         DOUBLE,
    bbox_ymax_nm         DOUBLE,
    area_px              UBIGINT,
    area_nm2             DOUBLE,
    perimeter_px         DOUBLE,
    perimeter_nm         DOUBLE,
    circularity          DOUBLE,
    solidity             DOUBLE,
    aspect_ratio         DOUBLE,
    roundness            DOUBLE,
    compactness          DOUBLE,
    major_axis_px        DOUBLE,
    minor_axis_px        DOUBLE,
    major_axis_nm        DOUBLE,
    minor_axis_nm        DOUBLE,
    major_axis_angle     DOUBLE,
    eccentricity         DOUBLE,
    feret_diameter_px    DOUBLE,
    min_feret_px         DOUBLE,
    feret_diameter_nm    DOUBLE,
    min_feret_nm         DOUBLE,
    touches_edge         BOOLEAN,
    pixel_size_x_nm      DOUBLE,
    pixel_size_y_nm      DOUBLE,
    pixel_size_z_nm      DOUBLE,
    image_bit_depth      UTINYINT,
    intensity_sum_normalized     DOUBLE[],
    intensity_sum_gray           DOUBLE[],
    intensity_mean_normalized    DOUBLE[],
    intensity_mean_gray          DOUBLE[],
    intensity_min_normalized     DOUBLE[],
    intensity_min_gray           DOUBLE[],
    intensity_max_normalized     DOUBLE[],
    intensity_max_gray           DOUBLE[],
    coloc_partner_ids            MAP(INTEGER, UUID[])
);

CREATE TABLE images (
    image_name      VARCHAR NOT NULL,
    image_rel_path  VARCHAR NOT NULL PRIMARY KEY,
    successful      BOOLEAN NOT NULL DEFAULT true,
    error_message   VARCHAR,
    disabled        BOOLEAN NOT NULL DEFAULT false,
    width           UINTEGER NOT NULL,
    height          UINTEGER NOT NULL,
    c_stacks        UINTEGER NOT NULL,
    z_stacks        UINTEGER NOT NULL,
    t_stacks        UINTEGER NOT NULL
);

CREATE TABLE classes (
    class_id  INTEGER NOT NULL PRIMARY KEY,
    name      VARCHAR NOT NULL,
    color     UINTEGER
);
"""

STATS = ["sum", "mean", "min", "max"]
# New column suffix -> the key suffix it was stored under in intensities_json.
KINDS = {"normalized": "raw", "gray": "scaled"}

# Columns copied unchanged (everything up to and including image_bit_depth
# except object_class_id, which gets its type converted).
COPIED_COLUMNS = [
    "image_name", "image_rel_path", "c_stack", "z_stack", "t_stack", "object_id",
    "seg_class_name", "seg_class_id", "object_class_name",
    # object_class_id: converted
    "parent_id", "children", "track_id",
    "centroid_x_px", "centroid_y_px", "centroid_x_nm", "centroid_y_nm",
    "bbox_xmin_px", "bbox_ymin_px", "bbox_xmax_px", "bbox_ymax_px",
    "bbox_xmin_nm", "bbox_ymin_nm", "bbox_xmax_nm", "bbox_ymax_nm",
    "area_px", "area_nm2", "perimeter_px", "perimeter_nm",
    "circularity", "solidity", "aspect_ratio", "roundness", "compactness",
    "major_axis_px", "minor_axis_px", "major_axis_nm", "minor_axis_nm",
    "major_axis_angle", "eccentricity",
    "feret_diameter_px", "min_feret_px", "feret_diameter_nm", "min_feret_nm",
    "touches_edge", "pixel_size_x_nm", "pixel_size_y_nm", "pixel_size_z_nm",
    "image_bit_depth",
]


def intensity_list_sql(stat: str, old_kind: str) -> str:
    """One per-channel list: position c + 1 = channel c's value (NULL if the
    object has no value for that channel), up to its highest channel."""
    ij = "CAST(intensities_json AS JSON)"
    channels = (
        f"range(0, COALESCE(list_max(list_transform(json_keys({ij}), "
        f"k -> TRY_CAST(k AS INTEGER))) + 1, 0))"
    )
    return (
        f"list_transform({channels}, "
        f"c -> CAST(({ij} -> CAST(c AS VARCHAR)) ->> '{stat}_{old_kind}' AS DOUBLE))"
    )


def coloc_map_sql() -> str:
    cj = "CAST(coloc_json AS JSON)"
    return (
        f"CAST(map_from_entries(list_transform(COALESCE(json_keys({cj}), []), "
        f"k -> {{'key': CASE WHEN k = 'unset' THEN -1 ELSE CAST(k AS INTEGER) END, "
        f"'value': CAST(CAST(({cj} -> k) AS VARCHAR[]) AS UUID[])}})) "
        f"AS MAP(INTEGER, UUID[]))"
    )


def objects_insert_sql() -> str:
    targets = COPIED_COLUMNS[:9] + ["object_class_id"] + COPIED_COLUMNS[9:]
    values = COPIED_COLUMNS[:9] + ["CAST(object_class_id AS INTEGER[])"] + COPIED_COLUMNS[9:]
    for stat in STATS:
        for new_kind, old_kind in KINDS.items():
            targets.append(f"intensity_{stat}_{new_kind}")
            values.append(intensity_list_sql(stat, old_kind))
    targets.append("coloc_partner_ids")
    values.append(coloc_map_sql())
    return (
        f"INSERT INTO objects ({', '.join(targets)})\n"
        f"SELECT {', '.join(values)}\nFROM src.objects"
    )


def object_columns(con, database: str) -> set:
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_catalog = ? AND table_name = 'objects'",
        [database],
    ).fetchall()
    return {row[0] for row in rows}


def verify(con) -> int:
    """Compares every converted value with the original; returns the number
    of mismatches."""
    checks = [
        ("objects missing", "SELECT COUNT(*) FROM src.objects o "
         "ANTI JOIN main.objects n USING (object_id)"),
        ("object classes", "SELECT COUNT(*) FROM src.objects o JOIN main.objects n USING (object_id) "
         "WHERE list_sort(CAST(o.object_class_id AS INTEGER[])) IS DISTINCT FROM list_sort(n.object_class_id)"),
    ]
    for stat in STATS:
        for new_kind, old_kind in KINDS.items():
            checks.append((
                f"intensity_{stat}_{new_kind}",
                "SELECT COUNT(*) FROM src.objects o JOIN main.objects n USING (object_id), "
                "range(0, len(n.intensity_mean_normalized)) r(c) "
                f"WHERE CAST((CAST(o.intensities_json AS JSON) -> CAST(c AS VARCHAR)) ->> '{stat}_{old_kind}' AS DOUBLE) "
                f"IS DISTINCT FROM n.intensity_{stat}_{new_kind}[c + 1]",
            ))
    checks.append((
        "colocalization",
        "SELECT COUNT(*) FROM src.objects o JOIN main.objects n USING (object_id) "
        "WHERE list_sort(list_transform(COALESCE(json_keys(CAST(o.coloc_json AS JSON)), []), "
        "k -> CASE WHEN k = 'unset' THEN -1 ELSE CAST(k AS INTEGER) END)) "
        "IS DISTINCT FROM list_sort(map_keys(n.coloc_partner_ids)) "
        "OR EXISTS (SELECT 1 FROM (SELECT UNNEST(json_keys(CAST(o.coloc_json AS JSON))) AS k) "
        "WHERE list_sort(CAST(CAST((CAST(o.coloc_json AS JSON) -> k) AS VARCHAR[]) AS UUID[])) "
        "IS DISTINCT FROM list_sort(n.coloc_partner_ids[CASE WHEN k = 'unset' THEN -1 ELSE CAST(k AS INTEGER) END]))",
    ))
    total = 0
    for name, sql in checks:
        mismatches = con.execute(sql).fetchone()[0]
        total += mismatches
        print(f"  {name:<28} {'ok' if mismatches == 0 else f'{mismatches} MISMATCHES'}")
    return total


def size_mb(path: str) -> float:
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0


class ConversionError(Exception):
    """The input can't be converted (or the result failed verification)."""


def convert(src: str, dst: str, verify_values: bool = False) -> bool:
    """Converts the results database `src` into a new file `dst`.

    Returns False if `src` already has the current format (`dst` is not
    written then). Raises `ConversionError` if `src` has an unknown format, the
    object count changes, or - with `verify_values` - any value differs.
    `src` itself is never changed (apart from merging a pending .wal into it).
    """
    for path in (dst, dst + ".wal"):
        if os.path.exists(path):
            os.remove(path)

    con = duckdb.connect(dst)
    try:
        # A pending .wal next to the input (results the app hadn't
        # checkpointed yet) can only be replayed by a writable attach; it is
        # merged into the input file, without changing any data.
        read_only = not os.path.exists(src + ".wal")
        con.execute(f"ATTACH '{src}' AS src {'(READ_ONLY)' if read_only else ''}")
        if not read_only:
            print(f"merged the pending {os.path.basename(src)}.wal into the input file")

        columns = object_columns(con, "src")
        if "intensity_mean_normalized" in columns:
            return False
        missing = {"intensities_json", "coloc_json", "object_class_id"} - columns
        if missing:
            raise ConversionError(
                f"{src} has an unknown format (missing columns: {', '.join(sorted(missing))})")

        start = time.time()
        con.execute(NEW_SCHEMA)
        con.execute(objects_insert_sql())
        con.execute("INSERT INTO images BY NAME SELECT * FROM src.images")
        con.execute("INSERT INTO classes BY NAME SELECT * FROM src.classes")
        con.execute("CHECKPOINT")
        objects_old = con.execute("SELECT COUNT(*) FROM src.objects").fetchone()[0]
        objects_new = con.execute("SELECT COUNT(*) FROM main.objects").fetchone()[0]
        print(f"converted {objects_new} objects in {time.time() - start:.1f} s")
        if objects_old != objects_new:
            raise ConversionError(
                f"object count differs: {objects_old} before, {objects_new} after")

        if verify_values:
            print("verifying every value:")
            if verify(con) != 0:
                raise ConversionError(f"verification failed - the conversion is in {dst}")
        return True
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", help="results database to convert (.evadb)")
    parser.add_argument("output", nargs="?", help="converted file (default: <input>_converted.evadb)")
    parser.add_argument("--in-place", action="store_true",
                        help="replace the input file; the original is kept as <input>.bak")
    parser.add_argument("--verify", action="store_true",
                        help="compare every converted value with the original afterwards")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = parser.parse_args()

    src = os.path.abspath(args.input)
    if not os.path.exists(src):
        sys.exit(f"{src} does not exist")
    if args.in_place and args.output:
        sys.exit("give either an output file or --in-place, not both")
    root, ext = os.path.splitext(src)
    dst = os.path.abspath(args.output) if args.output else (
        f"{root}.converting{ext}" if args.in_place else f"{root}_converted{ext}")
    if os.path.exists(dst) and not args.force:
        sys.exit(f"{dst} already exists (use --force to overwrite)")

    try:
        converted = convert(src, dst, verify_values=args.verify)
    except ConversionError as error:
        sys.exit(f"{error} - the input is unchanged")
    if not converted:
        if os.path.exists(dst):
            os.remove(dst)
        print(f"{src} already has the current format - nothing to do")
        return 0

    print(f"size: {size_mb(src):.0f} MB -> {size_mb(dst):.0f} MB")
    if args.in_place:
        backup = src + ".bak"
        os.replace(src, backup)
        os.replace(dst, src)
        print(f"replaced {src} (original kept as {backup})")
    else:
        print(f"written to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
