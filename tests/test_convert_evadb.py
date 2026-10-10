"""Tests for convert_evadb.py.

Run with:  python -m unittest tests/test_convert_evadb.py
(needs the DuckDB Python package: pip install duckdb)
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import duckdb
    import convert_evadb
except ImportError:  # pragma: no cover - reported as a skip below
    duckdb = None

# The `objects` table as results files had it before the conversion: the
# columns the converter reads, with the old types.
OLD_SCHEMA = """
CREATE TABLE objects (
    image_name VARCHAR NOT NULL, image_rel_path VARCHAR NOT NULL,
    c_stack INTEGER, z_stack INTEGER, t_stack INTEGER,
    object_id UUID NOT NULL,
    seg_class_name VARCHAR, seg_class_id INTEGER,
    object_class_name VARCHAR, object_class_id VARCHAR,
    parent_id VARCHAR, children VARCHAR, track_id UBIGINT,
    centroid_x_px DOUBLE, centroid_y_px DOUBLE, centroid_x_nm DOUBLE, centroid_y_nm DOUBLE,
    bbox_xmin_px UINTEGER, bbox_ymin_px UINTEGER, bbox_xmax_px UINTEGER, bbox_ymax_px UINTEGER,
    bbox_xmin_nm DOUBLE, bbox_ymin_nm DOUBLE, bbox_xmax_nm DOUBLE, bbox_ymax_nm DOUBLE,
    area_px UBIGINT, area_nm2 DOUBLE, perimeter_px DOUBLE, perimeter_nm DOUBLE,
    circularity DOUBLE, solidity DOUBLE, aspect_ratio DOUBLE, roundness DOUBLE, compactness DOUBLE,
    major_axis_px DOUBLE, minor_axis_px DOUBLE, major_axis_nm DOUBLE, minor_axis_nm DOUBLE,
    major_axis_angle DOUBLE, eccentricity DOUBLE,
    feret_diameter_px DOUBLE, min_feret_px DOUBLE, feret_diameter_nm DOUBLE, min_feret_nm DOUBLE,
    touches_edge BOOLEAN,
    pixel_size_x_nm DOUBLE, pixel_size_y_nm DOUBLE, pixel_size_z_nm DOUBLE,
    image_bit_depth UTINYINT,
    intensities_json JSON, coloc_json JSON
);
CREATE TABLE images (
    image_name VARCHAR NOT NULL, image_rel_path VARCHAR NOT NULL PRIMARY KEY,
    successful BOOLEAN NOT NULL DEFAULT true, error_message VARCHAR,
    disabled BOOLEAN NOT NULL DEFAULT false,
    width UINTEGER NOT NULL, height UINTEGER NOT NULL,
    c_stacks UINTEGER NOT NULL, z_stacks UINTEGER NOT NULL, t_stacks UINTEGER NOT NULL
);
CREATE TABLE classes (class_id INTEGER NOT NULL PRIMARY KEY, name VARCHAR NOT NULL, color UINTEGER);
"""

PARTNER_A = "00000000-0000-0000-0000-00000000000a"
PARTNER_B = "00000000-0000-0000-0000-00000000000b"

# object id suffix -> (object_class_id, intensities_json, coloc_json)
OBJECTS = {
    # Channels 0 and 2 measured, channel 1 not; two partner classes.
    1: ("[1,3]",
        '{"0":{"sum_raw":0.5,"sum_scaled":127.5,"mean_raw":0.25,"mean_scaled":63.75,'
        '"min_raw":0.1,"min_scaled":25.5,"max_raw":0.9,"max_scaled":229.5},'
        '"2":{"sum_raw":1.5,"sum_scaled":382.5,"mean_raw":0.75,"mean_scaled":191.25,'
        '"min_raw":0.2,"min_scaled":51.0,"max_raw":1.0,"max_scaled":255.0}}',
        f'{{"4":["{PARTNER_A}","{PARTNER_B}"],"7":["{PARTNER_A}"]}}'),
    # Nothing measured, no colocalization - stored as NULL.
    2: ("[]", None, None),
    # Nothing measured, no colocalization - stored as empty objects.
    3: ("[2]", "{}", "{}"),
    # A partner without an object class.
    4: ("[2]", "{}", f'{{"unset":["{PARTNER_B}"]}}'),
}


def object_id(n: int) -> str:
    return f"00000000-0000-0000-0000-{n:012d}"


@unittest.skipIf(duckdb is None, "needs the DuckDB Python package (pip install duckdb)")
class ConvertEvadbTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.dir.name, "old.evadb")
        self.dst = os.path.join(self.dir.name, "new.evadb")
        con = duckdb.connect(self.src)
        con.execute(OLD_SCHEMA)
        for n, (classes, intensities, coloc) in OBJECTS.items():
            con.execute(
                "INSERT INTO objects (image_name, image_rel_path, object_id, object_class_id, "
                "area_px, image_bit_depth, intensities_json, coloc_json) "
                "VALUES ('a.tif', 'a.tif', ?, ?, 10, 8, ?, ?)",
                [object_id(n), classes, intensities, coloc],
            )
        con.execute("INSERT INTO images (image_name, image_rel_path, width, height, c_stacks, "
                    "z_stacks, t_stacks) VALUES ('a.tif', 'a.tif', 64, 64, 3, 1, 1)")
        con.execute("INSERT INTO classes VALUES (1, 'Cell', 0), (2, 'Spot', 0)")
        con.close()

    def tearDown(self):
        self.dir.cleanup()

    def convert(self, verify_values=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return convert_evadb.convert(self.src, self.dst, verify_values=verify_values)

    def row(self, n: int, columns: str):
        with duckdb.connect(self.dst, read_only=True) as con:
            return con.execute(
                f"SELECT {columns} FROM objects WHERE object_id = ?", [object_id(n)]
            ).fetchone()

    def test_intensities_become_per_channel_lists_with_null_for_an_unmeasured_channel(self):
        self.convert()

        normalized, gray = self.row(1, "intensity_mean_normalized, intensity_mean_gray")
        self.assertEqual(normalized, [0.25, None, 0.75])
        self.assertEqual(gray, [63.75, None, 191.25])
        self.assertEqual(self.row(1, "intensity_sum_normalized")[0], [0.5, None, 1.5])
        self.assertEqual(self.row(1, "intensity_max_gray")[0], [229.5, None, 255.0])

    def test_objects_without_intensities_get_empty_lists(self):
        self.convert()

        self.assertEqual(self.row(2, "intensity_mean_normalized")[0], [])  # was NULL
        self.assertEqual(self.row(3, "intensity_mean_normalized")[0], [])  # was {}

    def test_colocalization_becomes_a_class_to_partner_map(self):
        self.convert()

        partners_of_4, partners_of_7 = self.row(
            1, "list_sort(CAST(coloc_partner_ids[4] AS VARCHAR[])), "
               "CAST(coloc_partner_ids[7] AS VARCHAR[])")
        self.assertEqual(partners_of_4, [PARTNER_A, PARTNER_B])
        self.assertEqual(partners_of_7, [PARTNER_A])
        self.assertEqual(self.row(1, "typeof(coloc_partner_ids)")[0], "MAP(INTEGER, UUID[])")

    def test_objects_without_colocalization_get_an_empty_map(self):
        self.convert()

        self.assertEqual(self.row(2, "cardinality(coloc_partner_ids)")[0], 0)  # was NULL
        self.assertEqual(self.row(3, "cardinality(coloc_partner_ids)")[0], 0)  # was {}

    def test_partners_without_a_class_are_stored_under_key_minus_one(self):
        self.convert()

        self.assertEqual(
            self.row(4, "CAST(coloc_partner_ids[-1] AS VARCHAR[])")[0], [PARTNER_B])

    def test_object_classes_become_an_integer_list(self):
        self.convert()

        self.assertEqual(self.row(1, "object_class_id")[0], [1, 3])
        self.assertEqual(self.row(2, "object_class_id")[0], [])

    def test_images_and_classes_are_copied(self):
        self.convert()

        with duckdb.connect(self.dst, read_only=True) as con:
            self.assertEqual(con.execute("SELECT c_stacks FROM images").fetchall(), [(3,)])
            self.assertEqual(
                con.execute("SELECT name FROM classes ORDER BY class_id").fetchall(),
                [("Cell",), ("Spot",)])

    def test_verification_passes_for_every_edge_case(self):
        # NULL and empty intensities/colocalization, a channel gap and an
        # unset partner class must all compare equal to their originals.
        self.assertTrue(self.convert(verify_values=True))

    def test_verification_catches_a_changed_value(self):
        self.convert()
        with duckdb.connect(self.dst) as con:
            con.execute("UPDATE objects SET intensity_mean_gray = [63.75, NULL, 999.0] "
                        "WHERE object_id = ?", [object_id(1)])
            con.execute(f"ATTACH '{self.src}' AS src (READ_ONLY)")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(convert_evadb.verify(con), 1)

    def test_verification_catches_a_missing_partner(self):
        self.convert()
        with duckdb.connect(self.dst) as con:
            con.execute("UPDATE objects SET coloc_partner_ids = MAP {} WHERE object_id = ?",
                        [object_id(1)])
            con.execute(f"ATTACH '{self.src}' AS src (READ_ONLY)")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(convert_evadb.verify(con), 1)

    def test_a_file_in_the_current_format_is_left_alone(self):
        self.convert()
        os.replace(self.dst, self.src)

        self.assertFalse(self.convert())

    def test_an_unknown_format_is_refused(self):
        with duckdb.connect(self.src) as con:
            con.execute("ALTER TABLE objects DROP COLUMN coloc_json")

        with self.assertRaises(convert_evadb.ConversionError):
            self.convert()


if __name__ == "__main__":
    unittest.main()
