# Test data

## multi-channel-4D-series.ome.tif

OME-TIFF sample data from the Open Microscopy Environment: the artificial
Bio-Formats dataset "multi-channel-4D-series" (3 channels, 5 focal planes,
7 time points, 439 × 167 pixels, 8 bit; every plane labeled with its
position).

- Source: https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/bioformats-artificial/multi-channel-4D-series.ome.tif
  (listed on https://docs.openmicroscopy.org/ome-model/5.4.0/ome-tiff/data.html),
  unchanged (MD5 `fbc9a4e40508a58e05a7b5bef9c82f07`, same as the download).
- Copyright: Sample OME-TIFF files (c) by the OME Consortium.
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
  (CC BY 4.0), as stated in the `readme.txt` and `COPYING` next to the file
  on downloads.openmicroscopy.org.

## slice_Z0_C0_T0.tif and the .raw files

Derived from `multi-channel-4D-series.ome.tif` (same CC BY 4.0 license and
attribution as above): `slice_Z0_C0_T0.tif` is its plane Z0 C0 T0 saved by
ImageJ (same 439 × 167 pixels, 8 bit), and the `.raw` files are 8-bit
planes (439 × 167 bytes each) the image reader's tests compare its reads
of the OME-TIFF against - `slice_Z0_C0_T0.raw` / `slice_Z1_C0_T0.raw`
single planes, `slice_Z0_C0_T0_{avg,min,max}_intensity.raw` its average,
minimum and maximum Z projections.

## The other files

`project_test/output/` holds images the job executor's tests write. The
`*.json` files are reference values for the measurement tests (see the
test that reads each one for how it was made).
