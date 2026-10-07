import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from numpy.testing import assert_allclose

import lkio
from lkio.tpf import detect_mission


def _raw_hdul(path):
    return fits.open(path)


class TestBothMissions:
    """Assertions that should hold for any mission's target pixel file."""

    @pytest.fixture(params=["tess_tpf", "kepler_tpf"])
    def path(self, request):
        return request.getfixturevalue(request.param)

    def test_flux_is_a_cube(self, path):
        ds = lkio.read_tpf(path)
        assert ds["FLUX"].array.ndim == 3

    def test_time_key_and_format(self, path):
        ds = lkio.read_tpf(path)
        assert ds.meta["TIME_KEY"] == "TIME"
        assert ds.meta["TIME_FORMAT"] in ("bkjd", "btjd")

    def test_aperture_shape_and_dtype(self, path):
        ds = lkio.read_tpf(path)
        nrow, ncol = ds["FLUX"].array.shape[-2:]
        assert np.shape(ds.meta["aperture"]) == (nrow, ncol)
        assert np.asarray(ds.meta["aperture"]).dtype == bool

    def test_quality_bitmask_filters_cadences(self, path):
        with _raw_hdul(path) as hdul:
            n_total = len(hdul[1].data)

        ds_none = lkio.read_tpf(path, quality_bitmask="none")
        ds_default = lkio.read_tpf(path, quality_bitmask="default")
        ds_hardest = lkio.read_tpf(path, quality_bitmask="hardest")

        assert len(ds_none.index) <= n_total
        assert len(ds_none.index) >= len(ds_default.index) >= len(ds_hardest.index)
        assert len(ds_default.index) == len(ds_default.meta["timemask"][0])

    def test_error_columns_matched_correctly(self, path):
        ds = lkio.read_tpf(path)
        with _raw_hdul(path) as hdul:
            timemask = ds.meta["timemask"]
            raw_flux_err = np.asarray(hdul[1].data["FLUX_ERR"])[timemask]
            raw_bkg_err = np.asarray(hdul[1].data["FLUX_BKG_ERR"])[timemask]

        assert_allclose(
            np.asarray(ds["FLUX"].uncertainty.array), raw_flux_err, equal_nan=True
        )
        assert_allclose(
            np.asarray(ds["FLUX_BKG"].uncertainty.array), raw_bkg_err, equal_nan=True
        )

    def test_mission_override_is_case_invariant(self, path):
        mission_name = lkio.read_tpf(path).meta["MISSION"]
        key = mission_name.lower()
        ds_lower = lkio.read_tpf(path, mission=key)
        ds_mixed = lkio.read_tpf(path, mission=key.upper())
        assert ds_lower.meta["MISSION"] == ds_mixed.meta["MISSION"] == mission_name

    def test_unsupported_mission_raises(self, path):
        with pytest.raises(ValueError, match="mission"):
            lkio.read_tpf(path, mission="hubble")


class TestTess:
    def test_time_format_is_btjd(self, tess_tpf):
        ds = lkio.read_tpf(tess_tpf)
        assert ds.meta["TIME_FORMAT"] == "btjd"
        assert ds.meta["MISSION"] == "TESS"

    def test_detect_mission(self, tess_tpf):
        with _raw_hdul(tess_tpf) as hdul:
            mission, time_format = detect_mission(hdul)
        assert mission == "TESS"
        assert time_format == "btjd"

    def test_column_row_match_header(self, tess_tpf):
        ds = lkio.read_tpf(tess_tpf)
        with _raw_hdul(tess_tpf) as hdul:
            header1 = hdul[1].header
        assert ds.meta["column"] == header1.get("1CRV5P", 0)
        assert ds.meta["row"] == header1.get("2CRV5P", 0)


class TestKepler:
    def test_time_format_is_bkjd(self, kepler_tpf):
        ds = lkio.read_tpf(kepler_tpf)
        assert ds.meta["TIME_FORMAT"] == "bkjd"
        assert ds.meta["MISSION"] in ("Kepler", "K2")

    def test_detect_mission(self, kepler_tpf):
        with _raw_hdul(kepler_tpf) as hdul:
            mission, time_format = detect_mission(hdul)
        assert mission in ("Kepler", "K2")
        assert time_format == "bkjd"

    def test_targetid_is_keplerid(self, kepler_tpf):
        ds = lkio.read_tpf(kepler_tpf)
        with _raw_hdul(kepler_tpf) as hdul:
            keplerid = hdul[0].header["KEPLERID"]
        assert ds.meta["TARGETID"] == keplerid

    def test_column_row_match_header(self, kepler_tpf):
        ds = lkio.read_tpf(kepler_tpf)
        with _raw_hdul(kepler_tpf) as hdul:
            header1 = hdul[1].header
        assert ds.meta["column"] == header1.get("1CRV5P", 0)
        assert ds.meta["row"] == header1.get("2CRV5P", 0)

    def test_flux_units_are_electron_per_s(self, kepler_tpf):
        ds = lkio.read_tpf(kepler_tpf)
        assert ds.meta["units"].get("FLUX") == "electron/s"

    def test_wcs_cross_check(self, kepler_tpf):
        """The APERTURE-extension WCS and the column-5 remap must agree.

        If this fails, the ext-2-first ordering in `_build_tpf_wcs` is wrong
        for Kepler products and the original ORIGIN-based heuristic needs to
        be restored for this mission.
        """
        with _raw_hdul(kepler_tpf) as hdul:
            wcs_ext2 = WCS(hdul[2].header)
            header1 = hdul[1].header
            remap = fits.Header()
            keymap = {
                "1CTYP5": "CTYPE1",
                "2CTYP5": "CTYPE2",
                "1CRPX5": "CRPIX1",
                "2CRPX5": "CRPIX2",
                "1CRVL5": "CRVAL1",
                "2CRVL5": "CRVAL2",
                "1CUNI5": "CUNIT1",
                "2CUNI5": "CUNIT2",
                "1CDLT5": "CDELT1",
                "2CDLT5": "CDELT2",
                "11PC5": "PC1_1",
                "12PC5": "PC1_2",
                "21PC5": "PC2_1",
                "22PC5": "PC2_2",
            }
            for old_key, new_key in keymap.items():
                if old_key in header1:
                    remap[new_key] = header1[old_key]
            wcs_remap = WCS(remap)

        sky_ext2 = wcs_ext2.all_pix2world([[2, 3]], 0)
        sky_remap = wcs_remap.all_pix2world([[2, 3]], 0)
        assert_allclose(sky_ext2, sky_remap, atol=1e-4)
