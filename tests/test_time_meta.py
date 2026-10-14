from pathlib import Path

import pytest
from astropy.io import fits

import lkio
from lkio.utils import detect_mission, get_time_scale

TESTFILES = Path(__file__).parent.parent / "testfiles"
TESS_LC = TESTFILES / "tess2020324010417-s0032-0000000685125879-0200-s_lc.fits"
UNKNOWN = TESTFILES / "wasp-155_190817_achromatic_k.fits"


def _require(path):
    if not path.exists():
        pytest.skip(f"{path} not found")
    return path


class TestDetectMission:
    def test_tess_lightcurve(self):
        with fits.open(_require(TESS_LC)) as hdul:
            assert detect_mission(hdul) == ("TESS", "btjd")

    def test_tess_tpf(self, tess_tpf):
        with fits.open(tess_tpf) as hdul:
            assert detect_mission(hdul) == ("TESS", "btjd")

    def test_kepler_tpf(self, kepler_tpf):
        with fits.open(kepler_tpf) as hdul:
            mission, time_format = detect_mission(hdul)
        assert mission in ("Kepler", "K2")
        assert time_format == "bkjd"

    def test_unknown(self):
        with fits.open(_require(UNKNOWN)) as hdul:
            assert detect_mission(hdul) == ("Unknown", "jd")


class TestGetTimeScale:
    def test_reads_timesys_lowercased(self):
        assert get_time_scale(fits.Header({"TIMESYS": "TDB"})) == "tdb"

    def test_defaults_to_tdb(self):
        assert get_time_scale(fits.Header()) == "tdb"


class TestReadGenericTimeMeta:
    def test_known_mission_sets_keys(self):
        ds = lkio.read_generic(_require(TESS_LC))
        assert ds.meta["TIME_FORMAT"] == "btjd"
        assert ds.meta["TIME_SCALE"] == "tdb"

    def test_unrecognized_file_leaves_keys_unset(self):
        ds = lkio.read_generic(_require(UNKNOWN))
        assert "TIME_FORMAT" not in ds.meta
        assert "TIME_SCALE" not in ds.meta
