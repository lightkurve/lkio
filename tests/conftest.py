from pathlib import Path

import pytest

TESTFILES = Path(__file__).parent / "data"


@pytest.fixture
def tess_tpf():
    """Path to a TESS target pixel file (a tess-asteroids moving-target product).

    Note: this file has no TICID, RA_OBJ/DEC_OBJ == 0, and bogus column-5 WCS
    reference values, so it cannot validate targetid/ra/dec/WCS correctness --
    only that the read path produces a well-formed DataSet.
    """
    path = TESTFILES / "tess25155310-s01-first-cadences.fits.gz"
    if not path.exists():
        pytest.skip(f"TESS test TPF not found at {path}")
    return path


@pytest.fixture
def kepler_tpf():
    """Path to a Kepler or K2 target pixel file, supplied by the user.

    No such file ships with the repository; this fixture skips when absent.
    """
    path = TESTFILES / "test-tpf-kplr-tabby-100-cadences.fits"
    if not path.exists():
        pytest.skip(f"Kepler test TPF not found at {path}")
    return path
