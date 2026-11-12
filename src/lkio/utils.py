import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

ppm = u.def_unit("ppm", 1e-6 * u.dimensionless_unscaled)


def detect_mission(hdul: fits.HDUList) -> tuple[str, str]:
    """Sniff the mission and time format from a Kepler, K2, or TESS file.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        The opened fits file.

    Returns
    -------
    mission, time_format : str, str
        `mission` is one of "Kepler", "K2", "TESS", or "Unknown".
        `time_format` is one of "bkjd", "btjd", or "jd".
    """
    header0 = hdul[0].header
    header1 = hdul[1].header if len(hdul) > 1 else fits.Header()

    bjdrefi = header1.get("BJDREFI")
    if bjdrefi == 2457000:
        return "TESS", "btjd"
    if bjdrefi == 2454833:
        if header0.get("MISSION") == "K2" or "CAMPAIGN" in header0:
            return "K2", "bkjd"
        return "Kepler", "bkjd"

    telescop = str(header0.get("TELESCOP", "")).upper()
    if telescop == "TESS":
        return "TESS", "btjd"
    if telescop == "KEPLER":
        if header0.get("MISSION") == "K2" or "CAMPAIGN" in header0:
            return "K2", "bkjd"
        return "Kepler", "bkjd"

    return "Unknown", "jd"


def get_time_scale(header: fits.Header) -> str:
    """Return the lowercased TIMESYS time scale of `header`, default "tdb"."""
    return str(header.get("TIMESYS", "tdb")).lower()


def build_aperture(hdul: fits.HDUList, shape: tuple[int, ...]) -> np.ndarray:
    """Build a 2-D pipeline aperture mask from the APERTURE extension.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        The opened target pixel file.
    nrow, ncol : int
        Shape to fall back to when the aperture extension is missing or
        unusable (e.g. early TESScut products stored floats in HDU 2).

    Returns
    -------
    aperture : np.ndarray of bool, shape (nrow, ncol)
    """
    try:
        return np.asarray(hdul[2].data & 2 > 0)
    except (IndexError, TypeError):
        return np.ones(shape, dtype=bool)


def load_cosmicray_extension(hdu: fits.HDUList) -> np.ndarray | None:
    """Function to load the cosmic ray extension into a datacube

    Parameters:
    -----------
    tpf: lk.TargetPixelFile
        Input Target Pixel File

    Returns:
    -------
    cr: np.ndarray
        Array containing cosmic ray fluxes of shape tpf.shape
    """
    cadenceno = hdu[1].data["CADENCENO"]

    # Get cosmic ray information
    l = np.where([h.name == "TARGET COSMIC RAY" for h in hdu])[0]
    if len(l) != 1:
        return None
    h = hdu[l[0]]
    c, x, y, f = [
        h.data[attr].copy() for attr in ["CADENCENO", "RAWX", "RAWY", "COSMIC_RAY"]
    ]
    if len(c) == 0:
        return None
    x -= hdu[1].header["1CRV5P"]  # tpf.column
    y -= hdu[1].header["2CRV5P"]  # tpf.row

    # Mask down to only cosmic rays in cadences within the TPF
    k = np.isin(c, cadenceno)
    c, x, y, f = c[k], x[k], y[k], f[k]

    # Map cosmic ray cadences to array indices
    r = {cadenceno[idx]: idx for idx in range(hdu[1].data["FLUX"].shape[0])}
    inv = np.asarray([r[c1] for c1 in c])

    # Build a cube
    cr = np.zeros(hdu[1].data["FLUX"].shape)
    cr[inv, y, x] = f
    return cr


def return_unit(tab: Table) -> None:
    # TODO: I think this will move to lkv3, not required for io.
    for colname in tab.colnames:
        # Ensure units have the correct astropy format
        # Speed-up: comparing units by their string representation is 1000x
        # faster than performing full-blown unit comparison
        unitstr = str(tab[colname].unit)
        if unitstr == "e-/s":
            tab[colname].unit = "electron/s"
        # Fixes issue #1504
        if unitstr == "electron / s":  # lightcurve.to_fits() saves as this string
            tab[colname].unit = "electron/s"
        elif unitstr == "pixels":
            tab[colname].unit = "pixel"
        elif unitstr == "ppm" and repr(tab[colname].unit).startswith("Unrecognized"):
            # Workaround for issue #956
            tab[colname].unit = ppm
        elif unitstr == "ADU":
            tab[colname].unit = "adu"
        elif unitstr.lower() == "unitless":
            tab[colname].unit = ""
        elif unitstr.lower() == "degcelcius":
            # CDIPS has non-astropy units
            tab[colname].unit = "deg_C"
        # Rename columns to lowercase
        tab.rename_column(colname, colname.lower())
