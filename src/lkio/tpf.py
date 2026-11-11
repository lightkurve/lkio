"""Target Pixel File (TPF) specific reading function.

Target pixel files from Kepler, K2, and TESS share a common structure: a
primary HDU, a binary table extension (EXT1) containing TIME/FLUX/QUALITY
columns plus a column-5 WCS, and (usually) an APERTURE image extension
(EXT2). This module knows how to sniff the mission from the file, apply the
right quality bitmask, build the pixel WCS, and populate all of the metadata
needed to support pixel-level science (aperture photometry, centroiding,
background estimation).
"""

import contextlib
import warnings
from pathlib import Path
from typing import Iterable, Optional, Tuple, Type, Union

import lkdata
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .generic import get_hdul, read_generic
from .quality import MISSION_QUALITY_FLAGS, QualityFlags
from .utils import (
    build_aperture,
    detect_mission,
    get_time_scale,
    load_cosmicray_extension,
)

__all__ = ["read_tpf"]

#: Column-5 WCS keywords (as stored in the EXT1 binary table header) mapped
#: onto the standard image-WCS keywords they correspond to.
_COLUMN5_WCS_KEYS = {
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


def _normalize_mission(
    mission: Optional[str], hdul: fits.HDUList
) -> Tuple[str, str, Type[QualityFlags]]:
    """Resolve the `mission` argument into a (name, time_format, flag-class) triple."""
    detected_name, detected_format = detect_mission(hdul)

    if mission is None:
        mission_name, time_format = detected_name, detected_format
    else:
        key = mission.strip().lower()
        if key not in MISSION_QUALITY_FLAGS:
            valid_options = tuple(MISSION_QUALITY_FLAGS.keys())
            msg = f"mission='{mission}' is not supported, expected one of {valid_options}"
            raise ValueError(msg)
        mission_name = "TESS" if key == "tess" else ("K2" if key == "k2" else "Kepler")
        time_format = "btjd" if key == "tess" else "bkjd"
        if detected_name != "Unknown" and detected_format != time_format:
            msg = (
                f"mission='{mission}' was requested but the file's BJDREFI "
                f"header suggests '{detected_name}' ({detected_format}); "
                "using the requested mission's time format."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)

    if mission_name == "Unknown":
        warnings.warn(
            "Could not detect the mission for this target pixel file; "
            "no quality bitmask filtering will be applied.",
            UserWarning,
            stacklevel=3,
        )
        return mission_name, time_format, QualityFlags

    qf_cls = MISSION_QUALITY_FLAGS[mission_name.lower()]
    return mission_name, time_format, qf_cls


def _build_tpf_wcs(hdul: fits.HDUList) -> Optional[WCS]:
    """Build the pixel-to-sky WCS for a target pixel file.

    Prefers the full celestial WCS on the APERTURE extension (EXT2), which
    both Kepler/K2 and SPOC/TESScut products carry. Falls back to remapping
    the column-5 WCS keywords stored in the EXT1 binary table header.
    """
    if len(hdul) > 2:
        try:
            header2 = hdul[2].header
            if "CTYPE1" in header2:
                return WCS(header2)
        except (IndexError, KeyError, ValueError):
            pass

    try:
        header1 = hdul[1].header
        new_header = fits.Header()
        for old_key, new_key in _COLUMN5_WCS_KEYS.items():
            if old_key in header1:
                new_header[new_key] = header1[old_key]
        new_header["NAXIS1"] = header1.get("NAXIS1")
        new_header["NAXIS2"] = header1.get("NAXIS2")
        return WCS(new_header)
    except (IndexError, KeyError, ValueError):
        return None


def _normalize_units(units: dict) -> dict:
    """Map non-standard astropy unit strings onto their canonical form."""
    remap = {"e-/s": "electron/s", "electron / s": "electron/s"}
    return {key: remap.get(value, value) for key, value in units.items()}


def read_tpf(
    file: Union[str, Path, fits.HDUList],
    mission: Optional[str] = None,
    quality_bitmask: Optional[Union[int, str]] = "default",
    flux_key: str = "FLUX",
    time_keys: Optional[Iterable[str]] = None,
    data_keys: Optional[Iterable[str]] = None,
    error_syntax: Optional[Union[str, Iterable[str]]] = "{key}_ERR",
    boolean_keys: Optional[Iterable[str]] = None,
    bitwise_keys: Optional[Iterable[str]] = None,
) -> lkdata.DataSet:
    """Read a Kepler, K2, or TESS target pixel file into an `lkdata.DataSet`.

    Parameters
    ----------
    file : str, pathlib.Path, astropy.io.fits.HDUList
        The path to the fits file or HDUList.
    mission : str, optional
        One of "Kepler", "K2", or "TESS" (case-insensitive). If not given,
        the mission is auto-detected from the file's header.
    quality_bitmask : int or str, optional
        Bitmask (int) or one of 'none', 'default', 'hard', or 'hardest'.
        Cadences flagged by the bitmask are dropped. By default 'default'.
    flux_key : str, optional
        The data column to use as the 'FLUX' product, by default 'FLUX'.
    time_keys : list, optional
        Time data contained in the fits file, by default
        ['TIME', 'TIMECORR', 'CADENCENO'].
    data_keys : list, optional
        Series or cube data contained by the fits file, by default
        ['FLUX', 'RAW_CNTS', 'FLUX_BKG', 'COSMIC_RAY', 'POS_CORR1', 'POS_CORR2'].
    boolean_keys : list, optional
        Boolean series or cube data contained by the fits file, by default [].
    bitwise_keys : list, optional
        Bitwise series or cube data, by default ['QUALITY'].

    Returns
    -------
    ds : lkdata.DataSet
        DataSet containing pixel data, time indices, and target/mission
        metadata from the target pixel file.
    """
    opened_here = not isinstance(file, fits.HDUList)
    hdul = get_hdul(file)
    try:
        if len(hdul) < 2 or not isinstance(hdul[1], (fits.TableHDU, fits.BinTableHDU)):
            msg = f"{file!r} does not contain a binary table in HDU 1"\
                  " -- is this a target pixel file?"
            raise ValueError(msg)
        column_names = hdul[1].columns.names
        if "FLUX" not in column_names or "QUALITY" not in column_names:
            msg = (
                f"{file!r} is missing FLUX and/or QUALITY columns in HDU 1 "
                "-- is this a target pixel file?"
            )
            raise ValueError(msg)

        mission_name, time_format, qf_cls = _normalize_mission(mission, hdul)

        quality_mask = qf_cls.create_quality_mask(
            hdul[1].data["QUALITY"], bitmask=quality_bitmask
        )
        if mission_name == "TESS" and quality_bitmask not in (0, "none", None):
            quality_mask = quality_mask & np.isfinite(hdul[1].data["TIME"])

        time_keys = time_keys or ["TIME", "TIMECORR", "CADENCENO"]
        bitwise_keys = bitwise_keys or ["QUALITY"]
        boolean_keys = boolean_keys or []
        data_keys = data_keys or [
            "FLUX",
            "RAW_CNTS",
            "FLUX_BKG",
            "COSMIC_RAY",
            "POS_CORR1",
            "POS_CORR2",
        ]

        ds = read_generic(
            file=hdul,
            ext=1,
            time_keys=time_keys,
            data_keys=data_keys,
            error_syntax=error_syntax,
            boolean_keys=boolean_keys,
            bitwise_keys=bitwise_keys,
            close_file=False,
            cadence_mask=quality_mask,
        )

        if flux_key not in ds.data_products:
            msg = (
                f"flux_key='{flux_key}' was not found among the read data "
                f"products {list(ds.data_products.keys())}."
            )
            raise ValueError(msg)

        shape = ds.data_products[flux_key].array.shape

        ds.meta["units"] = _normalize_units(ds.meta.get("units", {}))
        ds.meta["TIME_FORMAT"] = time_format
        ds.meta["TIME_SCALE"] = get_time_scale(hdul[1].header)
        ds.meta["TIME_KEY"] = "TIME"
        ds.meta["FLUX_ORIGIN"] = flux_key
        ds.meta["MISSION"] = mission_name
        ds.meta["QUALITY_BITMASK"] = quality_bitmask

        header0 = hdul[0].header
        if mission_name in ("Kepler", "K2"):
            targetid = header0.get("KEPLERID")
        elif mission_name == "TESS":
            targetid = header0.get("TICID")
        else:
            targetid = None
        if targetid is None:
            targetid = header0.get("OBJECT")
        ds.meta["TARGETID"] = targetid
        ds.meta["LABEL"] = header0.get("OBJECT") or (
            str(targetid) if targetid is not None else None
        )

        ds.meta["aperture"] = build_aperture(hdul, shape)
        ds.meta["wcs"] = _build_tpf_wcs(hdul)
        ds.meta["column"] = int(hdul[1].header.get("1CRV5P", 0))
        ds.meta["row"] = int(hdul[1].header.get("2CRV5P", 0))

        if "QUALITY" in ds.bitwise_products:
            with contextlib.suppress(TypeError, ValueError):
                ds.bitwise_products["QUALITY"] = lkdata.BitwiseSeries(
                    np.asarray(ds["QUALITY"].apply(int)),
                    code_dict=qf_cls.STRINGS,
                    index=ds.index,
                )

        if "COSMIC_RAY" not in ds.data_products:
            cr = load_cosmicray_extension(hdul)
            if cr is not None:
                ds.meta["cosmic ray"] = True
                ds.data_products["COSMIC_RAY"] = lkdata.DataCube(
                    data=cr[ds.meta["timemask"]]
                )
            else:
                ds.meta["cosmic ray"] = False
        else:
            ds.meta["cosmic ray"] = True

        return ds
    finally:
        if opened_here:
            hdul.close()
