"""QLP specific reading function"""

from collections.abc import Iterable
from pathlib import Path

import lkdata
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .generic import get_hdul, read_generic
from .utils import build_aperture


def read_qlp(
    file: str | Path | fits.HDUList,
    flux_key: str | None = None,
    time_keys: Iterable[str] | None = None,
    data_keys: Iterable[str] | None = None,
    error_syntax: str | Iterable[str] | None = "{key}_ERR",
    boolean_keys: Iterable[str] | None = None,
    bitwise_keys: Iterable[str] | None = None,
) -> lkdata.DataSet:
    hdu = get_hdul(file)
    time_keys = time_keys or ["TIME", "CADENCENO", "ORBITID"]
    data_keys = data_keys or [
        "SAP_FLUX",
        "DET_FLUX",
        "KSPSAP_FLUX",
        "SAP_X",
        "SAP_Y",
        "SAP_BKG",
        "KSPSAP_FLUX_SML",
        "KSPSAP_FLUX_LAG",
    ]
    boolean_keys = boolean_keys or []
    bitwise_keys = bitwise_keys or ["QUALITY"]

    ds = read_generic(
        file=hdu,
        ext=1,
        time_keys=time_keys,
        data_keys=data_keys,
        error_syntax=error_syntax,
        boolean_keys=boolean_keys,
        bitwise_keys=bitwise_keys,
        close_file=False,
    )

    ds.meta["wcs"] = WCS(hdu[1].header)

    # QLP changes for extended mission 2 (sector 56+). Now using “DET_FLUX” and “DET_FLUX_ERR” instead of “KSPSAP_FLUX” and “KSPSAP_FLUX_ERR” for the detrended flux and detrended flux error, respectively
    if not flux_key:
        flux_key = "DET_FLUX" if "DET_FLUX" in ds.data_products else "KSPSAP_FLUX"

    ds.data_products["FLUX"] = ds.data_products[flux_key]
    ds.meta["FLUX_ORIGIN"] = flux_key

    # `IndexError`/`TypeError` from a missing or unusable aperture extension
    # fall back to an all-True mask shaped like the aperture extension would
    # have been (QLP products carry no pixel cube to infer a shape from).
    if len(hdu) > 2 and hdu[2].data is not None:
        fallback_nrow, fallback_ncol = np.shape(hdu[2].data)
    else:
        fallback_nrow, fallback_ncol = (1, 1)
    ds.meta["aperture"] = build_aperture(hdu, (fallback_nrow, fallback_ncol))

    # TODO: I don't actually see units in the header, except for time?

    return ds
