"""QLP specific reading function"""

from pathlib import Path
from typing import Iterable, Optional, Union

from astropy.io import fits
from astropy.wcs import WCS

from .generic import get_hdul, read_generic


def read_qlp(
    file: Union[str, Path, fits.HDUList],
    flux_key: Optional[str] = None,
    time_keys: Optional[Iterable[str]] = None,
    data_keys: Optional[Iterable[str]] = None,
    error_syntax: Optional[Union[str, Iterable[str]]] = "{key}_ERR",
    boolean_keys: Optional[Iterable[str]] = None,
    bitwise_keys: Optional[Iterable[str]] = None,
):

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
        if "DET_FLUX" in ds.data_products.keys():
            flux_key = "DET_FLUX"
        else:
            flux_key = "KSPSAP_FLUX"

    ds.data_products["FLUX"] = ds.data_products[flux_key]
    ds.meta["FLUX_ORIGIN"] = flux_key

    # `IndexError` may be raised if the aperture extension (#2) is missing
    # `TypeError` may be raised because early versions of TESScut returned floats in HDU 2
    try:
        ds.meta["aperture"] = hdu[2].data & 2 > 0
    except (IndexError, TypeError):
        ds.meta["aperture"] = np.ones(ds.data_products["FLUX"].array.shape, dtype=bool)

    # TODO: I don't actually see units in the header, except for time?

    return ds
