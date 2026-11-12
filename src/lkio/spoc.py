"""SPOC specific reading function"""

from collections.abc import Iterable
from pathlib import Path

import lkdata
from astropy.io import fits
from astropy.wcs import WCS

from .generic import get_hdul, read_generic
from .utils import build_aperture, load_cosmicray_extension


def read_spoc(
    file: str | Path | fits.HDUList,
    flux_key: str | None = None,
    time_keys: Iterable[str] | None = None,
    data_keys: Iterable[str] | None = None,
    error_syntax: str | Iterable[str] | None = "{key}_ERR",
    boolean_keys: Iterable[str] | None = None,
    bitwise_keys: Iterable[str] | None = None,
) -> lkdata.DataSet:
    """Helper function to read in a TESS fits file processed by the SPOC pipeline

    Parameters
    ----------
    file : str, pathlib.Path, astropy.io.fits.HDUList
        The path to the fits file or HDUList
    flux : str, optional
        The parameter to use as the default flux value.
        If unspecified, this will be 'FLUX' for target pixel files and
        'PDCSAP_FLUX' for lightcurves
    time_keys : list, optional
        Time data contained in the fits file. These will be used as the time
        indices for the DataSet. By default ['TIME', 'TIMECORR', 'CADENCENO']
    data_keys : list, optional
        Series or cube data contained by the fits file. By default
        ['FLUX','RAW_CNTS','SAP_FLUX','SAP_BKG','PDCSAP_FLUX','FLUX_BKG',
        'PSF_CENTR1','PSF_CENTR2','MOM_CENTR1','MOM_CENTR2',
        'POS_CORR1','POS_CORR2']
    boolean_keys : list, optional
        boolean series or cube data contained by the fits file, by default []
    bitwise_keys : list, optional
        Bitwise series or cube data, by default ['QUALITY'] for the TESS data
        quality flags which are stored as a bitwise array

    Returns
    -------
    ds
        lkdata.DataSet containing data from the fits file
    """

    hdu = get_hdul(file)

    time_keys = time_keys or ["TIME", "TIMECORR", "CADENCENO"]
    data_keys = data_keys or [
        "FLUX",
        "RAW_CNTS",
        "SAP_FLUX",
        "SAP_BKG",
        "PDCSAP_FLUX",
        "FLUX_BKG",
        "PSF_CENTR1",
        "PSF_CENTR2",
        "MOM_CENTR1",
        "MOM_CENTR2",
        "POS_CORR1",
        "POS_CORR2",
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
    if not flux_key:
        flux_key = "PDCSAP_FLUX" if len(ds.cubes) == 0 else "FLUX"
    if flux_key != "FLUX":
        ds.data_products["FLUX"] = ds.data_products[flux_key]
        ds.meta["units"]["FLUX"] = ds.meta["units"][flux_key]

    ds.meta["FLUX_ORIGIN"] = flux_key

    shape = ds.data_products["FLUX"].array.shape
    ds.meta["aperture"] = build_aperture(hdu, shape)

    # 20-second TESS data products have a 'COSMIC RAY' extension to show the
    # pipeline-removed cosmic rays. We add the array as a DataCube as necessary.
    cr = load_cosmicray_extension(hdu)
    if cr is not None:
        ds.meta["cosmic ray"] = True
        ds.data_products["COSMIC_RAY"] = lkdata.DataCube(data=cr[ds.meta["timemask"]])
    else:
        ds.meta["cosmic ray"] = False

    hdu.close()
    return ds
