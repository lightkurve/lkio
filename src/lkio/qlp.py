from astropy.io import fits

from .generic import generic_io
from astropy.wcs import WCS


def read_qlp(
        file,
        flux = None,
        time_keys = ['TIME','CADENCENO', 'ORBITID'], 
        data_keys = ['SAP_FLUX','KSPSAP_FLUX','SAP_X','SAP_Y','SAP_BKG','KSPSAP_FLUX_SML','KSPSAP_FLUX_LAG'],
        boolean_keys = [],
        bitwise_keys = ['QUALITY'], 
        ):
    
    if isinstance(file, fits.HDUList):
        hdu = file
    elif isinstance(file, str):
        hdu = fits.open(file)
    else:
        warnings.warn('No file provided.')  


    ds = generic_io(
        file = file,
        time_keys = time_keys,
        data_keys=data_keys,
        error_syntax = '{key}_ERR',
        boolean_keys= boolean_keys,
        bitwise_keys = bitwise_keys,
        close_file=False,
        )

    # TODO: does QLP not have the aperture? I don't see it in my test file
    #meta['aperture'] = hdu[2].data
    ds.meta['wcs'] = WCS(hdu[1].header)
    
    # QLP changes for extended mission 2 (sector 56+). Now using “DET_FLUX” and “DET_FLUX_ERR” instead of “KSPSAP_FLUX” and “KSPSAP_FLUX_ERR” for the detrended flux and detrended flux error, respectively
    if flux is None:
        if "DET_FLUX" in ds.data_products.keys():
            flux = "DET_FLUX"
        else:
            flux = "KSPSAP_FLUX"

    ds.data_products["FLUX"] = ds.data_products[flux]
    ds.meta['FLUX_ORIGIN'] = flux

    # TODO: I don't actually see units in the header, except for time?

    return ds
