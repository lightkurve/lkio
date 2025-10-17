from astropy.io import fits

from .generic import generic_io
from astropy.wcs import WCS


def read_qlp(
        file = "/Users/nthom/.lightkurve/cache/mastDownload/HLSP/hlsp_qlp_tess_ffi_s0001-0000000183306762_tess_v01_llc/hlsp_qlp_tess_ffi_s0001-0000000183306762_tess_v01_llc.fits",
        hdu = None, 
        time = 'TIME',
        flux = 'KSPSAP_FLUX',
        time_keys = ['TIME','CADENCENO', 'ORBITID'], 
        data_keys = ['SAP_FLUX','KSPSAP_FLUX','SAP_X','SAP_Y','SAP_BKG','KSPSAP_FLUX_SML','KSPSAP_FLUX_LAG'],
        data_err_keys = None,
        boolean_keys = [],
        bitwise_keys = ['QUALITY'], 
        ):
    
    hdu = fits.open(file)


    '''time_series = ['TIME','CADENCENO', 'ORBITID']

    # Should I assume all error has <name>_ERR?
    data_series = [('SAP_FLUX', None),
                    ('KSPSAP_FLUX', 'KSPSAP_FLUX_ERR'),
                    ('SAP_X',None),
                    ('SAP_Y', None),
                    ('SAP_BKG','SAP_BKG_ERR'),
                    ('KSPSAP_FLUX_SML', None),
                    ('KSPSAP_FLUX_LAG', None),
                    ]

    bitwise_series = ['QUALITY'] #Bitwise data can't have uncertainties

    bool_series = []'''

    ds = generic_io(
        hdu = hdu,
        time_keys = time_keys,
        data_keys=data_keys,
        data_err_keys=data_err_keys,
        boolean_keys= boolean_keys,
        bitwise_keys = bitwise_keys,
        )

    # TODO: does QLP not have the aperture? I don't see it in my test file
    #meta['aperture'] = hdu[2].data
    ds.meta['wcs'] = WCS(hdu[1].header)
    #
    if flux != 'FLUX':
        ds.data_products["FLUX"] = ds.data_products[flux]
    ds.meta['FLUX_ORIGIN'] = flux

    # TODO: I don't actually see units in the header, except for time?

    return ds

