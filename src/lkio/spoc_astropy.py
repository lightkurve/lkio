from astropy.io import fits
import sys, os
#parent_dir_name = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath('lkdata'))))
import lkdata
from .generic import generic_io
from astropy.wcs import WCS
import warnings
from .utils import load_cosmicray_extension


def read_spoc(
        file,
        flux = None,
        time_keys = ['TIME', 'TIMECORR', 'CADENCENO'], 
        data_keys = ['FLUX','RAW_CNTS','SAP_FLUX','SAP_BKG','PDCSAP_FLUX','FLUX_BKG','PSF_CENTR1','PSF_CENTR2','MOM_CENTR1','MOM_CENTR2',
                     'POS_CORR1','POS_CORR2'],
        #alternatively, could get all data by using keys = hdu[1].header['TTYPE*']. This includes time columns though
        boolean_keys = [],
        bitwise_keys = ['QUALITY'], 
              ):
    """Helper function to read in a TESS fits file processed by the SPOC pipeline

    Parameters
    ----------
    file : str, astropy.io.fits.HDUList
        if string, the path to the fits file. Alternatively can pass an HDUList
    flux : _type_, optional
        the parameter to use as the default flux value. If unspecified, this will be 'FLUX' for target pixel files and 'PDCSAP_FLUX' for lightcurves
    time_keys : list, optional
        time data contained in the fits file. These will be used as the time indices for the DataSet. By default ['TIME', 'TIMECORR', 'CADENCENO']
    data_keys : list, optional
        series or cube data contained by the fits file, by default ['FLUX','RAW_CNTS','SAP_FLUX','SAP_BKG','PDCSAP_FLUX','FLUX_BKG','PSF_CENTR1','PSF_CENTR2','MOM_CENTR1','MOM_CENTR2', 'POS_CORR1','POS_CORR2']
    boolean_keys : list, optional
        boolean series or cube data contained by the fits file, by default []
    bitwise_keys : list, optional
        Bitwise series or cube data, by default ['QUALITY'] for the TESS data quality flags which are stores as a bitwise array

    Returns
    -------
    ds
        lkdata.DataSet containing data from the fits file
    """

    '''if isinstance(file, fits.HDUList):
        hdu = file
    elif isinstance(file, str):
        hdu = fits.open(file)
    else:
        warnings.warn('No file provided.')'''
    
    if isinstance(file, fits.HDUList):
        hdu = file
    elif isinstance(file, str):
        hdu = fits.open(file)
    else:
        warnings.warn('No file provided.')


    ds = generic_io(
        file = hdu,
        time_keys = time_keys,
        data_keys=data_keys,
        boolean_keys= boolean_keys,
        bitwise_keys = bitwise_keys,
        error_syntax = '{key}_ERR',
        close_file = False,
        )


    ds.meta['wcs'] = WCS(hdu[1].header)
    if not flux:
        if len(ds.cubes) == 0:
            flux = 'PDCSAP_FLUX'
        else:
            flux = 'FLUX'
    if flux != 'FLUX':
        ds.data_products["FLUX"] = ds.data_products[flux]
        ds.meta['units']["FLUX"] = ds.meta['units'][flux]

    ds.meta['FLUX_ORIGIN'] = flux



    # `IndexError` may be raised if the aperture extension (#2) is missing
    # `TypeError` may be raised because early versions of TESScut returned floats in HDU 2
    try:
        ds.meta['aperture'] = hdu[2].data & 2 > 0
    except (IndexError, TypeError):
        ds.meta['aperture'] = np.ones(ds.data_products['FLUX'].array.shape, dtype=bool)


    
    # 20-second TESS data products have a 'COSMIC RAY' extension to show the 
    # pipeline-removed cosmic rays. We add the array as a DataCube as necessary. 
    cr = load_cosmicray_extension(hdu)
    if cr is not None:
        ds.meta['cosmic ray'] = True
        ds.data_products['COSMIC_RAY'] = lkdata.DataCube(data=cr[ds.meta['timemask']])
    else:
        ds.meta['cosmic ray'] = False


    
    hdu.close()
    return ds

