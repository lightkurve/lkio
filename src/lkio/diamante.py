from astropy.io import fits
#parent_dir_name = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath('lkdata'))))
from .generic import read_generic
import warnings


def read_diamante(
        file,
        flux = 'LC1_AP3',
        time_keys = ['BTJD'], 
        data_keys = ['LC0_AP1','LC1_AP1','LC2_AP1',
                     'LC0_AP2','LC1_AP2','LC2_AP2',
                     'LC0_AP3','LC1_AP3','LC2_AP3',
                     'LC0_AP4','LC1_AP4','LC2_AP4',
                     'SKY', 
                     'RA_AP1', 'DEC_AP1','RA_AP2', 'DEC_AP2',
                     'RA_AP3', 'DEC_AP3','RA_AP4', 'DEC_AP4'],
        #error_keys = ['ELC0_AP1']
        #alternatively, could get all data by using keys = hdu[1].header['TTYPE*']. This includes time columns though
        boolean_keys = [],
        bitwise_keys =['FLAG_AP1','FLAG_AP2','FLAG_AP3','FLAG_AP4'], 
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

    # Diamante does not contain the required 'EXTNAME' keywork, so add it here
    hdu[0].header['EXTNAME'] = 'PRIMARY'
    hdu[1].header['EXTNAME'] = 'LIGHTCURVE'
    # Diamante breaks the generic reader because errors are E{key} rather than {key}_ERR. 
    # I'll have to re-think errors now. 

    ds = read_generic(
        file = hdu,
        time_keys = time_keys,
        data_keys=data_keys,
        boolean_keys= boolean_keys,
        bitwise_keys = bitwise_keys,
        error_syntax = 'E{key}',
        close_file = True,
        )


    if flux != 'FLUX':
        ds.data_products["FLUX"] = ds.data_products[flux]
        if flux.split('_')[0] == 'LCO':
            ds.meta['units']["FLUX"] = 'e-/s'
        else:
            ds.meta['units']["FLUX"] = None

    ds.meta['FLUX_ORIGIN'] = flux

    # No WCS as the header does not include the required keywords

    # diamente lc does not contain the aperture in the same way as SPOC
    # it only had the aperture radius. You can get this from the default flux column
    ap = flux.split('AP')[1]
    ds.meta['aperture'] = ap

    
    return ds

