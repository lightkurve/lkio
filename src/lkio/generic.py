from astropy.io import fits
import warnings
import numpy as np
import sys
sys.path.append('/Users/nthom/TESS_GI/gitNschanch/tessgi/lkdata/src/')
import lkdata
import fnmatch


#sys.path.append(parent_dir_name + "/src")
from lkdata import DataSeries,BoolSeries, BitwiseSeries, DataSet


def generic_io(
        file,
        time_keys = ['TIME', 'TIMECORR', 'CADENCENO'], 
        data_keys = None,
        #data_err_keys = None,
        error_syntax = '{key}_ERR',
        boolean_keys = [],
        bitwise_keys = [], 
        close_file = True,

    ):
    """
    Reads in a generic fits file and returns a lkdata.DataSet

    Parameters
    ----------
    file : str, astropy.io.fits.HDUList
        if string, the path to the fits file. Alternatively can pass an HDUList
    time_keys : list, optional
        time data contained in the fits file. These will be used as the time indices for the DataSet. By default ['TIME', 'TIMECORR', 'CADENCENO']
    data_keys : list, optional
        series or cube data contained by the fits file. If none ar specified, it will try to just read all data from the header
    data_err_keys : list, optional
        series or cube data contained by the fits file. If not specified, will look for data keys with '_ERR', eg, for the data key 'FLUX', it will search for 'FLUX_ERR'
    boolean_keys : list, optional
        boolean series or cube data contained by the fits file, by default []
    bitwise_keys : list, optional
        Bitwise series or cube data,

    Returns
    -------
    ds
        lkdata.DataSet containing containing Series, Cubes, and metadata
    """
    if isinstance(file, fits.HDUList):
        hdu = file
    elif isinstance(file, str):
        hdu = fits.open(file)
    else:
        warnings.warn('No file provided.')


    # Collect meta data from all extensions
    meta = {}
    # Get information from the primary header
    for i in range(len(hdu)):
        temp_meta = {}
        head = hdu[i].header
        for k in head.keys():
            temp_meta[k] = head[k]
        meta[f"{temp_meta['EXTNAME']}"] = temp_meta




    # If not specifics are provided, try to just read all data from the header
    header_keys = hdu[1].columns.names.copy()

    if not data_keys:
        data_keys = header_keys
        data_keys = [k for k in data_keys if k not in bitwise_keys]
        data_keys = [k for k in data_keys if k not in boolean_keys]
        data_keys = [k for k in data_keys if k not in time_keys]

    else:
        data_keys = np.atleast_1d(data_keys)
    
    unit_dict = {}
    for key in [k for k in data_keys if k in hdu[1].data.columns.names] :
        datakeys = [hdu[1].header['TTYPE*'][i] for i in range(len(hdu[1].header['TTYPE*']))]
        indexloc = datakeys.index(key) + 1

        try:
            unit_dict[key] = hdu[1].header[f'TUNIT{indexloc}']
        except KeyError:
            continue
    meta["units"] = unit_dict

    ts = {}
    for key in time_keys:
        if key in header_keys:
            ts[key] = hdu[1].data[key].byteswap().newbyteorder()

    # Some data products can have NaN time values. This is not allowed by DataSet, so remove them
    timemask = np.where(~np.isnan(np.array(list(ts.values()))).any(axis=0)) # Valid times
    ts = {key: value[timemask] for key, value in ts.items()}
    meta["timemask"] = timemask
    

    #if not data_err_keys:
    if error_syntax == '{key}_ERR':
        error_columns = [string for string in hdu[1].columns.names if string.endswith('ERR')]
    elif error_syntax == 'E{key}':
        error_columns = [string for string in hdu[1].columns.names if string.startswith('E')]
    data_keys = [item for item in data_keys if item not in error_columns] 



    bws = {}
    for key in bitwise_keys:
        if key in header_keys:
            bws[key] = lkdata.BitwiseSeries(data = np.array(hdu[1].data[key]).astype(int)[timemask],
                #time_indices = ts,
                )



    bos = {}
    for key in boolean_keys:
        if key in header_keys:
            bos[key] = lkdata.BoolSeries(data = np.array(hdu[1].data[key]).byteswap().newbyteorder()[timemask],
                #time_indices = ts,
            )  





    ds = {}
    for key in data_keys:
        if key in header_keys:
            d = np.array(hdu[1].data[key]).byteswap().newbyteorder()[timemask]
            #print(f"{key} dimensions: {d.ndim}")
            key_error = fnmatch.filter(error_columns, f"*{key}*")
            if len(key_error) > 0:
            #if f"{key}_ERR" in error_columns:
                if d.ndim == 1:
                    ds[key] = lkdata.DataSeries(data = d,
                        uncertainty = np.array(hdu[1].data[key_error[0]]).byteswap().newbyteorder()[timemask],
                        #time_indices = ts,
                        )
                elif d.ndim == 3:
                    ds[key] = lkdata.DataCube(data = d,
                        uncertainty = np.array(hdu[1].data[key_error[0]]).byteswap().newbyteorder()[timemask],
                        #time_indices = ts,
                        )

            else:
                if d.ndim == 1:
                    ds[key] = lkdata.DataSeries(data = np.array(hdu[1].data[key]).byteswap().newbyteorder()[timemask],
                        #time_indices = ts,        
                        )
                elif d.ndim == 3:
                    ds[key] = lkdata.DataCube(data = np.array(hdu[1].data[key]).byteswap().newbyteorder()[timemask],
                        #time_indices = ts,        
                        )

    
   


    dset = lkdata.DataSet(data_products=ds, 
        bool_products=bos,
        bitwise_products=bws,
        time_indices = ts,
        meta = meta,
        )
    
    
    
    if close_file:
        hdu.close()
    return dset # aperture and wcs added in specific data product readers