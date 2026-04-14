"""Generic read/write functions"""

import fnmatch
from typing import Optional, Union, Iterable
import logging

import lkdata
import numpy as np
from astropy.io import fits

log = logging.getLogger(__name__)

DEFAULT_TIME_KEYS = ["TIME", "TIMECORR", "CADENCENO"]
LK_CLASSES = {1: lkdata.DataSeries,
            2: lkdata.DataSeriesCollection,
            3: lkdata.DataCube}

def get_hdu(file) -> fits.HDUList:
    """Return the HDUList associated with the input"""
    assert isinstance(file, (fits.HDUList, str)), (
        "`file` must be a string or an HDUList."
    )
    if isinstance(file, fits.HDUList):
        hdu = file
    elif file.startswith("s3://"):
        # Filename is an S3 cloud URI
        log.debug("Reading file from AWS S3 cloud.")
        hdu = fits.open(file, use_fsspec=True, fsspec_kwargs={"anon": True})
    else:
        hdu = fits.open(file)

    return hdu


def make_keys_set(keys) -> set:
    """Return a set of keys"""
    keys = keys or []
    return set(np.atleast_1d(np.array(keys)))


def get_data_units(data_keys, hdu) -> dict:
    unit_dict = {}
    for key in data_keys:
        datakeys = list(hdu.header["TTYPE*"])
        indexloc = datakeys.index(key) + 1

        try:
            unit_dict[key] = hdu.header[f"TUNIT{indexloc}"]
        except KeyError:
            # no units provided
            continue
    return unit_dict


def read_primary_HDU(ext, data_keys):
    return


def generic_io(
    file: Union[str, fits.HDUList],
    time_keys: Optional[Iterable[str]] = None,
    data_keys: Optional[Iterable[str]] = None,
    error_syntax: Optional[Union[str, Iterable[str]]] = "{key}_ERR",
    boolean_keys: Optional[Iterable[str]] = None,
    bitwise_keys: Optional[Iterable[str]] = None,
    close_file: Optional[bool] = True,
) -> lkdata.DataSet:
    """
    Reads in a generic fits file and returns a lkdata.DataSet

    Parameters
    ----------
    file : str, astropy.io.fits.HDUList
        if string, the path to the fits file. Alternatively can pass an HDUList
    time_keys : list, optional
        time data contained in the fits file. These will be used as the time
        indices for the DataSet. By default ['TIME', 'TIMECORR', 'CADENCENO']
    data_keys : list, optional
        series or cube data contained by the fits file. If none are specified,
        it will try to read all data from the header
    error_syntax : str or list, optional
        If not specified, will look for data keys with '_ERR', eg, for the data
        key 'FLUX', it will search for 'FLUX_ERR'. If a list is given, the
        number of error keys must be the same as data keys.
    boolean_keys : list, optional
        boolean series or cube data contained by the fits file, by default None
    bitwise_keys : list, optional
        Bitwise series or cube data, by default None

    Returns
    -------
    ds
        lkdata.DataSet containing containing Series, Cubes, and metadata
    """

    time_keys = make_keys_set(time_keys)
    boolean_keys = make_keys_set(boolean_keys)
    bitwise_keys = make_keys_set(bitwise_keys)

    hdul = get_hdu(file)

    # Collect meta data from all extensions
    # Get information from the primary header
    meta = {
        f"{hdu.header['EXTNAME']}": hdu.header
        for hdu in hdul
        if isinstance(hdu, (fits.PrimaryHDU, fits.TableHDU, fits.ImageHDU))
    }

    # TODO: Revise following assumption
    # If no specifics are provided, try to just read all data from the header
    # of the first extension
    ext = 1
    hdu = hdul[ext]
    header_keys = hdu.columns.names.copy()

    # if not data_err_keys:
    if error_syntax == "{key}_ERR":
        error_keys = [
            string for string in hdu.columns.names if string.endswith("_ERR")
        ]
    elif error_syntax == "E{key}":
        error_keys = [
            string for string in hdu.columns.names if string.startswith("E")
        ]
    else:
        error_keys = []
    error_keys = set(error_keys)

    if not data_keys:
        data_keys = [
            k
            for k in hdu.columns.names
            if k not in (bitwise_keys | boolean_keys | time_keys | error_keys)
        ]
    data_keys = make_keys_set(data_keys)
    data_keys = data_keys.intersection(hdu.columns.names)

    meta["units"] = get_data_units(data_keys, hdu)

    times_dict = {key: hdu.data[key].byteswap().newbyteorder()
                  for key in (time_keys & header_keys)}

    # lkv3 products do not allow NaN time values
    timemask = np.where(
        ~np.isnan(np.array(list(times_dict.values()))).any(axis=0)
    )  # Valid times
    meta["timemask"] = timemask
    times_dict = {key: value[timemask] for key, value in times_dict.items()}

    bitwise_dict = {key: lkdata.BitwiseSeries(
        data=np.array(hdu.data[key]).astype(int)[timemask])
                    for key in (bitwise_keys & header_keys)}

    bool_dict = {key: lkdata.BoolSeries(
        data=np.array(hdu.data[key]).byteswap().newbyteorder()[timemask])
                 for key in (boolean_keys & header_keys)}

    data_series_dict = {}
    for key in (data_keys & header_keys):
        col_data = np.array(hdu.data[key]).byteswap().newbyteorder()[timemask]
        key_error = fnmatch.filter(error_keys, f"*{key}*")

        if len(key_error) > 0:
            data_series_dict[key] = LK_CLASSES[col_data.ndim](
                data=col_data,
                uncertainty=np.array(hdu.data[key_error[0]])
                .byteswap()
                .newbyteorder()[timemask],
            )

        else:
            data_series_dict[key] = LK_CLASSES[col_data.ndim](
                data=np.array(hdu.data[key])
                .byteswap()
                .newbyteorder()[timemask],
            )

    dset = lkdata.DataSet(
        data_products=data_series_dict,
        bool_products=bool_dict,
        bitwise_products=bitwise_dict,
        time_indices=times_dict,
        meta=meta,
    )

    if close_file:
        hdul.close()
    return dset 
