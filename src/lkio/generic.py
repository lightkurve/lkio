"""Generic read/write functions"""

import fnmatch
from typing import Optional, Union, Iterable
import logging

import lkdata
import numpy as np
from astropy.io import fits

log = logging.getLogger(__name__)

DEFAULT_TIME_KEYS = ["TIME", "TIMECORR", "CADENCENO"]
LK_CLASSES = {1: lkdata.DataSeries, 2: lkdata.DataSeriesCollection, 3: lkdata.DataCube}


def get_hdul(file) -> fits.HDUList:
    """Return the HDUList associated with the input"""
    assert isinstance(file, (fits.HDUList, str)), (
        "`file` must be a string or an HDUList."
    )
    if isinstance(file, fits.HDUList):
        hdul = file
    elif file.startswith("s3://"):
        # Filename is an S3 cloud URI
        log.debug("Reading file from AWS S3 cloud.")
        hdul = fits.open(file, use_fsspec=True, fsspec_kwargs={"anon": True})
    else:
        hdul = fits.open(file)

    return hdul


def check_headers(file):
    """Convenience function to get fits headers.

    Parameters
    ----------
    file : str or Path
        The file to read.

    Returns
    -------
    list[fits.header], list[dict]
        A tuple containing a list of the Header objects and a list of key
        dictionaries which likely correspond to their namesake.
    """
    with get_hdul(file) as hdul:
        headers = [hdu.header for hdu in hdul]

    def check_kw(kw):
        # Check column names for identifying keywords
        return [
            [header[key] for key in header["TTYPE*"] if kw in header[key].lower()]
            for header in headers
        ]

    time_keys = check_kw("time")
    error_syntax = check_kw("err")
    boolean_keys = check_kw("aperture")
    bitwise_keys = check_kw("quality")
    keys_list = []
    for i in range(len(headers)):
        keys_list.append({})
        if time_keys[i]:
            keys_list[i]["time_keys"] = time_keys[i]
        if error_syntax[i]:
            keys_list[i]["error_syntax"] = error_syntax[i]
        if boolean_keys[i]:
            keys_list[i]["boolean_keys"] = boolean_keys[i]
        if bitwise_keys[i]:
            keys_list[i]["bitwise_keys"] = bitwise_keys[i]

    return headers, keys_list


def make_keys_set(keys) -> set:
    """Return a set of keys"""
    keys = keys or []
    return set(np.atleast_1d(np.array(keys)))


def get_data_units(data_keys, hdu) -> dict:
    unit_dict = {}
    for key in data_keys:
        datakeys = list(hdu.header["TTYPE*"].values())
        indexloc = datakeys.index(key) + 1

        try:
            unit_dict[key] = hdu.header[f"TUNIT{indexloc}"]
        except KeyError:
            # no units provided
            continue
    return unit_dict


def read_table_HDU(hdu):
    header_keys = set(hdu.columns.names.copy())
    # if not data_err_keys:
    if error_syntax == "{key}_ERR":
        error_keys = set(
            [string for string in hdu.columns.names if string.endswith("_ERR")]
        )
    elif error_syntax == "E{key}":
        error_keys = set(
            [string for string in hdu.columns.names if string.startswith("E")]
        )
    else:
        error_keys = set()
    if not data_keys:
        data_keys = [
            k
            for k in hdu.columns.names
            if k not in (bitwise_keys | boolean_keys | time_keys | error_keys)
        ]
    data_keys = make_keys_set(data_keys)
    data_keys = data_keys.intersection(hdu.columns.names)
    times_dict = {
        key: hdu.data[key].byteswap().newbyteorder()
        for key in (time_keys & header_keys)
    }

    # lkv3 products do not allow NaN time values
    timemask = np.where(
        ~np.isnan(np.array(list(times_dict.values()))).any(axis=0)
    )  # Valid times
    meta["timemask"] = timemask
    times_dict = {key: value[timemask] for key, value in times_dict.items()}


def read_image_hdu(hdu, **meta) -> lkdata.DataSet:
    # Image data
    col_data = np.array(hdu.data).byteswap().newbyteorder()
    dcube = lkdata.DataCube(
        [col_data],
        nrow=col_data.shape[0],
        ncol=col_data.shape[1],
        **meta,
    )
    dset = lkdata.DataSet(dcube)
    return dset


def generic_io(
    file: Union[str, fits.HDUList],
    ext: int = 1,
    time_keys: Optional[Iterable[str]] = None,
    data_keys: Optional[Iterable[str]] = None,
    error_syntax: Optional[Union[str, Iterable[str]]] = "{key}_ERR",
    boolean_keys: Optional[Iterable[str]] = None,
    bitwise_keys: Optional[Iterable[str]] = None,
    close_file: Optional[bool] = True,
) -> Union[lkdata.DataSet, lkdata.DataCube]:
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

    time_keys = make_keys_set(time_keys) or set(["TIME"])
    boolean_keys = make_keys_set(boolean_keys)
    bitwise_keys = make_keys_set(bitwise_keys)

    hdul = get_hdul(file)
    hdu = hdul[ext]
    # Get information from the primary header
    meta = {"PRIMARY": hdul[0].header}
    if ext > 0:
        # Get information from the extension header
        name = hdu.header["EXTNAME"] or f"EXT{ext}"
        meta.update({name: hdu.header})
    if isinstance(hdu, (fits.TableHDU, fits.BinTableHDU)):
        result = read_table_hdu(
            time_keys, data_keys, error_syntax, boolean_keys, bitwise_keys, hdu, meta
        )
    else:
        result = read_image_hdu(hdu, **meta)

    if close_file:
        hdul.close()
    return result


def read_table_hdu(
    time_keys, data_keys, error_syntax, boolean_keys, bitwise_keys, hdu, meta
):
    header_keys = set(hdu.columns.names.copy())
    col_data = hdu.data["baseline"]
    # if not data_err_keys:
    # TODO: What happens with an error data length mismatch
    if error_syntax == "{key}_ERR":
        error_keys = set(
            [string for string in hdu.columns.names if string.endswith("_ERR")]
        )
    elif error_syntax == "E{key}":
        error_keys = set(
            [string for string in hdu.columns.names if string.startswith("E")]
        )
    else:
        error_keys = set()
    if not data_keys:
        data_keys = [
            k
            for k in hdu.columns.names
            if k not in (bitwise_keys | boolean_keys | time_keys | error_keys)
        ]
    data_keys = make_keys_set(data_keys)
    data_keys = data_keys.intersection(hdu.columns.names)
    meta["units"] = get_data_units(data_keys, hdu)
    times_dict = {
        key: hdu.data[key].byteswap().newbyteorder()
        for key in (time_keys & header_keys)
    }
    if times_dict:
        # lkv3 products do not allow NaN time values
        timemask = np.where(~np.isnan(np.array(list(times_dict.values()))).any(axis=0))
        meta["timemask"] = timemask
    else:
        # No valid time key, no mask
        timemask = slice(0, -1)

    times_dict = {key: value[timemask] for key, value in times_dict.items()} or None

    bitwise_dict = {
        key: np.array(hdu.data[key]).astype(int)[timemask]
        for key in (bitwise_keys & header_keys)
    }

    bool_dict = {
        key: np.array(hdu.data[key]).byteswap().newbyteorder()[timemask]
        for key in (boolean_keys & header_keys)
    }
    col_data = hdu.data["baseline"]
    data_series_dict = {}
    for key in data_keys & header_keys:
        col_data = hdu.data[key]
        col_data = col_data.byteswap().newbyteorder()
        col_data = np.array(col_data)[timemask]
        # col_data = np.array(hdu.data[key]).byteswap().newbyteorder()[timemask]
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
                data=np.array(hdu.data[key]).byteswap().newbyteorder()[timemask],
            )

    dset = lkdata.DataSet(
        data_products=data_series_dict,
        bool_products=bool_dict,
        bitwise_products=bitwise_dict,
        time_indices=times_dict,
        meta=meta,
    )
    return dset
