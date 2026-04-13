"""Generic read/write functions"""

import fnmatch
from typing import Optional, Union, Iterable
import logging

import lkdata
import numpy as np
from astropy.io import fits

log = logging.getLogger(__name__)

DEFAULT_TIME_KEYS = ["TIME", "TIMECORR", "CADENCENO"]


def get_hdu(file):
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
    data_keys = make_keys_set(data_keys)
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

    # If no specifics are provided, try to just read all data from the header
    ext = 1  # assuming primary hdu is extension 1...
    header_keys = hdul[ext].columns.names.copy()

    if not data_keys:
        data_keys = [
            k
            for k in hdul[ext].columns.names
            if k not in set(bitwise_keys) + set(boolean_keys) + set(time_keys)
        ]
    data_keys = np.atleast_1d(np.array(data_keys))

    unit_dict = {}
    for key in [k for k in data_keys if k in hdul[1].data.columns.names]:
        datakeys = [
            hdul[1].header["TTYPE*"][i] for i in range(len(hdul[1].header["TTYPE*"]))
        ]
        indexloc = datakeys.index(key) + 1

        try:
            unit_dict[key] = hdul[ext].header[f"TUNIT{indexloc}"]
        except KeyError:
            continue
    meta["units"] = unit_dict

    ts = {}
    for key in time_keys:
        if key in header_keys:
            ts[key] = hdul[ext].data[key].byteswap().newbyteorder()

    # Some data products can have NaN time values. This is not allowed by DataSet, so remove them
    timemask = np.where(
        ~np.isnan(np.array(list(ts.values()))).any(axis=0)
    )  # Valid times
    ts = {key: value[timemask] for key, value in ts.items()}
    meta["timemask"] = timemask

    # if not data_err_keys:
    if error_syntax == "{key}_ERR":
        error_columns = [
            string for string in hdul[1].columns.names if string.endswith("ERR")
        ]
    elif error_syntax == "E{key}":
        error_columns = [
            string for string in hdul[1].columns.names if string.startswith("E")
        ]
    data_keys = [item for item in data_keys if item not in error_columns]

    bws = {}
    for key in bitwise_keys:
        if key in header_keys:
            bws[key] = lkdata.BitwiseSeries(
                data=np.array(hdul[ext].data[key]).astype(int)[timemask],
                # time_indices = ts,
            )

    bos = {}
    for key in boolean_keys:
        if key in header_keys:
            bos[key] = lkdata.BoolSeries(
                data=np.array(hdul[1].data[key]).byteswap().newbyteorder()[timemask],
                # time_indices = ts,
            )

    ds = {}
    for key in data_keys:
        if key in header_keys:
            d = np.array(hdul[1].data[key]).byteswap().newbyteorder()[timemask]
            # print(f"{key} dimensions: {d.ndim}")
            key_error = fnmatch.filter(error_columns, f"*{key}*")
            if len(key_error) > 0:
                # if f"{key}_ERR" in error_columns:
                if d.ndim == 1:
                    ds[key] = lkdata.DataSeries(
                        data=d,
                        uncertainty=np.array(hdul[1].data[key_error[0]])
                        .byteswap()
                        .newbyteorder()[timemask],
                        # time_indices = ts,
                    )
                elif d.ndim == 3:
                    ds[key] = lkdata.DataCube(
                        data=d,
                        uncertainty=np.array(hdul[1].data[key_error[0]])
                        .byteswap()
                        .newbyteorder()[timemask],
                        # time_indices = ts,
                    )

            else:
                if d.ndim == 1:
                    ds[key] = lkdata.DataSeries(
                        data=np.array(hdul[1].data[key])
                        .byteswap()
                        .newbyteorder()[timemask],
                        # time_indices = ts,
                    )
                elif d.ndim == 3:
                    ds[key] = lkdata.DataCube(
                        data=np.array(hdul[1].data[key])
                        .byteswap()
                        .newbyteorder()[timemask],
                        # time_indices = ts,
                    )

    dset = lkdata.DataSet(
        data_products=ds,
        bool_products=bos,
        bitwise_products=bws,
        time_indices=ts,
        meta=meta,
    )

    if close_file:
        hdul.close()
    return dset  # aperture and wcs added in specific data product readers
