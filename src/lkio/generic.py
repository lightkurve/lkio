"""Generic read/write functions"""

import fnmatch
from typing import Optional, Union, Iterable
import logging
import pathlib

import lkdata
import numpy as np
from astropy.io import fits

log = logging.getLogger(__name__)

DEFAULT_TIME_KEYS = ["TIME", "TIMECORR", "CADENCENO"]
LK_CLASSES = {1: lkdata.DataSeries, 2: lkdata.DataSeriesCollection, 3: lkdata.DataCube}


def get_hdul(file) -> fits.HDUList:
    """Return the HDUList associated with the input"""
    assert isinstance(file, (fits.HDUList, str, pathlib.Path)), (
        "`file` must be a string or an HDUList."
    )
    if isinstance(file, fits.HDUList):
        hdul = file
    elif str(file).startswith("s3://"):
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


def _match_error_key(key, error_keys) -> Optional[str]:
    """Return the uncertainty column belonging to `key`, or None.

    An exact ``{key}_ERR`` match always wins.  Otherwise the candidates are
    sorted before picking one, because `error_keys` is a set and bare
    `fnmatch.filter` would otherwise return results in iteration order --
    e.g. 'FLUX' matches both 'FLUX_ERR' and 'FLUX_BKG_ERR' in a target pixel
    file, and the winner would vary between runs.
    """
    exact = f"{key}_ERR"
    if exact in error_keys:
        return exact
    candidates = sorted(fnmatch.filter(sorted(error_keys), f"*{key}*"))
    return candidates[0] if candidates else None


def read_image_hdu(hdu, **meta) -> lkdata.DataSet:
    # Image data
    col_data = np.array(hdu.data).byteswap().newbyteorder()
    if not hdu.data:
        raise AttributeError("Given HDU has no data.")
    dcube = lkdata.DataCube(
        [col_data],
        nrow=col_data.shape[0],
        ncol=col_data.shape[1],
    )
    dset = lkdata.DataSet(dcube, meta=meta)
    return dset


def read_generic(
    file: Union[str, fits.HDUList],
    ext: int = 1,
    time_keys: Optional[Iterable[str]] = None,
    data_keys: Optional[Iterable[str]] = None,
    error_syntax: Optional[Union[str, Iterable[str]]] = "{key}_ERR",
    boolean_keys: Optional[Iterable[str]] = None,
    bitwise_keys: Optional[Iterable[str]] = None,
    close_file: Optional[bool] = True,
    cadence_mask: Optional[np.ndarray] = None,
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
    cadence_mask : array of bool, optional
        Additional per-cadence mask (e.g. from quality flags) to AND into the
        NaN-time mask before any data is sliced, by default None

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
            time_keys,
            data_keys,
            error_syntax,
            boolean_keys,
            bitwise_keys,
            hdu,
            meta,
            cadence_mask=cadence_mask,
        )
    else:
        result = read_image_hdu(hdu, **meta)

    if close_file:
        hdul.close()
    return result


def read_table_hdu(
    time_keys,
    data_keys,
    error_syntax,
    boolean_keys,
    bitwise_keys,
    hdu,
    meta,
    cadence_mask=None,
):
    header_keys = set(hdu.columns.names.copy())
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
        good = ~np.isnan(np.array(list(times_dict.values()))).any(axis=0)
        if cadence_mask is not None:
            good = good & np.asarray(cadence_mask, dtype=bool)
        timemask = np.where(good)
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
    data_series_dict = {}
    for key in data_keys & header_keys:
        col_data = hdu.data[key]
        col_data = col_data.byteswap().newbyteorder()
        col_data = np.array(col_data)[timemask]
        key_error = _match_error_key(key, error_keys)

        if key_error is not None:
            data_series_dict[key] = LK_CLASSES[col_data.ndim](
                data=col_data,
                uncertainty=np.array(hdu.data[key_error])
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


# ── Write ──────────────────────────────────────────────────────────────────


def _fits_format(dtype) -> str:
    """Map a numpy dtype to a FITS column format code."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.float64):
        return "D"
    if np.issubdtype(dtype, np.float32):
        return "E"
    if np.issubdtype(dtype, np.signedinteger):
        return {8: "K", 4: "J", 2: "I", 1: "B"}.get(dtype.itemsize, "J")
    if np.issubdtype(dtype, np.unsignedinteger):
        return "J"
    if np.issubdtype(dtype, np.bool_):
        return "L"
    return "D"


def _build_primary_header(meta_val) -> fits.Header:
    """Return a fits.Header built from a dict or an existing Header."""
    if isinstance(meta_val, fits.Header):
        return meta_val.copy()
    header = fits.Header()
    if not isinstance(meta_val, dict):
        return header
    for key, val in meta_val.items():
        if not key or key in {"COMMENT", "HISTORY", "END"} or key.startswith("_"):
            continue
        if not isinstance(val, (bool, int, float, str)):
            continue
        try:
            header[key] = val
        except Exception:
            pass
    return header


def _is_image_dataset(dataset: lkdata.DataSet) -> bool:
    """True when the DataSet should be written as image HDU(s).

    Criteria: ntime == 1, every data product is a DataCube, every bool
    product is a BoolCube, and there are no bitwise products.
    """
    if dataset.ntime != 1:
        return False
    if dataset.bitwise_products:
        return False
    if not dataset.data_products and not dataset.bool_products:
        return False
    for p in dataset.data_products.values():
        if not isinstance(p, lkdata.DataCube):
            return False
    for p in dataset.bool_products.values():
        if not isinstance(p, lkdata.BoolCube):
            return False
    return True


def write_fits(
    dataset: lkdata.DataSet,
    file: Union[str, pathlib.Path],
    ext_name: str = "LIGHTCURVE",
    overwrite: bool = False,
) -> None:
    """Write a DataSet to a FITS file.

    **Image DataSets** (``ntime == 1``, all products are DataCube /
    BoolCube, no bitwise products) are written as image HDUs:

    * Single DataCube → ``PrimaryHDU`` with the 2-D array as data.
    * Multiple cubes → empty ``PrimaryHDU`` plus one ``ImageHDU`` per cube,
      named by the product key.  BoolCube products are written last as
      integer (0/1) image extensions.

    **All other DataSets** are written as a two-HDU file: a ``PrimaryHDU``
    (populated from ``dataset.meta['PRIMARY']`` when present) and a
    ``BinTableHDU`` named *ext_name* containing time-index columns, data
    products with their uncertainties, bitwise products, and bool products.

    Parameters
    ----------
    dataset : lkdata.DataSet
        DataSet to serialise.
    file : str or Path
        Destination file path.
    ext_name : str, optional
        EXTNAME for the binary table extension, by default "LIGHTCURVE".
        Ignored for image DataSets.
    overwrite : bool, optional
        Overwrite an existing file, by default False.
    """
    meta = getattr(dataset, "meta", None) or {}
    units: dict = meta.get("units", {}) if isinstance(meta, dict) else {}
    primary_header = _build_primary_header(meta.get("PRIMARY", {}))

    # ── Image path ─────────────────────────────────────────────────────────
    if _is_image_dataset(dataset):
        data_cubes = [
            (key, np.array(p.array)[0]) for key, p in dataset.data_products.items()
        ]
        bool_cubes = [
            (key, np.array(p.array)[0].astype(np.int32))
            for key, p in dataset.bool_products.items()
        ]
        all_images = data_cubes + bool_cubes

        if len(all_images) == 1:
            key, arr2d = all_images[0]
            hdul = fits.HDUList([fits.PrimaryHDU(data=arr2d, header=primary_header)])
        else:
            hdus: list = [fits.PrimaryHDU(header=primary_header)]
            for key, arr2d in all_images:
                ext_header = _build_primary_header(meta.get(key, {}))
                hdus.append(fits.ImageHDU(data=arr2d, header=ext_header, name=key))
            hdul = fits.HDUList(hdus)

        hdul.writeto(file, overwrite=overwrite)
        return

    # ── BinTable path ──────────────────────────────────────────────────────
    primary_hdu = fits.PrimaryHDU(header=primary_header)

    # ── Build table columns ────────────────────────────────────────────────
    columns: list = []

    # Time-index columns (all MultiIndex levels except the default integer sequence)
    index_df = dataset.index.to_frame(index=False)
    for col_name in index_df.columns:
        if col_name == "time_index":
            continue
        arr = index_df[col_name].to_numpy()
        columns.append(
            fits.Column(
                name=col_name,
                format=_fits_format(arr.dtype),
                array=arr,
                unit=units.get(col_name),
            )
        )

    # Data product columns
    for key, product in dataset.data_products.items():
        arr = np.array(product.array)
        unit = units.get(key)

        if arr.ndim == 1:
            columns.append(
                fits.Column(
                    name=key, format=_fits_format(arr.dtype), array=arr, unit=unit
                )
            )
            unc = getattr(product, "uncertainty", None)
            if unc is not None and bool(unc):
                err = np.array(unc.array)
                columns.append(
                    fits.Column(
                        name=f"{key}_ERR",
                        format=_fits_format(err.dtype),
                        array=err,
                        unit=unit,
                    )
                )

        elif arr.ndim == 3:
            ntime, nrow, ncol = arr.shape
            n_pix = nrow * ncol
            fmt = f"{n_pix}{_fits_format(arr.dtype)}"
            # FITS TDIM uses FORTRAN (column-major) order, so list ncol first
            dim = f"({ncol},{nrow})"
            columns.append(
                fits.Column(
                    name=key,
                    format=fmt,
                    array=arr.reshape(ntime, n_pix),
                    unit=unit,
                    dim=dim,
                )
            )
            unc = getattr(product, "uncertainty", None)
            if unc is not None and bool(unc):
                err = np.array(unc.array).reshape(ntime, n_pix)
                columns.append(
                    fits.Column(
                        name=f"{key}_ERR",
                        format=fmt,
                        array=err,
                        unit=unit,
                        dim=dim,
                    )
                )

    # Bitwise columns (QUALITY flags etc.) — cast BitSet objects to int32
    for key, product in dataset.bitwise_products.items():
        arr = product.apply(int).to_numpy().astype(np.int32)
        columns.append(fits.Column(name=key, format="J", array=arr))

    # Bool columns
    for key, product in dataset.bool_products.items():
        arr = np.array(product, dtype=bool)
        columns.append(fits.Column(name=key, format="L", array=arr))

    # ── BinTable HDU ───────────────────────────────────────────────────────
    table_hdu = fits.BinTableHDU.from_columns(columns, name=ext_name)

    # Restore non-structural observation-level header cards from the original
    # extension header, skipping anything auto-generated for the column layout.
    ext_meta = meta.get(ext_name, {}) if isinstance(meta, dict) else {}
    _STRUCTURAL = {
        "",
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "EXTNAME",
        "EXTVER",
    }
    if isinstance(ext_meta, fits.Header):
        items = [(c.keyword, c.value, c.comment) for c in ext_meta.cards]
    elif isinstance(ext_meta, dict):
        items = [(k, v, "") for k, v in ext_meta.items()]
    else:
        items = []
    for card_key, card_val, card_comment in items:
        if (
            card_key in _STRUCTURAL
            or card_key.startswith("T")
            or card_key in table_hdu.header
        ):
            continue
        if not isinstance(card_val, (bool, int, float, str)):
            continue
        try:
            table_hdu.header[card_key] = (card_val, card_comment)
        except Exception:
            pass

    fits.HDUList([primary_hdu, table_hdu]).writeto(file, overwrite=overwrite)
