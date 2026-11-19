from typing import Optional, Union, Iterable
import logging
import pathlib

import lkdata
import numpy as np

from .generic import read_generic
from .generic import read_generic_asdf
from .generic import read_generic_hdulist

import readers

from astropy.io.fits.hdu.base import _BaseHDU

def _choose_reader(data, reader=None):
    """Returns Kepler and TESS file types given a FITS object.

    This function will detect the file type by looking at both the TELESCOP and
    CREATOR keywords in the first extension of the FITS header. If the file is
    recognized as a Kepler or TESS data product, one of the following strings
    will be returned:

        * `'KeplerTargetPixelFile'`
        * `'TessTargetPixelFile'`
        * `'KeplerLightCurve'`
        * `'TessLightCurve'`
        * `'K2SFF'`
        * `'EVEREST'`
        * `'K2SC'`
        * `'K2VARCAT'`
        * `'QLP'`
        * `'GSFC-ELEANOR-LITE'`
        * `'PATHOS'`
        * `'TASOC'`
        * `'KEPSEISMIC'`
        * `'CDIPS'`
        * `'TGLC'`
        * `'TARS'`
        * `'Folded'`

    If the data product cannot be detected, `None` will be returned.

    Parameters
    ----------
    hdulist : astropy.io.fits.HDUList object
        A FITS file.

    Returns
    -------
    filetype : str or None
        A string describing the detected filetype. If the filetype is not
        recognized, `None` will be returned.
    """

    if(reader is None):
        # copied over from lightkurve/io/detect/detect_filetype
        # header should be hdulist[0].header == data[0] I think?
        # I think this should now return the reader function

        # The 'origin' fits header value gives institution responsible for creating the file
        # Is it a MIT/QLP TESS FFI Quicklook Pipeline light curve?
        # cf. http://archive.stsci.edu/hlsp/qlp
        if "mit/qlp" in data[0].get("origin", "").lower():
            return readers.qlp.read_qlp

        # Is it a TESS All-Sky Rotation Survey (TARS) light curve?
        # cf. https://archive.stsci.edu/hlsp/tars
        if data[0].get("HLSPID", "").strip().upper() == "TARS":
            #TODO
            return "TARS"

        # Is it a vanilla eleanor or GSFC-ELEANOR-LITE light curve?
        if (
            data[0].get("LITE") is not None
            and data[0].get("PCORIGIN") is not None
        ):  
            #TODO
            return "ELEANOR"

        # Is it a PATHOS TESS light curve?
        # cf. http://archive.stsci.edu/hlsp/pathos
        # The 'pathos' name doesn't stay in the filename if when we use fits.open
        # to download a file, so we have to check for all the important columns
        # This will cause problems if another HLSP has the exact same colnames...
        if all(
            x in hdulist[1].columns.names
            for x in [
                "PSF_FLUX_RAW",
                "PSF_FLUX_COR",
                "AP4_FLUX_RAW",
                "AP4_FLUX_COR",
                "SKY_LOCAL",
            ]
        ):
            #TODO
            return "PATHOS"

        # Is it a TASOC TESS light curve?
        # cf. https://tasoc.dk and https://archive.stsci.edu/hlsp/tasoc
        if data[0].get("ORIGIN") == "TASOC/Aarhus":
            #TODO
            return "TASOC"

        # Is it a CDIPS TESS light curve?
        # cf. http://archive.stsci.edu/hlsp/cdips
        if "cdips" in data[0].get("ORIGIN", "").lower():
            #TODO
            return "CDIPS"

        # Is it a K2VARCAT file?
        # There are no self-identifying keywords in the header, so go by filename.
        fname = hdulist.filename() if callable(hdulist.filename) else hdulist.filename
        if fname is not None:
            if "hlsp_k2varcat" in fname:
                #TODO
                return "K2VARCAT"

        # Is it a K2SC file?
        if "k2sc" in data[0].get("creator", "").lower():
            #TODO
            return "K2SC"

        # Is it a K2SFF file?
        try:
            # There are no metadata keywords identifying K2SFF FITS files,
            # so we go by structure.
            if (
                data[1].header.get("EXTNAME") == "BESTAPER"
                and data[1].header.get("TTYPE4") == "ARCLENGTH"
            ):
                #TODO
                return "K2SFF"
        except Exception:
            pass

        # Is it an EVEREST file?
        try:
            if "EVEREST" in str(data[0].get("COMMENT")):
                #TODO
                return "EVEREST"
        except Exception:
            pass

        # Is it a KEPSEISMIC file?
        if data[0].get("ORIGIN") == "CEA & SSI":
            #TODO
            return "KEPSEISMIC"

        # Is it a TGLC file?
        if data[0].get("ORIGIN") == "UCSB/TGLC":
            #TODO
            return "TGLC"

        
        # Is it an official data product?
        # Use `creator` keyword to determine tpf or lc
        creator = data[0]["creator"].lower()
        origin = data[0]["origin"].lower()
        # check if the file was a folded lightcurve, as saved by lightkurve.FoldedLightCurve().to_fits()
        if 'folded' in creator:
            #TODO
            return "Folded"
        # use `telescop` keyword to determine mission
        if "TELESCOP" in data[0]():
            telescop = data[0]["telescop"].lower()
        else:
            # Some old custom TESS data did not define the `TELESCOP` card
            telescop = data[0]["mission"].lower()
        if telescop == "kepler":
            # Kepler TPFs will contain "TargetPixelExporterPipelineModule"
            if "targetpixel" in creator:
                return readers.tpf.read_tpf
            # Kepler LCFs will contain "FluxExporter2PipelineModule"
            elif (
                "fluxexporter" in creator
                or "lightcurve" in creator
            ):
                #TODO
                return "KeplerLightCurve"
        elif telescop == "tess":
            # TESS TPFs will contain "TargetPixelExporterPipelineModule"
            if "targetpixel" in creator:
                return readers.tpf.read_tpf
            # TESS LCFs will contain "LightCurveExporterPipelineModule"
            elif "lightcurve" in creator:
                return readers.spoc.read_spoc
            # Early versions of TESScut did not set a good CREATOR keyword
            elif "stsci" in origin:
                return readers.tpf.read_tpf

    return reader

# fgure out asdf
def _read(file, reader=None):
    # Turn the file/hdu into a generic lkdata DataSet
    
    if(isinstance(file,str)):
        # this is jank.  sort for caps, compressed, anything else?
        if(file.split('.')[-1] == 'fits'):
            data = read_generic(file)
        if(file.split('.')[-1] == 'asdf'):
            data = read_generic_asdf(file)
    else:
        if(isinstance(file, )):
            data = read_generic_hdulist(file)

    # Choose a reader to customize a dataset
    reader = _choose_reader(data, reader=reader)
    if (reader is not None):
        data = reader(data)

    # This will effectively be our default return
    # This will result in returning the read_generic untouched
    return data

def read(files): # add mypy typing
    datasets = [_read(file) for file in files]
    #TODO add a check to see if all datasets are the same kind of thing,
    # throw a warning if not
    return datasets
