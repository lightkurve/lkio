"""Module to handle IO in the lkv3 ecosystem"""
import os

from .utils import mission_time as _time  # noqa: F401  (registers the bkjd/btjd astropy Time formats)
from .readers.diamante import read_diamante
from .generic.generic_fits import check_headers, read_generic, write_fits
from .readers.qlp import read_qlp
from .utils.mission_quality import KeplerQualityFlags, QualityFlags, TessQualityFlags
from .readers.spoc import read_spoc
from .readers.tpf import read_tpf

from .generic.generic_asdf import read_generic_asdf
from .generic.generic_hdulist import read_generic_hdulist

ROOTNAME = "lkv3"
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

__all__ = [
    "KeplerQualityFlags",
    "QualityFlags",
    "TessQualityFlags",
    "check_headers",
    "read_diamante",
    "read_generic",
    "read_qlp",
    "read_spoc",
    "read_tpf",
    "write_fits",
]
