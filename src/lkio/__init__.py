"""Module to handle IO in the lkv3 ecosystem"""
import os

from . import time as _time  # noqa: F401  (registers the bkjd/btjd astropy Time formats)
from .diamante import read_diamante
from .generic import check_headers, read_generic, write_fits
from .qlp import read_qlp
from .quality import KeplerQualityFlags, QualityFlags, TessQualityFlags
from .spoc import read_spoc
from .tpf import read_tpf

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
