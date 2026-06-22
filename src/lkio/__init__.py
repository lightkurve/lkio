"""Module to handle IO in the lkv3 ecosystem"""
import os
from .diamante import read_diamante
from .generic import check_headers, read_generic, write_fits
from .qlp import read_qlp
from .spoc import read_spoc

ROOTNAME = "lkv3"
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

__all__ = [
    "read_generic",
    "check_headers",
    "write_fits",
    "read_spoc",
    "read_diamante",
    "read_qlp",
]
