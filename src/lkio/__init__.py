"""Copyright (c) 2026 Daniel Giles. All rights reserved.

lkio: Takes you from fits to DataSets
"""

from __future__ import annotations

from pathlib import Path

from . import time as _time  # noqa: F401  (registers the bkjd/btjd astropy Time formats)
from ._version import version as __version__
from .diamante import read_diamante
from .generic import check_headers, read_generic, write_fits
from .qlp import read_qlp
from .quality import KeplerQualityFlags, QualityFlags, TessQualityFlags
from .spoc import read_spoc
from .tpf import read_tpf

ROOTNAME = "lkv3"
PACKAGEDIR = Path(__file__).resolve().parent

__all__ = [
    "KeplerQualityFlags",
    "QualityFlags",
    "TessQualityFlags",
    "__version__",
    "check_headers",
    "read_diamante",
    "read_generic",
    "read_qlp",
    "read_spoc",
    "read_tpf",
    "write_fits",
]
