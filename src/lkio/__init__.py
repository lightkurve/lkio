import os


ROOTNAME = "lkv3"
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .generic import generic_io
from .spoc_astropy import read_spoc
from .diamante import read_diamante
from .qlp import read_qlp