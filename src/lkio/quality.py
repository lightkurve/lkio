"""Mission quality-flag tables and cadence masking.

The QUALITY column of Kepler/K2/TESS data products is a bitfield.  The classes
below give each bit a name and a human-readable description, and provide the
``create_quality_mask`` helper used by the readers to drop bad cadences.
"""

import logging
from typing import Any, ClassVar

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "MISSION_QUALITY_FLAGS",
    "KeplerQualityFlags",
    "QualityFlags",
    "TessQualityFlags",
]


def _as_values(array: Any) -> Any:
    """Strip an ``astropy`` unit wrapper, if there is one, without importing astropy."""
    return getattr(array, "value", array)


class QualityFlags:
    """Abstract base class for the mission quality-flag tables."""

    STRINGS: ClassVar[dict[int, str]] = {}
    OPTIONS: ClassVar[dict[str, int]] = {}

    @classmethod
    def decode(cls, quality: Any) -> list[str]:
        """Convert a QUALITY value into a list of human-readable strings.

        This function takes the QUALITY bitstring that can be found for each
        cadence in Kepler/K2/TESS' pixel and light curve files and converts into
        a list of human-readable strings explaining the flags raised (if any).

        Parameters
        ----------
        quality : int
            Value from the 'QUALITY' column of a Kepler/K2/TESS pixel or
            lightcurve file.

        Returns
        -------
        flags : list of str
            List of human-readable strings giving a short description of the
            quality flags raised.  Returns an empty list if no flags raised.
        """
        quality = _as_values(quality)
        return [
            description
            for flag, description in cls.STRINGS.items()
            if quality & flag > 0
        ]

    @classmethod
    def create_quality_mask(
        cls, quality_array: Any, bitmask: int | str | None = None
    ) -> np.ndarray:
        """Return a boolean array which flags good cadences given a bitmask.

        Parameters
        ----------
        quality_array : array of int
            'QUALITY' column of a Kepler/K2/TESS target pixel or lightcurve file.
        bitmask : int or str
            Bitmask (int) or one of 'none', 'default', 'hard', or 'hardest'.
            ``None`` masks nothing.

        Returns
        -------
        boolean_mask : array of bool
            Boolean array in which `True` means the data is of good quality.
        """
        # Return an array filled with `True` by default (i.e. ignore nothing)
        if bitmask is None:
            return np.ones(len(quality_array), dtype=bool)
        quality_array = _as_values(quality_array)
        # A few pre-defined bitmasks can be specified as strings
        if isinstance(bitmask, str):
            try:
                bitmask = cls.OPTIONS[bitmask]
            except KeyError as err:
                valid_options = tuple(cls.OPTIONS.keys())
                msg = (
                    f"quality_bitmask='{bitmask}' is not supported, "
                    f"expected one of {valid_options}"
                )
                raise ValueError(msg) from err
        # The bitmask is applied using the bitwise AND operator
        quality_mask = (quality_array & bitmask) == 0
        # Log the quality masking as info or warning
        n_cadences = len(quality_array)
        n_cadences_masked = (~quality_mask).sum()
        percent_masked = 100.0 * n_cadences_masked / n_cadences
        logmsg = (
            f"{percent_masked:.0f}% ({n_cadences_masked}/{n_cadences}) of the "
            f"cadences will be ignored due to the quality mask "
            f"(quality_bitmask={bitmask})."
        )
        if percent_masked > 20:
            log.warning("Warning: %s", logmsg)
        else:
            log.info(logmsg)
        return np.asarray(quality_mask, dtype=bool)


class KeplerQualityFlags(QualityFlags):
    """
    This class encodes the meaning of the various Kepler QUALITY bitmask flags,
    as documented in the Kepler Archive Manual (Ref. [1], Table 2.3).

    References
    ----------
    .. [1] Kepler: A Search for Terrestrial Planets. Kepler Archive Manual.
        http://archive.stsci.edu/kepler/manuals/archive_manual.pdf
    """

    AttitudeTweak = 1
    SafeMode = 2
    CoarsePoint = 4
    EarthPoint = 8
    ZeroCrossing = 16
    Desat = 32
    Argabrightening = 64
    ApertureCosmic = 128
    ManualExclude = 256
    # Bit 2**10 = 512 is unused by Kepler
    SensitivityDropout = 1024
    ImpulsiveOutlier = 2048
    ArgabrighteningOnCCD = 4096
    CollateralCosmic = 8192
    DetectorAnomaly = 16384
    NoFinePoint = 32768
    NoData = 65536
    RollingBandInAperture = 131072
    RollingBandInMask = 262144
    PossibleThrusterFiring = 524288
    ThrusterFiring = 1048576

    #: DEFAULT bitmask identifies all cadences which are definitely useless.
    DEFAULT_BITMASK = (
        AttitudeTweak
        | SafeMode
        | CoarsePoint
        | EarthPoint
        | Desat
        | ManualExclude
        | DetectorAnomaly
        | NoData
        | ThrusterFiring
    )
    #: HARD bitmask is conservative and may identify cadences which are useful.
    HARD_BITMASK = (
        DEFAULT_BITMASK
        | SensitivityDropout
        | ApertureCosmic
        | CollateralCosmic
        | PossibleThrusterFiring
    )
    #: HARDEST bitmask identifies cadences with any flag set. Its use is not recommended.
    HARDEST_BITMASK = 2096639

    #: Dictionary which provides friendly names for the various bitmasks.
    OPTIONS: ClassVar[dict[str, int]] = {
        "none": 0,
        "default": DEFAULT_BITMASK,
        "hard": HARD_BITMASK,
        "hardest": HARDEST_BITMASK,
    }

    #: Pretty string descriptions for each flag
    STRINGS: ClassVar[dict[int, str]] = {
        1: "Attitude tweak",
        2: "Safe mode",
        4: "Coarse point",
        8: "Earth point",
        16: "Zero crossing",
        32: "Desaturation event",
        64: "Argabrightening",
        128: "Cosmic ray in optimal aperture",
        256: "Manual exclude",
        1024: "Sudden sensitivity dropout",
        2048: "Impulsive outlier",
        4096: "Argabrightening on CCD",
        8192: "Cosmic ray in collateral data",
        16384: "Detector anomaly",
        32768: "No fine point",
        65536: "No data",
        131072: "Rolling band in optimal aperture",
        262144: "Rolling band in full mask",
        524288: "Possible thruster firing",
        1048576: "Thruster firing",
    }


class TessQualityFlags(QualityFlags):
    """
    This class encodes the meaning of the various TESS QUALITY bitmask flags,
    as documented in the TESS Data Products Description Document (Ref. [1], Table 28).

    References
    ----------
    .. [1] TESS Science Data Products Description Document (EXP-TESS-ARC-ICD-0014)
        https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014.pdf
    """

    AttitudeTweak = 1
    SafeMode = 2
    CoarsePoint = 4
    EarthPoint = 8
    Argabrightening = 16
    Desat = 32
    ApertureCosmic = 64
    ManualExclude = 128
    Discontinuity = 256
    ImpulsiveOutlier = 512
    CollateralCosmic = 1024
    #: The first stray light flag is set manually by MIT based on visual inspection.
    Straylight = 2048
    #: The second stray light flag is set automatically by Ames/SPOC based on
    #: background level thresholds.
    Straylight2 = 4096
    # See TESS Science Data Products Description Document
    PlanetSearchExclude = 8192
    BadCalibrationExclude = 16384
    # Set in the sector 20 data release notes
    InsufficientTargets = 32768

    #: DEFAULT bitmask identifies all cadences which are definitely useless.
    # See https://outerspace.stsci.edu/display/TESS/2.0+-+Data+Product+Overview
    DEFAULT_BITMASK = (
        AttitudeTweak
        | SafeMode
        | CoarsePoint
        | EarthPoint
        | Argabrightening
        | Desat
        | ManualExclude
        | ImpulsiveOutlier
        | BadCalibrationExclude
    )
    #: HARD bitmask is conservative and may identify cadences which are useful.
    HARD_BITMASK = (
        DEFAULT_BITMASK | ApertureCosmic | CollateralCosmic | Straylight | Straylight2
    )
    #: HARDEST bitmask identifies cadences with any flag set. Its use is not recommended.
    HARDEST_BITMASK = 65535

    #: Dictionary which provides friendly names for the various bitmasks.
    OPTIONS: ClassVar[dict[str, int]] = {
        "none": 0,
        "default": DEFAULT_BITMASK,
        "hard": HARD_BITMASK,
        "hardest": HARDEST_BITMASK,
    }

    #: Pretty string descriptions for each flag
    STRINGS: ClassVar[dict[int, str]] = {
        1: "Attitude tweak",
        2: "Safe mode",
        4: "Coarse point",
        8: "Earth point",
        16: "Argabrightening",
        32: "Desaturation event",
        64: "Cosmic ray in optimal aperture",
        128: "Manual exclude",
        256: "Discontinuity corrected",
        512: "Impulsive outlier",
        1024: "Cosmic ray in collateral data",
        2048: "Straylight",
        4096: "Straylight2",
        8192: "Planet Search Exclude",
        16384: "Bad Calibration Exclude",
        32768: "Insufficient Targets for Error Correction Exclude",
    }


#: Maps a (lower-cased) mission name onto its quality-flag table.
MISSION_QUALITY_FLAGS: dict[str, type[QualityFlags]] = {
    "kepler": KeplerQualityFlags,
    "k2": KeplerQualityFlags,
    "tess": TessQualityFlags,
}
