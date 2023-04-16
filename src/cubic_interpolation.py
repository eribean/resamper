import numpy.typing as npt
import numba as nb


class CubicKeys:
    """Filter coefficients for cubic keys interpolation."""

    needs_prefilter: bool = False