"""Ghia, Ghia & Shin (1982) lid-driven cavity benchmark.

'High-Re solutions for incompressible flow using the Navier-Stokes equations and
a multigrid method', J. Comput. Phys. 48, 387-411. Tables I and II.

u is sampled along the vertical centreline, v along the horizontal centreline,
both normalised by the lid velocity.
"""

from __future__ import annotations

import numpy as np

# y, then u at Re = 100, 400, 1000
_U_TABLE = np.array(
    [
        [0.0000, 0.00000, 0.00000, 0.00000],
        [0.0547, -0.03717, -0.08186, -0.18109],
        [0.0625, -0.04192, -0.09266, -0.20196],
        [0.0703, -0.04775, -0.10338, -0.22220],
        [0.1016, -0.06434, -0.14612, -0.29730],
        [0.1719, -0.10150, -0.24299, -0.38289],
        [0.2813, -0.15662, -0.32726, -0.27805],
        [0.4531, -0.21090, -0.17119, -0.10648],
        [0.5000, -0.20581, -0.11477, -0.06080],
        [0.6172, -0.13641, 0.02135, 0.05702],
        [0.7344, 0.00332, 0.16256, 0.18719],
        [0.8516, 0.23151, 0.29093, 0.33304],
        [0.9531, 0.68717, 0.55892, 0.46604],
        [0.9609, 0.73722, 0.61756, 0.51117],
        [0.9688, 0.78871, 0.68439, 0.57492],
        [0.9766, 0.84123, 0.75837, 0.65928],
        [1.0000, 1.00000, 1.00000, 1.00000],
    ]
)

# x, then v at Re = 100, 400, 1000
_V_TABLE = np.array(
    [
        [0.0000, 0.00000, 0.00000, 0.00000],
        [0.0625, 0.09233, 0.18360, 0.27485],
        [0.0703, 0.10091, 0.19713, 0.29012],
        [0.0781, 0.10890, 0.20920, 0.30353],
        [0.0938, 0.12317, 0.22965, 0.32627],
        [0.1563, 0.16077, 0.28124, 0.37095],
        [0.2266, 0.17507, 0.30203, 0.33075],
        [0.2344, 0.17527, 0.30174, 0.32235],
        [0.5000, 0.05454, 0.05186, 0.02526],
        [0.8047, -0.24533, -0.38598, -0.31966],
        [0.8594, -0.22445, -0.44993, -0.42665],
        [0.9063, -0.16914, -0.23827, -0.51550],
        [0.9453, -0.10313, -0.22847, -0.39188],
        [0.9531, -0.08864, -0.19254, -0.33714],
        [0.9609, -0.07391, -0.15663, -0.27669],
        [0.9688, -0.05906, -0.12146, -0.21388],
        [1.0000, 0.00000, 0.00000, 0.00000],
    ]
)

_COLUMN = {100: 1, 400: 2, 1000: 3}
AVAILABLE_REYNOLDS = tuple(_COLUMN)


def _column(re: float) -> int:
    key = int(round(re))
    if key not in _COLUMN:
        raise KeyError(f"Ghia data available for Re in {AVAILABLE_REYNOLDS}, not {re}")
    return _COLUMN[key]


def u_profile(reynolds: float) -> tuple[np.ndarray, np.ndarray]:
    col = _column(reynolds)
    return _U_TABLE[:, 0].copy(), _U_TABLE[:, col].copy()


def v_profile(reynolds: float) -> tuple[np.ndarray, np.ndarray]:
    col = _column(reynolds)
    return _V_TABLE[:, 0].copy(), _V_TABLE[:, col].copy()


# Primary vortex centre (x, y) and the normalised stream function there.
# A scalar, geometry-level check that complements the centreline RMS: profiles can
# agree while the recirculation sits in the wrong place.
PRIMARY_VORTEX = {
    100: {"centre": (0.6172, 0.7344), "psi": -0.103423},
    400: {"centre": (0.5547, 0.6055), "psi": -0.113909},
    1000: {"centre": (0.5313, 0.5625), "psi": -0.117929},
}

SECONDARY_VORTEX_BR = {
    100: (0.9453, 0.0625),
    400: (0.8906, 0.1250),
    1000: (0.8594, 0.1094),
}

SECONDARY_VORTEX_BL = {
    100: (0.0313, 0.0391),
    400: (0.0508, 0.0469),
    1000: (0.0859, 0.0781),
}


def primary_vortex(reynolds: float) -> dict:
    key = int(round(reynolds))
    if key not in PRIMARY_VORTEX:
        raise KeyError(f"Ghia data available for Re in {AVAILABLE_REYNOLDS}, not {reynolds}")
    return PRIMARY_VORTEX[key]


def rms_error(
    coord: np.ndarray, values: np.ndarray, ref_coord: np.ndarray, ref_values: np.ndarray
) -> float:
    """RMS deviation of a computed profile from the benchmark, interpolated onto
    the benchmark's sample points."""
    interp = np.interp(ref_coord, coord, values)
    return float(np.sqrt(np.mean((interp - ref_values) ** 2)))
