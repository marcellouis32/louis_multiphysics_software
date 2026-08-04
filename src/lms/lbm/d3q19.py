"""D3Q19 lattice constants and equilibrium.

The 3D counterpart of `d2q9.py`, and deliberately the same shape so the two can be read
side by side. Index 0 is rest, 1-6 are the six axis directions, 7-18 the twelve
face diagonals. D3Q19 omits the eight corner directions of D3Q27; it is the standard
choice for wall-bounded incompressible flow, cheaper by 30% in memory and bandwidth,
and adequate for everything short of strongly anisotropic turbulence.

Velocities are ordered in opposite pairs, so `OPPOSITE[i]` is `i + 1` for odd `i` and
`i - 1` for even `i > 0`. That is not decoration: bounce-back indexes by it on every
solid node of every step, and a pairwise ordering makes it a arithmetic rather than a
lookup on the GPU.

The relaxation helpers live in `d2q9.py` and are reused unchanged -- `viscosity_to_omega`
and `trt_magic_omega` are statements about relaxation rates, not about lattice geometry,
so the magic parameter that pins the bounce-back wall halfway between nodes independent
of viscosity carries over verbatim.
"""

from __future__ import annotations

import numpy as np

# fmt: off
EX = np.array([0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0],
              dtype=np.int32)
EY = np.array([0,  0,  0,  1, -1,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0,  1, -1,  1, -1],
              dtype=np.int32)
EZ = np.array([0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1, -1, -1,  1,  1, -1, -1,  1],
              dtype=np.int32)
# fmt: on

W = np.array(
    [1 / 3]
    + [1 / 18] * 6
    + [1 / 36] * 12,
    dtype=np.float64,
)

OPPOSITE = np.array(
    [0] + [i + 1 if i % 2 else i - 1 for i in range(1, 19)],
    dtype=np.int32,
)

CS2 = 1.0 / 3.0
Q = 19


def equilibrium(
    rho: np.ndarray, ux: np.ndarray, uy: np.ndarray, uz: np.ndarray
) -> np.ndarray:
    """Second-order Maxwell-Boltzmann expansion. Returns shape (19, nz, ny, nx)."""
    usq = ux * ux + uy * uy + uz * uz
    feq = np.empty((Q,) + rho.shape, dtype=rho.dtype)
    for i in range(Q):
        eu = EX[i] * ux + EY[i] * uy + EZ[i] * uz
        feq[i] = W[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq)
    return feq


def macroscopic(
    f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = f.sum(axis=0)
    ux = np.tensordot(EX.astype(f.dtype), f, axes=(0, 0)) / rho
    uy = np.tensordot(EY.astype(f.dtype), f, axes=(0, 0)) / rho
    uz = np.tensordot(EZ.astype(f.dtype), f, axes=(0, 0)) / rho
    return rho, ux, uy, uz
