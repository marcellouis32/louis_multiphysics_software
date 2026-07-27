"""D2Q9 lattice constants and equilibrium.

Velocity ordering is the standard one; index 0 is rest, 1-4 are the axis
directions, 5-8 the diagonals. OPPOSITE[i] is the index of -e_i, which is what
bounce-back needs.
"""

from __future__ import annotations

import numpy as np

EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)

W = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=np.float64,
)

OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

CS2 = 1.0 / 3.0
Q = 9


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Second-order Maxwell-Boltzmann expansion. Returns shape (9, ny, nx)."""
    usq = ux * ux + uy * uy
    feq = np.empty((Q,) + rho.shape, dtype=rho.dtype)
    for i in range(Q):
        eu = EX[i] * ux + EY[i] * uy
        feq[i] = W[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq)
    return feq


def macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = f.sum(axis=0)
    ux = np.einsum("i,ijk->jk", EX.astype(f.dtype), f) / rho
    uy = np.einsum("i,ijk->jk", EY.astype(f.dtype), f) / rho
    return rho, ux, uy


def viscosity_to_omega(nu: float) -> float:
    tau = 3.0 * nu + 0.5
    return 1.0 / tau


def trt_magic_omega(omega_plus: float, magic: float = 0.25) -> float:
    """Antisymmetric relaxation rate for TRT.

    magic = 1/4 places the bounce-back wall exactly halfway between nodes,
    independent of viscosity. With plain BGK the wall location drifts with tau,
    which is the usual reason a validated case degrades when you change Re.
    """
    tau_plus = 1.0 / omega_plus
    tau_minus = magic / (tau_plus - 0.5) + 0.5
    return 1.0 / tau_minus
