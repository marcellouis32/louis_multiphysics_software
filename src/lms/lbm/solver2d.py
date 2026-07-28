"""2D D2Q9 solver with halfway bounce-back walls.

This is the reference implementation: pure NumPy, readable, deliberately not
optimised. It exists to pin down the algorithm against analytic and benchmark
solutions before the GPU port, and it stays in the tree afterwards as the oracle
the GPU kernels are tested against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from lms.lbm.d2q9 import (
    EX,
    EY,
    OPPOSITE,
    Q,
    W,
    equilibrium,
    macroscopic,
    trt_magic_omega,
    viscosity_to_omega,
)


def stream_function(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Stream function psi, defined by u = dpsi/dy and v = -dpsi/dx.

    Only meaningful for 2D incompressible flow. Integrated from psi = 0 at the
    origin: along the bottom edge using v, then up each column using u, both with
    the trapezoid rule. Iso-contours of psi are exact streamlines, which makes this
    the standard way to locate recirculation zones and compare vortex centres --
    a scalar check that profiles alone cannot provide, since a solver can match
    centreline velocities while putting the vortex in the wrong place.
    """
    ny, nx = ux.shape
    psi = np.zeros((ny, nx), dtype=np.float64)
    psi[0, :] = -np.concatenate([[0.0], np.cumsum(0.5 * (uy[0, 1:] + uy[0, :-1]))])
    psi[1:, :] = psi[0, :][None, :] + np.cumsum(0.5 * (ux[1:, :] + ux[:-1, :]), axis=0)
    return psi


def vortex_centre(psi: np.ndarray) -> tuple[float, float, float]:
    """Locate the strongest (most negative) circulation cell.

    Returns normalised (x, y) of the extremum and the psi value there.
    """
    idx = np.unravel_index(np.argmin(psi), psi.shape)
    ny, nx = psi.shape
    return (idx[1] + 0.5) / nx, (idx[0] + 0.5) / ny, float(psi[idx])


@dataclass
class SolverState:
    ux: np.ndarray
    uy: np.ndarray
    rho: np.ndarray
    steps: int
    converged: bool
    residuals: list[tuple[int, float]] = field(default_factory=list)

    @property
    def speed(self) -> np.ndarray:
        return np.hypot(self.ux, self.uy)

    def vorticity(self) -> np.ndarray:
        duy_dx = np.gradient(self.uy, axis=1)
        dux_dy = np.gradient(self.ux, axis=0)
        return duy_dx - dux_dy

    def stream_function(self) -> np.ndarray:
        return stream_function(self.ux, self.uy)


class D2Q9Solver:
    """Row index is y (row 0 = bottom), column index is x."""

    def __init__(
        self,
        shape: tuple[int, int],
        omega: float,
        solid: np.ndarray,
        collision: Literal["bgk", "trt"] = "trt",
        dtype=np.float64,
    ) -> None:
        ny, nx = shape
        self.shape = shape
        self.dtype = dtype
        self.collision = collision
        self.omega_plus = omega
        self.omega_minus = trt_magic_omega(omega) if collision == "trt" else omega
        self.solid = solid.astype(bool)
        self.fluid = ~self.solid

        self.rho = np.ones(shape, dtype=dtype)
        self.ux = np.zeros(shape, dtype=dtype)
        self.uy = np.zeros(shape, dtype=dtype)
        self.f = equilibrium(self.rho, self.ux, self.uy).astype(dtype)

        # Momentum injected by moving walls, added to the bounced-back populations.
        self.wall_forcing = np.zeros((Q,) + shape, dtype=dtype)

    def set_moving_wall(self, mask: np.ndarray, ux: float, uy: float = 0.0, rho_w: float = 1.0):
        """mask selects solid nodes that translate with velocity (ux, uy)."""
        for i in range(Q):
            eu = EX[i] * ux + EY[i] * uy
            self.wall_forcing[i][mask] = 6.0 * W[i] * rho_w * eu

    def _collide(self, feq: np.ndarray) -> np.ndarray:
        if self.collision == "bgk":
            return self.f - self.omega_plus * (self.f - feq)

        f_opp = self.f[OPPOSITE]
        feq_opp = feq[OPPOSITE]
        f_sym = 0.5 * (self.f + f_opp)
        f_asym = 0.5 * (self.f - f_opp)
        feq_sym = 0.5 * (feq + feq_opp)
        feq_asym = 0.5 * (feq - feq_opp)
        return (
            self.f
            - self.omega_plus * (f_sym - feq_sym)
            - self.omega_minus * (f_asym - feq_asym)
        )

    def step(self) -> None:
        rho, ux, uy = macroscopic(self.f)
        ux[self.solid] = 0.0
        uy[self.solid] = 0.0
        self.rho, self.ux, self.uy = rho, ux, uy

        fpost = self._collide(equilibrium(rho, ux, uy))

        # Halfway bounce-back: at a solid node, the population that arrived from a
        # fluid neighbour is reversed, then streaming carries it back. The wall
        # therefore sits midway between the solid node and its fluid neighbour.
        reversed_f = self.f[OPPOSITE] + self.wall_forcing
        fpost[:, self.solid] = reversed_f[:, self.solid]

        for i in range(Q):
            self.f[i] = np.roll(fpost[i], shift=(int(EY[i]), int(EX[i])), axis=(0, 1))

    def run(
        self,
        max_steps: int,
        tol: float = 1e-6,
        check_every: int = 500,
        callback: Callable[[int, "D2Q9Solver"], None] | None = None,
    ) -> SolverState:
        prev = np.hypot(self.ux, self.uy).copy()
        residuals: list[tuple[int, float]] = []
        converged = False
        step = 0

        for step in range(1, max_steps + 1):
            self.step()

            if not np.isfinite(self.rho).all():
                raise FloatingPointError(
                    f"solver diverged at step {step}. omega={self.omega_plus:.4f} "
                    f"(tau={1 / self.omega_plus:.4f}); tau below ~0.51 or lattice "
                    "velocity above ~0.1 will do this. Lower the lid velocity or "
                    "raise the resolution."
                )

            if step % check_every == 0:
                speed = np.hypot(self.ux, self.uy)
                denom = np.linalg.norm(speed) or 1.0
                res = float(np.linalg.norm(speed - prev) / denom)
                residuals.append((step, res))
                prev = speed.copy()
                if callback:
                    callback(step, self)
                if res < tol:
                    converged = True
                    break

        return SolverState(
            ux=self.ux.copy(),
            uy=self.uy.copy(),
            rho=self.rho.copy(),
            steps=step,
            converged=converged,
            residuals=residuals,
        )


def lid_driven_cavity(
    n: int = 256,
    reynolds: float = 1000.0,
    lid_velocity: float = 0.1,
    collision: Literal["bgk", "trt"] = "trt",
) -> D2Q9Solver:
    """Standard benchmark: square cavity, three no-slip walls, top lid sliding in +x.

    The cavity side length in lattice units is n-2, because the halfway bounce-back
    walls sit midway between the solid ring and the first fluid layer.
    """
    if lid_velocity > 0.15:
        raise ValueError(
            f"lid velocity {lid_velocity} is far into the compressible regime; "
            "LBM needs lattice Mach << 1, so keep this at or below ~0.1"
        )

    shape = (n, n)
    length = n - 2
    nu = lid_velocity * length / reynolds
    omega = viscosity_to_omega(nu)

    solid = np.zeros(shape, dtype=bool)
    solid[0, :] = solid[-1, :] = True
    solid[:, 0] = solid[:, -1] = True

    solver = D2Q9Solver(shape, omega, solid, collision=collision)

    lid = np.zeros(shape, dtype=bool)
    lid[-1, :] = True
    solver.set_moving_wall(lid, ux=lid_velocity)

    solver.reynolds = reynolds
    solver.lid_velocity = lid_velocity
    solver.cavity_length = length
    return solver


def cavity_centerlines(
    state: SolverState, lid_velocity: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract the two profiles Ghia et al. tabulate, in normalised coordinates.

    Returns (y, u_along_vertical_centerline, x, v_along_horizontal_centerline).
    """
    ny, nx = state.ux.shape
    interior = slice(1, -1)

    # Node centres sit half a lattice unit inside the walls.
    coord = (np.arange(1, ny - 1) - 0.5) / (ny - 2)

    mid_x = nx // 2
    mid_y = ny // 2
    u_line = state.ux[interior, mid_x] / lid_velocity
    v_line = state.uy[mid_y, interior] / lid_velocity
    return coord, u_line, coord, v_line
