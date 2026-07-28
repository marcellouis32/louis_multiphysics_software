"""Decaying Taylor-Green vortex: the order-of-accuracy check with no benchmark ceiling.

The lid-driven cavity tells us whether the solver agrees with Ghia et al., but Ghia
is a 129^2 solution from 1982 with its own error bar, so it cannot certify better
than about 1%. Taylor-Green has a closed-form solution of the incompressible
Navier-Stokes equations, so the error is exact at every resolution and the measured
order of accuracy means something.

The domain is periodic, which is the point: no walls, so this isolates the collision
and streaming operators from the boundary conditions. If the order comes out wrong
here, the bug is in the bulk scheme, not in bounce-back.

    ux = -U cos(kx) sin(ky) exp(-2 nu k^2 t)
    uy =  U sin(kx) cos(ky) exp(-2 nu k^2 t)
    p  = -(rho0 U^2 / 4) [cos(2kx) + cos(2ky)] exp(-4 nu k^2 t)

with kx = ky = k = 2 pi / n, which is what makes the field divergence-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lms.lbm.d2q9 import CS2, equilibrium, viscosity_to_omega
from lms.lbm.solver2d import D2Q9Solver


def analytic(n: int, u0: float, nu: float, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact velocity and density on an n x n periodic lattice at lattice time t."""
    k = 2.0 * np.pi / n
    j, i = np.mgrid[0:n, 0:n].astype(np.float64)
    decay = np.exp(-2.0 * nu * k * k * t)

    ux = -u0 * np.cos(k * i) * np.sin(k * j) * decay
    uy = u0 * np.sin(k * i) * np.cos(k * j) * decay
    p = -0.25 * u0 * u0 * (np.cos(2.0 * k * i) + np.cos(2.0 * k * j)) * decay**2
    rho = 1.0 + p / CS2
    return ux, uy, rho


@dataclass
class ConvergenceResult:
    n: int
    u0: float
    nu: float
    steps: int
    error: float
    """Relative L2 velocity error at the final time, normalised by the exact field."""

    reynolds: float
    mach: float


def run_case(
    n: int,
    u0: float,
    nu: float,
    steps: int,
    collision: str = "trt",
) -> ConvergenceResult:
    """Advance a Taylor-Green vortex `steps` lattice steps and measure the L2 error."""
    solid = np.zeros((n, n), dtype=bool)
    solver = D2Q9Solver((n, n), omega=viscosity_to_omega(nu), solid=solid, collision=collision)

    ux, uy, rho = analytic(n, u0, nu, 0.0)
    solver.rho, solver.ux, solver.uy = rho, ux, uy
    solver.f = equilibrium(rho, ux, uy)

    for _ in range(steps):
        solver.step()

    ex, ey, _ = analytic(n, u0, nu, float(steps))
    num = np.sqrt(np.sum((solver.ux - ex) ** 2 + (solver.uy - ey) ** 2))
    den = np.sqrt(np.sum(ex**2 + ey**2))

    return ConvergenceResult(
        n=n,
        u0=u0,
        nu=nu,
        steps=steps,
        error=float(num / den),
        reynolds=float(u0 * n / nu),
        mach=float(u0 / np.sqrt(CS2)),
    )


def diffusive_ladder(
    sizes: tuple[int, ...],
    u0: float = 0.05,
    nu: float = 0.01,
    steps0: int = 900,
    collision: str = "trt",
) -> list[ConvergenceResult]:
    """Refine with dt ~ dx^2: halve the lattice velocity when you double the grid.

    Viscosity in lattice units stays fixed (so tau is fixed), the Reynolds number is
    fixed, and the Mach number falls like 1/n. Because the LBM compressibility error
    is O(Ma^2) it then falls like 1/n^2, at the same rate as the discretisation error,
    and the scheme shows its true second order.

    Steps scale as n^2 so every grid is compared at the same physical time.
    """
    n0 = sizes[0]
    out = []
    for n in sizes:
        ratio = n / n0
        out.append(run_case(n, u0 / ratio, nu, round(steps0 * ratio**2), collision))
    return out


def acoustic_ladder(
    sizes: tuple[int, ...],
    u0: float = 0.05,
    reynolds: float = 160.0,
    steps0: int = 900,
    collision: str = "trt",
) -> list[ConvergenceResult]:
    """Refine with dt ~ dx: hold the lattice velocity fixed.

    This is the trap. Mach number is constant, so the O(Ma^2) compressibility error is
    a constant floor that refinement cannot touch. Once the discretisation error drops
    below that floor the measured order collapses toward first order or worse, and it
    looks like the solver is broken when it is only being measured wrongly.
    """
    n0 = sizes[0]
    out = []
    for n in sizes:
        ratio = n / n0
        nu = u0 * n / reynolds
        out.append(run_case(n, u0, nu, round(steps0 * ratio), collision))
    return out


def observed_order(results: list[ConvergenceResult]) -> float:
    """Least-squares slope of log(error) against log(n), negated."""
    logn = np.log(np.array([r.n for r in results], dtype=float))
    loge = np.log(np.array([r.error for r in results], dtype=float))
    slope = np.polyfit(logn, loge, 1)[0]
    return float(-slope)
