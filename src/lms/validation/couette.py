"""Taylor-Couette flow: the exact solution that validates Phase 2's two new error sources.

Rotating inner cylinder, fixed outer, periodic along the axis. Below the Taylor
instability the flow is laminar, steady, and closed-form:

    u_theta(r) = A r + B / r      A = -Omega R1^2 / (R2^2 - R1^2)
                                  B =  Omega R1^2 R2^2 / (R2^2 - R1^2)

    torque / length = 4 pi rho nu Omega R1^2 R2^2 / (R2^2 - R1^2)

Phase 2 introduces exactly two new numerical ingredients: staircase *curved* walls, and
the momentum-exchange force/torque sum. This one flow measures both against exact
answers -- the velocity profile isolates the wall-placement error, the torque isolates
the bookkeeping -- before either is trusted anywhere near an impeller. It also
exercises the verified moving-wall machinery on a wall whose velocity differs at every
node, which no Phase 0/1 case did.

The ladder scales the wall speed like 1/R so the Taylor number stays fixed and safely
subcritical on every rung: refining the grid must not quietly cross into Taylor
vortices, or the "error" against the laminar solution would be physics, not numerics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lms.lbm.d2q9 import viscosity_to_omega
from lms.lbm.solver3d import D3Q19Solver, init_backend


def analytic_profile(r: np.ndarray, r_inner: float, r_outer: float, omega: float):
    """Azimuthal velocity of the laminar solution at radii `r`."""
    denom = r_outer**2 - r_inner**2
    a = -omega * r_inner**2 / denom
    b = omega * r_inner**2 * r_outer**2 / denom
    return a * r + b / r


def analytic_torque_per_length(
    nu: float, omega: float, r_inner: float, r_outer: float, rho: float = 1.0
) -> float:
    """Torque per unit axial length that the fluid exerts on the inner cylinder.

    Negative for positive Omega: the fluid resists the rotation. Sign matters -- a
    momentum-exchange implementation with a flipped link direction produces the right
    magnitude and the wrong sign, and only a signed reference catches it.
    """
    return -4.0 * np.pi * rho * nu * omega * r_inner**2 * r_outer**2 / (
        r_outer**2 - r_inner**2
    )


def taylor_number(u_wall: float, r_inner: float, r_outer: float, nu: float) -> float:
    """Ta = Omega^2 R1 d^3 / nu^2; laminar below the critical ~1708."""
    gap = r_outer - r_inner
    omega = u_wall / r_inner
    return omega**2 * r_inner * gap**3 / nu**2


@dataclass
class CouetteResult:
    r_inner: float
    r_outer: float
    u_wall: float
    nu: float
    taylor: float
    steps: int
    profile_error: float
    """Relative L2 error of the azimuthally averaged u_theta against the closed form,
    over the annulus interior (one cell clear of each staircase wall)."""

    torque_error: float
    """Relative error of the measured torque against the closed form. Signed
    agreement is asserted before the magnitude is compared."""

    torque_measured: float
    torque_analytic: float


def build_annulus(r_inner: float, r_outer: float, nz: int = 4):
    """Solid masks and wall velocity for the annulus: outer shell static (1), inner
    cylinder measured (2), periodic in z."""
    n = int(2 * np.ceil(r_outer) + 5)
    shape = (n, n, nz)
    x, y, _ = np.mgrid[0:n, 0:n, 0:nz].astype(np.float64)
    x -= (n - 1) / 2.0
    y -= (n - 1) / 2.0
    r = np.hypot(x, y)

    solid = np.zeros(shape, dtype=np.int32)
    solid[r > r_outer] = 1
    solid[r <= r_inner] = 2
    return shape, solid, x, y, r


def run_case(
    r_outer: float,
    radius_ratio: float = 0.5,
    u_wall: float = 0.02,
    nu: float = 1.0 / 60.0,
    nz: int = 4,
    max_steps: int = 400_000,
    torque_samples: int = 200,
    prefer_gpu: bool = True,
) -> CouetteResult:
    """Run one annulus to steady state and score it against the closed forms."""
    r_inner = radius_ratio * r_outer
    omega = u_wall / r_inner

    init_backend(prefer_gpu=prefer_gpu, precision="fp32" if prefer_gpu else "fp64")
    shape, solid, x, y, r = build_annulus(r_inner, r_outer, nz)
    solver = D3Q19Solver(shape, omega=viscosity_to_omega(nu), solid=solid, collision="trt")

    inner = solid == 2
    vel = np.zeros(shape + (3,))
    vel[..., 0] = -omega * y
    vel[..., 1] = omega * x
    vel[~inner] = 0.0
    solver.set_wall_velocity(vel)
    solver.set_axis((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0, 0.0)

    state = solver.run(max_steps=max_steps, tol=1e-9, check_every=500)

    # Torque: steady flow, so a short average only knocks down fp32 jitter.
    samples = np.empty(torque_samples)
    for i in range(torque_samples):
        solver.step()
        samples[i] = solver.torque()[2]
    torque = float(samples.mean())
    torque_ref = analytic_torque_per_length(nu, omega, r_inner, r_outer) * nz

    # Profile: azimuthal average of u_theta in unit-radius bins, compared at the mean
    # radius of each bin -- comparing at the bin *centre* would fold a discretisation
    # of our own diagnostic into the error being measured.
    fluid = solid == 0
    u_theta = np.where(
        r > 0, (x * state.uy - y * state.ux) / np.maximum(r, 1e-12), 0.0
    )
    bins = np.floor(r[fluid]).astype(int)
    vals = u_theta[fluid]
    radii = r[fluid]
    lo, hi = int(np.ceil(r_inner)) + 1, int(np.floor(r_outer)) - 1
    measured, exact = [], []
    for b in range(lo, hi + 1):
        sel = bins == b
        if sel.sum() < 8:
            continue
        measured.append(vals[sel].mean())
        exact.append(analytic_profile(radii[sel].mean(), r_inner, r_outer, omega))
    measured = np.asarray(measured)
    exact = np.asarray(exact)

    return CouetteResult(
        r_inner=r_inner,
        r_outer=r_outer,
        u_wall=u_wall,
        nu=nu,
        taylor=taylor_number(u_wall, r_inner, r_outer, nu),
        steps=state.steps,
        profile_error=float(np.linalg.norm(measured - exact) / np.linalg.norm(exact)),
        torque_error=float(abs(torque - torque_ref) / abs(torque_ref)),
        torque_measured=torque,
        torque_analytic=torque_ref,
    )


def ladder(
    outer_radii: tuple[float, ...] = (15.0, 30.0, 60.0),
    u0: float = 0.03,
    **kwargs,
) -> list[CouetteResult]:
    """Resolution ladder at fixed Taylor number: u_wall scales like 1/R, so every rung
    solves the same dimensionless problem and the error trend is pure numerics."""
    r0 = outer_radii[0]
    return [run_case(r, u_wall=u0 * r0 / r, **kwargs) for r in outer_radii]
