"""Exact-solution validation for the scalar lattice, before it ever sees the tank.

Two closed forms, each isolating one failure mode:

**Pure diffusion of a Gaussian.** With u = 0 a Gaussian stays Gaussian and its variance
grows linearly: sigma^2(t) = sigma_0^2 + 2 D t. This pins the diffusivity-to-tau map --
a wrong cs^2 in the D3Q7 equilibrium shows up here as a wrong diffusion *rate* while
everything looks qualitatively fine.

**Advection of a Gaussian by a uniform flow.** The blob must translate at exactly u
while spreading no faster than D says. The measured excess spreading *is* the numerical
diffusion, reported as a diffusivity so it can be compared directly against the
physical one -- that ratio is the honest ceiling on the Schmidt numbers the lattice can
represent, and the reason the module docstring refuses to pretend Sc = 1000 is being
resolved.

Coupling the scalar to the tank before these pass would leave a wrong blend time with
two possible causes. After they pass it has one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lms.lbm.scalar3d import ScalarD3Q7
from lms.lbm.solver3d import _taichi


def gaussian_blob(n: int, sigma: float, centre=None) -> np.ndarray:
    """Periodic-safe Gaussian: sigma must stay well under n/6 or the images overlap."""
    if centre is None:
        centre = ((n - 1) / 2.0,) * 3
    x, y, z = np.mgrid[0:n, 0:n, 0:n].astype(np.float64)
    r2 = (x - centre[0]) ** 2 + (y - centre[1]) ** 2 + (z - centre[2]) ** 2
    return np.exp(-r2 / (2.0 * sigma**2))


def measured_moments(c: np.ndarray):
    """Centroid and isotropic variance of a concentration field, periodic-naive
    (valid while the blob stays far from the wrap)."""
    n = c.shape[0]
    x, y, z = np.mgrid[0:n, 0:n, 0:n].astype(np.float64)
    m = c.sum()
    cx, cy, cz = (x * c).sum() / m, (y * c).sum() / m, (z * c).sum() / m
    var = (((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) * c).sum() / m / 3.0
    return (cx, cy, cz), float(var)


@dataclass
class DiffusionResult:
    n: int
    d: float
    steps: int
    variance_error: float
    """Relative error of the measured variance growth against 2 D t."""

    profile_error: float
    """Relative L2 error against the exact Gaussian at the final time."""


def run_diffusion(n: int, d: float, steps: int, sigma0: float = 4.0) -> DiffusionResult:
    """Zero-velocity diffusion against the closed form."""
    ti = _taichi()
    vel = ti.Vector.field(3, ti.lang.impl.get_runtime().default_fp, shape=(n, n, n))
    vel.fill(0)
    solver = ScalarD3Q7((n, n, n), d_molecular=d, solid=np.zeros((n, n, n), bool),
                        vel=vel)
    solver.set_concentration(gaussian_blob(n, sigma0))

    for _ in range(steps):
        solver.step()

    c = solver.concentration()
    _, var = measured_moments(c)
    var_exact = sigma0**2 + 2.0 * d * steps

    sigma_t = np.sqrt(var_exact)
    amplitude = (sigma0**2 / var_exact) ** 1.5
    exact = amplitude * gaussian_blob(n, sigma_t)
    return DiffusionResult(
        n=n, d=d, steps=steps,
        variance_error=float(abs(var - var_exact) / var_exact),
        profile_error=float(np.linalg.norm(c - exact) / np.linalg.norm(exact)),
    )


@dataclass
class AdvectionResult:
    u: float
    d: float
    steps: int
    centroid_error: float
    """How far the blob's centroid is from u*t, in cells."""

    d_numerical: float
    """Excess spreading rate beyond the physical D -- the lattice's own diffusion."""

    schmidt_ceiling: float
    """nu_typical / d_numerical: above this Schmidt number the lattice's own
    diffusion, not the physical diffusivity, controls the mixing."""


def run_advection(
    n: int, u: float, d: float, steps: int, sigma0: float = 4.0, nu_ref: float = 1e-3
) -> AdvectionResult:
    """Uniform advection: translation must be exact-rate, spreading must be D's."""
    ti = _taichi()
    vel = ti.Vector.field(3, ti.lang.impl.get_runtime().default_fp, shape=(n, n, n))
    host = np.zeros((n, n, n, 3))
    host[..., 0] = u
    vel.from_numpy(host.astype(
        np.float32 if vel.dtype == ti.f32 else np.float64
    ))

    solver = ScalarD3Q7((n, n, n), d_molecular=d, solid=np.zeros((n, n, n), bool),
                        vel=vel)
    start = ((n - 1) / 2.0 - u * steps / 2.0, (n - 1) / 2.0, (n - 1) / 2.0)
    solver.set_concentration(gaussian_blob(n, sigma0, centre=start))
    _, var0 = measured_moments(solver.concentration())

    for _ in range(steps):
        solver.step()

    c = solver.concentration()
    (cx, _, _), var = measured_moments(c)
    d_total = (var - var0) / (2.0 * steps)
    d_num = max(d_total - d, 0.0)
    return AdvectionResult(
        u=u, d=d, steps=steps,
        centroid_error=float(abs(cx - (start[0] + u * steps))),
        d_numerical=float(d_num),
        schmidt_ceiling=float(nu_ref / d_num) if d_num > 0 else float("inf"),
    )
