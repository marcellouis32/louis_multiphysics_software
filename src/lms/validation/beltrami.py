"""Beltrami (ABC) flow: the 3D order-of-accuracy check with no benchmark ceiling.

This is the 3D counterpart of `taylor_green.py`, and it exists for the same reason: a
benchmark can only certify the solver to the benchmark's own error bar, whereas an exact
solution certifies it to whatever the solver can actually achieve.

**The 3D Taylor-Green vortex is not an analytic solution.** In 2D the nonlinear term
vanishes identically, which is what makes the 2D case exact. In 3D it does not -- the
3D Taylor-Green vortex is a transition-to-turbulence benchmark with reference DNS data,
not a closed form. Reaching for it by name would quietly cost the exact-solution
property that made the Phase 0 accuracy work conclusive.

The ABC (Arnold-Beltrami-Childress) flow is exact, fully three-dimensional, unsteady and
periodic:

    u = A sin(kz) + C cos(ky)
    v = B sin(kx) + A cos(kz)
    w = C sin(ky) + B cos(kx)

Two properties do all the work. It is divergence-free by inspection -- u has no x
dependence, v no y, w no z. And it is a Beltrami field, meaning curl(u) = k u: the
vorticity is everywhere parallel to the velocity. That kills the nonlinear term, because

    (u . grad) u = grad(|u|^2 / 2) - u x curl(u) = grad(|u|^2 / 2) - u x (k u)
                 = grad(|u|^2 / 2)

is a pure gradient and is absorbed into the pressure. What is left is a heat equation,
and since every component is a single Fourier mode with |k| = k, the Laplacian gives
-k^2 u and the whole field decays as a single exponential:

    u(t) = u(0) exp(-nu k^2 t),      p = p0 - |u(t)|^2 / 2

Unlike Taylor-Green in 2D, all three velocity components are non-trivial and all three
streaming directions are exercised, so a bug confined to the z direction -- the class of
bug the extruded-cavity test cannot see, because nothing there drives z -- shows up here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lms.lbm.d2q9 import viscosity_to_omega
from lms.lbm.d3q19 import CS2, EX, EY, EZ, Q, W
from lms.lbm.solver3d import D3Q19Solver, init_backend


def analytic(
    n: int, u0: float, nu: float, t: float, abc: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact velocity and density on an n^3 periodic lattice at lattice time t.

    Returns `(ux, uy, uz, rho)`. `abc` scales the three Beltrami amplitudes; the
    isotropic default (1, 1, 1) is the usual choice and keeps every direction equally
    exercised.
    """
    a, b, c = abc
    k = 2.0 * np.pi / n
    i, j, m = np.mgrid[0:n, 0:n, 0:n].astype(np.float64)
    decay = np.exp(-nu * k * k * t)

    ux = u0 * (a * np.sin(k * m) + c * np.cos(k * j)) * decay
    uy = u0 * (b * np.sin(k * i) + a * np.cos(k * m)) * decay
    uz = u0 * (c * np.sin(k * j) + b * np.cos(k * i)) * decay

    # The nonlinear term is the gradient of |u|^2/2, so that is exactly what the
    # pressure has to balance.
    p = -0.5 * (ux**2 + uy**2 + uz**2)
    rho = 1.0 + p / CS2
    return ux, uy, uz, rho


def analytic_strain(
    n: int, u0: float, nu: float, t: float, abc: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """Exact velocity-gradient tensor, shape (3, 3, n, n, n) with `grad[a, b] = d u_b / d x_a`.

    Available in closed form, which matters: this feeds the non-equilibrium part of the
    initial condition, and computing it by finite differences would contaminate a
    convergence study with the differencing error it is trying to measure.
    """
    a, b, c = abc
    k = 2.0 * np.pi / n
    i, j, m = np.mgrid[0:n, 0:n, 0:n].astype(np.float64)
    s = u0 * k * np.exp(-nu * k * k * t)
    z = np.zeros((n, n, n))

    # u = u0(a sin kz + c cos ky), v = u0(b sin kx + a cos kz), w = u0(c sin ky + b cos kx)
    grad = np.empty((3, 3) + (n, n, n))
    grad[0, 0], grad[1, 0], grad[2, 0] = z, -s * c * np.sin(k * j), s * a * np.cos(k * m)
    grad[0, 1], grad[1, 1], grad[2, 1] = s * b * np.cos(k * i), z, -s * a * np.sin(k * m)
    grad[0, 2], grad[1, 2], grad[2, 2] = -s * b * np.sin(k * i), s * c * np.cos(k * j), z
    return grad


def nonequilibrium(rho: np.ndarray, grad: np.ndarray, tau: float) -> np.ndarray:
    """First-order Chapman-Enskog populations, shape (Q, n, n, n).

    Starting a run from equilibrium alone throws away the part of the distribution that
    encodes the strain rate. The solver rebuilds it within a few steps, but the discarded
    stress radiates an acoustic transient first, and that transient does not shrink at
    second order under refinement -- it contaminates the measured order of accuracy while
    looking exactly like a solver defect.

        f_i^(1) = -(tau w_i rho / cs^2) (e_ia e_ib - cs^2 delta_ab) S_ab
    """
    e = np.stack([EX, EY, EZ]).astype(np.float64)
    strain = 0.5 * (grad + grad.transpose(1, 0, 2, 3, 4))

    out = np.empty((Q,) + rho.shape)
    for i in range(Q):
        contraction = np.zeros_like(rho)
        for a in range(3):
            for b in range(3):
                q = e[a, i] * e[b, i] - (CS2 if a == b else 0.0)
                contraction += q * strain[a, b]
        out[i] = -(tau * W[i] * rho / CS2) * contraction
    return out


@dataclass
class ConvergenceResult3D:
    n: int
    u0: float
    nu: float
    steps: int
    error: float
    """Relative L2 velocity error at the final time, normalised by the exact field."""

    reynolds: float
    mach: float
    precision: str


def run_case(
    n: int,
    u0: float,
    nu: float,
    steps: int,
    collision: str = "trt",
    prefer_gpu: bool = False,
    precision: str = "fp64",
    equilibrium_only: bool = False,
) -> ConvergenceResult3D:
    """Advance a Beltrami flow `steps` lattice steps and measure the L2 error.

    Defaults to the fp64 CPU path. An order-of-accuracy study drives the error down
    toward the precision floor by construction, so measuring it in fp32 eventually
    measures rounding rather than discretisation.
    """
    init_backend(prefer_gpu=prefer_gpu, precision=precision)
    solid = np.zeros((n, n, n), dtype=bool)   # fully periodic: no walls anywhere
    solver = D3Q19Solver(
        (n, n, n), omega=viscosity_to_omega(nu), solid=solid, collision=collision
    )

    ux, uy, uz, rho = analytic(n, u0, nu, 0.0)
    fneq = None
    if not equilibrium_only:
        tau = 1.0 / viscosity_to_omega(nu)
        fneq = nonequilibrium(rho, analytic_strain(n, u0, nu, 0.0), tau)
    solver.set_state(rho, ux, uy, uz, f_neq=fneq)

    for _ in range(steps):
        solver.step()

    _, gx, gy, gz = solver.macroscopic()
    ex, ey, ez, _ = analytic(n, u0, nu, float(steps))
    num = np.sqrt(np.sum((gx - ex) ** 2 + (gy - ey) ** 2 + (gz - ez) ** 2))
    den = np.sqrt(np.sum(ex**2 + ey**2 + ez**2))

    return ConvergenceResult3D(
        n=n,
        u0=u0,
        nu=nu,
        steps=steps,
        error=float(num / den),
        reynolds=float(u0 * n / nu),
        mach=float(u0 / np.sqrt(CS2)),
        precision=precision,
    )


def diffusive_ladder(
    sizes: tuple[int, ...],
    u0: float = 0.05,
    nu: float = 0.01,
    steps0: int = 400,
    collision: str = "trt",
    prefer_gpu: bool = False,
    precision: str = "fp64",
    equilibrium_only: bool = False,
) -> list[ConvergenceResult3D]:
    """Refine with dt ~ dx^2: halve the lattice velocity when you double the grid.

    Viscosity stays fixed in lattice units, so tau is fixed and the Reynolds number is
    fixed, while the Mach number falls like 1/n. The LBM compressibility error is
    O(Ma^2), so it falls like 1/n^2 -- the same rate as the discretisation error -- and
    the scheme shows its true second order.

    Phase 0 established this the hard way in 2D. Under acoustic scaling the same code
    measured order ~1.1 with the error eventually *rising* under refinement, purely
    because a constant Mach number pins a constant compressibility floor. Do not
    rediscover that here.

    Steps scale as n^2 so every grid is compared at the same physical time.
    """
    n0 = sizes[0]
    out = []
    for n in sizes:
        ratio = n / n0
        out.append(
            run_case(
                n, u0 / ratio, nu, round(steps0 * ratio**2),
                collision, prefer_gpu, precision, equilibrium_only,
            )
        )
    return out


def observed_order(results: list[ConvergenceResult3D]) -> float:
    """Least-squares slope of log(error) against log(n), negated."""
    logn = np.log(np.array([r.n for r in results], dtype=float))
    loge = np.log(np.array([r.error for r in results], dtype=float))
    return float(-np.polyfit(logn, loge, 1)[0])
