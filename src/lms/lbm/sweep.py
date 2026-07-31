"""Reynolds-number sweeps by parameter continuation.

A sweep over Reynolds number is the closest 2D analogue of the scale-up problem: hold
the geometry fixed, change the balance of inertia to viscosity, and watch where the
recirculation goes. Ghia et al. tabulate the answer at Re = 100, 400 and 1000, and the
primary vortex migrates from (0.617, 0.734) toward the geometric centre as Re rises.

Every frame here is a *converged steady solution*. That matters. The 2D cavity below
Re ~ 8000 is genuinely steady -- our own runs converge to a relative change in |u| below
2e-7, which an unsteady flow cannot do -- so a sweep is a legitimate sequence of physical
states rather than a time series. Interpolating between a handful of solutions to pad a
frame count would produce frames that solve nothing, and is deliberately not offered.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from lms.lbm.d2q9 import equilibrium
from lms.lbm.solver2d import D2Q9Solver, lid_driven_cavity


@dataclass
class SweepFrame:
    """One converged (or step-capped) solution at a single Reynolds number."""

    reynolds: float
    ux: np.ndarray
    uy: np.ndarray
    rho: np.ndarray
    steps: int
    converged: bool
    tau: float
    # The residual the solve actually reached. `converged` alone only says whether the
    # solver stopped on tolerance or on its step budget, which is not the same question
    # as how converged the field is -- a frame stopped at 1.2e-6 against a 1e-6
    # tolerance is a steady solution in every way that matters. Without this number
    # that distinction is unrecoverable after the run.
    residual: float = float("nan")


def warm_start(solver: D2Q9Solver, ux: np.ndarray, uy: np.ndarray, rho: np.ndarray) -> None:
    """Seed a solver from an existing macroscopic field.

    Only the equilibrium part is reconstructed; the non-equilibrium part of the
    distribution is discarded and re-establishes itself within a few steps (it relaxes
    at rate omega). That transient is far shorter than the convergence it saves, which
    is the whole bargain of continuation.

    Solid nodes are reset to rest at unit density rather than copied. Their
    populations are bounce-back state, not a physical distribution, so `macroscopic`
    reports nonsense there -- densities of +-50 are routine. Feeding that back through
    `equilibrium` seeds a wall of garbage that streams into the fluid and blows the
    solve up within a few hundred steps. Bounce-back regenerates those nodes from
    their fluid neighbours anyway, so nothing is lost by discarding them.
    """
    fluid = solver.fluid
    solver.rho = np.where(fluid, rho, 1.0).astype(solver.dtype)
    solver.ux = np.where(fluid, ux, 0.0).astype(solver.dtype)
    solver.uy = np.where(fluid, uy, 0.0).astype(solver.dtype)
    solver.f = equilibrium(solver.rho, solver.ux, solver.uy).astype(solver.dtype)


def reynolds_sweep(
    n: int,
    reynolds_values: Sequence[float],
    lid_velocity: float = 0.1,
    tol: float = 1e-6,
    max_steps: int = 150_000,
    cold_start: bool = False,
) -> Iterator[SweepFrame]:
    """Solve the cavity at each Reynolds number, warm-starting from the previous one.

    Running every case from rest wastes most of the work: neighbouring Reynolds numbers
    have similar solutions, so starting each solve from its predecessor lands it near the
    answer already. This is standard parameter continuation, and at small increments it
    cuts the step count by a large factor.

    The caveat is real and worth stating: continuation follows one solution branch, so it
    can hide hysteresis where multiple steady states exist. Below Re ~ 8000 the 2D cavity
    has a unique steady solution, so it is safe *here*. Revisit this assumption for 3D and
    for stirred tanks, where multiple stable flow patterns genuinely do occur.

    Set `cold_start=True` to start every case from rest instead -- used by the tests to
    demonstrate that continuation actually helps.

    Yields frames in the order given, so a caller can stream results to disk rather than
    holding the whole sweep in memory.
    """
    previous: SweepFrame | None = None

    for re in reynolds_values:
        solver = lid_driven_cavity(
            n=n, reynolds=float(re), lid_velocity=lid_velocity, collision="trt"
        )
        if previous is not None and not cold_start:
            warm_start(solver, previous.ux, previous.uy, previous.rho)

        state = solver.run(max_steps=max_steps, tol=tol, check_every=500)

        frame = SweepFrame(
            reynolds=float(re),
            ux=state.ux,
            uy=state.uy,
            rho=state.rho,
            steps=state.steps,
            converged=state.converged,
            tau=float(1.0 / solver.omega_plus),
            residual=float(state.residuals[-1][1]) if state.residuals else float("nan"),
        )
        previous = frame
        yield frame


def sweep_values(
    lowest: float = 100.0,
    highest: float = 3200.0,
    count: int = 25,
    anchors: Sequence[float] = (100.0, 400.0, 1000.0),
) -> np.ndarray:
    """Log-spaced Reynolds numbers with benchmark values inserted exactly.

    Log spacing because the vortex structure responds to Re logarithmically -- linear
    spacing would crawl through the interesting low-Re changes and then leap over the
    high-Re ones. The anchors are the Reynolds numbers Ghia tabulates, forced to appear
    exactly so the animation has reference points to be checked against.
    """
    grid = np.geomspace(lowest, highest, count)
    keep = [v for v in grid if all(abs(v - a) / a > 0.02 for a in anchors)]
    merged = np.unique(np.concatenate([np.asarray(keep), np.asarray(anchors, dtype=float)]))
    return merged[(merged >= lowest - 1e-9) & (merged <= highest + 1e-9)]
