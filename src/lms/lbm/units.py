"""Physical-to-lattice unit conversion: the bridge between the schema and the solver.

The schema speaks SI (metres, seconds, Pa·s); the solver speaks lattice units (dx = dt =
rho = 1). Every quantity crosses that bridge through exactly two choices:

    dx  = tank diameter / cells_across_tank        (resolution, from the schema)
    dt  set so the impeller tip moves at Ma * cs   (lattice_mach, from the schema)

Everything else -- viscosity, rotation rate, step counts -- follows by dimensional
analysis, and the dimensionless groups must survive the crossing untouched: the lattice
impeller Reynolds number equals the physical one exactly, or the simulation is quietly
running a different problem than the case describes. The tests assert that round trip
rather than trusting it.

A consequence worth seeing once: for water in a lab-scale tank the *molecular* lattice
viscosity is tiny (tau - 0.5 ~ 1e-5 for the standard Rushton case). The flow the solver
actually feels is the eddy viscosity; without the LES model and the regularized
collision operator from Phase 1 these conditions would be unrunnable, which is why the
phases came in the order they did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lms.lbm.d3q19 import CS2
from lms.schema.case import Case

CS = math.sqrt(CS2)


@dataclass(frozen=True)
class LatticeScales:
    """Everything the solver and scripts need to run a Case in lattice units."""

    dx_m: float
    dt_s: float

    nx: int
    ny: int
    nz: int
    """Domain including the one-cell solid rim on every face."""

    nu: float
    """Lattice kinematic viscosity (molecular; the LES model adds nu_t on top)."""

    tau: float
    u_tip: float
    """Impeller tip speed in lattice units = lattice_mach * cs, by construction."""

    impeller_radius: float
    """Cells. Fractional on purpose -- rounding it would change the Reynolds number."""

    omega_shaft: float
    """Shaft angular velocity, radians per step."""

    steps_per_revolution: float
    reynolds: float
    """Impeller Reynolds number rho N D^2 / mu, recomputed on the lattice side. Must
    equal `case.dimensionless().reynolds` to round-off; the tests hold it there."""

    mach: float

    def steps_for(self, seconds: float) -> int:
        return round(seconds / self.dt_s)


def scales_from_case(case: Case) -> LatticeScales:
    vessel = case.geometry.vessel
    imp = case.primary_impeller
    fluid = case.fluid

    n = case.numerics.cells_across_tank
    dx = case.numerics.dx(vessel.diameter_m)

    # dt from the Mach choice: the tip is the fastest thing in the tank, so pinning it
    # at Ma * cs is what keeps the whole flow inside the low-Mach envelope the scheme
    # needs. This is the same lesson Phase 0 paid for -- compressibility error is
    # O(Ma^2) -- applied at setup time instead of discovered afterwards.
    u_tip_phys = math.pi * imp.speed_hz * imp.diameter_m
    u_tip = case.numerics.lattice_mach * CS
    dt = dx * u_tip / u_tip_phys

    nu_phys = float(fluid.viscosity_pa_s) / fluid.density_kg_m3
    nu = nu_phys * dt / (dx * dx)

    revs_per_step = imp.speed_hz * dt
    d_lat = imp.diameter_m / dx

    nz_fluid = round(vessel.liquid_level_m / dx)
    return LatticeScales(
        dx_m=dx,
        dt_s=dt,
        nx=n + 2,
        ny=n + 2,
        nz=nz_fluid + 2,
        nu=nu,
        tau=3.0 * nu + 0.5,
        u_tip=u_tip,
        impeller_radius=d_lat / 2.0,
        omega_shaft=2.0 * math.pi * revs_per_step,
        steps_per_revolution=1.0 / revs_per_step,
        reynolds=revs_per_step * d_lat * d_lat / nu,
        mach=case.numerics.lattice_mach,
    )
