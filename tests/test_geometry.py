"""Tank voxelisation and unit-conversion tests.

The geometry module is the oracle the in-kernel rotating impeller will be checked
against, so it has to be right for reasons a human can verify: analytic volumes,
symmetries the physics demands, and dimensionless groups that survive the unit
conversion untouched.
"""

import numpy as np
import pytest

from lms.geometry.tank import tank_from_case, vessel_mask
from lms.lbm.units import scales_from_case
from lms.schema.case import load_case

EXAMPLE = "cases/examples/rushton_standard.yaml"


@pytest.fixture(scope="module")
def case():
    return load_case(EXAMPLE)


@pytest.fixture(scope="module")
def scales(case):
    return scales_from_case(case)


@pytest.fixture(scope="module")
def tank(case, scales):
    return tank_from_case(case, scales)


class TestUnits:
    def test_reynolds_number_survives_the_conversion(self, case, scales):
        """The one identity that matters most: the lattice is running the same
        dimensionless problem the case describes, exactly, not approximately."""
        assert scales.reynolds == pytest.approx(case.dimensionless().reynolds, rel=1e-12)

    def test_tip_mach_is_what_the_schema_asked_for(self, case, scales):
        assert scales.mach == case.numerics.lattice_mach
        assert scales.u_tip == pytest.approx(scales.mach / np.sqrt(3.0), rel=1e-12)

    def test_tip_speed_round_trips_to_si(self, case, scales):
        u_tip_si = scales.u_tip * scales.dx_m / scales.dt_s
        assert u_tip_si == pytest.approx(case.tip_speed_m_s(), rel=1e-12)

    def test_steps_per_revolution_is_consistent(self, case, scales):
        seconds_per_rev = scales.steps_per_revolution * scales.dt_s
        assert seconds_per_rev == pytest.approx(1.0 / case.primary_impeller.speed_hz,
                                                rel=1e-12)
        # One revolution of omega_shaft is exactly 2 pi.
        assert scales.omega_shaft * scales.steps_per_revolution == pytest.approx(
            2.0 * np.pi, rel=1e-12
        )

    def test_the_molecular_tau_really_is_that_small(self, scales):
        """Documents the regime rather than guards a computation: water at lab scale
        gives tau - 0.5 ~ 1e-5, which is why the regularized operator and LES are
        prerequisites for this phase rather than luxuries."""
        assert 0.5 < scales.tau < 0.5001


class TestVessel:
    def test_fluid_fraction_converges_to_pi_over_four(self):
        """The bore is a circle inscribed in the n x n cross-section, so the interior
        fluid fraction must approach pi/4, with the staircase error shrinking like the
        perimeter-to-area ratio, ~1/n."""
        errors = []
        for n in (32, 64, 128):
            shape = (n + 2, n + 2, 6)
            fluid = ~vessel_mask(shape, radius=n / 2.0)
            fraction = fluid[:, :, 3].sum() / (n + 2) ** 2
            errors.append(abs(fraction - np.pi / 4.0 * (n / (n + 2)) ** 2))
        assert errors[2] < errors[0]
        assert errors[2] < 0.01

    def test_bottom_and_lid_are_solid(self, tank):
        assert tank.static_solid[:, :, 0].all()
        assert tank.static_solid[:, :, -1].all()

    def test_no_fluid_outside_the_bore(self, tank):
        nx, ny, _ = tank.shape
        x, y = np.mgrid[0:nx, 0:ny].astype(float)
        r = np.hypot(x - (nx - 1) / 2, y - (ny - 1) / 2)
        outside = r > tank.radius + 1.0
        assert tank.static_solid[outside, :].all()


class TestBaffles:
    def test_four_baffles_have_four_fold_symmetry(self, tank):
        """rot90 about the domain centre is exact on this grid, so a 4-baffle mask must
        equal its own quarter-turn exactly -- not approximately."""
        mid = tank.shape[2] // 2
        plane = tank.static_solid[:, :, mid]
        assert np.array_equal(plane, np.rot90(plane))


class TestImpeller:
    def test_six_blades_have_six_fold_symmetry(self, tank):
        """mask(theta) must equal mask(theta + 2 pi / 6): rotating by one blade pitch
        is the identity. This is the symmetry the torque signal will inherit."""
        a = tank.impeller_mask(0.3)
        b = tank.impeller_mask(0.3 + 2.0 * np.pi / 6.0)
        assert np.array_equal(a, b)

    def test_quarter_turn_equals_rotating_the_array(self, tank):
        """The strongest available check of the point test: rotating the impeller by
        90 degrees must equal rotating the voxel array by 90 degrees, cell for cell,
        because the axis sits at a grid-rotation-symmetric point."""
        a = tank.impeller_mask(0.0)
        b = tank.impeller_mask(np.pi / 2.0)
        assert np.array_equal(b, np.rot90(a, axes=(0, 1)))

    def test_blade_count_reads_from_the_mask(self, tank):
        """Count blade crossings on a ring at blade mid-radius: six blades, six
        solid arcs."""
        p = tank.impeller["z_centre"]
        # Between the disc rim and the blade tips: the only solid on that ring is
        # blade. Sampling at blade mid-radius would land on the disc and count 1 arc.
        mid_r = 0.5 * (tank.impeller["disc_radius"] + tank.impeller["blade_outer"])
        mask = tank.impeller_mask(0.0)[:, :, round(p)]
        angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
        nx, ny, _ = tank.shape
        xs = np.clip(np.round((nx - 1) / 2 + mid_r * np.cos(angles)).astype(int), 0, nx - 1)
        ys = np.clip(np.round((ny - 1) / 2 + mid_r * np.sin(angles)).astype(int), 0, ny - 1)
        ring = mask[xs, ys]
        crossings = int((ring.astype(int) - np.roll(ring, 1).astype(int) == 1).sum())
        assert crossings == 6

    def test_volume_matches_the_analytic_solid(self, case, scales, tank):
        """Voxel count against the closed-form volume of disc + hub + blades + shaft.
        Loose tolerance: plates are ~1 cell thick, so the staircase error on them is
        genuinely large -- this catches a wrong shape, not a wrong fifth digit."""
        p = tank.impeller
        mask = tank.impeller_mask(0.0)
        disc = np.pi * p["disc_radius"] ** 2 * p["thickness"]
        hub = np.pi * p["hub_radius"] ** 2 * p["blade_height"]
        blades = (
            p["n_blades"]
            * (p["blade_outer"] - p["blade_inner"])
            * p["thickness"] * p["blade_height"]
        )
        shaft_h = tank.shape[2] - 1 - p["z_centre"]
        shaft = np.pi * p["shaft_radius"] ** 2 * shaft_h
        analytic = disc + hub + blades + shaft
        assert mask.sum() == pytest.approx(analytic, rel=0.35)

    def test_wall_velocity_is_rigid_rotation(self, tank):
        """u = omega x r: azimuthal, proportional to radius, zero z-component, and
        exactly zero off the impeller."""
        mask = tank.impeller_mask(0.0)
        vel = tank.impeller_wall_velocity(mask)
        assert np.all(vel[~mask] == 0.0)

        x, y, _ = np.mgrid[0:tank.shape[0], 0:tank.shape[1], 0:tank.shape[2]].astype(float)
        x -= (tank.shape[0] - 1) / 2
        y -= (tank.shape[1] - 1) / 2
        # Radial component of the wall velocity must vanish: rigid rotation moves
        # every point along its circle, never across it.
        radial = vel[..., 0] * x + vel[..., 1] * y
        assert np.abs(radial[mask]).max() < 1e-12
        speed = np.hypot(vel[..., 0], vel[..., 1])
        r = np.hypot(x, y)
        on = mask & (r > 0)
        assert np.allclose(speed[on], tank.omega_shaft * r[on], rtol=1e-12)

    def test_tip_speed_matches_the_scales(self, scales, tank):
        """The fastest wall node must move at u_tip to within half a cell of radius --
        the staircase places the outermost node just inside the true tip."""
        mask = tank.impeller_mask(0.0)
        vel = tank.impeller_wall_velocity(mask)
        fastest = np.hypot(vel[..., 0], vel[..., 1]).max()
        expected = scales.omega_shaft * scales.impeller_radius
        assert fastest == pytest.approx(expected, abs=tank.omega_shaft * 0.75)
        assert expected == pytest.approx(scales.u_tip, rel=1e-12)


class TestCouette:
    """Fast pieces of the Taylor-Couette validation; the full ladder lives in
    scripts/validate_couette.py. These pin the analytic forms and the properties of
    the momentum-exchange sum that do not need a converged flow."""

    def test_analytic_profile_satisfies_both_boundary_conditions(self):
        from lms.validation.couette import analytic_profile

        assert analytic_profile(np.array([7.5]), 7.5, 15.0, 0.004)[0] == pytest.approx(
            0.03, rel=1e-12
        )
        assert analytic_profile(np.array([15.0]), 7.5, 15.0, 0.004)[0] == pytest.approx(
            0.0, abs=1e-15
        )

    def test_analytic_torque_resists_the_rotation(self):
        from lms.validation.couette import analytic_torque_per_length

        assert analytic_torque_per_length(0.02, 0.004, 7.5, 15.0) < 0.0

    def test_measured_torque_matches_the_closed_form(self):
        """One small converged annulus, end to end: geometry, moving wall, momentum
        exchange, sign and magnitude. The resolution ladder refines this; the test
        guards it."""
        from lms.validation.couette import run_case

        r = run_case(12.0, u_wall=0.03, prefer_gpu=False, max_steps=40_000,
                     torque_samples=50)
        assert r.taylor < 1708, "test case must stay laminar to compare with the closed form"
        assert r.torque_measured < 0.0
        assert r.torque_error < 0.05
        assert r.profile_error < 0.05

    def test_equilibrium_fluid_exerts_no_force(self):
        """At rest every population is exactly its weight, so the momentum-exchange sum
        must vanish identically -- including the restored 2 w_q shift terms. A sign or
        bookkeeping error here produces a spurious net force long before it produces a
        wrong torque."""
        from lms.lbm.solver3d import D3Q19Solver, init_backend

        init_backend(prefer_gpu=False, precision="fp64")
        n = 20
        solid = np.zeros((n, n, n), dtype=np.int32)
        x, y, _ = np.mgrid[0:n, 0:n, 0:n].astype(float)
        solid[np.hypot(x - (n - 1) / 2, y - (n - 1) / 2) <= 4.0] = 2
        s = D3Q19Solver((n, n, n), omega=1.6, solid=solid)
        for _ in range(5):
            s.step()
        assert np.abs(s.force()).max() < 1e-12
        assert np.abs(s.torque()).max() < 1e-11
