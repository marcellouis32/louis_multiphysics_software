import numpy as np
import pytest

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
from lms.lbm.solver2d import (
    D2Q9Solver,
    cavity_centerlines,
    lid_driven_cavity,
    stream_function,
    vortex_centre,
)
from lms.lbm import solver2d, sweep
from lms.validation import ghia, taylor_green


class TestLattice:
    def test_weights_sum_to_one(self):
        assert W.sum() == pytest.approx(1.0)

    def test_velocity_set_is_symmetric(self):
        assert EX.sum() == 0 and EY.sum() == 0

    def test_opposite_is_an_involution(self):
        assert np.array_equal(OPPOSITE[OPPOSITE], np.arange(Q))
        assert np.array_equal(EX[OPPOSITE], -EX)
        assert np.array_equal(EY[OPPOSITE], -EY)

    def test_second_moment_is_isotropic(self):
        """sum_i w_i e_ia e_ib = cs^2 delta_ab. Without this the recovered
        equations are not Navier-Stokes."""
        assert (W * EX * EX).sum() == pytest.approx(1 / 3)
        assert (W * EY * EY).sum() == pytest.approx(1 / 3)
        assert (W * EX * EY).sum() == pytest.approx(0.0, abs=1e-15)


class TestEquilibrium:
    def test_rest_state_recovers_weights(self):
        rho = np.ones((3, 3))
        feq = equilibrium(rho, np.zeros((3, 3)), np.zeros((3, 3)))
        for i in range(Q):
            assert feq[i] == pytest.approx(W[i])

    def test_moments_match_inputs(self):
        rng = np.random.default_rng(0)
        rho = 1.0 + 0.01 * rng.standard_normal((8, 8))
        ux = 0.02 * rng.standard_normal((8, 8))
        uy = 0.02 * rng.standard_normal((8, 8))
        r, x, y = macroscopic(equilibrium(rho, ux, uy))
        assert r == pytest.approx(rho, abs=1e-12)
        assert x == pytest.approx(ux, abs=1e-12)
        assert y == pytest.approx(uy, abs=1e-12)


class TestRelaxation:
    def test_viscosity_roundtrip(self):
        for nu in (0.001, 0.01, 0.1):
            omega = viscosity_to_omega(nu)
            assert (1.0 / omega - 0.5) / 3.0 == pytest.approx(nu)

    def test_trt_magic_parameter_holds(self):
        for omega_plus in (0.6, 1.0, 1.7, 1.9):
            omega_minus = trt_magic_omega(omega_plus, magic=0.25)
            product = (1 / omega_plus - 0.5) * (1 / omega_minus - 0.5)
            assert product == pytest.approx(0.25)


class TestSolver:
    def test_quiescent_fluid_stays_quiescent(self):
        solid = np.zeros((16, 16), dtype=bool)
        solver = D2Q9Solver((16, 16), omega=1.0, solid=solid)
        for _ in range(50):
            solver.step()
        assert np.abs(solver.ux).max() < 1e-14
        assert solver.rho == pytest.approx(np.ones((16, 16)), abs=1e-13)

    def test_mass_is_conserved(self):
        solver = lid_driven_cavity(n=32, reynolds=100.0)
        initial = solver.f.sum()
        for _ in range(300):
            solver.step()
        assert solver.f.sum() == pytest.approx(initial, rel=1e-11)

    def test_lid_drives_flow_in_positive_x(self):
        solver = lid_driven_cavity(n=48, reynolds=100.0)
        solver.run(max_steps=2000, tol=0.0, check_every=2000)
        top_half = solver.ux[-8:-1, 1:-1]
        assert top_half.mean() > 0

    def test_rejects_compressible_lid_velocity(self):
        with pytest.raises(ValueError, match="compressible regime"):
            lid_driven_cavity(n=32, lid_velocity=0.4)

    @pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
    def test_diverging_run_reports_actionable_error(self):
        """Kelvin-Helmholtz at near-zero viscosity and high lattice Mach is the
        classic BGK blow-up. The transverse perturbation matters: without it the
        shear layer is one-dimensional, the nonlinear term vanishes, and the case
        is linearly stable no matter how thin the viscosity.

        The solver should say why it failed, not emit a wall of NaN warnings.
        """
        n = 32
        solid = np.zeros((n, n), dtype=bool)
        solver = D2Q9Solver((n, n), omega=1.9999, solid=solid, collision="bgk")
        band = (np.arange(n) // 4) % 2 == 0
        solver.ux[:] = np.where(band, 0.4, -0.4)[:, None]
        solver.uy[:] = 0.05 * np.sin(2 * np.pi * np.arange(n) / n)[None, :]
        solver.f = equilibrium(solver.rho, solver.ux, solver.uy)
        with pytest.raises(FloatingPointError, match="lattice velocity"):
            solver.run(max_steps=20_000, check_every=100)


class TestStreamFunction:
    @staticmethod
    def analytic(n: int):
        """psi = sin(pi x) sin(pi y) on a unit square, expressed in index space,
        with the velocity field derived from it exactly."""
        j, i = np.mgrid[0:n, 0:n].astype(float)
        k = np.pi / (n - 1)
        psi = np.sin(k * i) * np.sin(k * j)
        ux = k * np.sin(k * i) * np.cos(k * j)
        uy = -k * np.cos(k * i) * np.sin(k * j)
        return psi, ux, uy

    def test_recovers_analytic_stream_function(self):
        psi_exact, ux, uy = self.analytic(64)
        psi = stream_function(ux, uy)
        assert np.abs(psi - psi_exact).max() < 2e-3

    def test_is_second_order_accurate(self):
        errors = []
        for n in (32, 64, 128):
            psi_exact, ux, uy = self.analytic(n)
            errors.append(np.abs(stream_function(ux, uy) - psi_exact).max())
        order = np.log2(errors[0] / errors[1])
        assert 1.7 < order < 2.3, f"trapezoid integration should be 2nd order, got {order:.2f}"

    def test_constant_flow_gives_linear_stream_function(self):
        n = 16
        ux = np.ones((n, n))
        uy = np.zeros((n, n))
        psi = stream_function(ux, uy)
        assert psi[:, 0] == pytest.approx(np.arange(n, dtype=float))

    def test_vortex_centre_finds_the_extremum(self):
        n = 32
        j, i = np.mgrid[0:n, 0:n].astype(float)
        psi = (i - 8.0) ** 2 + (j - 20.0) ** 2
        x, y, value = vortex_centre(psi)
        assert x == pytest.approx((8 + 0.5) / n)
        assert y == pytest.approx((20 + 0.5) / n)
        assert value == pytest.approx(0.0)


class TestTaylorGreen:
    """Order of accuracy against a closed-form solution.

    The cavity can only ever be checked against Ghia's 1982 129^2 result, which has
    its own error bar. Taylor-Green is exact, so these are the tests that would catch
    a subtly wrong collision operator -- one that still converges, still conserves
    mass, still matches Ghia to 1%, but at first order instead of second.
    """

    SIZES = (16, 32, 64)

    def test_analytic_field_is_divergence_free(self):
        """Periodic central differences, not np.gradient: the one-sided stencils
        np.gradient uses at the array edge are wrong for a periodic field and would
        report a spurious divergence there."""
        n = 64
        ux, uy, _ = taylor_green.analytic(n, u0=0.05, nu=0.01, t=0.0)
        dux_dx = 0.5 * (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1))
        duy_dy = 0.5 * (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0))
        assert np.abs(dux_dx + duy_dy).max() < 1e-15

    def test_analytic_field_decays_at_the_viscous_rate(self):
        n, nu = 64, 0.01
        k = 2 * np.pi / n
        t = 500.0
        ux0, _, _ = taylor_green.analytic(n, u0=0.05, nu=nu, t=0.0)
        uxt, _, _ = taylor_green.analytic(n, u0=0.05, nu=nu, t=t)
        assert np.abs(uxt).max() / np.abs(ux0).max() == pytest.approx(
            np.exp(-2 * nu * k * k * t)
        )

    def test_diffusive_scaling_is_second_order(self):
        results = taylor_green.diffusive_ladder(self.SIZES, steps0=150)
        order = taylor_green.observed_order(results)
        assert 1.8 < order < 2.4, f"expected second order under diffusive scaling, got {order:.2f}"

    def test_diffusive_scaling_holds_tau_and_reynolds_fixed(self):
        """The property that makes the ladder valid: only the mesh changes."""
        results = taylor_green.diffusive_ladder(self.SIZES, steps0=50)
        assert len({round(r.nu, 12) for r in results}) == 1
        assert all(r.reynolds == pytest.approx(results[0].reynolds) for r in results)
        assert results[-1].mach < results[0].mach / 3

    def test_acoustic_scaling_stalls_on_the_mach_error_floor(self):
        """Documents the trap rather than the solver.

        Holding lattice velocity fixed leaves the O(Ma^2) compressibility error
        constant under refinement. Past the point where discretisation error drops
        below it, refining stops helping -- and on a fine enough grid it actively
        hurts, because tau grows with n at fixed Re. This is why the cavity study
        measured order ~1.1 with a solver that is genuinely second order.

        Run at an elevated Mach (u0 = 0.12) so the floor is reachable on grids small
        enough to keep this in the fast suite. At u0 = 0.05 the same collapse happens,
        it just takes until n = 128.
        """
        results = taylor_green.acoustic_ladder(self.SIZES, u0=0.12, steps0=150)
        assert len({round(r.mach, 12) for r in results}) == 1
        order = taylor_green.observed_order(results)
        assert order < 1.8, f"acoustic scaling should not reach second order, got {order:.2f}"
        assert results[-1].error > results[-2].error


class TestGhiaData:
    def test_endpoints_are_walls_and_lid(self):
        y, u = ghia.u_profile(1000)
        assert u[0] == pytest.approx(0.0)
        assert u[-1] == pytest.approx(1.0)
        x, v = ghia.v_profile(1000)
        assert v[0] == pytest.approx(0.0) and v[-1] == pytest.approx(0.0)

    def test_primary_vortex_sign_structure(self):
        """The Re=1000 primary vortex drives backflow in the lower half."""
        y, u = ghia.u_profile(1000)
        assert u[(y > 0.05) & (y < 0.4)].max() < 0

    def test_unknown_reynolds_rejected(self):
        with pytest.raises(KeyError, match="Ghia data available"):
            ghia.u_profile(2500)


@pytest.mark.slow
def test_cavity_matches_ghia_re1000():
    solver = lid_driven_cavity(n=192, reynolds=1000.0, lid_velocity=0.1)
    state = solver.run(max_steps=180_000, tol=2e-7, check_every=1000)

    y, u, x, v = cavity_centerlines(state, solver.lid_velocity)
    gy, gu = ghia.u_profile(1000)
    gx, gv = ghia.v_profile(1000)

    assert ghia.rms_error(y, u, gy, gu) < 0.025
    assert ghia.rms_error(x, v, gx, gv) < 0.025


class TestReynoldsSweep:
    """Parameter continuation. The sweep is the basis of the animated deliverable, so
    the ordering guarantee and the warm-start payoff are both asserted rather than
    assumed."""

    def test_frames_come_back_in_the_requested_order(self):
        values = [100.0, 160.0, 250.0]
        frames = list(
            sweep.reynolds_sweep(n=24, reynolds_values=values, tol=1e-4, max_steps=20_000)
        )
        assert [f.reynolds for f in frames] == values
        for f in frames:
            assert f.ux.shape == f.uy.shape == (24, 24)
            assert np.isfinite(f.ux).all() and np.isfinite(f.uy).all()
            assert f.tau > 0.5
            assert f.steps > 0

    def test_warm_start_converges_in_fewer_steps_than_starting_from_rest(self):
        """The entire justification for continuation, made executable. If this ever
        fails, the warm start has stopped helping and the extra machinery is dead
        weight."""
        values = [100.0, 160.0]
        kwargs = {"n": 24, "reynolds_values": values, "tol": 3e-5, "max_steps": 40_000}
        warm = list(sweep.reynolds_sweep(**kwargs))
        cold = list(sweep.reynolds_sweep(**kwargs, cold_start=True))

        # First case starts from rest either way, so it is the control.
        assert warm[0].steps == cold[0].steps
        assert warm[1].steps < cold[1].steps
        assert all(f.converged for f in warm + cold)

    def test_warm_start_discards_the_nonphysical_density_on_solid_nodes(self):
        """Solid nodes hold bounce-back populations, so `macroscopic` reports wild
        densities there. Copying them into the new solver's equilibrium seeds a wall
        of garbage that streams into the fluid and diverges the solve."""
        solver = solver2d.lid_driven_cavity(n=16, reynolds=100.0, lid_velocity=0.1)
        wild_rho = np.full((16, 16), 1.0)
        wild_rho[solver.solid] = -50.0
        sweep.warm_start(solver, np.zeros((16, 16)), np.zeros((16, 16)), wild_rho)

        assert np.all(solver.rho[solver.solid] == 1.0)
        assert np.isfinite(solver.f).all()
        assert solver.f.min() > 0.0

    def test_sweep_values_pin_the_ghia_benchmarks_exactly(self):
        values = sweep.sweep_values(lowest=100.0, highest=3200.0, count=25)
        for anchor in (100.0, 400.0, 1000.0):
            assert np.isclose(values, anchor).any()
        assert (np.diff(values) > 0).all()
        assert values[0] >= 100.0 and values[-1] <= 3200.0
        # Log spacing: the ratio between neighbours should be roughly constant, unlike
        # a linear ladder which would crawl at low Re and leap at high Re.
        ratios = values[1:] / values[:-1]
        assert ratios.max() / ratios.min() < 2.0
