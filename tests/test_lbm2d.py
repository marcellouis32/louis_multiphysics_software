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
from lms.lbm.solver2d import D2Q9Solver, cavity_centerlines, lid_driven_cavity
from lms.validation import ghia


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
