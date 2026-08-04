"""D3Q19 lattice tests.

These mirror the D2Q9 lattice tests in `test_lbm2d.py`. A wrong weight or a mispaired
opposite index does not crash -- it produces a solver that runs happily and is quietly
wrong about viscosity or wall position, which is far more expensive to find later.
"""

import numpy as np
import pytest

from lms.lbm import d3q19
from lms.lbm.d2q9 import trt_magic_omega, viscosity_to_omega


class TestLattice:
    def test_weights_sum_to_one(self):
        assert d3q19.W.sum() == pytest.approx(1.0, abs=1e-15)

    def test_weight_groups_are_the_standard_d3q19_set(self):
        assert d3q19.W[0] == pytest.approx(1 / 3)
        assert np.allclose(d3q19.W[1:7], 1 / 18)
        assert np.allclose(d3q19.W[7:19], 1 / 36)

    def test_velocity_set_has_the_right_shells(self):
        """One rest vector, six at distance 1, twelve at distance sqrt(2). The eight
        corner directions of D3Q27 must be absent."""
        speeds = d3q19.EX**2 + d3q19.EY**2 + d3q19.EZ**2
        assert (speeds == 0).sum() == 1
        assert (speeds == 1).sum() == 6
        assert (speeds == 2).sum() == 12
        assert (speeds == 3).sum() == 0

    def test_velocities_are_distinct(self):
        vecs = {tuple(v) for v in zip(d3q19.EX, d3q19.EY, d3q19.EZ)}
        assert len(vecs) == d3q19.Q

    def test_first_moment_vanishes(self):
        """Sum of w_i e_i must be zero or the lattice has a built-in drift."""
        for e in (d3q19.EX, d3q19.EY, d3q19.EZ):
            assert (d3q19.W * e).sum() == pytest.approx(0.0, abs=1e-15)

    def test_second_moment_is_exactly_one_third(self):
        """This is what fixes the speed of sound, and through it the viscosity. It has
        to be exact, not approximately right."""
        for e in (d3q19.EX, d3q19.EY, d3q19.EZ):
            assert (d3q19.W * e * e).sum() == pytest.approx(d3q19.CS2, abs=1e-15)

    def test_second_moment_is_isotropic(self):
        """Cross terms must vanish, or the lattice is anisotropic and the recovered
        stress tensor picks up a preferred direction."""
        for a, b in ((d3q19.EX, d3q19.EY), (d3q19.EX, d3q19.EZ), (d3q19.EY, d3q19.EZ)):
            assert (d3q19.W * a * b).sum() == pytest.approx(0.0, abs=1e-15)

    def test_fourth_moment_isotropy(self):
        """The Navier-Stokes limit needs sum w e_a e_b e_c e_d isotropic. For the
        diagonal-in-pairs component that means <e_x^2 e_y^2> = cs^4 = 1/9."""
        assert (d3q19.W * d3q19.EX**2 * d3q19.EY**2).sum() == pytest.approx(1 / 9, abs=1e-15)
        assert (d3q19.W * d3q19.EX**4).sum() == pytest.approx(1 / 3, abs=1e-15)

    def test_opposite_is_an_involution(self):
        assert np.array_equal(d3q19.OPPOSITE[d3q19.OPPOSITE], np.arange(d3q19.Q))

    def test_opposite_actually_negates_the_velocity(self):
        """Bounce-back is only bounce-back if OPPOSITE[i] really is -e_i."""
        for e in (d3q19.EX, d3q19.EY, d3q19.EZ):
            assert np.array_equal(e[d3q19.OPPOSITE], -e)

    def test_opposite_preserves_weight(self):
        assert np.allclose(d3q19.W[d3q19.OPPOSITE], d3q19.W)


class TestEquilibrium:
    def test_recovers_density_and_velocity(self):
        rng = np.random.default_rng(0)
        shape = (4, 5, 6)
        rho = 1.0 + 0.01 * rng.standard_normal(shape)
        ux = 0.02 * rng.standard_normal(shape)
        uy = 0.02 * rng.standard_normal(shape)
        uz = 0.02 * rng.standard_normal(shape)

        feq = d3q19.equilibrium(rho, ux, uy, uz)
        r, a, b, c = d3q19.macroscopic(feq)

        assert np.allclose(r, rho, atol=1e-14)
        # Second-order equilibrium recovers momentum exactly; the error is O(u^3)
        # and shows up only in the third moment.
        assert np.allclose(a, ux, atol=1e-14)
        assert np.allclose(b, uy, atol=1e-14)
        assert np.allclose(c, uz, atol=1e-14)

    def test_equilibrium_at_rest_is_the_weights(self):
        shape = (3, 3, 3)
        feq = d3q19.equilibrium(np.ones(shape), *(np.zeros(shape),) * 3)
        for i in range(d3q19.Q):
            assert np.allclose(feq[i], d3q19.W[i])

    def test_momentum_flux_is_the_euler_stress(self):
        """Sum_i f_eq e_a e_b must equal rho*cs^2*delta_ab + rho*u_a*u_b. This is the
        tensor the Chapman-Enskog expansion needs; getting it wrong gives the wrong
        pressure or the wrong advection."""
        shape = (3, 3, 3)
        rho = np.full(shape, 1.02)
        ux, uy, uz = (np.full(shape, v) for v in (0.03, -0.02, 0.01))
        feq = d3q19.equilibrium(rho, ux, uy, uz)

        exp = {"x": (d3q19.EX, ux), "y": (d3q19.EY, uy), "z": (d3q19.EZ, uz)}
        for a, (ea, ua) in exp.items():
            for b, (eb, ub) in exp.items():
                got = np.tensordot(ea * eb, feq, axes=(0, 0))
                want = rho * ua * ub + (rho * d3q19.CS2 if a == b else 0.0)
                assert np.allclose(got, want, atol=1e-14), f"component {a}{b}"

    def test_is_galilean_covariant_in_the_low_mach_limit(self):
        """Shifting the frame by a small uniform velocity must shift the recovered
        velocity by the same amount."""
        shape = (4, 4, 4)
        rho = np.ones(shape)
        zero = np.zeros(shape)
        shift = 0.01
        feq = d3q19.equilibrium(rho, zero + shift, zero, zero)
        _, ux, uy, uz = d3q19.macroscopic(feq)
        assert np.allclose(ux, shift, atol=1e-14)
        assert np.allclose(uy, 0.0, atol=1e-14)
        assert np.allclose(uz, 0.0, atol=1e-14)


class TestRelaxationHelpersAreDimensionIndependent:
    """d3q19 deliberately does not define its own; these are properties of the
    relaxation rates, not the lattice."""

    def test_viscosity_round_trips(self):
        for nu in (1e-4, 0.01, 0.1):
            omega = viscosity_to_omega(nu)
            assert (1.0 / omega - 0.5) / 3.0 == pytest.approx(nu, rel=1e-12)

    def test_magic_parameter_holds_at_quarter(self):
        for nu in (1e-3, 0.05):
            wp = viscosity_to_omega(nu)
            wm = trt_magic_omega(wp)
            lam = (1.0 / wp - 0.5) * (1.0 / wm - 0.5)
            assert lam == pytest.approx(0.25, rel=1e-12)


# --------------------------------------------------------------------------- GPU solver

taichi = pytest.importorskip("taichi", reason="3D solver needs the gpu extra")


class TestSolver3D:
    """The kernel is checked against the Phase 0 2D solver, which was itself verified
    against Ghia and an analytic decay. That makes these comparisons against a known
    answer rather than against another implementation of the same guess.

    Kept small: Taichi has to compile kernels on first call, which dominates runtime.
    """

    N, NZ, RE, STEPS = 24, 4, 100.0, 1200

    def _oracle(self):
        from lms.lbm.solver2d import lid_driven_cavity

        s = lid_driven_cavity(n=self.N, reynolds=self.RE, lid_velocity=0.1, collision="trt")
        return s.run(max_steps=self.STEPS, tol=0.0, check_every=self.STEPS)

    def _run3d(self, prefer_gpu, precision, periodic_z=True):
        from lms.lbm.solver3d import init_backend, lid_driven_cavity_3d

        init_backend(prefer_gpu=prefer_gpu, precision=precision)
        s = lid_driven_cavity_3d(
            n=self.N, nz=self.NZ, reynolds=self.RE, lid_velocity=0.1,
            collision="trt", periodic_z=periodic_z,
        )
        return s.run(max_steps=self.STEPS, tol=0.0, check_every=self.STEPS)

    def test_fp64_reproduces_the_2d_oracle_to_machine_precision(self):
        """The port test. In fp64 the 3D kernel is running the same algorithm on the
        same data, so anything above rounding noise is a bug in the kernel."""
        ref = self._oracle()
        state = self._run3d(prefer_gpu=False, precision="fp64")
        # 2D is (y, x); 3D is (x, y, z).
        assert np.abs(state.ux[:, :, 0].T - ref.ux).max() < 1e-12
        assert np.abs(state.uy[:, :, 0].T - ref.uy).max() < 1e-12

    def test_a_z_uniform_problem_stays_z_uniform(self):
        """Nothing drives motion along z, so every z-plane must be identical and u_z
        must stay at zero. A streaming bug in the z direction shows up here and
        nowhere else."""
        state = self._run3d(prefer_gpu=False, precision="fp64")
        assert np.ptp(state.ux, axis=2).max() < 1e-14
        assert np.abs(state.uz).max() < 1e-14

    def test_mass_is_conserved(self):
        """Checked on the populations, not on `rho`. `rho` reports 1.0 at solid nodes
        by design -- they hold bounce-back state, not a physical distribution -- so its
        mean is a display quantity, not the conserved one."""
        from lms.lbm.solver3d import init_backend, lid_driven_cavity_3d

        init_backend(prefer_gpu=False, precision="fp64")
        s = lid_driven_cavity_3d(n=self.N, nz=self.NZ, reynolds=self.RE, lid_velocity=0.1)
        before = s.total_mass()
        s.run(max_steps=self.STEPS, tol=0.0, check_every=self.STEPS)
        assert s.total_mass() == pytest.approx(before, rel=1e-12)

    def test_fluid_density_matches_the_2d_oracle(self):
        """Density is the pressure field, and getting velocity right while getting
        pressure wrong is a real failure mode -- so it is checked separately."""
        ref = self._oracle()
        state = self._run3d(prefer_gpu=False, precision="fp64")
        fluid = ~np.array(
            [[(i in (0, self.N - 1)) or (j in (0, self.N - 1))
              for i in range(self.N)] for j in range(self.N)]
        )
        assert np.abs(state.rho[:, :, 0].T - ref.rho)[fluid].max() < 1e-12

    def test_end_walls_break_the_two_dimensional_solution(self):
        """With periodic_z the cavity is 2D in disguise. Close the z ends and the flow
        must stop being z-uniform -- that difference is the first thing this codebase
        can say that 2D could not."""
        state = self._run3d(prefer_gpu=False, precision="fp64", periodic_z=False)
        assert np.ptp(state.ux, axis=2).max() > 1e-3
        assert np.abs(state.uz).max() > 1e-5

    def test_rejects_fp64_on_metal_with_a_useful_message(self):
        """Metal has no float64. Failing early with an explanation beats failing deep
        inside Taichi's SPIR-V builder."""
        from lms.lbm.solver3d import init_backend

        with pytest.raises(ValueError, match="does not support float64"):
            init_backend(prefer_gpu=True, precision="fp64")

    def test_lid_velocity_guard_matches_the_2d_solver(self):
        from lms.lbm.solver3d import init_backend, lid_driven_cavity_3d

        init_backend(prefer_gpu=False, precision="fp64")
        with pytest.raises(ValueError, match="compressible"):
            lid_driven_cavity_3d(n=16, nz=4, lid_velocity=0.5)


class TestBeltrami:
    """The exact-solution checks. These verify the *analytic* field's defining
    properties first, because a convergence study against a wrong reference solution
    measures nothing and looks entirely healthy while doing it."""

    N, U0, NU = 16, 0.05, 0.01

    @staticmethod
    def _dx(f, axis):
        """Periodic central difference. np.gradient uses one-sided stencils at the
        array edges, which are simply wrong for a periodic field."""
        return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / 2.0

    def test_analytic_field_is_divergence_free(self):
        from lms.validation.beltrami import analytic

        ux, uy, uz, _ = analytic(self.N, self.U0, self.NU, 0.0)
        div = self._dx(ux, 0) + self._dx(uy, 1) + self._dx(uz, 2)
        # Each component is independent of its own coordinate, so this is not merely
        # small -- it is identically zero.
        assert np.abs(div).max() == 0.0

    def test_analytic_field_is_beltrami(self):
        """curl(u) parallel to u is what kills the nonlinear term and makes the flow an
        exact solution. The discrete central difference of sin(kx) carries sin(k) rather
        than k, so the comparison uses sin(k)."""
        from lms.validation.beltrami import analytic

        ux, uy, uz, _ = analytic(self.N, self.U0, self.NU, 0.0)
        k_discrete = np.sin(2.0 * np.pi / self.N)
        cx = self._dx(uz, 1) - self._dx(uy, 2)
        cy = self._dx(ux, 2) - self._dx(uz, 0)
        cz = self._dx(uy, 0) - self._dx(ux, 1)
        for got, want in ((cx, ux), (cy, uy), (cz, uz)):
            assert np.abs(got - k_discrete * want).max() < 1e-15

    def test_nonlinear_term_is_a_pure_gradient(self):
        """u x curl(u) = 0 is the reason the Navier-Stokes equations collapse to a heat
        equation here. If this were not zero the 'exact' solution would not be exact."""
        from lms.validation.beltrami import analytic

        ux, uy, uz, _ = analytic(self.N, self.U0, self.NU, 0.0)
        cx = self._dx(uz, 1) - self._dx(uy, 2)
        cy = self._dx(ux, 2) - self._dx(uz, 0)
        cz = self._dx(uy, 0) - self._dx(ux, 1)
        cross = (uy * cz - uz * cy, uz * cx - ux * cz, ux * cy - uy * cx)
        scale = float(np.abs(ux).max()) ** 2
        assert max(np.abs(c).max() for c in cross) / scale < 1e-14

    def test_field_decays_at_the_viscous_rate(self):
        from lms.validation.beltrami import analytic

        k = 2.0 * np.pi / self.N
        t = 500.0
        a, *_ = analytic(self.N, self.U0, self.NU, 0.0)
        b, *_ = analytic(self.N, self.U0, self.NU, t)
        expected = np.exp(-self.NU * k * k * t)
        assert np.abs(b).max() / np.abs(a).max() == pytest.approx(expected, rel=1e-12)

    def test_analytic_strain_matches_finite_differences(self):
        """The closed-form gradient feeds the non-equilibrium initial state. If it were
        wrong the initialisation would be wrong in a way nothing else would catch."""
        from lms.validation.beltrami import analytic, analytic_strain

        n = 32
        ux, uy, uz, _ = analytic(n, self.U0, self.NU, 0.0)
        grad = analytic_strain(n, self.U0, self.NU, 0.0)
        k = 2.0 * np.pi / n
        # Central differencing scales the exact derivative by sin(k)/k.
        factor = np.sin(k) / k
        for b, comp in enumerate((ux, uy, uz)):
            for a in range(3):
                assert np.abs(self._dx(comp, a) - factor * grad[a, b]).max() < 1e-15

    def test_nonequilibrium_carries_the_viscous_stress(self):
        """Sum_i e_ia e_ib f_i^(1) must equal -2 tau rho cs^2 S_ab. That identity is
        what makes the correction the viscous stress rather than an arbitrary tweak."""
        from lms.lbm import d3q19
        from lms.validation.beltrami import analytic, analytic_strain, nonequilibrium

        n, tau = 16, 0.8
        _, _, _, rho = analytic(n, self.U0, self.NU, 0.0)
        grad = analytic_strain(n, self.U0, self.NU, 0.0)
        strain = 0.5 * (grad + grad.transpose(1, 0, 2, 3, 4))
        fneq = nonequilibrium(rho, grad, tau)

        e = np.stack([d3q19.EX, d3q19.EY, d3q19.EZ]).astype(float)
        # Relative to the size of the stress itself: the components are O(5e-3), so an
        # absolute bound would be measuring float64's exponent rather than the identity.
        scale = 2.0 * tau * d3q19.CS2 * np.abs(strain).max()
        for a in range(3):
            for b in range(3):
                got = np.tensordot(e[a] * e[b], fneq, axes=(0, 0))
                want = -2.0 * tau * rho * d3q19.CS2 * strain[a, b]
                assert np.abs(got - want).max() / scale < 1e-13, f"component {a}{b}"

    def test_nonequilibrium_start_beats_equilibrium_after_one_step(self):
        """The whole justification for computing f_neq at all. One step is enough,
        because this is an initialisation error, not an accumulated one."""
        from lms.validation.beltrami import run_case

        with_neq = run_case(24, self.U0, self.NU, steps=1)
        without = run_case(24, self.U0, self.NU, steps=1, equilibrium_only=True)
        assert with_neq.error < 0.2 * without.error

    def test_diffusive_refinement_holds_tau_and_reynolds_fixed(self):
        from lms.validation.beltrami import diffusive_ladder

        results = diffusive_ladder(sizes=(16, 24), u0=0.05, nu=0.01, steps0=60)
        assert results[0].nu == results[1].nu           # same tau
        assert results[0].reynolds == pytest.approx(results[1].reynolds, rel=1e-12)
        # Mach falls like 1/n, which is what drives the O(Ma^2) error down at 1/n^2.
        assert results[1].mach == pytest.approx(results[0].mach * 16 / 24, rel=1e-12)
