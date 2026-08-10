"""D3Q7 scalar lattice tests: moments, conservation, and the two closed forms."""

import numpy as np
import pytest

taichi = pytest.importorskip("taichi", reason="scalar solver needs the gpu extra")

from lms.lbm.scalar3d import CS2_7, EX7, EY7, EZ7, OPP7, W7, ScalarD3Q7
from lms.lbm.solver3d import init_backend


class TestLatticeD3Q7:
    def test_weights_sum_to_one(self):
        assert W7.sum() == pytest.approx(1.0, abs=1e-15)

    def test_second_moment_is_the_advertised_cs2(self):
        """cs^2 = 1/4 is baked into the equilibrium's 4 e.u factor and the
        diffusivity map D = (tau - 1/2)/4; all three must agree or the scalar
        diffuses at a rate other than the one requested."""
        for e in (EX7, EY7, EZ7):
            assert (W7 * e * e).sum() == pytest.approx(CS2_7, abs=1e-15)

    def test_opposite_negates_velocities(self):
        for e in (EX7, EY7, EZ7):
            assert np.array_equal(e[OPP7], -e)


class TestScalarPhysics:
    @staticmethod
    def _still_fluid(n, solid=None):
        init_backend(prefer_gpu=False, precision="fp64")
        import taichi as ti

        vel = ti.Vector.field(3, ti.f64, shape=(n, n, n))
        vel.fill(0)
        if solid is None:
            solid = np.zeros((n, n, n), dtype=bool)
        return ScalarD3Q7((n, n, n), d_molecular=0.02, solid=solid, vel=vel)

    def test_scalar_is_conserved_with_walls(self):
        """Bounce-back walls are zero-flux: total content constant to fp64 round-off.
        The CoV endpoint of the blend-time measurement is meaningless without this."""
        n = 20
        solid = np.zeros((n, n, n), dtype=bool)
        solid[0], solid[-1] = True, True
        solid[:, 0], solid[:, -1] = True, True
        solid[:, :, 0], solid[:, :, -1] = True, True
        s = self._still_fluid(n, solid)
        rng = np.random.default_rng(0)
        c0 = rng.random((n, n, n))
        c0[solid] = 0.0
        s.set_concentration(c0)
        before = s.total()
        for _ in range(200):
            s.step()
        assert s.total() == pytest.approx(before, rel=1e-12)

    def test_gaussian_diffuses_at_the_requested_rate(self):
        from lms.validation.scalar import run_diffusion

        r = run_diffusion(32, d=0.02, steps=200)
        assert r.variance_error < 0.01
        assert r.profile_error < 0.02

    def test_uniform_advection_translates_exactly(self):
        from lms.validation.scalar import run_advection

        r = run_advection(48, u=0.03, d=0.005, steps=300)
        assert r.centroid_error < 0.01
        # Numerical diffusion exists and must be small next to the physical D --
        # measured, because in the tank it competes with the *modelled* nu_t/Sc_t.
        assert r.d_numerical < 0.05 * r.d

    def test_concentration_stays_within_physical_bounds(self):
        """Advection-diffusion of c in [0, 1] must not manufacture overshoots beyond
        the mild lattice ringing; a sign error in the equilibrium blows this up."""
        s = self._still_fluid(24)
        c0 = np.zeros((24, 24, 24))
        c0[8:16, 8:16, 8:16] = 1.0
        s.set_concentration(c0)
        for _ in range(150):
            s.step()
        c = s.concentration()
        assert c.min() > -0.05
        assert c.max() < 1.05
