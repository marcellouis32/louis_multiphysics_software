import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from lms.viz.fields2d import _equalize, _upsample_mask, line_integral_convolution
from lms.viz.style import house_style, signed_asinh_norm, symmetric_norm


def solid_body_rotation(n=24):
    y, x = np.mgrid[0:n, 0:n].astype(float)
    cy = cx = (n - 1) / 2
    return -(y - cy), (x - cx)


class TestEqualize:
    def test_output_is_uniform_on_unit_interval(self):
        rng = np.random.default_rng(0)
        out = _equalize(rng.standard_normal((40, 40)))
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)
        # A uniform distribution has its quartiles at 0.25 / 0.5 / 0.75.
        assert np.percentile(out, 50) == pytest.approx(0.5, abs=0.02)
        assert np.percentile(out, 25) == pytest.approx(0.25, abs=0.02)

    def test_preserves_ordering(self):
        data = np.array([[3.0, 1.0], [4.0, 2.0]])
        out = _equalize(data)
        assert np.array_equal(np.argsort(out.ravel()), np.argsort(data.ravel()))


class TestLIC:
    def test_shape_follows_upsample_factor(self):
        ux, uy = solid_body_rotation(16)
        tex = line_integral_convolution(ux, uy, n_steps=8, upsample=3)
        assert tex.shape == (48, 48)

    def test_output_is_bounded(self):
        ux, uy = solid_body_rotation(16)
        tex = line_integral_convolution(ux, uy, n_steps=8, upsample=2)
        assert tex.min() >= 0.0 and tex.max() <= 1.0

    def test_is_deterministic_for_a_given_seed(self):
        ux, uy = solid_body_rotation(16)
        a = line_integral_convolution(ux, uy, n_steps=8, upsample=2, seed=7)
        b = line_integral_convolution(ux, uy, n_steps=8, upsample=2, seed=7)
        assert np.array_equal(a, b)

    def test_zero_field_does_not_blow_up(self):
        zeros = np.zeros((12, 12))
        tex = line_integral_convolution(zeros, zeros, n_steps=6, upsample=2)
        assert np.isfinite(tex).all()

    def test_texture_is_smoother_along_the_flow_than_across_it(self):
        """The point of LIC: correlation length should be much longer parallel to
        the streamlines than perpendicular to them."""
        n = 64
        ux = np.ones((n, n))
        uy = np.zeros((n, n))
        tex = line_integral_convolution(ux, uy, n_steps=30, step_size=1.0, upsample=1)
        along = np.mean(np.abs(np.diff(tex, axis=1)))
        across = np.mean(np.abs(np.diff(tex, axis=0)))
        assert along < 0.5 * across


class TestNorms:
    def test_symmetric_norm_is_centred_on_zero(self):
        norm = symmetric_norm(np.array([-3.0, 1.0, 2.0, 8.0]))
        assert norm.vmin == pytest.approx(-norm.vmax)
        assert norm(0.0) == pytest.approx(0.5)

    def test_asinh_norm_is_centred_and_ordered(self):
        data = np.concatenate([np.linspace(-0.01, 0.01, 500), [5.0, -5.0]])
        norm = signed_asinh_norm(data)
        assert norm(0.0) == pytest.approx(0.5, abs=1e-6)
        assert norm(0.005) > norm(0.001) > norm(0.0)

    def test_asinh_norm_ignores_masked_cells_when_scaling(self):
        data = np.array([0.1, 0.2, 0.15, 900.0])
        mask = np.array([False, False, False, True])
        assert signed_asinh_norm(data, mask=mask).vmax < 1.0

    def test_narrow_linear_region_lifts_a_weak_uniform_core_off_the_neutral_colour(self):
        """The lid-driven cavity failure mode, reduced to its essentials.

        A field where most cells sit at |x| ~ 2 but a thin boundary layer reaches 80.
        With a wide linear region the core lands almost on the dark centre of a
        diverging colormap and disappears. Narrowing it must push the core out far
        enough to take on colour.
        """
        # The interior needs a spread, not a single value: percentiles of a delta
        # are all equal, so the knob would have nothing to grip.
        interior = -np.linspace(0.05, 4.0, 4000)
        layer = np.linspace(20.0, 80.0, 200)
        data = np.concatenate([interior, layer])

        wide = signed_asinh_norm(data, percentile=99.5, linear_percentile=60.0)
        narrow = signed_asinh_norm(data, percentile=97.0, linear_percentile=20.0)

        def offset(norm):
            """Distance of the core from the neutral colour, as a fraction of the
            half-range. Below ~0.2 a dark-centred colormap shows nothing."""
            return abs(norm(-2.0) - 0.5) / 0.5

        assert offset(narrow) > 1.5 * offset(wide)
        assert offset(narrow) > 0.3
        assert narrow.vmax < wide.vmax


class TestMask:
    def test_upsample_preserves_coverage_fraction(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, :] = True
        up = _upsample_mask(mask, (30, 30))
        assert up.shape == (30, 30)
        assert up.mean() == pytest.approx(mask.mean(), abs=1e-9)


def test_house_style_restores_global_rc():
    before = matplotlib.rcParams["axes.facecolor"]
    with house_style():
        pass
    assert matplotlib.rcParams["axes.facecolor"] == before
