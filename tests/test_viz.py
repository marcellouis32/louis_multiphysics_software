import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.colors import PowerNorm

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


class TestContourAnimation:
    """The GIF is a deliverable, so these assert the properties that would silently
    ruin one: wrong frame count, and a colour scale that moves between frames."""

    @staticmethod
    def _frames(count=3, n=16):
        y, x = np.mgrid[0:n, 0:n].astype(float) / (n - 1)
        # A blob that migrates across the box, so frames are genuinely different.
        return [np.exp(-40 * ((x - 0.2 - 0.3 * k) ** 2 + (y - 0.5) ** 2)) for k in range(count)]

    def test_writes_a_gif_with_one_frame_per_field(self, tmp_path):
        from PIL import Image

        from lms.viz.animate import contour_animation

        fields = self._frames(3)
        out = contour_animation(
            fields,
            reynolds=[100.0, 400.0, 1000.0],
            label="test",
            norm=PowerNorm(gamma=1.0, vmin=0.0, vmax=1.0),
            fps=4,
            dpi=40,
            figsize=(2.4, 2.2),
            save=tmp_path / "anim.gif",
        )
        assert out.exists()
        with Image.open(out) as img:
            assert img.format == "GIF"
            assert img.n_frames == len(fields)
            # Constant dimensions: a "tight" savefig bbox would vary these per frame
            # and produce a shivering or malformed GIF.
            sizes = set()
            for i in range(img.n_frames):
                img.seek(i)
                sizes.add(img.size)
            assert len(sizes) == 1

    def test_rejects_mismatched_frame_and_label_counts(self, tmp_path):
        from lms.viz.animate import contour_animation

        with pytest.raises(ValueError, match="Reynolds"):
            contour_animation(
                self._frames(3),
                reynolds=[100.0],
                label="test",
                norm=PowerNorm(gamma=1.0, vmin=0.0, vmax=1.0),
                save=tmp_path / "bad.gif",
            )

    def test_the_supplied_norm_is_not_mutated(self, tmp_path):
        """The flicker guarantee. If the animator recomputed or rescaled the norm per
        frame, identical data would take different colours as the sweep progressed."""
        from lms.viz.animate import contour_animation

        norm = PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0)
        before = (norm.vmin, norm.vmax, norm.gamma)
        contour_animation(
            self._frames(3),
            reynolds=[100.0, 400.0, 1000.0],
            label="test",
            norm=norm,
            dpi=40,
            figsize=(2.4, 2.2),
            save=tmp_path / "anim.gif",
        )
        assert (norm.vmin, norm.vmax, norm.gamma) == before

    def test_pooled_norm_spans_every_frame_not_just_one(self):
        """Vorticity grows with Re, so a norm fitted to the first frame alone would
        saturate the rest of the animation."""
        from lms.viz.animate import pooled_norm

        fields = [np.full((8, 8), 1.0), np.full((8, 8), 5.0)]
        norm = pooled_norm(fields, lambda pool: PowerNorm(1.0, 0.0, float(pool.max())))
        assert norm.vmax == pytest.approx(5.0)

    def test_pooled_norm_ignores_masked_solid_cells(self):
        from lms.viz.animate import pooled_norm

        solid = np.zeros((8, 8), dtype=bool)
        solid[0, :] = True
        fields = [np.ones((8, 8))]
        fields[0][0, :] = 99.0
        norm = pooled_norm(fields, lambda pool: PowerNorm(1.0, 0.0, float(pool.max())), solid=solid)
        assert norm.vmax == pytest.approx(1.0)


class TestSlices3D:
    """Slicing is where a 3D renderer lies most easily: an off-by-one axis or the wrong
    velocity pair produces a picture that looks entirely plausible and describes a flow
    that does not exist."""

    @staticmethod
    def _ramp():
        """A field whose value encodes its own coordinates, so a mis-sliced plane is
        detectable by inspection rather than by eye."""
        i, j, k = np.mgrid[0:6, 0:8, 0:10].astype(float)
        return 100 * i + 10 * j + k

    def test_slice_takes_the_requested_plane(self):
        from lms.viz.fields3d import slice_plane

        f = self._ramp()
        assert slice_plane(f, "x", 2)[0, 0] == 200.0
        assert slice_plane(f, "y", 3)[0, 0] == 30.0
        assert slice_plane(f, "z", 4)[0, 0] == 4.0

    def test_slice_defaults_to_the_centre_plane(self):
        from lms.viz.fields3d import slice_plane

        f = self._ramp()
        assert np.array_equal(slice_plane(f, "z"), slice_plane(f, "z", 5))

    def test_slice_is_transposed_for_drawing(self):
        """Shape (nx, ny, nz) sliced normal to z leaves (nx, ny), which must come back
        as (ny, nx) so matplotlib's row axis is the vertical one."""
        from lms.viz.fields3d import slice_plane

        assert slice_plane(self._ramp(), "z").shape == (8, 6)
        assert slice_plane(self._ramp(), "x").shape == (10, 8)

    def test_rejects_an_unknown_axis(self):
        from lms.viz.fields3d import slice_plane

        with pytest.raises(ValueError, match="axis must be one of"):
            slice_plane(self._ramp(), "w")

    def test_in_plane_components_pick_the_right_velocity_pair(self):
        """Drawing streamlines from the out-of-plane component is the classic 3D
        slicing error, and it is silent."""
        from lms.viz.fields3d import in_plane_components

        ux, uy, uz = (np.full((4, 5, 6), v) for v in (1.0, 2.0, 3.0))
        assert [c.flat[0] for c in in_plane_components(ux, uy, uz, "x")] == [2.0, 3.0]
        assert [c.flat[0] for c in in_plane_components(ux, uy, uz, "y")] == [1.0, 3.0]
        assert [c.flat[0] for c in in_plane_components(ux, uy, uz, "z")] == [1.0, 2.0]

    def test_orthogonal_slices_share_one_colour_scale(self):
        """Three independently normalised panels would invite the reader to compare
        colours that mean different things."""
        from lms.viz.fields3d import plot_orthogonal_slices

        norm = PowerNorm(gamma=1.0, vmin=0.0, vmax=1.0)
        fig = plot_orthogonal_slices(self._ramp() / 1000.0, label="t", norm=norm)
        assert (norm.vmin, norm.vmax) == (0.0, 1.0)   # not mutated
        assert len(fig.axes) >= 3

    def test_orthogonal_slices_rejects_a_2d_field(self):
        from lms.viz.fields3d import plot_orthogonal_slices

        with pytest.raises(ValueError, match="expected a 3D field"):
            plot_orthogonal_slices(np.zeros((4, 4)), label="t")

    def test_span_profiles_draws_reference_and_every_curve(self):
        from lms.viz.fields3d import plot_span_profiles

        coord = np.linspace(0, 1, 20)
        profiles = {f"span {s}": (np.sin(coord * s), coord) for s in (1, 2, 3)}
        fig = plot_span_profiles(profiles, reference=(np.cos(coord), coord))
        ax = fig.axes[0]
        assert len(ax.lines) >= len(profiles)
        assert len(ax.collections) >= 1          # the reference scatter


class TestAnimationOptions:
    @staticmethod
    def _frames(count=3, n=12):
        y, x = np.mgrid[0:n, 0:n].astype(float) / (n - 1)
        return [np.exp(-30 * ((x - 0.3 - 0.2 * k) ** 2 + (y - 0.5) ** 2)) for k in range(count)]

    def test_isolines_can_be_switched_off(self, tmp_path):
        """On a turbulent slice the contour tracer finds structure at every scale, so
        overlaid isolines are noise rather than information -- and they roughly double
        the GIF size."""
        from lms.viz.animate import contour_animation

        out = contour_animation(
            self._frames(), reynolds=[1.0, 2.0, 3.0], label="t",
            norm=PowerNorm(1.0, 0.0, 1.0), n_lines=0,
            dpi=40, figsize=(2.2, 2.0), save=tmp_path / "a.gif",
        )
        assert out.exists()

    def test_frame_captions_can_be_supplied_directly(self, tmp_path):
        """The animator was written for Reynolds sweeps, but decaying turbulence needs
        elapsed time in the caption instead."""
        from lms.viz.animate import contour_animation

        out = contour_animation(
            self._frames(), reynolds=[0.0, 0.5, 1.0], label="t",
            norm=PowerNorm(1.0, 0.0, 1.0),
            labels=["t/T = 0.00", "t/T = 0.50", "t/T = 1.00"],
            dpi=40, figsize=(2.2, 2.0), save=tmp_path / "b.gif",
        )
        assert out.exists()
