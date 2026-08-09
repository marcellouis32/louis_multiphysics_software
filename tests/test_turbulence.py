"""Isotropic-turbulence initial conditions and statistics.

The initial field and the spectral machinery are checked before any solver runs on
them. A wrong initial condition or a mis-normalised spectrum would not crash -- it
would produce a plausible-looking validation of nothing.
"""

import numpy as np
import pytest

from lms.validation.turbulence import (
    coarsen_spectral,
    energy_spectrum,
    inertial_slope,
    kinetic_energy,
    model_spectrum,
    solenoidal_field,
    statistics,
    wavenumbers,
)

N = 32
U_RMS = 0.05


@pytest.fixture(scope="module")
def field():
    return solenoidal_field(N, u_rms=U_RMS, k_peak=4.0, seed=0)


class TestSolenoidalField:
    def test_divergence_free_spectrally(self, field):
        """Checked as k . u_hat, not with a finite difference. A second-order stencil
        represents the derivative as sin(k) rather than k, so it reports a residual of
        order k^2 on a field that is genuinely solenoidal -- testing it that way would
        measure the stencil, not the field."""
        kx, ky, kz, kmag = wavenumbers(N)
        hats = [np.fft.fftn(c) for c in field]
        residual = np.abs(kx * hats[0] + ky * hats[1] + kz * hats[2])
        scale = (kmag * np.abs(hats[0])).max()
        assert residual.max() / scale < 1e-12

    def test_field_is_real_and_has_no_mean_flow(self, field):
        for comp in field:
            assert np.isrealobj(comp)
            assert abs(float(comp.mean())) < 1e-15

    def test_rms_is_what_was_requested(self, field):
        rms = np.sqrt(np.mean(sum(c**2 for c in field)) / 3.0)
        assert rms == pytest.approx(U_RMS, rel=1e-12)

    def test_energy_sits_near_the_requested_peak(self, field):
        k, e = energy_spectrum(*field)
        assert k[np.argmax(e)] == pytest.approx(4, abs=1)

    def test_is_statistically_isotropic(self, field):
        energies = [float(np.mean(c**2)) for c in field]
        assert max(energies) / min(energies) < 1.3

    def test_seed_is_reproducible(self):
        a = solenoidal_field(16, seed=3)
        b = solenoidal_field(16, seed=3)
        c = solenoidal_field(16, seed=4)
        assert np.array_equal(a[0], b[0])
        assert not np.array_equal(a[0], c[0])


class TestSpectrum:
    def test_spectrum_sums_to_the_kinetic_energy(self, field):
        """The normalisation is chosen so these are two views of one number; if they
        ever disagree, one of them is wrong."""
        _, e = energy_spectrum(*field)
        assert e.sum() == pytest.approx(kinetic_energy(*field), rel=1e-12)

    def test_model_spectrum_vanishes_at_zero_and_peaks_where_asked(self):
        k = np.arange(0, 40, dtype=float)
        e = model_spectrum(k, k_peak=6.0)
        assert e[0] == 0.0
        assert k[np.argmax(e)] == pytest.approx(6.0, abs=1.0)

    def test_inertial_slope_recovers_a_planted_power_law(self):
        k = np.arange(0, 64, dtype=float)
        e = np.zeros_like(k)
        e[1:] = k[1:] ** (-5 / 3)
        slope, window = inertial_slope(k, e, 4, 32)
        assert slope == pytest.approx(-5 / 3, abs=1e-9)
        assert window == (4, 32)

    def test_inertial_slope_refuses_a_window_too_narrow_to_fit(self):
        k = np.arange(0, 64, dtype=float)
        e = np.zeros_like(k)
        e[1:] = k[1:] ** (-5 / 3)
        with pytest.raises(ValueError, match="usable wavenumbers"):
            inertial_slope(k, e, 10, 11)


class TestSpectralCoarsening:
    def test_resolved_shells_are_preserved_exactly(self, field):
        """A sharp filter must not touch the scales it keeps. Truncating asymmetrically
        would break the Hermitian symmetry of a real field's transform, np.real() would
        silently discard the imaginary part, and the filter would quietly stop being
        exact while still looking plausible."""
        _, e_fine = energy_spectrum(*field)
        coarse = [coarsen_spectral(c, 16) for c in field]
        _, e_coarse = energy_spectrum(*coarse)
        upto = 16 // 2 - 1
        assert np.abs(e_coarse[1:upto] - e_fine[1:upto]).max() < 1e-15

    def test_coarsening_stays_real(self, field):
        for c in (coarsen_spectral(f, 16) for f in field):
            assert np.isrealobj(c)

    def test_coarsening_removes_energy_rather_than_creating_it(self, field):
        _, e_fine = energy_spectrum(*field)
        coarse = [coarsen_spectral(c, 8) for c in field]
        _, e_coarse = energy_spectrum(*coarse)
        assert e_coarse.sum() <= e_fine.sum() + 1e-18

    def test_identity_when_sizes_match(self, field):
        assert np.array_equal(coarsen_spectral(field[0], N), field[0])

    def test_refusing_to_refine(self, field):
        with pytest.raises(ValueError, match="cannot coarsen"):
            coarsen_spectral(field[0], 64)


class TestStatistics:
    def test_reynolds_lambda_is_consistent_with_its_parts(self, field):
        nu = 0.01
        s = statistics(*field, nu=nu)
        assert s.reynolds_lambda == pytest.approx(s.u_rms * s.taylor_microscale / nu, rel=1e-12)
        assert s.u_rms == pytest.approx(U_RMS, rel=1e-9)

    def test_dissipation_grows_with_viscosity(self, field):
        a = statistics(*field, nu=0.001)
        b = statistics(*field, nu=0.01)
        assert b.dissipation == pytest.approx(10.0 * a.dissipation, rel=1e-9)


class TestVorticity:
    """Vorticity on a periodic box, and the colour-map choice it implies."""

    def test_curl_of_a_known_field(self):
        """Solid-body rotation about z has omega = (0, 0, 2). Central differencing
        represents the derivative as sin(k) rather than k, but for a linear field the
        stencil is exact, so this pins the operator with no discretisation excuse."""
        from lms.validation.turbulence import vorticity

        n = 16
        i, j, _ = np.mgrid[0:n, 0:n, 0:n].astype(float)
        # Restrict to the interior so the periodic wrap of a non-periodic field does
        # not contaminate the check.
        wx, wy, wz = vorticity(-(j - n / 2), (i - n / 2), np.zeros((n, n, n)))
        core = (slice(1, -1), slice(1, -1), slice(1, -1))
        assert np.allclose(wz[core], 2.0)
        assert np.allclose(wx[core], 0.0)
        assert np.allclose(wy[core], 0.0)

    def test_curl_uses_periodic_wraparound(self):
        """np.gradient falls back to one-sided stencils at the array edges, which
        invents a boundary on a periodic domain. A pure Fourier mode must give the same
        answer on the faces as in the middle."""
        from lms.validation.turbulence import vorticity

        n = 16
        _, j, _ = np.mgrid[0:n, 0:n, 0:n].astype(float)
        k = 2 * np.pi / n
        _, _, wz = vorticity(np.sin(k * j), np.zeros((n, n, n)), np.zeros((n, n, n)))
        # The face values must match the interior pattern, not be one-sided artefacts.
        assert np.abs(wz[:, 0, :] - wz[:, 0, :].mean()).max() < 1e-12
        assert wz[:, 0, :].mean() == pytest.approx(-np.sin(k) * np.cos(0.0), abs=1e-12)

    def test_enstrophy_is_non_negative(self, field):
        """Which is why it must be rendered with a sequential colormap: a diverging map
        centred on zero would spend half its range on values that cannot occur."""
        from lms.validation.turbulence import enstrophy

        assert enstrophy(*field).min() >= 0.0


class TestDissipationScaling:
    """The dissipation rate sets the Taylor microscale, Re_lambda and the Kolmogorov
    scale, so an error in it silently mis-states how turbulent a run was."""

    def test_matches_a_direct_strain_rate_computation(self):
        """Independent route to the same number: eps = 2 nu <S_ab S_ab>, with exact
        spectral derivatives. Agreement to a few percent (shell binning is the residual)
        is what pins the wavenumber convention."""
        from lms.validation.turbulence import dissipation_rate, solenoidal_field

        n, nu = 64, 0.01
        ux, uy, uz = solenoidal_field(n, u_rms=0.05, k_peak=4.0, seed=0)

        k1 = np.fft.fftfreq(n) * n
        kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
        scale = 2 * np.pi / n
        hats = [np.fft.fftn(c) for c in (ux, uy, uz)]
        comps = [kx, ky, kz]

        direct = 0.0
        for a in range(3):
            for b in range(3):
                dab = np.fft.ifftn(1j * comps[a] * scale * hats[b])
                dba = np.fft.ifftn(1j * comps[b] * scale * hats[a])
                strain = 0.5 * np.real(dab + dba)
                direct += 2 * nu * np.mean(strain * strain)

        assert dissipation_rate(ux, uy, uz, nu) == pytest.approx(direct, rel=0.05)

    def test_uses_physical_wavenumbers_not_shell_indices(self):
        """The specific bug this guards: using the integer shell index inflates epsilon
        by (n / 2 pi)^2 -- 101x at n = 64 -- and drags Re_lambda down by n / 2 pi."""
        from lms.validation.turbulence import (
            dissipation_rate,
            energy_spectrum,
            solenoidal_field,
        )

        n, nu = 64, 0.01
        ux, uy, uz = solenoidal_field(n, u_rms=0.05, seed=0)
        k, e = energy_spectrum(ux, uy, uz)
        naive = 2.0 * nu * np.sum(k**2 * e)

        ratio = naive / dissipation_rate(ux, uy, uz, nu)
        assert ratio == pytest.approx((n / (2 * np.pi)) ** 2, rel=1e-9)
