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
        e = np.where(k > 0, k ** (-5 / 3), 0.0)
        slope, window = inertial_slope(k, e, 4, 32)
        assert slope == pytest.approx(-5 / 3, abs=1e-9)
        assert window == (4, 32)

    def test_inertial_slope_refuses_a_window_too_narrow_to_fit(self):
        k = np.arange(0, 64, dtype=float)
        e = np.where(k > 0, k ** (-5 / 3), 0.0)
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
