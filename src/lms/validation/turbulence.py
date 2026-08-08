"""Isotropic-turbulence initial conditions and statistics.

Decaying isotropic turbulence is the LES test case that needs no external tables. Three
things can be checked against theory or against our own data:

  * the inertial range should follow Kolmogorov's k^-5/3 -- a prediction, not a
    tabulated measurement,
  * total kinetic energy must fall monotonically, since a subgrid model that injects
    energy is broken,
  * a coarse LES run should track a resolved run of the same flow over the scales the
    coarse grid can represent. That is precisely what LES claims to do, stated as
    something falsifiable.

Everything here is spectral, which is the natural language for the first two and the
only honest way to do the third: comparing two grids in physical space compares
different sampling of the same field, while comparing them mode by mode compares the
same physical scales.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wavenumbers(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integer wavevector components and their magnitude on an n^3 lattice.

    Uses FFT ordering, so index 0 is the mean mode and the upper half of each axis
    carries the negative frequencies. `np.fft.fftfreq(n) * n` gives exactly that.
    """
    k1 = np.fft.fftfreq(n) * n
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    return kx, ky, kz, np.sqrt(kx**2 + ky**2 + kz**2)


def model_spectrum(k: np.ndarray, k_peak: float = 4.0) -> np.ndarray:
    """E(k) ~ (k/k0)^4 exp(-2 (k/k0)^2), the standard synthetic initial spectrum.

    The k^4 rise at low wavenumber and the Gaussian roll-off keep the initial field
    smooth and free of energy at the grid scale, so the first few steps are turbulence
    developing rather than the solver digesting a discontinuity.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(k > 0, k / k_peak, 0.0)
    return ratio**4 * np.exp(-2.0 * ratio**2)


def solenoidal_field(
    n: int, u_rms: float = 0.05, k_peak: float = 4.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random divergence-free velocity field with a prescribed energy spectrum.

    Built in Fourier space so incompressibility is exact rather than approximate:
    projecting out the component parallel to k makes `k . u_hat` vanish to machine
    precision. A field assembled in physical space and then "mostly" divergence-free
    would radiate a pressure transient for the first few hundred steps and contaminate
    the early spectrum.

    Note the divergence is exactly zero *spectrally*, not under a finite-difference
    stencil: a second-order central difference represents the derivative as sin(k)
    rather than k, so it reports a residual of order k^2 on a field that is genuinely
    solenoidal. Check this field with `k . u_hat`, not with `np.roll`.

    The projection uses real coefficients and so preserves the Hermitian symmetry of the
    transform of a real field, which is what keeps the result real after inversion.
    """
    rng = np.random.default_rng(seed)
    kx, ky, kz, kmag = wavenumbers(n)

    # Random phases with the right symmetry: transform a real Gaussian field.
    hats = [np.fft.fftn(rng.standard_normal((n, n, n))) for _ in range(3)]

    # Helmholtz projection: remove the component along k.
    k2 = np.where(kmag > 0, kmag**2, 1.0)
    dot = (kx * hats[0] + ky * hats[1] + kz * hats[2]) / k2
    hats = [h - c * dot for h, c in zip(hats, (kx, ky, kz))]

    # Impose the target spectrum shell by shell. The random field has a k^2 density of
    # states, so amplitudes are scaled by sqrt(E(k) / (4 pi k^2)) to land on E(k).
    shell = np.where(kmag > 0, 4.0 * np.pi * kmag**2, 1.0)
    amp = np.sqrt(np.where(kmag > 0, model_spectrum(kmag, k_peak) / shell, 0.0))
    hats = [h * amp for h in hats]
    for h in hats:
        h[0, 0, 0] = 0.0                      # no mean flow
        # Zero the Nyquist planes. There, index n/2 carries k = -n/2 with no +n/2
        # partner, so it is its own conjugate and must be real -- but the projection
        # does not preserve that, np.real() below then discards the leftover imaginary
        # part, and the field stops being exactly solenoidal at those modes. The model
        # spectrum puts almost no energy there, so dropping them costs nothing and
        # buys exactness.
        half = n // 2
        h[half, :, :] = 0.0
        h[:, half, :] = 0.0
        h[:, :, half] = 0.0

    field = [np.real(np.fft.ifftn(h)) for h in hats]

    # Normalise to the requested rms. u_rms is per component, so <|u|^2> = 3 u_rms^2.
    current = np.sqrt(np.mean(sum(c**2 for c in field)) / 3.0)
    scale = u_rms / current if current > 0 else 1.0
    return tuple(c * scale for c in field)


def energy_spectrum(
    ux: np.ndarray, uy: np.ndarray, uz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Shell-averaged E(k), normalised so `E.sum()` is the mean kinetic energy.

    Returns `(k, E)` with integer shell centres. That normalisation is deliberate: it
    makes the spectrum and `kinetic_energy` two views of one number, so a mismatch
    between them is immediately visible rather than a matter of convention.
    """
    n = ux.shape[0]
    _, _, _, kmag = wavenumbers(n)
    e = np.zeros_like(ux, dtype=np.float64)
    for comp in (ux, uy, uz):
        hat = np.fft.fftn(comp) / comp.size
        e += 0.5 * np.abs(hat) ** 2

    shells = np.rint(kmag).astype(int)
    kmax = n // 2
    spectrum = np.bincount(shells.ravel(), weights=e.ravel(), minlength=kmax + 1)
    return np.arange(kmax + 1), spectrum[: kmax + 1]


def kinetic_energy(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray) -> float:
    """Volume-averaged kinetic energy per unit mass."""
    return float(0.5 * np.mean(ux**2 + uy**2 + uz**2))


def dissipation_rate(
    ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, nu: float
) -> float:
    """Molecular dissipation, computed spectrally as 2 nu sum k^2 E(k).

    Spectral rather than by finite differences, because a difference stencil
    systematically under-resolves the highest wavenumbers -- exactly where dissipation
    lives -- and would flatter the solver.
    """
    k, e = energy_spectrum(ux, uy, uz)
    return float(2.0 * nu * np.sum(k**2 * e))


@dataclass
class TurbulenceStats:
    step: int
    energy: float
    dissipation: float
    u_rms: float
    taylor_microscale: float
    reynolds_lambda: float


def statistics(
    ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, nu: float, step: int = 0
) -> TurbulenceStats:
    """The standard scalar diagnostics for isotropic turbulence."""
    energy = kinetic_energy(ux, uy, uz)
    eps = dissipation_rate(ux, uy, uz, nu)
    u_prime = float(np.sqrt(2.0 * energy / 3.0))
    lam = float(np.sqrt(15.0 * nu * u_prime**2 / eps)) if eps > 0 else float("inf")
    return TurbulenceStats(
        step=step,
        energy=energy,
        dissipation=eps,
        u_rms=u_prime,
        taylor_microscale=lam,
        reynolds_lambda=float(u_prime * lam / nu) if nu > 0 else float("inf"),
    )


def coarsen_spectral(field: np.ndarray, n_coarse: int) -> np.ndarray:
    """Truncate a field to a coarser grid by discarding high Fourier modes.

    This is a *sharp spectral filter*, which is the definition of the LES field the
    coarse run is supposed to reproduce. Coarsening by subsampling or box-averaging
    instead would alias the discarded scales back into the resolved ones, so the coarse
    run would start from a field the fine run never contained and the comparison would
    be against the wrong target.
    """
    n = field.shape[0]
    if n_coarse > n:
        raise ValueError(f"cannot coarsen {n} to {n_coarse}")
    if n_coarse == n:
        return field.copy()

    hat = np.fft.fftn(field)
    half = n_coarse // 2
    # k from -half to half-1, the modes an n_coarse grid can represent.
    keep = np.concatenate([np.arange(half), np.arange(n - half, n)])
    small = hat[np.ix_(keep, keep, keep)]

    # Zero the Nyquist planes. Mode -half has no positive counterpart on the coarse
    # grid, so keeping it breaks the Hermitian symmetry of a real field's transform --
    # the inverse comes back complex, np.real() silently discards the imaginary part,
    # and the filter quietly stops being exact. Dropping it costs the single highest
    # wavenumber and keeps everything below it exact.
    small[half, :, :] = 0.0
    small[:, half, :] = 0.0
    small[:, :, half] = 0.0

    # numpy's inverse transform divides by the new element count, so without this
    # rescaling the coarse field comes back with the wrong amplitude.
    return np.real(np.fft.ifftn(small)) * (n_coarse / n) ** 3


def inertial_slope(
    k: np.ndarray, e: np.ndarray, k_lo: int, k_hi: int
) -> tuple[float, tuple[int, int]]:
    """Least-squares slope of log E against log k over a stated window.

    The window is returned alongside the slope, and callers are expected to report it.
    At 64^3-256^3 the inertial range spans only a handful of wavenumbers, so a -5/3
    quoted without its fitting window is not a measurement -- it is a choice of window
    dressed up as a result.
    """
    band = (k >= k_lo) & (k <= k_hi) & (e > 0)
    if band.sum() < 3:
        raise ValueError(f"only {band.sum()} usable wavenumbers in [{k_lo}, {k_hi}]")
    slope = np.polyfit(np.log(k[band]), np.log(e[band]), 1)[0]
    return float(slope), (k_lo, k_hi)


__all__ = [
    "TurbulenceStats",
    "coarsen_spectral",
    "dissipation_rate",
    "energy_spectrum",
    "inertial_slope",
    "kinetic_energy",
    "model_spectrum",
    "solenoidal_field",
    "statistics",
    "wavenumbers",
]
