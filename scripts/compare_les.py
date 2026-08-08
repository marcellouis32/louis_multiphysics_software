"""Does the coarse LES run track the resolved run on the scales they share?

    python scripts/compare_les.py runs/turbulence

That question is what LES claims, stated so it can fail. The comparison is made mode by
mode rather than cell by cell: the two grids sample the same physical box differently, so
only the spectra are directly comparable, and only up to the coarse grid's Nyquist
wavenumber. Beyond that the coarse run has nothing to say and the resolved run is the
only source of truth.

Reads every `turb_*.npz` in the directory, treats the finest grid as the reference, and
compares each coarser run against it at matched eddy-turnover time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.colors import PowerNorm

from lms.validation.turbulence import enstrophy, inertial_slope
from lms.viz.fields3d import plot_decay, plot_energy_spectra, plot_orthogonal_slices
from lms.viz.style import FLOW


def load(path: Path) -> dict:
    d = np.load(path)
    return {
        "label": f"{int(d['n'])}³  Cs={float(d['smagorinsky']):g}",
        "n": int(d["n"]),
        "cs": float(d["smagorinsky"]),
        "k": d["k"],
        "spectra": d["spectra"],
        "steps": d["steps"],
        "turnover": float(d["turnover"]),
        "energy": d["energy"],
        "nu": float(d["nu"]),
        "nu_t_mean": d["nu_t_mean"],
        "ux": d["ux"], "uy": d["uy"], "uz": d["uz"],
    }


def at_time(run: dict, t_over_T: float) -> np.ndarray:
    """Spectrum at the sample nearest a given turnover time."""
    times = run["steps"] / run["turnover"]
    return run["spectra"][int(np.argmin(np.abs(times - t_over_T)))]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path, nargs="?", default=Path("runs/turbulence"))
    p.add_argument("--at", type=float, default=1.0, help="turnover time to compare at")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out = args.out or args.directory
    runs = [load(f) for f in sorted(args.directory.glob("turb_*.npz"))]
    if not runs:
        raise SystemExit(f"no turb_*.npz in {args.directory}")

    reference = max(runs, key=lambda r: r["n"])
    coarse = [r for r in runs if r is not reference]
    print(f"reference: {reference['label']}   "
          f"comparing at t/T = {args.at:g}\n")

    # ---------------------------------------------------------------- spectra
    curves = {}
    for run in [reference, *coarse]:
        curves[run["label"]] = (run["k"], at_time(run, args.at))

    e_ref = at_time(reference, args.at)
    print(f"{'run':>18} {'k_nyq':>6} {'band':>10} {'mean |dE|/E':>12} {'max |dE|/E':>11}")
    for run in coarse:
        k_nyq = run["n"] // 2
        e = at_time(run, args.at)
        # Compare only where the coarse grid has something to say, and drop the last
        # octave below its Nyquist: that is where the sharp filter's own truncation
        # dominates and neither run is describing the same thing.
        hi = max(k_nyq // 2, 2)
        band = slice(1, hi + 1)
        rel = np.abs(e[band] - e_ref[band]) / np.maximum(e_ref[band], 1e-30)
        print(f"{run['label']:>18} {k_nyq:>6} {f'1-{hi}':>10} "
              f"{rel.mean():>12.3f} {rel.max():>11.3f}")

    # Inertial-range slope, always with its window stated.
    lo, hi = 4, max(reference["n"] // 8, 8)
    print(f"\ninertial slope over k in [{lo}, {hi}]  (window stated because at these")
    print("resolutions it is a choice, and a slope without it is not a measurement)")
    for run in [reference, *coarse]:
        try:
            slope, _ = inertial_slope(run["k"], at_time(run, args.at), lo, hi)
            print(f"  {run['label']:>18}  {slope:+.3f}   (Kolmogorov: -1.667)")
        except ValueError as exc:
            print(f"  {run['label']:>18}  n/a: {exc}")

    plot_energy_spectra(
        curves,
        title="Energy spectrum — coarse LES against a resolved run",
        subtitle=f"decaying isotropic turbulence   ·   t/T = {args.at:g}",
        guide_window=(lo, hi),
        reference=reference["label"],
        save=out / "spectra.png",
    )

    # ------------------------------------------------------------ decay history
    plot_decay(
        {r["label"]: (r["steps"] / r["turnover"], r["energy"]) for r in [reference, *coarse]},
        ylabel="kinetic energy",
        title="Energy decay",
        subtitle="a subgrid model that injects energy would show here",
        save=out / "energy_decay.png",
    )

    for run in coarse:
        if run["nu_t_mean"].max() > 0:
            ratio = run["nu_t_mean"] / run["nu"]
            print(f"\n{run['label']}: nu_t/nu ranges {ratio.min():.2f} to {ratio.max():.2f}")
            if ratio.max() > 5:
                print("  the model is supplying most of the dissipation, so this is a")
                print("  statement about the model at least as much as about the solver")

    # ------------------------------------------------------------------ slices
    for run in [reference, *coarse]:
        omega = enstrophy(run["ux"], run["uy"], run["uz"])
        # Sequential, not diverging. |omega| cannot be negative, so a map centred on
        # zero would spend half its range on values that never occur and put the
        # quiet regions -- most of the box -- on the neutral midpoint.
        plot_orthogonal_slices(
            omega, label=r"$|\omega|$",
            title=f"Vorticity magnitude — {run['label']}",
            subtitle="decaying isotropic turbulence   ·   three centre planes, shared scale",
            cmap=FLOW,
            norm=PowerNorm(gamma=0.55, vmin=0.0,
                           vmax=float(np.percentile(omega, 99.5))),
            save=out / f"vorticity_{run['n']}_cs{run['cs']:g}.png",
        )

    print(f"\nfigures written to {out}/")


if __name__ == "__main__":
    main()
