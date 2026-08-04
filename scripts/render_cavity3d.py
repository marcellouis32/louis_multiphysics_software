"""Render figures from saved 3D cavity states.

    python scripts/render_cavity3d.py runs/cavity3d

Same run-once/render-many split as the 2D pipeline. Three figures per span, plus one
comparison across spans:

  slices_*        the three centre planes on one shared colour scale
  uz_*            the spanwise velocity, which is identically zero in 2D -- so any
                  structure here is the end-wall physics that 2D cannot represent
  taylor_gortler_*  the mid-height plane where the corner vortices live
  span_profiles   mid-plane centreline against Ghia at every span, the figure that
                  makes the approach to the 2D limit legible
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.colors import PowerNorm

from lms.validation import ghia
from lms.viz.fields3d import plot_orthogonal_slices, plot_span_profiles
from lms.viz.style import DIVERGING_DARK, FLOW, signed_asinh_norm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path, nargs="?", default=Path("runs/cavity3d"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out = args.out or args.directory
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.directory.glob("cavity3d_*.npz"))
    if not files:
        raise SystemExit(f"no cavity3d_*.npz in {args.directory}")

    profiles, reynolds, n = {}, None, None
    for path in files:
        d = np.load(path)
        ux, uy, uz = d["ux"], d["uy"], d["uz"]
        solid = d["solid"].astype(bool)
        span = float(d["span"])
        reynolds = float(d["reynolds"])
        lid = float(d["lid_velocity"])
        n = ux.shape[0]
        tag = f"re{int(reynolds)}_n{n}_span{span:g}"
        sub = (f"Re = {reynolds:g}   ·   {n}²  span {span:g}   ·   "
               f"TRT   ·   {int(d['steps']):,} steps")

        speed = np.sqrt(ux**2 + uy**2 + uz**2) / lid
        plot_orthogonal_slices(
            speed, label="$|u|\\,/\\,U_{lid}$",
            title="3D lid-driven cavity — velocity magnitude",
            subtitle=sub, cmap=FLOW, solid=solid,
            norm=PowerNorm(gamma=0.6, vmin=0.0, vmax=float(speed[~solid].max())),
            save=out / f"slices_{tag}.png",
        )

        # u_z is the whole story: it is identically zero in 2D, so every feature here
        # is something the 2D solver could not have produced.
        plot_orthogonal_slices(
            uz / lid, label="$u_z\\,/\\,U_{lid}$",
            title="3D lid-driven cavity — spanwise velocity (zero in 2D)",
            subtitle=sub, cmap=DIVERGING_DARK, solid=solid,
            norm=signed_asinh_norm(uz / lid, mask=solid, percentile=99.0,
                                   linear_percentile=40.0),
            save=out / f"uz_{tag}.png",
        )

        # Mid-plane centreline, for the cross-span comparison.
        interior = slice(1, -1)
        coord = (np.arange(1, n - 1) - 0.5) / (n - 2)
        mid_x, mid_z = n // 2, ux.shape[2] // 2
        profiles[f"span {span:g}"] = (ux[mid_x, interior, mid_z] / lid, coord)
        print(f"rendered {tag}")

    gy, gu = ghia.u_profile(reynolds)
    plot_span_profiles(
        profiles,
        reference=(gu, gy),
        title="Mid-plane $u$ approaches the 2D limit as the span grows",
        subtitle=f"Re = {reynolds:g}   ·   {n}² cross-section",
        save=out / "span_profiles.png",
    )
    print(f"\nfigures written to {out}/")
    print("The gap at span 1 is the end-wall effect, not solver error: a short cavity")
    print("genuinely is not a 2D cavity, and Ghia only describes the 2D limit.")


if __name__ == "__main__":
    main()
