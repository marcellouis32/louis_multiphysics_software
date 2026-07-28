"""Re-render figures from a saved cavity state, without re-running the solver.

    python scripts/render_cavity.py runs/cavity/state_re1000_n192.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.colors import PowerNorm

from lms.lbm.solver2d import stream_function, vortex_centre
from lms.validation import ghia
from lms.viz.fields2d import plot_contours, plot_streamfunction
from lms.viz.style import DIVERGING_DARK, FLOW, signed_asinh_norm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("state", type=Path)
    p.add_argument("--lid-velocity", type=float, default=0.1)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    data = np.load(args.state)
    ux, uy, solid = data["ux"], data["uy"], data["solid"].astype(bool)
    reynolds = float(data["reynolds"])
    steps = int(data["steps"])
    out = args.out or args.state.parent
    n = ux.shape[0]
    tag = f"re{int(reynolds)}_n{n}"
    subtitle = f"Re = {reynolds:g}   ·   {n}²   ·   TRT   ·   {steps:,} steps"

    interior = (slice(1, -1), slice(1, -1))
    ux_i, uy_i = ux[interior], uy[interior]
    length = n - 2

    speed = np.hypot(ux_i, uy_i) / args.lid_velocity
    vort = (np.gradient(uy, axis=1) - np.gradient(ux, axis=0))[interior]
    vort *= length / args.lid_velocity
    psi = stream_function(ux_i, uy_i) / (args.lid_velocity * length)

    plot_contours(
        speed,
        label="$|u|\\,/\\,U_{lid}$",
        title="Lid-driven cavity — velocity magnitude",
        subtitle=subtitle,
        cmap=FLOW,
        norm=PowerNorm(gamma=0.6, vmin=0.0, vmax=float(speed.max())),
        save=out / f"contour_speed_{tag}.png",
    )

    plot_contours(
        vort,
        label="$\\omega_z L\\,/\\,U_{lid}$",
        title="Lid-driven cavity — vorticity",
        subtitle=subtitle,
        cmap=DIVERGING_DARK,
        # Clip at the 97th percentile and keep the linear region tight: the lid
        # boundary layer reaches |omega| ~ 80 but the core rotates at ~2, and it is
        # the core we came to look at. Costs saturation in ~3% of cells, all of them
        # inside the wall layers.
        norm=signed_asinh_norm(vort, percentile=97.0, linear_percentile=20.0),
        save=out / f"contour_vorticity_{tag}.png",
    )

    cx, cy, psi_min = vortex_centre(psi)

    ref = ghia.primary_vortex(reynolds)
    gx, gy = ref["centre"]
    offset = float(np.hypot(cx - gx, cy - gy))
    print(f"primary vortex centre   LBM ({cx:.4f}, {cy:.4f})   Ghia ({gx:.4f}, {gy:.4f})")
    print(f"  offset {offset:.4f} L  ({offset * length:.1f} cells)")
    print(f"  psi_min  LBM {psi_min:.6f}   Ghia {ref['psi']:.6f}   "
          f"({abs(psi_min - ref['psi']) / abs(ref['psi']) * 100:.2f}% error)")

    plot_streamfunction(
        psi,
        title="Lid-driven cavity — stream function",
        subtitle=subtitle,
        markers={
            "LBM vortex centre": (cx, cy),
            "Ghia et al. 1982": (gx, gy),
        },
        save=out / f"contour_streamfunction_{tag}.png",
    )
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
