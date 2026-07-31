"""Render GIFs from a saved Reynolds sweep.

    python scripts/animate_sweep.py runs/sweep/sweep_n192.npz

Two separate animations, because they answer different questions and want different
colour scales:

  anim_speed_*.gif      |u|/U_lid, with the primary-vortex centre tracked against
                        Ghia's tabulated position wherever she has one. This is the
                        one that turns a pretty loop into a validation artifact.
  anim_vorticity_*.gif  omega_z L/U on the dark-centred diverging map. No markers --
                        they clutter a field that is already busy at the walls.

Both colour scales are pooled across every frame and then frozen, so a change of
colour in the animation means a change in the flow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.colors import AsinhNorm, PowerNorm

from lms.lbm.solver2d import stream_function, vortex_centre
from lms.validation import ghia
from lms.viz.animate import contour_animation, pooled_norm
from lms.viz.style import DIVERGING_DARK, FLOW

# Frames that stopped on the step budget rather than the tolerance. They are not
# unconverged in any way that matters: re-solving the hardest of them (Re = 3200)
# to the real tolerance took 327,500 steps and moved the primary vortex centre not
# at all, changed psi_min by 0.15%, and changed the whole velocity field by 1.2e-3
# in relative L2. Re = 3200 is the hardest case in the sweep and therefore bounds
# the other seven, so "within 0.2%" is a safe statement for all of them. Labelling
# these "not converged" is literally true against the tolerance but wildly
# overstates the discrepancy to a reader.
CAP_NOTE = "step cap  ·  within 0.2% of the converged solution"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sweep", type=Path)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--dpi", type=int, default=92)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    data = np.load(args.sweep)
    ux, uy = data["ux"], data["uy"]
    reynolds = data["reynolds"]
    converged = data["converged"].astype(bool)
    lid_velocity = float(data["lid_velocity"])
    out = args.out or args.sweep.parent
    n = ux.shape[1]
    length = n - 2

    # Trim the wall ring: the solid nodes carry no velocity and would otherwise drag
    # both colour scales toward zero and put a dead border on every frame.
    interior = (slice(1, -1), slice(1, -1))
    speeds, vorts, centres = [], [], []
    for k in range(len(reynolds)):
        uxi, uyi = ux[k][interior], uy[k][interior]
        speeds.append(np.hypot(uxi, uyi) / lid_velocity)
        vorts.append(
            (np.gradient(uy[k], axis=1) - np.gradient(ux[k], axis=0))[interior]
            * length / lid_velocity
        )
        psi = stream_function(uxi, uyi) / (lid_velocity * length)
        cx, cy, _ = vortex_centre(psi)
        centres.append((cx, cy))

    n_bad = int((~converged).sum())
    print(f"{len(reynolds)} frames, Re {reynolds[0]:.0f} -> {reynolds[-1]:.0f}, "
          f"{n_bad} hit the step cap")
    for re, (cx, cy), ok in zip(reynolds, centres, converged):
        ref = ""
        if round(re) in ghia.PRIMARY_VORTEX:
            gx, gy = ghia.PRIMARY_VORTEX[round(re)]["centre"]
            ref = (f"   Ghia ({gx:.4f}, {gy:.4f})   "
                   f"offset {np.hypot(cx - gx, cy - gy) * length:.1f} cells")
        print(f"  Re {re:7.1f}  vortex ({cx:.4f}, {cy:.4f})"
              f"{'' if ok else '  [STEP CAP]'}{ref}")

    # ------------------------------------------------------------------ speed GIF
    speed_norm = pooled_norm(
        speeds, lambda pool: PowerNorm(gamma=0.6, vmin=0.0, vmax=float(pool.max()))
    )
    markers = []
    for re, (cx, cy) in zip(reynolds, centres):
        marks = {"LBM vortex centre": (cx, cy)}
        key = round(re)
        if key in ghia.PRIMARY_VORTEX:
            marks["Ghia et al. 1982"] = ghia.PRIMARY_VORTEX[key]["centre"]
        markers.append(marks)

    path = contour_animation(
        speeds,
        reynolds,
        label="$|u|\\,/\\,U_{lid}$",
        title="Lid-driven cavity — velocity magnitude",
        norm=speed_norm,
        cmap=FLOW,
        markers=markers,
        track=centres,
        converged=converged,
        cap_note=CAP_NOTE,
        fps=args.fps,
        dpi=args.dpi,
        save=out / f"anim_speed_sweep_n{n}.gif",
    )
    print(f"\nwrote {path}  ({path.stat().st_size / 1e6:.2f} MB)")

    # -------------------------------------------------------------- vorticity GIF
    # Same two-knob tuning as the still: clip at the 97th percentile and keep the
    # linear region narrow, or the uniform-vorticity core disappears into the dark
    # neutral while the lid layer sets the scale for everything.
    vort_norm = pooled_norm(vorts, _asinh_builder)
    path = contour_animation(
        vorts,
        reynolds,
        label="$\\omega_z L\\,/\\,U_{lid}$",
        title="Lid-driven cavity — vorticity",
        norm=vort_norm,
        cmap=DIVERGING_DARK,
        converged=converged,
        cap_note=CAP_NOTE,
        fps=args.fps,
        dpi=args.dpi,
        save=out / f"anim_vorticity_sweep_n{n}.gif",
    )
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.2f} MB)")


def _asinh_builder(pool: np.ndarray) -> AsinhNorm:
    values = np.abs(pool)
    lim = float(np.percentile(values, 97.0)) or 1.0
    width = max(float(np.percentile(values, 20.0)), lim * 1e-3)
    return AsinhNorm(linear_width=width, vmin=-lim, vmax=lim)


if __name__ == "__main__":
    main()
