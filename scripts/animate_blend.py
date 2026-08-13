"""Animate the blend-time run: dye homogenising, and the flow that does it.

    python scripts/animate_blend.py runs/blend/blend_rushton_standard_lab_n96.npz

Two artifacts:

  anim_dye_*      the dye field alone on the vertical mid-plane -- the in-silico
                  version of the dye-and-stopwatch experiment.
  anim_mixing_*   the synchronised two-panel: velocity magnitude left, dye right,
                  same timeline. The flow that does the mixing and the mixing it
                  does, in one loop. This is the presentation artifact.

Style decisions, all from the plan and the figure-style skill:

  * Dye rides `DYE` (magma lifted off its black foot) and speed rides `EMBER`
    (inferno, same lift): sequential for non-negative quantities, visually distinct
    from each other, and neither dissolves into the dark canvas the way a ramp with
    a near-black foot does.
  * The dye norm is fixed from the *initial* patch concentration (vmax = 1) and
    frozen: the fade toward the mixed value IS the result. The caption carries CoV
    and t/theta_95 so it reads as mixing rather than as a colour trick.
  * The two panels carry separate norms and separate colorbars -- sharing a scale
    between different quantities would be exactly the error the shared scale
    prevents between same-quantity panels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.colors import PowerNorm

from lms.viz.animate import contour_animation, pooled_norm, side_by_side_animation
from lms.viz.style import DYE, EMBER


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("state", type=Path)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--dpi", type=int, default=105)
    p.add_argument("--max-frames", type=int, default=110)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    d = np.load(args.state)
    out = args.out or args.state.parent
    n = int(d["n"])
    theta95_s = float(d["theta95_s"])
    grenville = float(d["grenville_s"])
    revs = d["frame_revs"]
    cov = d["cov"]                      # (rev, CoV) pairs
    solid = d["static_solid"].astype(bool)
    mid_y = solid.shape[1] // 2

    keep = np.arange(len(revs))
    if len(keep) > args.max_frames:
        keep = np.linspace(0, len(revs) - 1, args.max_frames).round().astype(int)

    dye = [d["frames_c"][i].T for i in keep]
    speed = [d["frames_u"][i].T for i in keep]
    rz_solid = solid[:, mid_y, :].T

    # theta_95 in revolutions comes straight from the CoV history's crossing row, so
    # the caption's t/theta_95 needs no unit conversion at all.
    crossing = cov[cov[:, 1] < 0.05]
    theta95_revs = float(crossing[0, 0]) if len(crossing) else float(revs[-1])
    captions = []
    for i in keep:
        c_here = np.interp(revs[i], cov[:, 0], cov[:, 1])
        captions.append(
            f"t/θ₉₅ = {revs[i] / theta95_revs:4.2f}   ·   CoV = {c_here:.3f}"
        )

    dye_norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)
    speed_norm = pooled_norm(
        speed, lambda pool: PowerNorm(gamma=0.6, vmin=0.0,
                                      vmax=float(np.percentile(pool, 99.5))),
    )

    path = contour_animation(
        dye, reynolds=list(revs[keep]), labels=captions,
        label="dye volume fraction",
        title="Blend time — tracer homogenisation, vertical mid-plane",
        subtitle=f"{n}³ tank   ·   θ₉₅ = {theta95_s:.1f} s   ·   "
                 f"Grenville {grenville:.1f} s",
        norm=dye_norm, cmap=DYE, renderer="image", n_filled=44, n_lines=0,
        solid=rz_solid,
        fps=args.fps, dpi=args.dpi,
        save=out / f"anim_dye_n{n}.gif",
    )
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.2f} MB)")

    path = side_by_side_animation(
        speed, dye,
        titles=("velocity magnitude", "dye"),
        label=r"$|u|\,/\,U_{tip}$",
        norm=speed_norm, cmap=EMBER,
        right_norm=dye_norm, right_cmap=DYE,
        right_label="dye volume fraction",
        captions=captions,
        title="The flow that does the mixing, and the mixing it does",
        subtitle=f"{n}³   ·   θ₉₅ = {theta95_s:.1f} s",
        n_filled=44, n_lines=0, renderer="image",
        fps=args.fps, dpi=args.dpi,
        save=out / f"anim_mixing_n{n}.gif",
    )
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
