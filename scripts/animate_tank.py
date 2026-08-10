"""Animate a stirred-tank run: the classic Rushton picture, moving.

    python scripts/animate_tank.py runs/tank/tank_rushton_standard_lab_n128.npz

Two views, two different questions:

  anim_tank_rz_*   vertical mid-plane. The radial discharge jet off the blade tips
                   splitting into the upper and lower circulation loops -- the frame
                   that says "stirred tank" to anyone who has seen one.
  anim_tank_xy_*   horizontal plane at impeller height: trailing vortices behind the
                   blades and their interaction with the baffles.

Style per the house standard: |u|/U_tip is non-negative, so sequential FLOW with a
PowerNorm; one norm pooled over all frames of both views and frozen; static solids
masked so the geometry reads. The caption carries the revolution count and the running
torque so the eye can watch the flow establish -- the torque settling IS the spin-up
ending.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.colors import PowerNorm

from lms.viz.animate import contour_animation, pooled_norm
from lms.viz.style import FLOW


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("state", type=Path)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--dpi", type=int, default=92)
    p.add_argument("--max-frames", type=int, default=120,
                   help="subsample to keep the GIF under the 5 MB budget")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    d = np.load(args.state)
    out = args.out or args.state.parent
    n = int(d["n"])
    spr = float(d["steps_per_rev"])
    torque = d["torque"]
    steps = d["frame_steps"]
    solid = d["static_solid"].astype(bool)
    z_imp = int(d["z_impeller"])
    mid_y = solid.shape[1] // 2

    keep = np.arange(len(steps))
    if len(keep) > args.max_frames:
        keep = np.linspace(0, len(steps) - 1, args.max_frames).round().astype(int)

    # Frames are (x, z) and (x, y); matplotlib draws rows vertically, so transpose.
    rz = [d["frames_rz"][i].T for i in keep]
    xy = [d["frames_xy"][i].T for i in keep]
    rz_solid = solid[:, mid_y, :].T
    xy_solid = solid[:, :, z_imp].T

    # Trailing-revolution torque average per frame: the number that says whether the
    # flow has established, carried where the eye is already looking.
    captions = []
    for i in keep:
        s = int(steps[i])
        window = torque[max(0, s - round(spr)):s + 1]
        mean_t = np.abs(window).mean() if len(window) else 0.0
        captions.append(f"rev {s / spr:5.1f}   ·   |T| = {mean_t:.3e}")

    # One norm across BOTH views: comparing the jet in one panel with the trailing
    # vortices in the other only means something on a shared scale.
    norm = pooled_norm(
        rz + xy,
        lambda pool: PowerNorm(gamma=0.6, vmin=0.0,
                               vmax=float(np.percentile(pool, 99.5))),
    )

    for tag, frames, mask, title in (
        ("rz", rz, rz_solid, "Stirred tank — velocity magnitude, vertical mid-plane"),
        ("xy", xy, xy_solid, "Stirred tank — velocity magnitude, impeller plane"),
    ):
        path = contour_animation(
            frames,
            reynolds=[0.0] * len(frames),
            labels=captions,
            label=r"$|u|\,/\,U_{tip}$",
            title=title,
            subtitle=f"{n}³ tank   ·   Re = {float(d['reynolds']):,.0f}   ·   regularized + LES",
            norm=norm,
            cmap=FLOW,
            n_filled=44,
            n_lines=0,
            solid=mask,
            fps=args.fps,
            dpi=args.dpi,
            save=out / f"anim_tank_{tag}_n{n}.gif",
        )
        print(f"wrote {path}  ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
