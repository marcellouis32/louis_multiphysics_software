"""Animate decaying turbulence: the one case here that is genuinely time-dependent.

    python scripts/animate_turbulence.py runs/turbulence/turb_n256_re2000_cs0.npz

The Reynolds-sweep animations from Phase 0 had to be honest about *not* being time
evolution -- the 2D cavity is steady, so those frames were a walk through parameter
space. Decaying turbulence is the real thing: every frame is a later moment in one flow,
and the structures visibly coarsen as small scales dissipate and the surviving eddies
merge.

Two animations, because they answer different questions and want different colour maps:

  anim_enstrophy_*   |omega|, a strictly non-negative magnitude, on the sequential FLOW
                     ramp. Shows where the vorticity *is*: bright filaments on a dark
                     field, thinning and fading as the cascade runs down.
  anim_wz_*          omega_z, which is signed, on the dark-centred diverging map. Shows
                     which way each eddy turns, so counter-rotating pairs read as
                     adjacent warm and cool patches rather than as one bright blob.

Both colour scales are pooled across every frame and then frozen, so the fading is the
flow losing energy rather than the colorbar rescaling under it. That matters more here
than in a parameter sweep: energy falls by two orders of magnitude over the run, and a
per-frame scale would show a flow that never changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.colors import AsinhNorm, PowerNorm

from lms.viz.animate import contour_animation, pooled_norm
from lms.viz.style import DIVERGING_DARK, FLOW


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("state", type=Path)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--dpi", type=int, default=92)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    d = np.load(args.state)
    if "slices_enstrophy" not in d:
        raise SystemExit(
            f"{args.state} has no slice snapshots. Re-run decaying_turbulence.py; "
            "older files stored only the final field."
        )

    out = args.out or args.state.parent
    n = int(d["n"])
    cs = float(d["smagorinsky"])
    turnover = float(d["turnover"])
    steps = d["steps"]
    times = steps / turnover
    energy = d["energy"]

    enstrophy = list(d["slices_enstrophy"])
    wz = list(d["slices_wz"])
    tag = f"n{n}_cs{cs:g}"

    # Energy falls by orders of magnitude, so the caption carries it: without a number
    # the eye reads "the picture got dimmer" and cannot tell decay from a colour trick.
    captions = [
        f"t/T = {t:4.2f}   ·   E/E₀ = {e / energy[0]:.3f}"
        for t, e in zip(times, energy)
    ]
    subtitle = f"{n}³   ·   Re = {float(d['reynolds']):g}   ·   Cs = {cs:g}"

    # ------------------------------------------------------------- enstrophy
    # Sequential: |omega| is non-negative, so a diverging map would waste half its range
    # and put the quiet majority of the box on the neutral midpoint.
    # 99.9 rather than the 99.5 used for the cavity. Turbulence is far more
    # intermittent than a laminar recirculation: the rare intense filaments are the
    # structure worth seeing, and clipping at 99.5 saturates them into featureless
    # patches. The cost is that half a percent of cells clip instead of a tenth.
    ens_norm = pooled_norm(
        enstrophy,
        lambda pool: PowerNorm(gamma=0.6, vmin=0.0, vmax=float(np.percentile(pool, 99.9))),
    )
    path = contour_animation(
        enstrophy, reynolds=list(times), labels=captions,
        label=r"$|\omega|$",
        title="Decaying turbulence — vorticity magnitude",
        subtitle=subtitle,
        norm=ens_norm, cmap=FLOW, n_filled=44, n_lines=0,
        fps=args.fps, dpi=args.dpi,
        save=out / f"anim_enstrophy_{tag}.gif",
    )
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.2f} MB)")

    # ------------------------------------------------------------ signed w_z
    lim = float(np.percentile(np.abs(np.concatenate([w.ravel() for w in wz])), 99.0))
    width = float(np.percentile(np.abs(np.concatenate([w.ravel() for w in wz])), 35.0))
    path = contour_animation(
        wz, reynolds=list(times), labels=captions,
        label=r"$\omega_z$",
        title="Decaying turbulence — spanwise vorticity",
        subtitle=subtitle,
        norm=AsinhNorm(linear_width=max(width, lim * 1e-3), vmin=-lim, vmax=lim),
        cmap=DIVERGING_DARK, n_filled=44, n_lines=0,
        fps=args.fps, dpi=args.dpi,
        save=out / f"anim_wz_{tag}.gif",
    )
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.2f} MB)")

    print(f"\n{len(enstrophy)} frames over {times[-1]:.2f} eddy turnovers")
    print(f"energy fell to {energy[-1] / energy[0] * 100:.1f}% of its initial value; the")
    print("colour scale is fixed across frames, so that fade is the flow, not the scale.")


if __name__ == "__main__":
    main()
