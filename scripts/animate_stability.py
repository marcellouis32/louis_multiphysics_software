"""TRT against regularized, side by side, at a Reynolds number that kills TRT.

    python scripts/animate_stability.py --n 128 --reynolds 8000

The whole argument for Phase 1c in one loop. Identical initial field, identical Reynolds
number, identical everything except the collision operator: one run tears itself apart,
the other stays coherent.

Two honesty constraints, and both cut against making the result look good:

**The colour scale is fitted to the surviving run, then frozen.** A diverging run reaches
absurd velocities within a few steps. Pooling the norm over both would let that excursion
set the scale, rendering the healthy run as a uniform dark square and the blow-up as
merely bright -- which reads as "nothing much happened either way".

**The dead panel stops and says so.** No NaNs painted as colour, no stale last-good frame
held on screen. A divergence is a measured result and should look like one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.colors import PowerNorm

from lms.lbm.d2q9 import viscosity_to_omega
from lms.lbm.solver3d import D3Q19Solver, init_backend
from lms.validation.turbulence import enstrophy, solenoidal_field
from lms.viz.animate import side_by_side_animation
from lms.viz.style import FLOW


def run(n, reynolds, collision, u_rms, k_peak, steps, every, seed=0):
    """Advance one case, keeping a mid-plane enstrophy slice per sample.

    Returns `(slices, last_valid_index)`. Sampling stops at the first non-finite field:
    everything after it is meaningless, and keeping it would only give the animation
    something misleading to draw.
    """
    nu = u_rms * n / reynolds
    solver = D3Q19Solver(
        (n, n, n), omega=viscosity_to_omega(nu),
        solid=np.zeros((n, n, n), dtype=bool), collision=collision,
    )
    solver.set_state(np.ones((n, n, n)), *solenoidal_field(n, u_rms=u_rms, k_peak=k_peak,
                                                           seed=seed))
    mid, slices, last = n // 2, [], None
    for step in range(steps + 1):
        if step % every == 0:
            _, gx, gy, gz = solver.macroscopic()
            if not np.isfinite(gx).all() or np.abs(gx).max() > 1.0:
                print(f"  {collision}: diverged by step {step:,}")
                break
            slices.append(enstrophy(gx, gy, gz)[:, :, mid])
            last = len(slices) - 1
        if step < steps:
            solver.step()
    return slices, last


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--reynolds", type=float, default=8000.0)
    p.add_argument("--u-rms", type=float, default=0.05)
    p.add_argument("--k-peak", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("runs/turb1c"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    init_backend(prefer_gpu=True, precision="fp32")
    every = max(args.steps // args.samples, 1)
    nu = args.u_rms * args.n / args.reynolds

    print(f"{args.n}^3   Re = {args.reynolds:g}   tau = {3 * nu + 0.5:.4f}   "
          f"no turbulence model")
    trt, trt_last = run(args.n, args.reynolds, "trt", args.u_rms, args.k_peak,
                        args.steps, every)
    reg, reg_last = run(args.n, args.reynolds, "regularized", args.u_rms, args.k_peak,
                        args.steps, every)
    print(f"  trt survived {len(trt)} samples, regularized {len(reg)}")

    frames = max(len(trt), len(reg))
    captions = [f"step {i * every:,}" for i in range(frames)]

    # Fitted to the survivor, never pooled: see the module docstring.
    survivor = reg if len(reg) >= len(trt) else trt
    norm = PowerNorm(
        gamma=0.6, vmin=0.0,
        vmax=float(np.percentile(np.concatenate([s.ravel() for s in survivor]), 99.9)),
    )

    path = side_by_side_animation(
        trt, reg,
        titles=("TRT", "regularized"),
        label=r"$|\omega|$",
        norm=norm, cmap=FLOW,
        captions=captions,
        dead_after=(trt_last, reg_last),
        dead_note="diverged",
        title=f"Same flow, same Re = {args.reynolds:g}, no model — only the collision "
              f"operator differs   ({args.n}³)",
        # Kept short: the right panel's lower-right corner is close to the colorbar,
        # and a longer string runs under it.
        subtitle=f"τ = {3 * nu + 0.5:.4f}",
        fps=args.fps,
        save=args.out / f"anim_stability_n{args.n}_re{int(args.reynolds)}.gif",
    )
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
