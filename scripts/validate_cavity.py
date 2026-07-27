"""Run the lid-driven cavity benchmark and emit the verification figure set.

    python scripts/validate_cavity.py --n 192 --reynolds 1000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from lms.lbm.solver2d import cavity_centerlines, lid_driven_cavity
from lms.validation import ghia
from lms.viz import plot_convergence, plot_ghia_comparison, plot_lic, plot_vorticity


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=192)
    p.add_argument("--reynolds", type=float, default=1000.0)
    p.add_argument("--lid-velocity", type=float, default=0.1)
    p.add_argument("--collision", choices=["bgk", "trt"], default="trt")
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--tol", type=float, default=2e-7)
    p.add_argument("--out", type=Path, default=Path("runs/cavity"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    solver = lid_driven_cavity(
        n=args.n,
        reynolds=args.reynolds,
        lid_velocity=args.lid_velocity,
        collision=args.collision,
    )
    tau = 1.0 / solver.omega_plus
    print(
        f"cavity {args.n}x{args.n}  Re={args.reynolds:g}  U={args.lid_velocity}  "
        f"tau={tau:.4f}  collision={args.collision}"
    )

    start = time.perf_counter()

    def progress(step: int, s) -> None:
        if step % 20_000 == 0:
            res = s and None
            del res
            elapsed = time.perf_counter() - start
            print(f"  step {step:>7,}   {step / elapsed:,.0f} steps/s")

    state = solver.run(
        max_steps=args.max_steps, tol=args.tol, check_every=1000, callback=progress
    )
    elapsed = time.perf_counter() - start
    print(
        f"{'converged' if state.converged else 'stopped'} after {state.steps:,} steps "
        f"in {elapsed:.1f}s ({state.steps / elapsed:,.0f} steps/s)"
    )

    y, u, x, v = cavity_centerlines(state, solver.lid_velocity)
    gy, gu = ghia.u_profile(args.reynolds)
    gx, gv = ghia.v_profile(args.reynolds)
    rms_u = ghia.rms_error(y, u, gy, gu)
    rms_v = ghia.rms_error(x, v, gx, gv)
    print(f"RMS vs Ghia:  u = {rms_u:.4f}   v = {rms_v:.4f}")

    tag = f"re{int(args.reynolds)}_n{args.n}"
    subtitle = (
        f"Re = {args.reynolds:g}   ·   {args.n}²   ·   "
        f"{args.collision.upper()}   ·   {state.steps:,} steps"
    )

    np.savez_compressed(
        args.out / f"state_{tag}.npz",
        ux=state.ux, uy=state.uy, rho=state.rho, solid=solver.solid,
        reynolds=args.reynolds, steps=state.steps,
    )

    plot_ghia_comparison(
        y, u, x, v, gy, gu, gx, gv,
        reynolds=args.reynolds, resolution=args.n,
        rms_u=rms_u, rms_v=rms_v,
        save=args.out / f"ghia_{tag}.png",
    )
    plot_lic(
        state.ux, state.uy,
        title="Lid-driven cavity — velocity field",
        subtitle=subtitle,
        solid=solver.solid,
        save=args.out / f"lic_{tag}.png",
    )
    plot_vorticity(
        state.vorticity(),
        title="Lid-driven cavity — vorticity",
        subtitle=subtitle,
        solid=solver.solid,
        save=args.out / f"vorticity_{tag}.png",
    )
    if state.residuals:
        plot_convergence(state.residuals, save=args.out / f"convergence_{tag}.png")

    print(f"figures written to {args.out}/")


if __name__ == "__main__":
    main()
