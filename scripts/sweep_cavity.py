"""Run a Reynolds sweep of the lid-driven cavity and save every converged solution.

    nohup .venv/bin/python -u scripts/sweep_cavity.py --n 192 \
      > runs/sweep_192.log 2>&1 &

Solving only, no rendering. That is the same run-once/render-many split as
`validate_cavity.py` -> `render_cavity.py`, and it has already paid for itself: the
sweep is hours of compute and the colour scale is something you want to re-tune in
seconds.

Redirect straight to a file rather than through a pipe. A pipe buffers, and a long run
then looks dead when it is only silent.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from lms.lbm.sweep import reynolds_sweep, sweep_values


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=192)
    p.add_argument("--lowest", type=float, default=100.0)
    p.add_argument("--highest", type=float, default=3200.0)
    p.add_argument("--count", type=int, default=25)
    p.add_argument("--lid-velocity", type=float, default=0.1)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--max-steps", type=int, default=150_000)
    p.add_argument("--cold-start", action="store_true", help="disable continuation")
    p.add_argument("--out", type=Path, default=Path("runs/sweep"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    values = sweep_values(lowest=args.lowest, highest=args.highest, count=args.count)

    print(
        f"cavity sweep  {args.n}x{args.n}  U={args.lid_velocity}  "
        f"{len(values)} points  Re {values[0]:.0f} -> {values[-1]:.0f}  "
        f"{'cold start' if args.cold_start else 'warm-started (continuation)'}"
    )

    frames = []
    start = time.perf_counter()
    for i, frame in enumerate(
        reynolds_sweep(
            n=args.n,
            reynolds_values=values,
            lid_velocity=args.lid_velocity,
            tol=args.tol,
            max_steps=args.max_steps,
            cold_start=args.cold_start,
        )
    ):
        frames.append(frame)
        elapsed = time.perf_counter() - start
        flag = "converged" if frame.converged else "STEP CAP"
        print(
            f"  [{i + 1:2d}/{len(values)}]  Re = {frame.reynolds:7.1f}  "
            f"tau = {frame.tau:.4f}  {frame.steps:>7,} steps  {flag}  "
            f"residual {frame.residual:.2e}  ({elapsed / 60:.1f} min elapsed)",
            flush=True,
        )

    path = args.out / f"sweep_n{args.n}.npz"
    np.savez_compressed(
        path,
        ux=np.stack([f.ux for f in frames]),
        uy=np.stack([f.uy for f in frames]),
        # rho is needed to restart a solve from a stored frame. Dropping it makes the
        # saved sweep un-resumable and un-auditable: you cannot afterwards ask how
        # converged a frame was, because you cannot reconstruct the solver state.
        rho=np.stack([f.rho for f in frames]),
        reynolds=np.array([f.reynolds for f in frames]),
        steps=np.array([f.steps for f in frames]),
        converged=np.array([f.converged for f in frames]),
        residual=np.array([f.residual for f in frames]),
        tau=np.array([f.tau for f in frames]),
        solid=_solid(args.n, args.lid_velocity, values[0]),
        lid_velocity=args.lid_velocity,
    )

    n_bad = sum(not f.converged for f in frames)
    total = (time.perf_counter() - start) / 60
    print(f"\n{len(frames)} frames in {total:.1f} min; {n_bad} hit the step cap")
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def _solid(n: int, lid_velocity: float, reynolds: float) -> np.ndarray:
    """The wall mask is identical for every frame, so store it once."""
    from lms.lbm.solver2d import lid_driven_cavity

    return lid_driven_cavity(
        n=n, reynolds=float(reynolds), lid_velocity=lid_velocity, collision="trt"
    ).solid


if __name__ == "__main__":
    main()
