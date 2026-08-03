"""Run a Reynolds sweep of the lid-driven cavity and save every converged solution.

    nohup .venv/bin/python -u scripts/sweep_cavity.py --n 192 \
      > runs/sweep_192.log 2>&1 &

Solving only, no rendering. That is the same run-once/render-many split as
`validate_cavity.py` -> `render_cavity.py`, and it has already paid for itself: the
sweep is hours of compute and the colour scale is something you want to re-tune in
seconds.

Redirect straight to a file rather than through a pipe. A pipe buffers, and a long run
then looks dead when it is only silent.

Each frame is written to disk the moment it finishes, and consolidated into a single
archive at the end. A sweep at 192² is around three hours; accumulating everything in
memory and writing once at the end means a crash, a full disk or an accidental Ctrl-C
throws away the entire run. Pass `--resume` to pick up from the frames already on disk.
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
    p.add_argument("--resume", action="store_true", help="skip frames already on disk")
    p.add_argument("--out", type=Path, default=Path("runs/sweep"))
    args = p.parse_args()

    parts = args.out / f"frames_n{args.n}"
    parts.mkdir(parents=True, exist_ok=True)
    values = sweep_values(lowest=args.lowest, highest=args.highest, count=args.count)

    done = _existing(parts, values) if args.resume else {}
    todo = [v for i, v in enumerate(values) if i not in done]
    seed = None
    if done:
        # Resume from the last completed frame so the continuation chain is unbroken.
        last = done[max(done)]
        seed = (last["ux"], last["uy"], last["rho"])
        print(f"resuming: {len(done)} frames already on disk, {len(todo)} to solve")

    print(
        f"cavity sweep  {args.n}x{args.n}  U={args.lid_velocity}  "
        f"{len(values)} points  Re {values[0]:.0f} -> {values[-1]:.0f}  "
        f"{'cold start' if args.cold_start else 'warm-started (continuation)'}"
    )

    start = time.perf_counter()
    for frame in reynolds_sweep(
        n=args.n,
        reynolds_values=todo,
        lid_velocity=args.lid_velocity,
        tol=args.tol,
        max_steps=args.max_steps,
        cold_start=args.cold_start,
        seed=seed,
    ):
        index = int(np.argmin(np.abs(values - frame.reynolds)))
        _write_frame(parts, index, frame)
        elapsed = time.perf_counter() - start
        flag = "converged" if frame.converged else "STEP CAP"
        print(
            f"  [{index + 1:2d}/{len(values)}]  Re = {frame.reynolds:7.1f}  "
            f"tau = {frame.tau:.4f}  {frame.steps:>7,} steps  {flag}  "
            f"residual {frame.residual:.2e}  ({elapsed / 60:.1f} min elapsed)",
            flush=True,
        )

    path = _consolidate(parts, args.out, args.n, values, args.lid_velocity)
    total = (time.perf_counter() - start) / 60
    print(f"\nsolved in {total:.1f} min")
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def _frame_path(parts: Path, index: int) -> Path:
    return parts / f"frame_{index:03d}.npz"


def _write_frame(parts: Path, index: int, frame) -> None:
    """Write one frame, atomically.

    Written to a temporary name and renamed, because a crash partway through a write
    would otherwise leave a truncated file that `--resume` would happily load as if it
    were a real solution. Rename is atomic on POSIX; the file either exists complete or
    does not exist.
    """
    tmp = _frame_path(parts, index).with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        ux=frame.ux, uy=frame.uy, rho=frame.rho,
        reynolds=frame.reynolds, steps=frame.steps,
        converged=frame.converged, residual=frame.residual, tau=frame.tau,
    )
    tmp.replace(_frame_path(parts, index))


def _existing(parts: Path, values: np.ndarray) -> dict[int, dict]:
    """Load the contiguous run of frames already on disk, keyed by index into `values`.

    Only a prefix is reused, and the scan stops at the first gap. Continuation is a
    chain, so reusing frames from beyond a gap would mean seeding a case from a solution
    several Reynolds numbers away and quietly changing what the resumed run computes.

    A frame whose stored Reynolds number no longer matches the requested grid also stops
    the scan, rather than being reused -- otherwise changing `--count` between runs would
    splice solutions from a different sweep into this one.
    """
    found: dict[int, dict] = {}
    for index in range(len(values)):
        path = _frame_path(parts, index)
        if not path.exists():
            break
        data = np.load(path)
        if abs(float(data["reynolds"]) - float(values[index])) > 1e-6 * float(values[index]):
            print(f"  stopping resume at {path.name}: Re {float(data['reynolds']):.1f} "
                  f"does not match {float(values[index]):.1f}")
            break
        found[index] = data
    return found


def _consolidate(
    parts: Path, out: Path, n: int, values: np.ndarray, lid_velocity: float
) -> Path:
    """Gather the per-frame files into the single archive the renderer reads."""
    frames = [np.load(_frame_path(parts, i)) for i in range(len(values))]
    path = out / f"sweep_n{n}.npz"
    np.savez_compressed(
        path,
        ux=np.stack([f["ux"] for f in frames]),
        uy=np.stack([f["uy"] for f in frames]),
        # rho is needed to restart a solve from a stored frame. Dropping it makes the
        # saved sweep un-resumable and un-auditable: you cannot afterwards ask how
        # converged a frame was, because you cannot reconstruct the solver state.
        rho=np.stack([f["rho"] for f in frames]),
        reynolds=np.array([float(f["reynolds"]) for f in frames]),
        steps=np.array([int(f["steps"]) for f in frames]),
        converged=np.array([bool(f["converged"]) for f in frames]),
        residual=np.array([float(f["residual"]) for f in frames]),
        tau=np.array([float(f["tau"]) for f in frames]),
        solid=_solid(n, lid_velocity, float(values[0])),
        lid_velocity=lid_velocity,
    )
    n_bad = sum(not bool(f["converged"]) for f in frames)
    print(f"\n{len(frames)} frames; {n_bad} hit the step cap")
    return path


def _solid(n: int, lid_velocity: float, reynolds: float) -> np.ndarray:
    """The wall mask is identical for every frame, so store it once."""
    from lms.lbm.solver2d import lid_driven_cavity

    return lid_driven_cavity(
        n=n, reynolds=float(reynolds), lid_velocity=lid_velocity, collision="trt"
    ).solid


if __name__ == "__main__":
    main()
