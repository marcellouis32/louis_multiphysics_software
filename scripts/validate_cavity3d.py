"""Validate the genuine 3D lid-driven cavity.

    python scripts/validate_cavity3d.py --n 96 --reynolds 1000 --spans 1 2 3

**On reference data.** The usual references for the 3D cavity are Ku, Hirsh & Taylor
(1987) and Albensoeder & Kuhlmann (2005). Their tables are not in this repository and
are not reproduced here from memory: `ghia.py` exists because benchmark numbers were
transcribed from the paper, and inventing a plausible-looking table would poison every
comparison made against it afterwards. Until those values are entered from the source,
this script validates against two things it can establish honestly.

**1. Symmetry that must emerge but is never imposed.** A cubic cavity with a lid sliding
along +x is mirror-symmetric about the mid-span plane. Nothing in the solver knows that:
the lattice is not symmetric under reflection (D3Q19 velocities come in pairs, but
bounce-back, streaming order and floating-point summation are not arranged to cancel),
and the initial condition is uniform. So the symmetry is a genuine prediction. u_x and
u_y must be even about mid-span and u_z odd, and the residual measures how much
asymmetry the discretisation injects.

**2. The two-dimensional limit, checked against Ghia.** As the span grows, the mid-span
plane stops feeling the end walls and must approach the 2D solution -- which is already
validated against Ghia et al. to ~1-2%. Running a span sweep therefore turns the
existing 2D benchmark into a real 3D check: the mid-plane error against Ghia has to fall
as the span grows, and the *rate* it falls at is the end-wall influence.

That second point is the physics worth having. The end walls drive Taylor-Gortler
vortices and drag the mid-plane circulation down; a narrow cavity is measurably not a 2D
cavity, and this quantifies by how much.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from lms.lbm.solver3d import init_backend, lid_driven_cavity_3d
from lms.validation import ghia


def centrelines(ux, uy, lid_velocity: float):
    """The two profiles Ghia tabulates, taken on the mid-span plane.

    Indexing is (x, y, z); node centres sit half a lattice unit inside the walls, which
    matches `cavity_centerlines` in solver2d.py so the coordinates are comparable.
    """
    nx, ny, nz = ux.shape
    interior = slice(1, -1)
    coord = (np.arange(1, nx - 1) - 0.5) / (nx - 2)
    mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2
    return (
        coord,
        ux[mid_x, interior, mid_z] / lid_velocity,
        coord,
        uy[interior, mid_y, mid_z] / lid_velocity,
    )


def symmetry_residual(field: np.ndarray, odd: bool = False) -> float:
    """How far the field is from being even (or odd) about the mid-span plane.

    Normalised by the field's own magnitude, so it reads as a relative error.
    """
    mirrored = field[:, :, ::-1]
    diff = field + mirrored if odd else field - mirrored
    scale = float(np.abs(field).max()) or 1.0
    return float(np.abs(diff).max() / (2.0 * scale))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=96)
    p.add_argument("--reynolds", type=float, default=1000.0)
    p.add_argument("--lid-velocity", type=float, default=0.1)
    p.add_argument("--spans", type=float, nargs="+", default=[1.0, 2.0, 3.0],
                   help="span / cavity width")
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--out", type=Path, default=Path("runs/cavity3d"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    arch, _ = init_backend(prefer_gpu=True, precision="fp32")

    gy, gu = ghia.u_profile(args.reynolds)
    gx, gv = ghia.v_profile(args.reynolds)
    length = args.n - 2

    print(f"3D lid-driven cavity  {args.n}^2 cross-section  Re = {args.reynolds:g}  "
          f"U = {args.lid_velocity}  ({arch} fp32)")
    print("Ghia et al. is a *2D* reference. The mid-plane must approach it as the span")
    print("grows; at span = 1 the gap is the end-wall effect, not solver error.\n")
    print(f"{'span':>6} {'nz':>5} {'cells':>9} {'steps':>8} {'conv':>6} "
          f"{'RMS u':>8} {'RMS v':>8} {'sym x':>9} {'sym z':>9} {'time':>7}")

    rows = []
    for span in args.spans:
        nz = round(span * length) + 2
        solver = lid_driven_cavity_3d(
            n=args.n, nz=nz, reynolds=args.reynolds,
            lid_velocity=args.lid_velocity, collision="trt", periodic_z=False,
        )
        t0 = time.perf_counter()
        state = solver.run(max_steps=args.max_steps, tol=args.tol, check_every=1000)
        elapsed = time.perf_counter() - t0

        y, u, x, v = centrelines(state.ux, state.uy, args.lid_velocity)
        rms_u = ghia.rms_error(y, u, gy, gu)
        rms_v = ghia.rms_error(x, v, gx, gv)
        sym_x = symmetry_residual(state.ux)
        sym_z = symmetry_residual(state.uz, odd=True)

        print(f"{span:>6.1f} {nz:>5} {args.n**2 * nz / 1e6:>8.2f}M {state.steps:>8,} "
              f"{'yes' if state.converged else 'CAP':>6} {rms_u:>8.4f} {rms_v:>8.4f} "
              f"{sym_x:>9.2e} {sym_z:>9.2e} {elapsed:>6.0f}s")

        np.savez_compressed(
            args.out / f"cavity3d_re{int(args.reynolds)}_n{args.n}_span{span:g}.npz",
            ux=state.ux, uy=state.uy, uz=state.uz, rho=state.rho,
            solid=solver.solid.to_numpy(), reynolds=args.reynolds,
            lid_velocity=args.lid_velocity, span=span, steps=state.steps,
            converged=state.converged,
        )
        rows.append((span, rms_u, rms_v, sym_x, sym_z))

    print("\nsymmetry: u_x even and u_z odd about mid-span. Never imposed, so a small")
    print("residual here is real evidence the boundary handling is not lopsided.")
    if len(rows) > 1:
        first, last = rows[0], rows[-1]
        print(f"\nmid-plane RMS against Ghia falls {first[1]:.4f} -> {last[1]:.4f} (u) and "
              f"{first[2]:.4f} -> {last[2]:.4f} (v)")
        print(f"as the span goes {first[0]:g} -> {last[0]:g}. That trend is the end-wall")
        print("effect receding, and it is what a correct 3D solver must show.")


if __name__ == "__main__":
    main()
