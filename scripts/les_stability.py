"""Where does TRT actually break, and how much does LES buy?

    python scripts/les_stability.py --n 64

This is the measurement that decides whether Phase 1c needs MRT. Phase 1b was scoped as
"LES only, measure before adding MRT" precisely so this table exists before a second
collision operator is written on the assumption that it is needed.

The lattice relaxation time is tau = 3 nu + 0.5, and fixing the Reynolds number at a
given resolution fixes nu, so pushing Re up drives tau toward 0.5 where LBM becomes
unstable. Eddy viscosity raises the local tau, so LES should extend the range. By how
much is an empirical question, and the answer sets the ceiling on what this solver can
do to a stirred tank before the collision operator has to change.

Reported alongside stability is Re_lambda, because a run that is stable but has
Re_lambda ~ 10 is not turbulent -- it is a viscous flow that happens not to have blown
up, and it would be misleading to count it as a success.
"""

from __future__ import annotations

import argparse

import numpy as np

from lms.lbm.d2q9 import viscosity_to_omega
from lms.lbm.solver3d import D3Q19Solver, init_backend
from lms.validation.turbulence import solenoidal_field, statistics


def attempt(n: int, reynolds: float, smagorinsky: float, u_rms: float,
            k_peak: float, steps: int, seed: int = 0):
    """Run briefly and report survival plus the turbulence level reached."""
    nu = u_rms * n / reynolds
    solver = D3Q19Solver(
        (n, n, n), omega=viscosity_to_omega(nu), solid=np.zeros((n, n, n), dtype=bool),
        collision="trt", smagorinsky=smagorinsky,
    )
    ux, uy, uz = solenoidal_field(n, u_rms=u_rms, k_peak=k_peak, seed=seed)
    solver.set_state(np.ones((n, n, n)), ux, uy, uz)

    re_lambda = float("nan")
    for step in range(steps):
        solver.step()
        if step % 200 == 0 or step == steps - 1:
            _, gx, gy, gz = solver.macroscopic()
            if not np.isfinite(gx).all() or np.abs(gx).max() > 1.0:
                return False, step, re_lambda, nu
            re_lambda = statistics(gx, gy, gz, nu).reynolds_lambda
    return True, steps, re_lambda, nu


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--u-rms", type=float, default=0.05)
    p.add_argument("--k-peak", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--reynolds", type=float, nargs="+",
                   default=[500, 1000, 2000, 4000, 8000, 16000, 32000])
    p.add_argument("--cs", type=float, nargs="+", default=[0.0, 0.1, 0.17])
    args = p.parse_args()

    arch, _ = init_backend(prefer_gpu=True, precision="fp32")
    print(f"TRT stability sweep  {args.n}^3  ({arch} fp32)  {args.steps:,} steps per case")
    print(f"u_rms = {args.u_rms}   k_peak = {args.k_peak:g}\n")

    header = f"{'Re':>8} {'nu':>9} {'tau':>8}"
    for cs in args.cs:
        header += f" {'Cs=' + format(cs, 'g'):>18}"
    print(header)

    for re in args.reynolds:
        nu = args.u_rms * args.n / re
        row = f"{re:>8.0f} {nu:>9.2e} {3 * nu + 0.5:>8.4f}"
        for cs in args.cs:
            ok, step, re_lambda, _ = attempt(
                args.n, re, cs, args.u_rms, args.k_peak, args.steps
            )
            cell = f"Re_l {re_lambda:5.1f}" if ok else f"died @{step:,}"
            row += f" {cell:>18}"
        print(row, flush=True)

    print("\nA case is only useful if it both survives and reaches a Re_lambda worth")
    print("calling turbulent. Stability at Re_lambda ~ 10 is a viscous flow that did not")
    print("blow up, not a turbulence result.")


if __name__ == "__main__":
    main()
