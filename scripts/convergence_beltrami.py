"""Order-of-accuracy study for the 3D solver, on an exact Beltrami solution.

    python scripts/convergence_beltrami.py --sizes 24 32 48 64 96

The 3D counterpart of `convergence_study.py`. Refinement is diffusive -- lattice
velocity falls like 1/n while tau and Reynolds number stay fixed -- because Phase 0
established in 2D that acoustic refinement pins the Mach number, leaves an O(Ma^2)
compressibility floor that refinement cannot cross, and reports order ~1.1 for a scheme
that is genuinely second order.

The initial state includes the first-order Chapman-Enskog non-equilibrium part. Starting
from equilibrium alone is 25x less accurate after a single step and, worse, leaves two
competing error sources so the measured order wanders instead of converging.
"""

from __future__ import annotations

import argparse
import math
import time

from lms.validation.beltrami import diffusive_ladder, observed_order


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[24, 32, 48, 64])
    p.add_argument("--u0", type=float, default=0.05)
    p.add_argument("--nu", type=float, default=0.01)
    p.add_argument("--steps0", type=int, default=900)
    p.add_argument("--gpu", action="store_true", help="fp32 on Metal instead of fp64 on CPU")
    p.add_argument(
        "--equilibrium-only", action="store_true",
        help="drop the non-equilibrium initial state, to show what it costs",
    )
    args = p.parse_args()

    sizes = tuple(sorted(args.sizes))
    mode = "metal fp32" if args.gpu else "cpu fp64"
    print(f"Beltrami (ABC) flow, diffusive refinement, {mode}")
    print(f"u0 = {args.u0} at n = {sizes[0]}, nu = {args.nu}, "
          f"{'equilibrium-only start' if args.equilibrium_only else 'with f_neq start'}\n")

    t0 = time.perf_counter()
    results = diffusive_ladder(
        sizes=sizes, u0=args.u0, nu=args.nu, steps0=args.steps0,
        prefer_gpu=args.gpu, precision="fp32" if args.gpu else "fp64",
        equilibrium_only=args.equilibrium_only,
    )

    print(f"{'n':>5} {'tau':>7} {'Ma':>8} {'Re':>8} {'steps':>8} {'L2 error':>12} {'order':>7}")
    previous = None
    for r in results:
        tau = 3.0 * r.nu + 0.5
        order = ""
        if previous is not None:
            local = math.log(previous.error / r.error) / math.log(r.n / previous.n)
            order = f"{local:7.2f}"
        print(f"{r.n:>5} {tau:>7.4f} {r.mach:>8.4f} {r.reynolds:>8.1f} "
              f"{r.steps:>8,} {r.error:>12.4e} {order:>7}")
        previous = r

    print(f"\nfitted order over all sizes      {observed_order(results):.3f}")
    if len(results) > 2:
        print(f"fitted over the finest three     {observed_order(results[-3:]):.3f}")
    print(f"\n{time.perf_counter() - t0:.0f}s")
    print("\nLocal order should approach 2 from above as the coarse-grid terms drop out.")
    print("tau and Re must be identical on every row -- that is what makes this diffusive")
    print("rather than acoustic refinement, and it is the whole reason the order is")
    print("measurable at all.")


if __name__ == "__main__":
    main()
