"""Taylor-Couette resolution ladder: the go/no-go measurement for Phase 2b.

    python scripts/validate_couette.py

Staircase curved walls and momentum-exchange torque are the two numerical ingredients
Phase 2 adds, and the power number depends on both. This ladder scores them against the
closed-form Couette solution at fixed Taylor number, so the trend across rungs is pure
numerics.

The decision this feeds: if the torque error at Rushton-relevant resolution does not
leave room inside the 10% Np budget, Bouzidi interpolated bounce-back happens *before*
the impeller is built, not after a wrong power number.
"""

from __future__ import annotations

import argparse
import math
import time

from lms.validation.couette import ladder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--radii", type=float, nargs="+", default=[15.0, 30.0, 60.0],
                   help="outer cylinder radius in cells, per rung")
    p.add_argument("--u0", type=float, default=0.03,
                   help="wall speed at the coarsest rung; scaled 1/R after that")
    p.add_argument("--cpu", action="store_true", help="fp64 CPU instead of fp32 Metal")
    args = p.parse_args()

    t0 = time.perf_counter()
    results = ladder(tuple(args.radii), u0=args.u0, prefer_gpu=not args.cpu)

    print(f"\n{'R1':>6} {'R2':>6} {'u_wall':>8} {'Ta':>6} {'steps':>9} "
          f"{'profile':>9} {'order':>6} {'torque':>9} {'order':>6}")
    prev = None
    for r in results:
        po = to = ""
        if prev is not None:
            ratio = math.log(r.r_outer / prev.r_outer)
            po = f"{math.log(prev.profile_error / r.profile_error) / ratio:6.2f}"
            to = f"{math.log(prev.torque_error / r.torque_error) / ratio:6.2f}"
        print(f"{r.r_inner:>6.1f} {r.r_outer:>6.1f} {r.u_wall:>8.4f} {r.taylor:>6.0f} "
              f"{r.steps:>9,} {r.profile_error:>9.4f} {po:>6} "
              f"{r.torque_error:>9.4f} {to:>6}")

    print(f"\n{time.perf_counter() - t0:.0f}s")
    worst = max(r.torque_error for r in results)
    print(f"worst torque error on the ladder: {worst * 100:.2f}%")
    print("Rushton context: the impeller tip radius at the milestone resolution is")
    print("~21 cells, between the first two rungs. The go/no-go is whether the torque")
    print("error there leaves room inside the 10% Np budget; if not, Bouzidi happens")
    print("before the impeller does.")


if __name__ == "__main__":
    main()
