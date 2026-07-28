"""Measure the solver's order of accuracy on the Taylor-Green vortex.

Runs the same physical problem twice, once under each refinement scaling, to show
that the scaling choice - not the solver - is what sets the measured order.

    python scripts/convergence_study.py --sizes 32 64 128
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from lms.validation.taylor_green import acoustic_ladder, diffusive_ladder, observed_order
from lms.viz.fields2d import plot_order_of_accuracy


def _report(name: str, results) -> float:
    order = observed_order(results)
    print(f"\n{name}")
    print(f"  {'n':>5} {'U':>8} {'nu':>8} {'tau':>7} {'Ma':>7} {'steps':>7} {'L2 error':>11}")
    for r in results:
        tau = 3.0 * r.nu + 0.5
        print(
            f"  {r.n:5d} {r.u0:8.4f} {r.nu:8.4f} {tau:7.4f} {r.mach:7.4f} "
            f"{r.steps:7d} {r.error:11.3e}"
        )
    for a, b in zip(results, results[1:]):
        local = -math.log(b.error / a.error) / math.log(b.n / a.n)
        print(f"  {a.n:>3d} -> {b.n:<3d}  local order {local:.2f}")
    print(f"  fitted order {order:.2f}")
    return order


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--out", type=Path, default=Path("runs/convergence"))
    args = p.parse_args()

    sizes = tuple(sorted(args.sizes))
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    diffusive = diffusive_ladder(sizes)
    d_order = _report("diffusive scaling   dt ~ dx^2,  U ~ 1/n,  tau fixed", diffusive)

    acoustic = acoustic_ladder(sizes)
    a_order = _report("acoustic scaling    dt ~ dx,    U fixed,  tau ~ n", acoustic)

    print(f"\ntotal {time.time() - t0:.1f}s")

    re = diffusive[0].reynolds
    plot_order_of_accuracy(
        {
            "diffusive  ($U \\propto 1/n$)": (
                [r.n for r in diffusive], [r.error for r in diffusive], d_order,
            ),
            "acoustic  ($U$ fixed)": (
                [r.n for r in acoustic], [r.error for r in acoustic], a_order,
            ),
        },
        title="Taylor-Green vortex — order of accuracy",
        subtitle=f"Re = {re:g}   ·   TRT, $\\Lambda = 1/4$   ·   exact solution, "
                 f"no benchmark ceiling",
        save=args.out / "order_of_accuracy.png",
    )
    print(f"figure written to {args.out}/order_of_accuracy.png")


if __name__ == "__main__":
    main()
