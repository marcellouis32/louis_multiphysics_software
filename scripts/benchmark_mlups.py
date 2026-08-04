"""Measure solver throughput in MLUPS, against the machine's bandwidth ceiling.

    python scripts/benchmark_mlups.py --sizes 64 128 192

LBM is memory-bandwidth-bound, so a raw MLUPS figure means little on its own. D3Q19
with double buffering moves 19 reads + 19 writes per cell per step -- 152 bytes in fp32
-- so the achievable rate follows directly from memory bandwidth. Reporting the
percentage of peak says whether the kernel is good or the machine is small.

Apple M4 Pro peak is ~273 GB/s, giving a hard ceiling near 1.8 GLUPS. A spike measuring
pure D3Q19 streaming with no collision reached ~500-600 MLUPS, so that is the number
this has to be read against: the gap between it and the figures here is what collision
costs.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from lms.lbm import d3q19
from lms.lbm.solver3d import D3Q19Solver, init_backend

PEAK_GB_S = 273.0


def benchmark(n: int, prefer_gpu: bool, precision: str, steps: int, collision: str):
    init_backend(prefer_gpu=prefer_gpu, precision=precision)
    solid = np.zeros((n, n, n), dtype=bool)
    solid[0, :, :] = solid[-1, :, :] = True
    solid[:, 0, :] = solid[:, -1, :] = True

    solver = D3Q19Solver((n, n, n), omega=1.8, solid=solid, collision=collision)
    lid = np.zeros((n, n, n), dtype=bool)
    lid[:, -1, :] = True
    solver.set_moving_wall(lid, ux=0.05)

    solver.step()          # compile before timing
    _sync(solver)

    t0 = time.perf_counter()
    for _ in range(steps):
        solver.step()
    _sync(solver)
    dt = time.perf_counter() - t0

    cells = n**3
    mlups = cells * steps / dt / 1e6
    width = 4 if precision == "fp32" else 8
    gb_s = 2 * d3q19.Q * cells * steps * width / dt / 1e9
    return mlups, gb_s


def _sync(solver) -> None:
    """Taichi launches are asynchronous; without this the timer measures queueing."""
    solver.ti.sync()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 192])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--collision", choices=["bgk", "trt"], default="trt")
    p.add_argument("--cpu", action="store_true", help="also benchmark the fp64 CPU path")
    args = p.parse_args()

    print(f"D3Q19 {args.collision.upper()}, {args.steps} steps per case")
    print(f"peak bandwidth {PEAK_GB_S:.0f} GB/s  ->  ceiling "
          f"{PEAK_GB_S * 1e9 / (2 * d3q19.Q * 4) / 1e6:,.0f} MLUPS in fp32\n")
    print(f"{'grid':>8} {'backend':>12} {'MLUPS':>9} {'GB/s':>9} {'% peak':>8} {'memory':>9}")

    runs = [(True, "fp32")] + ([(False, "fp64")] if args.cpu else [])
    for n in args.sizes:
        for prefer_gpu, precision in runs:
            width = 4 if precision == "fp32" else 8
            mem = 2 * d3q19.Q * n**3 * width / 1e9
            try:
                mlups, gb_s = benchmark(n, prefer_gpu, precision, args.steps, args.collision)
                label = f"{'metal' if prefer_gpu else 'cpu'} {precision}"
                print(f"{n:>6}^3 {label:>12} {mlups:>9,.0f} {gb_s:>9.1f} "
                      f"{gb_s / PEAK_GB_S * 100:>7.1f}% {mem:>8.2f}G")
            except Exception as exc:  # noqa: BLE001
                print(f"{n:>6}^3 {precision:>12}   FAILED: {type(exc).__name__}: "
                      f"{str(exc)[:50]}")


if __name__ == "__main__":
    main()
