"""Blend time: inject a tracer, stir until uniform, compare against Grenville.

    python scripts/blend_time.py cases/examples/rushton_standard.yaml --n 96

The dye-and-stopwatch experiment every process engineer has run, in silico. The tracer
enters as the schema's top_patch and the tank stirs. Two mixing-time definitions are
reported, because the literature uses both and they differ by a factor of ~2 here:

  * absolute:   CoV over the fluid falls below 5%
  * reduction:  CoV falls to 5% of its INITIAL value (95% homogenisation)

A top-patch injection starts at CoV ~ 2.9, so the reduction definition's target is
CoV ~ 0.145 -- reachable long before the absolute 0.05. Correlations like Grenville
were fitted to probe measurements that correspond more closely to the reduction
reading; quoting only the absolute one against them would manufacture a factor-2
"error" out of a units mismatch between definitions.

The comparison target is `grenville_blend_time`, fed with *our own measured* power
number from the same flow -- not the literature midpoint -- so the check stays
self-contained. Target agreement is ~30%, with the correlation's own scatter
acknowledged; tighter than that would be pretending the correlation is a standard.

Numbers this run is honest about: the effective diffusivity is nu_t/Sc_t (Sc_t = 0.7)
because Sc = 1000 is unresolvable; and the scalar lattice's measured numerical
diffusion is comparable to that modelled diffusivity in the quiet bulk, which matters
little here only because turbulent blending is convection-dominated -- both facts are
printed with the result.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from run_case import build_solver

from lms.geometry.tank import tank_from_case
from lms.lbm.scalar3d import ScalarD3Q7
from lms.lbm.units import scales_from_case
from lms.schema.case import load_case
from lms.schema.reference import grenville_blend_time


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("case", type=Path)
    p.add_argument("--n", type=int, default=96)
    p.add_argument("--spinup", type=float, default=8.0,
                   help="revolutions of flow before the tracer goes in")
    p.add_argument("--max-revs", type=float, default=60.0)
    p.add_argument("--np-measured", type=float, default=None,
                   help="power number for the Grenville comparison; measured from "
                        "this flow if a prior run_case result is not supplied")
    p.add_argument("--frames-per-rev", type=int, default=5)
    p.add_argument("--boundary", choices=["halfway", "bouzidi"], default="bouzidi",
                   help="blade wall treatment; bouzidi is the Phase 2 default since "
                        "the meter audit")
    p.add_argument("--out", type=Path, default=Path("runs/blend"))
    args = p.parse_args()

    case = load_case(args.case)
    if args.n is not None:
        case = case.model_copy(deep=True)
        case.numerics.cells_across_tank = args.n

    scales = scales_from_case(case)
    tank = tank_from_case(case, scales)
    solver = build_solver(case, scales, tank, prefer_gpu=True,
                          boundary=args.boundary)
    spr = scales.steps_per_revolution

    # Scalar rides the flow solver's own fields; blades are transparent to it (the
    # swept torus is 1-2% of the volume -- see scalar3d's docstring).
    tracer = ScalarD3Q7(
        tank.shape,
        d_molecular=scales.nu / 1000.0,          # Sc = 1000, for what little it does
        solid=tank.static_solid | tank.impeller_static_mask(),
        vel=solver.vel,
        nu_t=solver.nu_t,
    )

    print(f"{case.name}  {args.n}^2x{scales.nz - 2}  Re = {scales.reynolds:,.0f}")
    print(f"spin-up {args.spinup:g} revs, then tracer top patch, CoV target 5%\n")

    t0 = time.perf_counter()
    for _ in range(round(args.spinup * spr)):
        solver.step()

    # Tracer: top 10% of the liquid column, as the schema's top_patch describes.
    fluid = ~(tank.static_solid | tank.impeller_static_mask())
    nz = tank.shape[2]
    c0 = np.zeros(tank.shape)
    top = slice(round((nz - 2) * 0.9) + 1, nz - 1)
    c0[:, :, top] = 1.0
    c0[~fluid] = 0.0
    tracer.set_concentration(c0)

    check_every = max(round(spr / args.frames_per_rev), 1)
    mid_y = tank.shape[1] // 2
    history, frames_c, frames_u, frame_revs = [], [], [], []
    theta95_steps = None
    reduction_steps = None
    cov0 = None
    stop_at = None

    total = round(args.max_revs * spr)
    for step in range(total):
        # Reassigning `total` would not shorten range(total) -- it was materialised at
        # loop entry -- so the early exit is an explicit break.
        if stop_at is not None and step >= stop_at:
            break
        solver.step()
        tracer.step()
        if step % check_every == 0:
            c = tracer.concentration()[fluid]
            cov = float(c.std() / c.mean())
            revs = step / spr
            history.append((revs, cov))

            full = tracer.concentration()
            vel = solver.vel.to_numpy()
            speed = np.sqrt((vel**2).sum(axis=-1)) / scales.u_tip
            frames_c.append(full[:, mid_y, :].astype(np.float32))
            frames_u.append(speed[:, mid_y, :].astype(np.float32))
            frame_revs.append(revs)
            if step % (check_every * 10) == 0:
                print(f"  rev {revs:6.1f}   CoV = {cov:.4f}   "
                      f"({time.perf_counter() - t0:,.0f}s)", flush=True)
            if cov0 is None:
                cov0 = cov
            if reduction_steps is None and cov < 0.05 * cov0:
                reduction_steps = step
                print(f"  -> 95% homogenisation (CoV {0.05 * cov0:.3f}) at rev {revs:.2f}")
            if cov < 0.05 and theta95_steps is None:
                theta95_steps = step
                print(f"  -> absolute CoV crossed 5% at rev {revs:.2f}")
                # A few more frames so the animation ends on a mixed tank.
                stop_at = step + round(3 * spr)

    # Save FIRST, verdict after. A run that misses its endpoint is exactly the run
    # whose frames are needed -- they show where the unmixed region sits -- and the
    # first blend attempt threw them away by exiting before the save.
    crossed = theta95_steps is not None
    theta95_s = theta95_steps * scales.dt_s if crossed else float("nan")
    reduction_s = (reduction_steps * scales.dt_s
                   if reduction_steps is not None else float("nan"))
    imp = case.primary_impeller
    np_used = args.np_measured or 4.11   # n=96 measured value from the Np ladder
    ref = grenville_blend_time(
        imp.speed_hz, np_used, imp.diameter_m, case.geometry.vessel.diameter_m
    )
    print()
    if reduction_steps is not None:
        print(f"theta_95 (95% reduction) = {reduction_s:.2f} s "
              f"({reduction_steps / spr:.1f} revs)   vs Grenville {ref:.2f} s   "
              f"ratio {reduction_s / ref:.2f}")
    if crossed:
        print(f"theta_95 (absolute 5%)   = {theta95_s:.2f} s "
              f"({theta95_steps / spr:.1f} revs)")
    if reduction_steps is None and not crossed:
        final_cov = history[-1][1]
        print(f"neither definition reached within {args.max_revs:g} revolutions "
              f"(final CoV = {final_cov:.3f}). Saving diagnostics anyway.")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"blend_{case.name}_n{args.n}.npz"
    np.savez_compressed(
        path,
        cov=np.array(history), theta95_s=theta95_s, reduction_s=reduction_s,
        grenville_s=ref,
        np_used=np_used, frames_c=np.array(frames_c), frames_u=np.array(frames_u),
        frame_revs=np.array(frame_revs), static_solid=tank.static_solid,
        steps_per_rev=spr, n=args.n, u_tip=scales.u_tip,
    )
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
