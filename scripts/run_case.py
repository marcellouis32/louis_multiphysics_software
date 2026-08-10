"""Run a stirred-tank case from YAML to a power number.

    python scripts/run_case.py cases/examples/rushton_standard.yaml

The first end-to-end schema-to-numbers path in the project: the Case that the eventual
AI setup agent will fill, driven through voxelisation, unit conversion, the rotating
impeller, and out to the quantity a process engineer actually asks for.

    Np = 2 pi <T> / (rho N^2 D^5)        (all in lattice units, rho = 1)

The torque history is saved whole, not just its mean. It is the diagnostic for two
separate things: fresh-node refill artifacts (spikes), and the staircase blade-volume
oscillation (periodic ripple at the blade-passing frequency). An Np quoted without the
history behind it would hide both.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from lms.geometry.tank import tank_from_case
from lms.lbm.d2q9 import viscosity_to_omega
from lms.lbm.solver3d import D3Q19Solver, init_backend
from lms.lbm.units import scales_from_case
from lms.schema.case import load_case
from lms.schema.reference import REFERENCE_POWER_NUMBER


def build_solver(case, scales, tank, prefer_gpu: bool = True) -> D3Q19Solver:
    init_backend(prefer_gpu=prefer_gpu, precision="fp32" if prefer_gpu else "fp64")

    solid = tank.static_solid.astype(np.int32)
    # The axisymmetric impeller parts are static solids, measured (2). Blades are the
    # rotor, evaluated in-kernel.
    imp_static = tank.impeller_static_mask()
    solid[imp_static] = 2

    turb = case.physics.momentum.turbulence
    smagorinsky = turb.smagorinsky_constant if turb.model != "none" else 0.0

    solver = D3Q19Solver(
        tank.shape,
        omega=viscosity_to_omega(scales.nu),
        solid=solid,
        collision="regularized",
        smagorinsky=smagorinsky,
        rotor=tank.rotor_params(),
    )
    solver.set_wall_velocity(tank.impeller_wall_velocity(imp_static))
    solver.set_axis((tank.shape[0] - 1) / 2.0, (tank.shape[1] - 1) / 2.0, 0.0)
    return solver


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("case", type=Path)
    p.add_argument("--n", type=int, default=None, help="override cells_across_tank")
    p.add_argument("--spinup", type=float, default=10.0, help="revolutions discarded")
    p.add_argument("--average", type=float, default=30.0, help="revolutions averaged")
    p.add_argument("--frames-per-rev", type=int, default=10)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", type=Path, default=Path("runs/tank"))
    args = p.parse_args()

    case = load_case(args.case)
    if args.n is not None:
        case = case.model_copy(deep=True)
        case.numerics.cells_across_tank = args.n

    scales = scales_from_case(case)
    tank = tank_from_case(case, scales)
    solver = build_solver(case, scales, tank, prefer_gpu=not args.cpu)

    spr = scales.steps_per_revolution
    n_spin = round(args.spinup * spr)
    n_avg = round(args.average * spr)
    total = n_spin + n_avg
    frame_every = max(round(spr / args.frames_per_rev), 1)

    print(f"{case.name}   {case.numerics.cells_across_tank}^2 x "
          f"{scales.nz - 2} cells   Re = {scales.reynolds:,.0f}   "
          f"tau = {scales.tau:.6f}   Cs = {solver.smagorinsky:g}")
    print(f"{spr:,.0f} steps/rev   spin-up {args.spinup:g} revs, average "
          f"{args.average:g} revs -> {total:,} steps\n")

    mid_y = tank.shape[1] // 2
    z_imp = round(tank.impeller["z_centre"])
    # Torque is logged on the device and drained in chunks: reading it back every
    # step costs a pipeline sync per step, measured at 6x total slowdown.
    torque_chunks = []
    frames_rz, frames_xy, frame_steps = [], [], []

    t0 = time.perf_counter()
    for step in range(total):
        solver.step()
        if step % 20_000 == 0 and step > 0:
            torque_chunks.append(solver.drain_torque_log())

        if step % frame_every == 0:
            vel = solver.vel.to_numpy()
            speed = np.sqrt((vel**2).sum(axis=-1)) / scales.u_tip
            if not np.isfinite(speed).any() or speed.max() > 5.0:
                raise FloatingPointError(f"diverged at step {step:,}")
            frames_rz.append(speed[:, mid_y, :].astype(np.float32))
            frames_xy.append(speed[:, :, z_imp].astype(np.float32))
            frame_steps.append(step)
            if step % (frame_every * 20) == 0:
                revs = step / spr
                print(f"  rev {revs:6.1f}   ({time.perf_counter() - t0:,.0f}s)",
                      flush=True)

    torque_chunks.append(solver.drain_torque_log())
    # The log records each step's torque one step late, so entry 0 is the pre-run
    # zero and the final step's value is still in the accumulator.
    torque = np.concatenate(torque_chunks)[1:]
    torque = np.append(torque, solver.torque()[2])

    elapsed = time.perf_counter() - t0
    mlups = tank.shape[0] * tank.shape[1] * tank.shape[2] * total / elapsed / 1e6

    # Power number, from the averaging window only.
    t_avg = np.abs(torque[n_spin:]).mean()
    n_lat = scales.omega_shaft / (2.0 * np.pi)
    d_lat = 2.0 * scales.impeller_radius
    np_measured = 2.0 * np.pi * t_avg / (n_lat**2 * d_lat**5)

    imp_type = case.primary_impeller.type
    band = REFERENCE_POWER_NUMBER.get(imp_type)
    print(f"\n<|T|> over the averaging window = {t_avg:.4e}  "
          f"(ripple: std/mean = {np.abs(torque[n_spin:]).std() / t_avg:.3f})")
    print(f"power number Np = {np_measured:.2f}")
    if band:
        lo, hi = band
        inside = lo <= np_measured <= hi
        within10 = 0.9 * lo <= np_measured <= 1.1 * hi
        verdict = ("inside the published band" if inside
                   else "within 10% of the band" if within10 else "OUTSIDE the band")
        print(f"published {imp_type.value}: {lo}-{hi}  ->  {verdict}")
    print(f"{elapsed:,.0f}s at {mlups:,.0f} MLUPS")

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{case.name}_n{case.numerics.cells_across_tank}"
    path = args.out / f"tank_{tag}.npz"
    vel = solver.vel.to_numpy()
    np.savez_compressed(
        path,
        torque=torque, spinup_steps=n_spin, steps_per_rev=spr,
        np_measured=np_measured, reynolds=scales.reynolds,
        u_tip=scales.u_tip, omega_shaft=scales.omega_shaft,
        frames_rz=np.array(frames_rz), frames_xy=np.array(frames_xy),
        frame_steps=np.array(frame_steps),
        static_solid=tank.static_solid, z_impeller=z_imp,
        ux=vel[..., 0].astype(np.float32), uy=vel[..., 1].astype(np.float32),
        uz=vel[..., 2].astype(np.float32),
        theta=solver.theta, n=case.numerics.cells_across_tank,
    )
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
