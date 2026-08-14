"""Close the angular-momentum budget, channel by channel.

    python scripts/audit_ledger.py --n 48 --burn 150 --window 100

For a window of steps, every route by which angular momentum enters or leaves the
resolved fluid is metered:

    dL (from the fields)  =?  -int(T_impeller) - int(T_static) + inject - remove

A residual near zero says the torque meter can be trusted; a large one says momentum
is moving through a channel nobody is watching -- and its size and sign say which.
This instrument exists because the Bouzidi meter read 3-6x the fluid's actual dL/dt
and six rounds of hypothesis-elimination could not localise the discrepancy without
it. The audit runs both boundaries so the trusted halfway case doubles as the control.
"""

from __future__ import annotations

import argparse

import numpy as np

from lms.geometry.tank import tank_from_case
from lms.lbm.d2q9 import viscosity_to_omega
from lms.lbm.solver3d import D3Q19Solver, init_backend
from lms.lbm.units import scales_from_case
from lms.schema.case import load_case


def audit(boundary: str, n: int, burn: int, window: int):
    case = load_case("cases/examples/rushton_standard.yaml").model_copy(deep=True)
    case.numerics.cells_across_tank = n
    scales = scales_from_case(case)
    tank = tank_from_case(case, scales)

    solid = tank.static_solid.astype(np.int32)
    solid[tank.impeller_static_mask()] = 2
    solver = D3Q19Solver(
        tank.shape, omega=viscosity_to_omega(scales.nu), solid=solid,
        collision="regularized", smagorinsky=0.1,
        rotor=tank.rotor_params(boundary=boundary), ledger=True,
    )
    solver.set_wall_velocity(tank.impeller_wall_velocity(tank.impeller_static_mask()))
    solver.set_axis((tank.shape[0] - 1) / 2.0, (tank.shape[1] - 1) / 2.0, 0.0)

    for _ in range(burn):
        solver.step()
    solver.drain_torque_log()
    solver.reset_ledger()
    l0 = solver.angular_momentum_z()

    for _ in range(window):
        solver.step()

    dl = solver.angular_momentum_z() - l0
    # Ring log holds steps 1..window-1 of the window (each logged one step late);
    # the final step's torque is still in the accumulator.
    t_imp = float(solver.drain_torque_log()[1:].sum() + solver.torque()[2])
    led = solver.ledger_report()

    predicted = -t_imp - led["static_torque_z"] + led["inject_Lz"]
    residual = dl - predicted
    scale = max(abs(dl), abs(t_imp), 1e-30)

    print(f"\n{boundary} (n={n}, window={window} steps after {burn} burn-in)")
    print(f"  dL (fields)          {dl:+.4e}")
    print(f"  -int T_impeller      {-t_imp:+.4e}")
    print(f"  -int T_static        {-led['static_torque_z']:+.4e}")
    print(f"  +inject (refills)    {led['inject_Lz']:+.4e}")
    print(f"  predicted dL         {predicted:+.4e}")
    print(f"  RESIDUAL             {residual:+.4e}   ({abs(residual) / scale:.1%} of scale)")
    return residual, scale


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--burn", type=int, default=150)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--boundary", choices=["halfway", "bouzidi", "both"], default="both")
    args = p.parse_args()

    init_backend(prefer_gpu=True, precision="fp32")
    kinds = ["halfway", "bouzidi"] if args.boundary == "both" else [args.boundary]
    for boundary in kinds:
        audit(boundary, args.n, args.burn, args.window)


if __name__ == "__main__":
    main()
