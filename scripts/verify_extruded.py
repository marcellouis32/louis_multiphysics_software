"""Verify the 3D GPU kernel against the verified 2D NumPy solver.

    python scripts/verify_extruded.py --n 48 --steps 6000

Run the lid-driven cavity in 3D on a domain that is periodic and uniform in z. Nothing
drives motion along z, so the solution must be the 2D solution repeated in every plane.
The 2D solver was verified in Phase 0 against Ghia and against an analytic Taylor-Green
decay, which makes this a comparison against a known-correct answer rather than against
another guess.

Two numbers come out of it:

  fp64 on CPU   -- should agree to machine precision. Anything larger is a porting bug,
                   and because the physics is identical there is nowhere else to hide.
  fp32 on Metal -- the price of single precision, which is the number that justifies or
                   refutes running production in fp32. Compare it against the
                   discretisation error (Ghia RMS ~ 0.02); if it is orders of magnitude
                   smaller, fp32 costs nothing that matters.
"""

from __future__ import annotations

import argparse

import numpy as np

from lms.lbm.solver2d import lid_driven_cavity
from lms.lbm.solver3d import init_backend, lid_driven_cavity_3d


def compare(state3d, ux2, uy2, plane: int = 0):
    """2D is indexed (y, x); 3D is indexed (x, y, z). Transpose to line them up."""
    ux3 = state3d.ux[:, :, plane].T
    uy3 = state3d.uy[:, :, plane].T
    scale = float(np.abs(ux2).max()) or 1.0
    return {
        "dux": float(np.abs(ux3 - ux2).max()),
        "duy": float(np.abs(uy3 - uy2).max()),
        "relative": float(np.abs(ux3 - ux2).max() / scale),
        "z_spread": float(np.ptp(state3d.ux, axis=2).max()),
        "uz": float(np.abs(state3d.uz).max()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--nz", type=int, default=4)
    p.add_argument("--reynolds", type=float, default=100.0)
    p.add_argument("--lid-velocity", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=6000)
    args = p.parse_args()

    print(f"cavity {args.n}x{args.n} (x{args.nz} in z)  Re={args.reynolds:g}  "
          f"{args.steps:,} steps")

    oracle = lid_driven_cavity(
        n=args.n, reynolds=args.reynolds, lid_velocity=args.lid_velocity, collision="trt"
    )
    ref = oracle.run(max_steps=args.steps, tol=0.0, check_every=args.steps)
    print(f"2D oracle: |u|max = {np.abs(ref.ux).max():.6f}\n")

    for prefer_gpu, precision in ((False, "fp64"), (True, "fp32")):
        arch, _ = init_backend(prefer_gpu=prefer_gpu, precision=precision)
        solver = lid_driven_cavity_3d(
            n=args.n, nz=args.nz, reynolds=args.reynolds,
            lid_velocity=args.lid_velocity, collision="trt",
        )
        state = solver.run(max_steps=args.steps, tol=0.0, check_every=args.steps)
        r = compare(state, ref.ux, ref.uy)
        print(f"{arch:6s} {precision}")
        print(f"   max |du_x| vs oracle   {r['dux']:.3e}")
        print(f"   max |du_y| vs oracle   {r['duy']:.3e}")
        print(f"   relative               {r['relative']:.3e}")
        print(f"   spread across z        {r['z_spread']:.3e}   (must be ~0)")
        print(f"   max |u_z|              {r['uz']:.3e}   (nothing drives z)\n")

    print("Compare the fp32 figure against the discretisation error: the Phase 0 cavity")
    print("sits at RMS ~0.02 against Ghia, so fp32 noise several orders below that is")
    print("not what limits accuracy.")


if __name__ == "__main__":
    main()
