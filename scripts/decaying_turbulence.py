"""Decaying isotropic turbulence: the LES validation case that needs no external tables.

    # resolved reference, ~10 min
    python scripts/decaying_turbulence.py --n 256 --turnovers 4

    # coarse LES of the same physical flow, seconds
    python scripts/decaying_turbulence.py --n 64 --init-n 256 --smagorinsky 0.1 --turnovers 4

    # the same coarse grid with no model, for contrast
    python scripts/decaying_turbulence.py --n 64 --init-n 256 --smagorinsky 0 --turnovers 4

**Matching two resolutions of one physical flow.** The box is the same physical size at
every resolution, so shell index k means the same physical scale on both grids and the
spectra are directly comparable. Lattice velocity is held fixed (acoustic scaling), which
keeps the Mach number -- and therefore the compressibility error -- the same on both
grids, so a difference between them is the subgrid model rather than a different amount
of artificial compressibility. Fixing the Reynolds number then forces

    nu = u_rms * n / Re          and       steps per turnover = n / u_rms

so the coarse grid runs at a smaller nu, a smaller tau, and fewer steps. That is the
point: the coarse grid is the one that cannot resolve the dissipation and therefore the
one that needs a model.

`--init-n` generates the initial field at the fine resolution and applies a sharp
spectral filter down to the run resolution, so the coarse run starts from exactly the
field the fine run contains on the scales they share -- not from an independently drawn
random field that merely has the same statistics.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from lms.lbm.d2q9 import viscosity_to_omega
from lms.lbm.solver3d import D3Q19Solver, init_backend
from lms.validation.turbulence import (
    coarsen_spectral,
    energy_spectrum,
    enstrophy,
    solenoidal_field,
    statistics,
    vorticity,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--init-n", type=int, default=None,
                   help="generate the field at this size, then spectrally filter to --n")
    p.add_argument("--reynolds", type=float, default=1280.0, help="u_rms * n / nu")
    p.add_argument("--u-rms", type=float, default=0.05)
    p.add_argument("--k-peak", type=float, default=4.0)
    p.add_argument("--smagorinsky", type=float, default=0.0)
    p.add_argument("--turnovers", type=float, default=4.0)
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", type=Path, default=Path("runs/turbulence"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    n = args.n
    nu = args.u_rms * n / args.reynolds
    tau = 3.0 * nu + 0.5
    turnover = n / args.u_rms
    total_steps = round(args.turnovers * turnover)
    every = max(total_steps // args.samples, 1)

    arch, _ = init_backend(prefer_gpu=not args.cpu, precision="fp32" if not args.cpu else "fp64")
    tag = f"n{n}_re{int(args.reynolds)}_cs{args.smagorinsky:g}"

    print(f"decaying isotropic turbulence  {n}^3  ({arch})")
    print(f"  Re = {args.reynolds:g}   nu = {nu:.5f}   tau = {tau:.4f}   "
          f"Cs = {args.smagorinsky:g}")
    print(f"  u_rms = {args.u_rms}   k_peak = {args.k_peak:g}   "
          f"turnover = {turnover:,.0f} steps")
    if tau < 0.51:
        print(f"  NOTE tau = {tau:.4f} is very close to 0.5; without a model this is "
              "expected to be unstable")

    # Initial field, optionally filtered down from a finer draw so that two resolutions
    # start from the same physical field rather than merely the same statistics.
    source_n = args.init_n or n
    ux, uy, uz = solenoidal_field(source_n, u_rms=args.u_rms, k_peak=args.k_peak,
                                  seed=args.seed)
    if source_n != n:
        ux, uy, uz = (coarsen_spectral(c, n) for c in (ux, uy, uz))
        # Filtering removes energy above the coarse Nyquist, so u_rms drops. That is
        # correct -- the filtered field genuinely has less energy -- and is reported
        # rather than renormalised away.
        kept = np.sqrt(np.mean(ux**2 + uy**2 + uz**2) / 3.0)
        print(f"  filtered {source_n}^3 -> {n}^3: u_rms {args.u_rms:.5f} -> {kept:.5f} "
              f"({(kept / args.u_rms) ** 2 * 100:.1f}% of the energy retained)")

    rho = np.ones((n, n, n))
    solver = D3Q19Solver(
        (n, n, n), omega=viscosity_to_omega(nu), solid=np.zeros((n, n, n), dtype=bool),
        collision="trt", smagorinsky=args.smagorinsky,
    )
    solver.set_state(rho, ux, uy, uz)

    # Mid-plane slices per sample, for the animation. Stored as 2D slices rather than
    # whole volumes: 25 snapshots of a 256^3 field would be 5 GB, while the slices are a
    # few MB and are all the animation ever shows. Vorticity is computed from the full
    # 3D field before slicing -- a single plane cannot give the out-of-plane derivatives.
    spectra, stats, sample_steps, nu_t_mean = [], [], [], []
    snap_enstrophy, snap_wz = [], []
    mid = n // 2
    t0 = time.perf_counter()
    for step in range(total_steps + 1):
        if step % every == 0:
            _, gx, gy, gz = solver.macroscopic()
            if not np.isfinite(gx).all():
                print(f"  DIVERGED at step {step:,}")
                break
            k, e = energy_spectrum(gx, gy, gz)
            spectra.append(e)
            stats.append(statistics(gx, gy, gz, nu, step))
            sample_steps.append(step)
            nu_t_mean.append(float(solver.eddy_viscosity().mean()))
            snap_enstrophy.append(enstrophy(gx, gy, gz)[:, :, mid].astype(np.float32))
            snap_wz.append(vorticity(gx, gy, gz)[2][:, :, mid].astype(np.float32))
            if step % (every * 8) == 0:
                s = stats[-1]
                print(f"  step {step:>7,}  t/T {step / turnover:5.2f}  "
                      f"E {s.energy:.4e}  eps {s.dissipation:.3e}  "
                      f"Re_lambda {s.reynolds_lambda:6.1f}  nu_t/nu {nu_t_mean[-1] / nu:5.2f}")
        if step < total_steps:
            solver.step()

    elapsed = time.perf_counter() - t0
    _, gx, gy, gz = solver.macroscopic()

    energies = np.array([s.energy for s in stats])
    drops = np.diff(energies)
    print(f"\n{len(stats)} samples in {elapsed:.0f}s")
    print(f"energy {energies[0]:.4e} -> {energies[-1]:.4e} "
          f"({energies[-1] / energies[0] * 100:.1f}% retained)")
    print(f"monotone decay: {'yes' if np.all(drops <= 0) else 'NO -- energy was created'}")

    path = args.out / f"turb_{tag}.npz"
    np.savez_compressed(
        path,
        k=k, spectra=np.array(spectra), steps=np.array(sample_steps),
        energy=energies,
        dissipation=np.array([s.dissipation for s in stats]),
        reynolds_lambda=np.array([s.reynolds_lambda for s in stats]),
        taylor_microscale=np.array([s.taylor_microscale for s in stats]),
        nu_t_mean=np.array(nu_t_mean),
        slices_enstrophy=np.array(snap_enstrophy),
        slices_wz=np.array(snap_wz),
        ux=gx, uy=gy, uz=gz,
        n=n, nu=nu, reynolds=args.reynolds, smagorinsky=args.smagorinsky,
        u_rms=args.u_rms, turnover=turnover, k_peak=args.k_peak,
    )
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
