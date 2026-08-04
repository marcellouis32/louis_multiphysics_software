"""3D D3Q19 solver on the GPU, via Taichi.

Same physics as `solver2d.py` -- TRT collision, halfway bounce-back, moving-wall
momentum injection -- which is the entire point. The 2D NumPy solver was verified in
Phase 0 against Ghia and against an analytic Taylor-Green decay, so running this on a
z-uniform domain and comparing cell by cell turns "is the port correct" into a question
with an exact answer.

Two implementation choices worth knowing about before reading the kernels:

**Populations are stored as `f - w_i`, not `f`.** In fp32 this is not cosmetic. A
population sits at O(0.01-0.3) while the non-equilibrium part that carries all the
physics is O(1e-6). fp32 has about seven significant digits, so storing `f` directly
puts the interesting part within a couple of digits of the rounding noise. Storing the
offset from the rest-equilibrium keeps the small quantity small and recovers most of
the lost precision. The transformation is exact and costs one add per population.

**Streaming is pull, into a second buffer (AB).** Simplest correct thing. In-place
patterns (AA, Esoteric Twist) halve memory and are the route to 512^3, but they make
every intermediate state harder to reason about and belong after this kernel is trusted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from lms.lbm import d3q19
from lms.lbm.d2q9 import trt_magic_omega, viscosity_to_omega

_TI = None


def _taichi():
    """Import and return the taichi module, with a useful error if it is missing.

    Imported lazily so that `import lms.lbm` stays cheap and so the 2D solver, the
    schema and the whole test suite keep working on a machine without the gpu extra.
    """
    global _TI
    if _TI is None:
        try:
            import taichi as ti
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "the 3D solver needs taichi. Install it with: pip install -e '.[gpu]'"
            ) from exc
        _TI = ti
    return _TI


def init_backend(prefer_gpu: bool = True, precision: str | None = None):
    """Start Taichi and return `(arch_name, float_dtype)`.

    Metal has no float64 -- Taichi raises "Type f64 not supported" from its SPIR-V
    builder -- so the GPU path is fp32 and the CPU path defaults to fp64. That split is
    deliberate: the CPU path exists to tell a precision effect apart from a real bug,
    which is only possible if one of the two is exact enough to be the reference.
    """
    ti = _taichi()
    arch = ti.metal if prefer_gpu else ti.cpu

    if precision is None:
        dtype = ti.f32 if prefer_gpu else ti.f64
    else:
        if precision not in ("fp32", "fp64"):
            raise ValueError(f"precision must be 'fp32' or 'fp64', got {precision!r}")
        if precision == "fp64" and prefer_gpu:
            raise ValueError(
                "Metal does not support float64. Use precision='fp32' on the GPU, or "
                "prefer_gpu=False to get an fp64 reference run on the CPU."
            )
        dtype = ti.f32 if precision == "fp32" else ti.f64

    ti.init(arch=arch, default_fp=dtype, random_seed=0)
    return ("metal" if prefer_gpu else "cpu"), dtype


@dataclass
class SolverState3D:
    """Mirrors `solver2d.SolverState` so downstream code reads the same either way."""

    ux: np.ndarray
    uy: np.ndarray
    uz: np.ndarray
    rho: np.ndarray
    steps: int
    converged: bool
    residuals: list[tuple[int, float]] = field(default_factory=list)

    @property
    def speed(self) -> np.ndarray:
        return np.sqrt(self.ux**2 + self.uy**2 + self.uz**2)


class D3Q19Solver:
    """Index order is (x, y, z). Call `init_backend()` before constructing one."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        omega: float,
        solid: np.ndarray,
        collision: Literal["bgk", "trt"] = "trt",
        dtype=None,
    ) -> None:
        ti = _taichi()
        self.ti = ti
        self.shape = shape
        self.dtype = dtype if dtype is not None else ti.lang.impl.get_runtime().default_fp
        self.collision = collision
        self.omega_plus = float(omega)
        self.omega_minus = trt_magic_omega(omega) if collision == "trt" else float(omega)

        Q = d3q19.Q

        # Q-minor (populations contiguous per cell). The spike measured this against
        # Q-major and found them within ~10% for the streaming pattern that dominates,
        # so this is chosen for readability, not because it won a benchmark. Revisit
        # with `scripts/benchmark_mlups.py` once collision is in the loop.
        self.f = ti.Vector.field(Q, self.dtype, shape=shape)
        self.f_new = ti.Vector.field(Q, self.dtype, shape=shape)

        self.rho = ti.field(self.dtype, shape=shape)
        self.vel = ti.Vector.field(3, self.dtype, shape=shape)
        self.solid = ti.field(ti.i32, shape=shape)
        self.wall_vel = ti.Vector.field(3, self.dtype, shape=shape)

        self.solid.from_numpy(solid.astype(np.int32))
        self.wall_vel.fill(0)

        self._build_kernels()
        self.reset()

    # ------------------------------------------------------------------ kernels

    def _build_kernels(self) -> None:
        ti = self.ti
        Q = d3q19.Q
        EX, EY, EZ = d3q19.EX, d3q19.EY, d3q19.EZ
        W, OPP = d3q19.W, d3q19.OPPOSITE
        nx, ny, nz = self.shape
        w_plus, w_minus = self.omega_plus, self.omega_minus
        is_trt = self.collision == "trt"

        @ti.func
        def feq_shifted(rho, u):
            """Equilibrium minus the rest weights, matching how f is stored."""
            out = ti.Vector.zero(self.dtype, Q)
            usq = u.dot(u)
            for q in ti.static(range(Q)):
                eu = EX[q] * u[0] + EY[q] * u[1] + EZ[q] * u[2]
                out[q] = W[q] * (rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq) - 1.0)
            return out

        @ti.kernel
        def initialise():
            for i, j, k in self.f:
                self.rho[i, j, k] = 1.0
                self.vel[i, j, k] = ti.Vector.zero(self.dtype, 3)
                # f = w_i at rest, so the shifted representation is exactly zero.
                self.f[i, j, k] = ti.Vector.zero(self.dtype, Q)
                self.f_new[i, j, k] = ti.Vector.zero(self.dtype, Q)

        def make_step(src, dst):
            """Build one collide-and-stream kernel reading `src` and writing `dst`.

            Two kernels are compiled, one for each buffer direction, and `step()`
            alternates them. The obvious alternative -- one kernel plus a Python-level
            `self.f, self.f_new = self.f_new, self.f` -- silently does nothing: Taichi
            resolves field references when the kernel is compiled, so the swapped
            attributes never reach the compiled code and it reads the same buffer
            forever. That failure is quiet, not loud; the solver runs at full speed and
            returns a velocity field of exactly zero.
            """

            @ti.kernel
            def step_kernel():
                for i, j, k in src:
                    # --- pull: gather each population from the neighbour it came from
                    g = ti.Vector.zero(self.dtype, Q)
                    for q in ti.static(range(Q)):
                        si = (i - EX[q] + nx) % nx
                        sj = (j - EY[q] + ny) % ny
                        sk = (k - EZ[q] + nz) % nz
                        g[q] = src[si, sj, sk][q]

                    if self.solid[i, j, k] != 0:
                        # Halfway bounce-back. The population that arrived from a fluid
                        # neighbour is sent straight back the way it came, so the wall sits
                        # midway between this node and that neighbour. A moving wall adds
                        # 6*w_i*(e_i . u_wall), which is the momentum the wall imparts.
                        uw = self.wall_vel[i, j, k]
                        b = ti.Vector.zero(self.dtype, Q)
                        for q in ti.static(range(Q)):
                            eu = EX[q] * uw[0] + EY[q] * uw[1] + EZ[q] * uw[2]
                            b[q] = g[OPP[q]] + 6.0 * W[q] * eu
                        dst[i, j, k] = b
                        self.rho[i, j, k] = 1.0
                        self.vel[i, j, k] = ti.Vector.zero(self.dtype, 3)
                    else:
                        # --- macroscopic. The +1 undoes the w_i shift: sum(f) = sum(g) + 1
                        # because the weights sum to one. Momentum needs no correction,
                        # since sum_i w_i e_i vanishes by construction.
                        r = 1.0
                        mx = 0.0
                        my = 0.0
                        mz = 0.0
                        for q in ti.static(range(Q)):
                            r += g[q]
                            mx += EX[q] * g[q]
                            my += EY[q] * g[q]
                            mz += EZ[q] * g[q]
                        u = ti.Vector([mx / r, my / r, mz / r])

                        self.rho[i, j, k] = r
                        self.vel[i, j, k] = u

                        # --- collide
                        eq = feq_shifted(r, u)
                        out = ti.Vector.zero(self.dtype, Q)
                        if ti.static(is_trt):
                            for q in ti.static(range(Q)):
                                qo = OPP[q]
                                f_sym = 0.5 * (g[q] + g[qo])
                                f_asym = 0.5 * (g[q] - g[qo])
                                e_sym = 0.5 * (eq[q] + eq[qo])
                                e_asym = 0.5 * (eq[q] - eq[qo])
                                out[q] = (
                                    g[q]
                                    - w_plus * (f_sym - e_sym)
                                    - w_minus * (f_asym - e_asym)
                                )
                        else:
                            for q in ti.static(range(Q)):
                                out[q] = g[q] - w_plus * (g[q] - eq[q])
                        dst[i, j, k] = out

            return step_kernel

        # Closes over the field rather than taking it as an argument: this module uses
        # `from __future__ import annotations`, which turns every annotation into a
        # string, and Taichi cannot read `"ti.template()"` as a type.
        speed = ti.field(self.dtype, shape=self.shape)

        @ti.kernel
        def speed_field():
            for i, j, k in speed:
                speed[i, j, k] = self.vel[i, j, k].norm()

        self._initialise = initialise
        self._steps = (make_step(self.f, self.f_new), make_step(self.f_new, self.f))
        self._parity = 0
        self._speed_field = speed_field
        self._speed = speed

    # ------------------------------------------------------------------ api

    def reset(self) -> None:
        self._initialise()
        self._parity = 0

    def set_state(
        self,
        rho: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        uz: np.ndarray,
        f_neq: np.ndarray | None = None,
    ) -> None:
        """Seed the solver from a macroscopic field, at equilibrium.

        `f_neq` supplies the non-equilibrium part, which carries the strain rate. It is
        optional but rarely optional in practice: omitting it discards the stress, and
        the solver rebuilds it only after radiating an acoustic transient whose error
        does not shrink at second order under refinement. In a convergence study that
        shows up as a degraded order and looks exactly like a solver defect.

        Both buffers are written, because `step()` alternates between two compiled
        kernels and which one runs first depends on the parity.
        """
        feq = d3q19.equilibrium(
            np.ascontiguousarray(rho, dtype=np.float64),
            np.ascontiguousarray(ux, dtype=np.float64),
            np.ascontiguousarray(uy, dtype=np.float64),
            np.ascontiguousarray(uz, dtype=np.float64),
        )
        if f_neq is not None:
            # `f_neq` is the pre-collision Chapman-Enskog part, but these buffers hold
            # *post-collision* populations: `step()` pulls from the buffer, then
            # collides, then writes. Collision has already acted on what is stored, so
            # the non-equilibrium part must be scaled by (1 - omega). Omitting the
            # factor does not merely shrink the correction -- with omega near 1.9 it
            # flips its sign, so the initial stress is applied backwards and the run
            # starts further from the truth than equilibrium alone.
            # The strain-rate part is symmetric under i -> opposite(i), so under TRT it
            # relaxes at omega_plus.
            feq = feq + (1.0 - self.omega_plus) * f_neq
        # Stored shifted by the rest weights, and laid out (x, y, z, Q) for Taichi.
        shifted = (feq - d3q19.W[:, None, None, None]).transpose(1, 2, 3, 0)
        packed = np.ascontiguousarray(shifted, dtype=_np_dtype(self.ti, self.dtype))
        self.f.from_numpy(packed)
        self.f_new.from_numpy(packed)
        self.rho.from_numpy(np.ascontiguousarray(rho, dtype=packed.dtype))
        vel = np.stack([ux, uy, uz], axis=-1)
        self.vel.from_numpy(np.ascontiguousarray(vel, dtype=packed.dtype))
        self._parity = 0

    def set_moving_wall(self, mask: np.ndarray, ux=0.0, uy=0.0, uz=0.0) -> None:
        """Give the selected solid nodes a translation velocity."""
        vel = self.wall_vel.to_numpy()
        vel[mask.astype(bool)] = (ux, uy, uz)
        self.wall_vel.from_numpy(vel.astype(_np_dtype(self.ti, self.dtype)))

    def step(self) -> None:
        self._steps[self._parity]()
        self._parity ^= 1

    def total_mass(self) -> float:
        """Sum of the real populations over every node, the exactly conserved quantity.

        Streaming is a permutation and TRT collision conserves mass, and bounce-back
        adds `6*w_q*(e_q . u_wall)` whose sum over q vanishes because `sum_q w_q e_q`
        is zero. So this number must not drift, and it is the cheapest global check
        that the kernel is not leaking.

        It cannot be recovered from `self.rho`, which deliberately reports 1.0 at solid
        nodes. Those nodes hold bounce-back state rather than a physical distribution,
        and reporting the raw sum there produces densities of +-50 that are meaningless
        to render and actively dangerous to restart a solve from.
        """
        f = self.f.to_numpy() if self._parity == 0 else self.f_new.to_numpy()
        # Stored shifted: real f = g + w_i, and the weights sum to one per node.
        return float(f.sum() + np.prod(self.shape))

    def macroscopic(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        vel = self.vel.to_numpy()
        return self.rho.to_numpy(), vel[..., 0], vel[..., 1], vel[..., 2]

    def run(
        self,
        max_steps: int,
        tol: float = 1e-6,
        check_every: int = 500,
        callback: Callable[[int, D3Q19Solver], None] | None = None,
    ) -> SolverState3D:
        """Advance to steady state, or until the step budget runs out.

        Convergence is the relative change in the speed field between checks, the same
        criterion the 2D solver uses, so the two are directly comparable.
        """
        self._speed_field()
        prev = self._speed.to_numpy()
        residuals: list[tuple[int, float]] = []
        converged = False
        step = 0

        for step in range(1, max_steps + 1):
            self.step()

            if step % check_every == 0:
                self._speed_field()
                speed = self._speed.to_numpy()
                if not np.isfinite(speed).all():
                    raise FloatingPointError(
                        f"solver diverged at step {step}. omega={self.omega_plus:.4f} "
                        f"(tau={1 / self.omega_plus:.4f}); tau below ~0.51 or lattice "
                        "velocity above ~0.1 will do this. Lower the velocity or raise "
                        "the resolution."
                    )
                denom = np.linalg.norm(speed) or 1.0
                res = float(np.linalg.norm(speed - prev) / denom)
                residuals.append((step, res))
                prev = speed
                if callback:
                    callback(step, self)
                if res < tol:
                    converged = True
                    break

        rho, ux, uy, uz = self.macroscopic()
        return SolverState3D(
            ux=ux, uy=uy, uz=uz, rho=rho,
            steps=step, converged=converged, residuals=residuals,
        )


def _np_dtype(ti, dtype):
    return np.float32 if dtype == ti.f32 else np.float64


def lid_driven_cavity_3d(
    n: int = 64,
    nz: int = 4,
    reynolds: float = 1000.0,
    lid_velocity: float = 0.1,
    collision: Literal["bgk", "trt"] = "trt",
    periodic_z: bool = True,
) -> D3Q19Solver:
    """The Phase 0 cavity, extruded along z.

    With `periodic_z=True` there are no walls in z, so the solution is uniform in z and
    must reproduce the 2D solver exactly. That is the porting test. Setting it False
    adds end walls and gives the genuine 3D cavity, where those end walls drive
    Taylor-Gortler vortices that 2D cannot produce.

    Cavity side length is n-2 in lattice units, matching `lid_driven_cavity` in
    solver2d.py: the halfway bounce-back wall sits midway between the solid ring and
    the first fluid layer.
    """
    if lid_velocity > 0.15:
        raise ValueError(
            f"lid velocity {lid_velocity} is far into the compressible regime; "
            "LBM needs lattice Mach << 1, so keep this at or below ~0.1"
        )

    shape = (n, n, nz)
    length = n - 2
    nu = lid_velocity * length / reynolds
    omega = viscosity_to_omega(nu)

    solid = np.zeros(shape, dtype=bool)
    solid[0, :, :] = solid[-1, :, :] = True
    solid[:, 0, :] = solid[:, -1, :] = True
    if not periodic_z:
        solid[:, :, 0] = solid[:, :, -1] = True

    solver = D3Q19Solver(shape, omega, solid, collision=collision)

    lid = np.zeros(shape, dtype=bool)
    lid[:, -1, :] = True
    solver.set_moving_wall(lid, ux=lid_velocity)

    solver.reynolds = reynolds
    solver.lid_velocity = lid_velocity
    solver.cavity_length = length
    return solver
