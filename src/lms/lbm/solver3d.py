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
    stopped_on: str = "step cap"
    """Why the run ended: "tolerance", "stagnation" or "step cap".

    Worth distinguishing. In fp32 the residual floors around 1e-5, so a stricter
    tolerance is unreachable and a converged solution would otherwise be labelled a
    failure -- which understates a perfectly good result rather than overstating it,
    but is still wrong.
    """

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
        collision: Literal["bgk", "trt", "regularized"] = "trt",
        dtype=None,
        smagorinsky: float = 0.0,
        rotor: dict | None = None,
        ledger: bool = False,
    ) -> None:
        """`rotor`, when given, adds a rotating blade set evaluated analytically in the
        kernel (see `Tank.rotor_params`). Keys: z_centre, blade_inner, blade_outer,
        half_t, half_h, n_blades, omega (radians per step). The axisymmetric parts of
        an impeller -- disc, hub, shaft -- do NOT belong in it: they rotate without
        changing shape, so they go in `solid` (marked 2 to be measured) with their
        wall velocity set once via `set_wall_velocity`."""
        ti = _taichi()
        self.ti = ti
        self.shape = shape
        self.dtype = dtype if dtype is not None else ti.lang.impl.get_runtime().default_fp
        self.collision = collision
        self.omega_plus = float(omega)
        self.omega_minus = trt_magic_omega(omega) if collision == "trt" else float(omega)
        if collision not in ("bgk", "trt", "regularized"):
            raise ValueError(
                f"collision must be bgk, trt or regularized, got {collision!r}"
            )
        # 0.0 disables the model entirely and must reproduce the laminar solver exactly;
        # that equivalence is the regression the whole LES path rests on.
        self.smagorinsky = float(smagorinsky)
        if self.smagorinsky < 0.0:
            raise ValueError(f"smagorinsky constant must be >= 0, got {smagorinsky}")

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

        # Eddy viscosity, written per step when LES is active. Diagnostic only -- the
        # collision uses the local rate directly -- but it is what makes the model
        # inspectable rather than a black box inside the kernel.
        self.nu_t = ti.field(self.dtype, shape=shape)
        self.nu_t.fill(0)

        # Solid semantics: 0 fluid, 1 solid, 2 solid-and-measured. Nodes marked 2 have
        # the momentum they exchange with the fluid accumulated into force and torque
        # each step -- the machinery the power number stands on. A plain bool mask
        # casts to 0/1 and measures nothing, so every pre-Phase-2 caller is unchanged.
        solid_int = solid.astype(np.int32)
        self.solid.from_numpy(solid_int)
        self.wall_vel.fill(0)

        # The momentum ledger: cumulative meters for every channel of angular
        # momentum into or out of the resolved fluid. Off by default -- it adds an
        # exchange loop over ALL wall nodes -- and switched on for audits, where the
        # question is not "what is the torque" but "does the budget close". Built
        # because the Bouzidi torque meter read 3-6x the fluid's actual dL/dt and the
        # only honest way to locate such a discrepancy is to meter every channel and
        # let the residual point at what is missing.
        self.ledger = bool(ledger)
        self._led_static = ti.field(self.dtype, shape=())   # torque-z on class-1 walls
        self._led_inject = ti.field(self.dtype, shape=())   # L_z injected by refills
        self._led_remove = ti.field(self.dtype, shape=())   # L_z removed by coverage
        self._led_static[None] = 0.0
        self._led_inject[None] = 0.0
        self._led_remove[None] = 0.0

        self._rotor = dict(rotor) if rotor is not None else None
        self.theta = 0.0
        """Host-side mirror of the shaft angle, for bookkeeping (revolution counts,
        oracle comparisons). The authoritative angle lives on the device and is
        advanced by a kernel, so stepping involves no host traffic at all."""
        # 0-d fields rather than kernel arguments: this module uses
        # `from __future__ import annotations`, which turns argument annotations into
        # strings Taichi cannot resolve -- the same trap ti.template() hit in Phase 1a.
        self._theta_now = ti.field(self.dtype, shape=())
        self._theta_prev = ti.field(self.dtype, shape=())
        self._theta_now[None] = 0.0
        self._theta_prev[None] = 0.0

        self._measure = bool(np.any(solid_int == 2)) or self._rotor is not None
        self._link_force = ti.Vector.field(3, self.dtype, shape=())
        self._link_torque = ti.Vector.field(3, self.dtype, shape=())
        # Torque history lives on the device and is drained rarely. Reading the
        # accumulator back every step costs a full pipeline sync per step -- measured
        # at 6x total slowdown on the first tank run -- for a number nobody needs
        # until the averaging window closes.
        self._log_ring = 1 << 16
        self._torque_log = ti.field(self.dtype, shape=self._log_ring)
        self._log_head = ti.field(ti.i32, shape=())
        self._log_head[None] = 0
        self._log_read = 0
        # Torque reference axis: a point it passes through (the torque is reported as a
        # full vector, so the axis *direction* is the caller's projection to take).
        # Defaults to the domain centreline, which is the tank shaft by construction.
        self.axis = ti.Vector.field(3, self.dtype, shape=())
        self.axis[None] = [(shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0, 0.0]

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
        is_regularized = self.collision == "regularized"
        cs2 = d3q19.CS2
        smag = self.smagorinsky
        les_on = smag > 0.0
        measure = self._measure
        has_rotor = self._rotor is not None
        rot = self._rotor or {}
        rot_zc = float(rot.get("z_centre", 0.0))
        rot_in = float(rot.get("blade_inner", 0.0))
        rot_out = float(rot.get("blade_outer", 0.0))
        rot_half_t = float(rot.get("half_t", 0.0))
        rot_half_h = float(rot.get("half_h", 0.0))
        rot_omega = float(rot.get("omega", 0.0))
        rot_sector = 2.0 * np.pi / int(rot.get("n_blades", 1))
        # Loose radial guards; correctness comes from the exact test inside them.
        rot_r2_lo = max(rot_in - 1.5, 0.0) ** 2
        rot_r2_hi = (rot_out + 1.5) ** 2
        ledger_on = self.ledger
        is_bouzidi = has_rotor and rot.get("boundary") == "bouzidi"
        # Plain-python constants for the Bouzidi block: int()/float() inside a kernel
        # body are rewritten into runtime casts by Taichi's AST transformer, so numpy
        # scalars must be converted OUT here, where ordinary Python still applies.
        OPP_I = [int(v) for v in OPP]
        EX_F = [float(v) for v in EX]
        EY_F = [float(v) for v in EY]
        EZ_F = [float(v) for v in EZ]
        rot_ht_true = float(rot.get("half_t_true", rot_half_t))
        # Band for the Bouzidi link scan: one halo cell beyond the blade box, since
        # only fluid nodes with a blade-covered neighbour participate.
        bz_r2_lo = max(rot_in - 1.8, 0.0) ** 2
        bz_r2_hi = (rot_out + 1.8) ** 2
        bz_dz = rot_half_h + 1.8
        axis_x = (nx - 1) / 2.0
        axis_y = (ny - 1) / 2.0
        ring = self._log_ring

        @ti.func
        def q_norm(fneq):
            """|Q| = sqrt(2 Q_ab Q_ab) where Q_ab = sum_i e_ia e_ib f_i^neq.

            This is the second moment of the non-equilibrium populations, and Phase 1a
            proved it equals -2 tau rho cs^2 S_ab. So the strain rate is available
            *locally*: no finite differences, no neighbour access, no extra memory
            traffic. It is the single best reason LES and LBM fit together.

            Note it needs no tau, which is what lets the Smagorinsky closure below be
            explicit rather than iterative.
            """
            qxx = 0.0
            qyy = 0.0
            qzz = 0.0
            qxy = 0.0
            qxz = 0.0
            qyz = 0.0
            for q in ti.static(range(Q)):
                qxx += EX[q] * EX[q] * fneq[q]
                qyy += EY[q] * EY[q] * fneq[q]
                qzz += EZ[q] * EZ[q] * fneq[q]
                qxy += EX[q] * EY[q] * fneq[q]
                qxz += EX[q] * EZ[q] * fneq[q]
                qyz += EY[q] * EZ[q] * fneq[q]
            # Off-diagonals counted twice, since Q is symmetric.
            return ti.sqrt(
                2.0 * (qxx * qxx + qyy * qyy + qzz * qzz)
                + 4.0 * (qxy * qxy + qxz * qxz + qyz * qyz)
            )

        @ti.func
        def les_omega(rho, fneq):
            """Local relaxation rate under the Smagorinsky closure.

            The eddy viscosity depends on the strain, the strain depends on tau, and tau
            depends on the eddy viscosity. The loop closes in one square root because
            |Q| is tau-free (Hou et al. 1996):

                tau = 0.5 [ tau_0 + sqrt(tau_0^2 + 18 Cs^2 |Q| / rho) ]
            """
            tau0 = 1.0 / w_plus
            tau = 0.5 * (tau0 + ti.sqrt(tau0 * tau0 + 18.0 * smag * smag * q_norm(fneq) / rho))
            return 1.0 / tau

        @ti.func
        def magic_partner(wp):
            """Antisymmetric rate holding Lambda = (tau+ - 1/2)(tau- - 1/2) at 1/4.

            Recomputed *per cell*, which is the whole point. Lambda = 1/4 is what pins
            the bounce-back wall exactly halfway between nodes independent of viscosity;
            Phase 0 established that letting it drift makes a case validated at one
            Reynolds number degrade at another. LES makes tau vary per cell and per
            step, so a single scalar omega_minus computed once would give every
            turbulent cell the wrong Lambda -- a wall whose position depends on how
            turbulent the flow near it happens to be.
            """
            tau_plus = 1.0 / wp
            return 1.0 / (0.25 / (tau_plus - 0.5) + 0.5)

        @ti.func
        def feq_shifted(rho, u):
            """Equilibrium minus the rest weights, matching how f is stored."""
            out = ti.Vector.zero(self.dtype, Q)
            usq = u.dot(u)
            for q in ti.static(range(Q)):
                eu = EX[q] * u[0] + EY[q] * u[1] + EZ[q] * u[2]
                out[q] = W[q] * (rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq) - 1.0)
            return out

        @ti.func
        def blade_at(x, y, z, th):
            """Is the point (relative to the shaft axis) inside a blade at angle th?

            Angular folding instead of a loop over blades: the node's polar angle is
            wrapped into the nearest blade's sector, so one sin/cos pair decides
            membership regardless of blade count. Blades cannot overlap angularly at
            r >= blade_inner, so the nearest blade is the only candidate. The algebra
            is identical to `Tank.blade_mask` -- xb = r cos(phi - a), yb = r sin(phi
            - a) -- including the tie epsilon carried inside half_t, which is what
            makes the mask-for-mask oracle comparison exact in fp64.
            """
            inside = 0
            if ti.abs(z - rot_zc) <= rot_half_h:
                r2 = x * x + y * y
                if rot_r2_lo <= r2 <= rot_r2_hi:
                    phi = ti.atan2(y, x)
                    d = phi - th
                    d = d - rot_sector * ti.round(d / rot_sector)
                    r = ti.sqrt(r2)
                    xb = r * ti.cos(d)
                    yb = r * ti.sin(d)
                    if ti.abs(yb) <= rot_half_t and rot_in <= xb <= rot_out:
                        inside = 1
            return inside

        @ti.func
        def blade_q(x, y, z, ejx, ejy, ejz, th):
            """Fraction q in (0, 1] of the link from (x, y, z) along e_j at which the
            blade surface sits, or -1.0 if the ray misses the box.

            Each blade is an axis-aligned box in its own frame, so this is an exact
            slab test after the same fold `blade_at` uses. The box takes the *true*
            half thickness: the tie epsilon makes node membership deterministic, but
            the wall is where the wall is. Mirrors `Tank.blade_link_fraction`, which
            is the NumPy oracle it is tested against.
            """
            phi = ti.atan2(y, x)
            ang = th + rot_sector * ti.round((phi - th) / rot_sector)
            ca = ti.cos(ang)
            sa = ti.sin(ang)
            px = x * ca + y * sa
            py = -x * sa + y * ca
            ex = ejx * ca + ejy * sa
            ey = -ejx * sa + ejy * ca

            t_lo = 0.0
            t_hi = 1.0e30
            ok = 1
            # x-slab [rot_in, rot_out]
            if ti.abs(ex) < 1e-12:
                if px < rot_in or px > rot_out:
                    ok = 0
            else:
                t1 = (rot_in - px) / ex
                t2 = (rot_out - px) / ex
                t_lo = ti.max(t_lo, ti.min(t1, t2))
                t_hi = ti.min(t_hi, ti.max(t1, t2))
            # y-slab [-ht_true, ht_true]
            if ti.abs(ey) < 1e-12:
                if py < -rot_ht_true or py > rot_ht_true:
                    ok = 0
            else:
                t1 = (-rot_ht_true - py) / ey
                t2 = (rot_ht_true - py) / ey
                t_lo = ti.max(t_lo, ti.min(t1, t2))
                t_hi = ti.min(t_hi, ti.max(t1, t2))
            # z-slab
            if ti.abs(ejz) < 1e-12:
                if z < rot_zc - rot_half_h or z > rot_zc + rot_half_h:
                    ok = 0
            else:
                t1 = (rot_zc - rot_half_h - z) / ejz
                t2 = (rot_zc + rot_half_h - z) / ejz
                t_lo = ti.max(t_lo, ti.min(t1, t2))
                t_hi = ti.min(t_hi, ti.max(t1, t2))

            out = -1.0
            if ok == 1 and t_lo <= t_hi and t_lo <= 1.0:
                out = ti.min(ti.max(t_lo, 1e-6), 1.0)
            return out

        @ti.kernel
        def begin_step():
            # Everything the host used to do between steps, moved onto the device so
            # stepping involves no host traffic at all: log the previous step's
            # torque into the ring, zero the accumulators, advance the shaft angle.
            head = self._log_head[None]
            self._torque_log[head % ring] = self._link_torque[None][2]
            self._log_head[None] = head + 1
            self._link_force[None] = ti.Vector.zero(self.dtype, 3)
            self._link_torque[None] = ti.Vector.zero(self.dtype, 3)
            if ti.static(has_rotor):
                # Wrapped by 2 pi to keep fp32 honest over long runs: at ~1e-3
                # rad/step an unwrapped angle passes 1e4 within minutes and fp32
                # starts eating the increment.
                self._theta_prev[None] = self._theta_now[None]
                nxt = self._theta_now[None] + rot_omega
                if nxt > 6.283185307179586:
                    nxt -= 6.283185307179586
                    self._theta_prev[None] -= 6.283185307179586
                self._theta_now[None] = nxt

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

                    # Blade membership is a function, not a field: solid(x, t)
                    # evaluated analytically per node, so rotation involves no mask
                    # updates and no host traffic.
                    xr = i - axis_x
                    yr = j - axis_y
                    dyn = 0
                    if ti.static(has_rotor):
                        dyn = blade_at(xr, yr, k, self._theta_now[None])

                    if self.solid[i, j, k] != 0 or dyn != 0:
                        # Halfway bounce-back. The population that arrived from a fluid
                        # neighbour is sent straight back the way it came, so the wall sits
                        # midway between this node and that neighbour. A moving wall adds
                        # 6*w_i*(e_i . u_wall), which is the momentum the wall imparts.
                        uw = self.wall_vel[i, j, k]
                        if ti.static(has_rotor):  # noqa: SIM102 -- static/runtime split
                            if dyn != 0:
                                # Rigid rotation: u = omega x r, fresh every step, so a
                                # blade's wall velocity is exact at every node it covers.
                                uw = ti.Vector([-rot_omega * yr, rot_omega * xr, 0.0])
                        b = ti.Vector.zero(self.dtype, Q)
                        for q in ti.static(range(Q)):
                            eu = EX[q] * uw[0] + EY[q] * uw[1] + EZ[q] * uw[2]
                            b[q] = g[OPP[q]] + 6.0 * W[q] * eu
                        dst[i, j, k] = b
                        self.rho[i, j, k] = 1.0
                        self.vel[i, j, k] = ti.Vector.zero(self.dtype, 3)

                        # --- momentum exchange, on measured solids only.
                        # Per boundary link, the momentum handed to the wall in one step
                        # is e_q (f_in + f_out) -- what arrives plus what is thrown back.
                        # The populations are stored shifted by w_q, and the shift does
                        # NOT cancel per link: f_in + f_out = (g_in + g_out) + 2 w_q. It
                        # is restored explicitly here. (Summed over a closed body the
                        # 2 w_q terms telescope to zero, which makes forgetting them
                        # invisible on exactly the symmetric test cases one debugs with
                        # -- the Couette analytic torque exists to catch that.)
                        # Deliberately nested rather than one `and`: the outer test is
                        # compile-time (ti.static) and elides the whole block from
                        # kernels that measure nothing; folding it into a runtime
                        # conjunction would put the cost on every solid node forever.
                        if ti.static(ledger_on):  # noqa: SIM102 -- static/runtime split
                            # Ledger channel: class-1 walls (vessel, baffles, lid,
                            # bottom). Same link exchange as the measured surfaces --
                            # these are the silent absorbers the audit exists to hear.
                            if self.solid[i, j, k] == 1:
                                for q in ti.static(range(1, Q)):
                                    lsi = (i - EX[q] + nx) % nx
                                    lsj = (j - EY[q] + ny) % ny
                                    lsk = (k - EZ[q] + nz) % nz
                                    lnf = self.solid[lsi, lsj, lsk] == 0
                                    if ti.static(has_rotor):  # noqa: SIM102
                                        if lnf and blade_at(
                                            xr - EX[q], yr - EY[q], k - EZ[q],
                                            self._theta_now[None],
                                        ) != 0:
                                            lnf = False
                                    if lnf:
                                        leu = (EX[q] * uw[0] + EY[q] * uw[1]
                                               + EZ[q] * uw[2])
                                        ldp = 2.0 * g[q] - 6.0 * W[q] * leu
                                        lfx = EX[q] * ldp
                                        lfy = EY[q] * ldp
                                        lrx = i - 0.5 * EX[q] - self.axis[None][0]
                                        lry = j - 0.5 * EY[q] - self.axis[None][1]
                                        self._led_static[None] += lrx * lfy - lry * lfx

                            # Coverage is deliberately NOT a ledger channel. The
                            # first audit metered it and the budget refused to close:
                            # under halfway bounce-back a covered node returns its
                            # gathered momentum to the fluid next step (a one-step
                            # inventory, not a flux), and under Bouzidi the loss is
                            # already inside the one-sided link meter, which counts
                            # fj out with only the reconstructed gk back. Metering it
                            # again double-books ~the entire impeller torque.

                        if ti.static(measure):  # noqa: SIM102
                            if self.solid[i, j, k] == 2 or (
                                dyn != 0 and ti.static(not is_bouzidi)
                            ):
                                for q in ti.static(range(1, Q)):
                                    si = (i - EX[q] + nx) % nx
                                    sj = (j - EY[q] + ny) % ny
                                    sk = (k - EZ[q] + nz) % nz
                                    nbr_fluid = self.solid[si, sj, sk] == 0
                                    if ti.static(has_rotor):  # noqa: SIM102 -- static/runtime split
                                        if nbr_fluid and blade_at(
                                            xr - EX[q], yr - EY[q], k - EZ[q],
                                            self._theta_now[None],
                                        ) != 0:
                                            nbr_fluid = False
                                    if nbr_fluid:
                                        eu = (EX[q] * uw[0] + EY[q] * uw[1]
                                              + EZ[q] * uw[2])
                                        # f_in = g[q] + w_q arrived along e_q; the
                                        # bounced partner leaves along -e_q as
                                        # b[OPP[q]] + w_q = g[q] - 6 w_q eu + w_q.
                                        # No +2W term: that is the isotropic pressure
                                        # background, whose net force/torque on a
                                        # closed body is exactly zero in the continuum.
                                        # On a staircase link set it does not cancel
                                        # exactly, and per link it is ~100x the
                                        # physical signal -- the ledger measured a
                                        # rotated blade's background non-closure at
                                        # several times the true torque. Dropping it
                                        # removes pure gauge, no physics.
                                        dp = 2.0 * g[q] - 6.0 * W[q] * eu
                                        # Galilean correction for a MOVING wall (Wen
                                        # et al. 2014): the naive e (f_in + f_out)
                                        # exchange is derived in the wall's rest
                                        # frame; in the lab frame the transferred
                                        # momentum is (e - u_w) f_in - (-e - u_w)
                                        # f_out, adding -u_w (f_in - f_out). Our
                                        # bounce-back gives f_in - f_out = 6 w eu
                                        # exactly. Drag and this term are both
                                        # quadratic in wall speed, so omitting it is
                                        # an O(1) relative error on moving-blade
                                        # torque -- while cancelling by symmetry on
                                        # bodies of revolution, which is why Couette
                                        # (and every static case) never saw it.
                                        cw = 6.0 * W[q] * eu
                                        fx = EX[q] * dp - uw[0] * cw
                                        fy = EY[q] * dp - uw[1] * cw
                                        fz = EZ[q] * dp - uw[2] * cw
                                        self._link_force[None] += ti.Vector([fx, fy, fz])
                                        # Torque arm: the link midpoint, where the wall
                                        # actually sits under the halfway convention.
                                        rx = i - 0.5 * EX[q] - self.axis[None][0]
                                        ry = j - 0.5 * EY[q] - self.axis[None][1]
                                        rz = k - 0.5 * EZ[q] - self.axis[None][2]
                                        self._link_torque[None] += ti.Vector([
                                            ry * fz - rz * fy,
                                            rz * fx - rx * fz,
                                            rx * fy - ry * fx,
                                        ])
                    else:
                        fresh = 0
                        if ti.static(has_rotor):  # noqa: SIM102 -- static/runtime split
                            # A node the blade just vacated holds bounce-back state,
                            # not a fluid distribution. Refill at equilibrium with the
                            # departing wall's velocity; the non-equilibrium part
                            # re-establishes within a few steps (omega ~ 1.99). Any
                            # artifact this leaves shows up as spikes in the torque
                            # history, which the power-number run records anyway --
                            # the diagnostic is free.
                            if blade_at(xr, yr, k, self._theta_prev[None]) != 0:
                                uwf = ti.Vector([-rot_omega * yr, rot_omega * xr, 0.0])
                                gnew = feq_shifted(1.0, uwf)
                                if ti.static(ledger_on):
                                    # Ledger channel: refill injection. Setting the
                                    # node to equilibrium replaces whatever streamed
                                    # in -- momentum created from nothing, as far as
                                    # the resolved fluid is concerned.
                                    dpx = 0.0
                                    dpy = 0.0
                                    for q in ti.static(range(Q)):
                                        dpx += EX[q] * (gnew[q] - g[q])
                                        dpy += EY[q] * (gnew[q] - g[q])
                                    self._led_inject[None] += xr * dpy - yr * dpx
                                g = gnew
                                fresh = 1

                        if ti.static(is_bouzidi):
                            # Sub-cell wall placement on blade links (Bouzidi, Firdaouss
                            # & Lallemand 2001, linear variant). The staircase blade was
                            # measured to under-drive the flow -- six suspects
                            # eliminated -- and this is the escalation the plan
                            # reserved. Fluid-centric: a population arriving from a
                            # blade-covered upstream node is reconstructed from
                            # post-collision values at this node and its next fluid
                            # neighbour, with the wall at its exact fraction q of the
                            # link. Both branches are affine with coefficients summing
                            # to one, so the f - w_i storage shift passes through
                            # exactly as it does for halfway bounce-back.
                            th_b = self._theta_now[None]
                            r2_here = xr * xr + yr * yr
                            in_band = (
                                fresh == 0
                                and ti.abs(k - rot_zc) <= bz_dz
                                and r2_here >= bz_r2_lo
                                and r2_here <= bz_r2_hi
                            )
                            if in_band:
                                for q in ti.static(range(1, Q)):
                                    if blade_at(
                                        xr - EX[q], yr - EY[q], k - EZ[q], th_b
                                    ) != 0:
                                        # ti.static keeps jd a Python int: a bare
                                        # assignment would create a runtime Expr and
                                        # the list subscripts below would fail.
                                        jd = ti.static(OPP_I[q])
                                        qq = blade_q(
                                            xr, yr, k,
                                            EX_F[jd], EY_F[jd], EZ_F[jd], th_b,
                                        )
                                        # Wall velocity at the actual wall point.
                                        wxp = xr + qq * EX_F[jd]
                                        wyp = yr + qq * EY_F[jd]
                                        uwx = -rot_omega * wyp
                                        uwy = rot_omega * wxp
                                        term = 6.0 * W[q] * (EX[q] * uwx + EY[q] * uwy)

                                        fj_here = src[i, j, k][jd]
                                        gk_new = 0.0
                                        used1 = 0
                                        if qq >= 0.0 and qq <= 0.5:
                                            s2i = (i + EX[q] + nx) % nx
                                            s2j = (j + EY[q] + ny) % ny
                                            s2k = (k + EZ[q] + nz) % nz
                                            far_ok = self.solid[s2i, s2j, s2k] == 0
                                            if far_ok and blade_at(
                                                xr + EX[q], yr + EY[q], k + EZ[q], th_b
                                            ) == 0:
                                                gk_new = (
                                                    2.0 * qq * fj_here
                                                    + (1.0 - 2.0 * qq)
                                                    * src[s2i, s2j, s2k][jd]
                                                    + term
                                                )
                                                used1 = 1
                                        if used1 == 0:
                                            # q > 1/2 branch; also the fallback for a
                                            # missing second neighbour or a tie-zone
                                            # miss (qq < 0), where q = 1/2 reduces it
                                            # to exact halfway bounce-back.
                                            qz = ti.max(qq, 0.5)
                                            inv = 1.0 / (2.0 * qz)
                                            gk_new = (
                                                inv * fj_here
                                                + (1.0 - inv) * src[i, j, k][q]
                                                + inv * term
                                            )
                                        g[q] = gk_new

                                        if ti.static(measure):
                                            # Same link exchange as the solid-centric
                                            # path, with the real (unshifted) in/out
                                            # pair and the arm at the exact wall point.
                                            # Isotropic background dropped; see the
                                            # solid-centric meter's comment.
                                            dp = fj_here + gk_new
                                            diff = fj_here - gk_new
                                            fxl = -EX[q] * dp - uwx * diff
                                            fyl = -EY[q] * dp - uwy * diff
                                            fzl = -EZ[q] * dp
                                            self._link_force[None] += ti.Vector(
                                                [fxl, fyl, fzl]
                                            )
                                            ax0 = self.axis[None][0]
                                            ay0 = self.axis[None][1]
                                            rxl = (i - ax0) + qq * EX_F[jd]
                                            ryl = (j - ay0) + qq * EY_F[jd]
                                            self._link_torque[None] += ti.Vector([
                                                0.0, 0.0, rxl * fyl - ryl * fxl,
                                            ])

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

                        # The w_i shifts cancel exactly in the difference, so this is
                        # the true non-equilibrium part despite both terms being shifted.
                        wp = w_plus
                        wm = w_minus
                        if ti.static(les_on):
                            fneq = g - eq
                            wp = les_omega(r, fneq)
                            wm = magic_partner(wp) if ti.static(is_trt) else wp
                            self.nu_t[i, j, k] = (1.0 / wp - 1.0 / w_plus) * cs2

                        out = ti.Vector.zero(self.dtype, Q)
                        if ti.static(is_regularized):
                            # Rebuild the whole non-equilibrium part from its momentum
                            # flux, discarding every higher Hermite moment. Those ghost
                            # modes carry no hydrodynamics but do carry the instability
                            # that kills LBM as tau approaches 1/2.
                            pxx = 0.0; pyy = 0.0; pzz = 0.0
                            pxy = 0.0; pxz = 0.0; pyz = 0.0
                            for q in ti.static(range(Q)):
                                d = g[q] - eq[q]
                                pxx += EX[q] * EX[q] * d
                                pyy += EY[q] * EY[q] * d
                                pzz += EZ[q] * EZ[q] * d
                                pxy += EX[q] * EY[q] * d
                                pxz += EX[q] * EZ[q] * d
                                pyz += EY[q] * EZ[q] * d
                            for q in ti.static(range(Q)):
                                # Off-diagonal terms appear twice in the contraction
                                # because Pi is symmetric, hence the factor of two.
                                contraction = (
                                    (EX[q] * EX[q] - cs2) * pxx
                                    + (EY[q] * EY[q] - cs2) * pyy
                                    + (EZ[q] * EZ[q] - cs2) * pzz
                                    + 2.0 * EX[q] * EY[q] * pxy
                                    + 2.0 * EX[q] * EZ[q] * pxz
                                    + 2.0 * EY[q] * EZ[q] * pyz
                                )
                                reg = W[q] / (2.0 * cs2 * cs2) * contraction
                                out[q] = eq[q] + (1.0 - wp) * reg
                        elif ti.static(is_trt):
                            for q in ti.static(range(Q)):
                                qo = OPP[q]
                                f_sym = 0.5 * (g[q] + g[qo])
                                f_asym = 0.5 * (g[q] - g[qo])
                                e_sym = 0.5 * (eq[q] + eq[qo])
                                e_asym = 0.5 * (eq[q] - eq[qo])
                                out[q] = (
                                    g[q]
                                    - wp * (f_sym - e_sym)
                                    - wm * (f_asym - e_asym)
                                )
                        else:
                            for q in ti.static(range(Q)):
                                out[q] = g[q] - wp * (g[q] - eq[q])
                        dst[i, j, k] = out

            return step_kernel

        # Diagnostic: the full strain-rate tensor, six independent components in
        # (xx, yy, zz, xy, xz, yz) order. Not used by the collision -- which needs only
        # |Q| -- but it is what lets the local strain be checked against an analytic
        # field, and that check is the only reason to trust the model's input.
        strain = ti.Vector.field(6, self.dtype, shape=self.shape)

        @ti.kernel
        def strain_field():
            for i, j, k in strain:
                if self.solid[i, j, k] != 0:
                    strain[i, j, k] = ti.Vector.zero(self.dtype, 6)
                else:
                    g = ti.Vector.zero(self.dtype, Q)
                    for q in ti.static(range(Q)):
                        si = (i - EX[q] + nx) % nx
                        sj = (j - EY[q] + ny) % ny
                        sk = (k - EZ[q] + nz) % nz
                        g[q] = self.f[si, sj, sk][q]

                    r = 1.0
                    mx = 0.0
                    my = 0.0
                    mz = 0.0
                    for q in ti.static(range(Q)):
                        r += g[q]
                        mx += EX[q] * g[q]
                        my += EY[q] * g[q]
                        mz += EZ[q] * g[q]
                    fneq = g - feq_shifted(r, ti.Vector([mx / r, my / r, mz / r]))

                    acc = ti.Vector.zero(self.dtype, 6)
                    for q in ti.static(range(Q)):
                        acc[0] += EX[q] * EX[q] * fneq[q]
                        acc[1] += EY[q] * EY[q] * fneq[q]
                        acc[2] += EZ[q] * EZ[q] * fneq[q]
                        acc[3] += EX[q] * EY[q] * fneq[q]
                        acc[4] += EX[q] * EZ[q] * fneq[q]
                        acc[5] += EY[q] * EZ[q] * fneq[q]
                    # S_ab = -Q_ab / (2 rho cs^2 tau), the Phase 1a identity inverted.
                    scale = -1.0 / (2.0 * r * cs2 / w_plus)
                    strain[i, j, k] = acc * scale

        # Closes over the field rather than taking it as an argument: this module uses
        # `from __future__ import annotations`, which turns every annotation into a
        # string, and Taichi cannot read `"ti.template()"` as a type.
        speed = ti.field(self.dtype, shape=self.shape)

        @ti.kernel
        def speed_field():
            for i, j, k in speed:
                speed[i, j, k] = self.vel[i, j, k].norm()

        snap = ti.field(ti.i32, shape=self.shape) if has_rotor else None
        if has_rotor:

            @ti.kernel
            def snapshot_solid():
                for i, j, k in snap:
                    d = blade_at(i - axis_x, j - axis_y, k, self._theta_now[None])
                    v = 0
                    if self.solid[i, j, k] != 0 or d != 0:
                        v = 1
                    snap[i, j, k] = v

            self._snapshot_solid = snapshot_solid
        else:
            self._snapshot_solid = None
        self._solid_snap = snap
        self._begin_step = begin_step if measure else None

        self._initialise = initialise
        self._steps = (make_step(self.f, self.f_new), make_step(self.f_new, self.f))
        self._parity = 0
        self._speed_field = speed_field
        self._speed = speed
        self._strain_field = strain_field
        self._strain = strain

    # ------------------------------------------------------------------ api

    def reset(self) -> None:
        self._initialise()
        self._parity = 0
        self.theta = 0.0
        self._theta_now[None] = 0.0
        self._theta_prev[None] = 0.0
        self._log_head[None] = 0
        self._log_read = 0

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
        if self._begin_step is not None:
            self._begin_step()
            if self._rotor is not None:
                self.theta += self._rotor["omega"]
        self._steps[self._parity]()
        self._parity ^= 1

    def reset_ledger(self) -> None:
        self._led_static[None] = 0.0
        self._led_inject[None] = 0.0
        self._led_remove[None] = 0.0

    def ledger_report(self) -> dict:
        """Cumulative ledger channels since the last reset. Requires ledger=True."""
        if not self.ledger:
            raise RuntimeError("construct the solver with ledger=True to audit")
        return {
            "static_torque_z": float(self._led_static[None]),
            "inject_Lz": float(self._led_inject[None]),
            "remove_Lz": float(self._led_remove[None]),
        }

    def angular_momentum_z(self) -> float:
        """Resolved fluid angular momentum about the set axis, from the fields."""
        rho = self.rho.to_numpy()
        vel = self.vel.to_numpy()
        nx, ny, _ = self.shape
        ax = float(self.axis[None][0]) if hasattr(self, "axis") else (nx - 1) / 2.0
        ay = float(self.axis[None][1]) if hasattr(self, "axis") else (ny - 1) / 2.0
        x = np.arange(nx)[:, None, None] - ax
        y = np.arange(ny)[None, :, None] - ay
        return float((rho * (x * vel[..., 1] - y * vel[..., 0])).sum())

    def drain_torque_log(self) -> np.ndarray:
        """Torque-z history accumulated on the device since the last drain.

        Each `step()` logs the *previous* step's torque, so after N steps the log
        holds N entries: a leading zero from before the first step, then steps
        1..N-1. The current step's torque is still in the accumulator; read it with
        `torque()` after the final step if the last entry matters. Drain at least
        every `_log_ring` steps or the ring wraps over unread history -- the assert
        is loud about it, because silent overwrite would bias the Np average."""
        head = int(self._log_head[None])
        fresh = head - self._log_read
        if fresh == 0:
            return np.array([])
        assert fresh <= self._log_ring, (
            f"torque ring wrapped: {fresh} unread entries > ring {self._log_ring}; "
            "drain more often"
        )
        log = self._torque_log.to_numpy()
        idx = np.arange(self._log_read, head) % self._log_ring
        self._log_read = head
        return log[idx]

    def solid_snapshot(self) -> np.ndarray:
        """Combined static + blade solid mask at the current angle, for comparison
        against the NumPy oracle (`Tank.blade_mask` union `static`)."""
        if self._snapshot_solid is None:
            raise RuntimeError("solid_snapshot needs a rotor")
        self._snapshot_solid()
        return self._solid_snap.to_numpy().astype(bool)

    def set_wall_velocity(self, velocity: np.ndarray) -> None:
        """Per-node wall velocity, shape (nx, ny, nz, 3). The vector generalisation of
        `set_moving_wall`, needed as soon as a wall rotates: u = omega x r differs at
        every node."""
        self.wall_vel.from_numpy(
            np.ascontiguousarray(velocity, dtype=_np_dtype(self.ti, self.dtype))
        )

    def set_axis(self, cx: float, cy: float, cz: float = 0.0) -> None:
        """Point the torque arm is measured from, in node coordinates."""
        self.axis[None] = [float(cx), float(cy), float(cz)]

    def force(self) -> np.ndarray:
        """Momentum handed to the measured solid in the last step (lattice units)."""
        return self._link_force[None].to_numpy()

    def torque(self) -> np.ndarray:
        """Torque on the measured solid in the last step, about `axis` (lattice
        units). For the tank, the power number wants the z component."""
        return self._link_torque[None].to_numpy()

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

    def strain_rate(self) -> np.ndarray:
        """Strain-rate tensor per cell, shape (nx, ny, nz, 3, 3).

        Recovered from the populations via the Phase 1a identity
        `sum_i e_ia e_ib f_i^neq = -2 tau rho cs^2 S_ab`, so it costs no finite
        differences. Solid nodes report zero -- their populations are bounce-back state,
        not a distribution a strain rate can be read from.
        """
        six = self._strain.to_numpy()
        out = np.empty(self.shape + (3, 3), dtype=six.dtype)
        xx, yy, zz, xy, xz, yz = (six[..., i] for i in range(6))
        out[..., 0, 0], out[..., 1, 1], out[..., 2, 2] = xx, yy, zz
        out[..., 0, 1] = out[..., 1, 0] = xy
        out[..., 0, 2] = out[..., 2, 0] = xz
        out[..., 1, 2] = out[..., 2, 1] = yz
        return out

    def compute_strain(self) -> np.ndarray:
        """Run the diagnostic strain kernel and return the tensor."""
        self._strain_field()
        return self.strain_rate()

    def eddy_viscosity(self) -> np.ndarray:
        """Smagorinsky eddy viscosity from the most recent step. Zero when LES is off."""
        return self.nu_t.to_numpy()

    def macroscopic(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        vel = self.vel.to_numpy()
        return self.rho.to_numpy(), vel[..., 0], vel[..., 1], vel[..., 2]

    @property
    def residual_floor(self) -> float:
        """Roughly the smallest residual this precision can express.

        The residual is a relative change in the speed field between checks, so it
        cannot fall below the rounding noise of the arithmetic producing it. Measured on
        the 3D cavity: fp32 plateaus near 1.1e-5 and stays there indefinitely, while
        fp64 keeps falling past 1e-12. A tolerance below this figure can never be met,
        and a solver asked to meet one will burn its whole step budget and then report
        failure on a solution that stopped changing long ago.
        """
        return 1e-5 if self.dtype == self.ti.f32 else 1e-11

    def run(
        self,
        max_steps: int,
        tol: float = 1e-6,
        check_every: int = 500,
        callback: Callable[[int, D3Q19Solver], None] | None = None,
        stagnation_window: int = 8,
        stagnation_ratio: float = 0.02,
    ) -> SolverState3D:
        """Advance to steady state, or until the step budget runs out.

        Convergence is the relative change in the speed field between checks, the same
        criterion the 2D solver uses, so the two are directly comparable.

        There are two ways to converge, and in fp32 the second is the one that fires.
        `tol` is the usual absolute threshold. But in single precision the residual
        floors at `residual_floor` -- around 1e-5 -- so any stricter tolerance is
        unreachable no matter how long the run continues. Stagnation detection catches
        that case: once the residual stops improving by more than `stagnation_ratio`
        between consecutive windows of `stagnation_window` checks, the field has stopped
        changing and the run is done. Verified directly on the 3D cavity, where 900,000
        steps gave the same profile as 250,000 to four decimal places while never once
        satisfying a 1e-6 tolerance.

        Set `stagnation_window=0` to disable and use `tol` alone.
        """
        self._speed_field()
        prev = self._speed.to_numpy()
        residuals: list[tuple[int, float]] = []
        converged = False
        stopped_on = "step cap"
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
                    converged, stopped_on = True, "tolerance"
                    break
                if _has_stagnated(residuals, stagnation_window, stagnation_ratio):
                    converged, stopped_on = True, "stagnation"
                    break

        rho, ux, uy, uz = self.macroscopic()
        return SolverState3D(
            ux=ux, uy=uy, uz=uz, rho=rho,
            steps=step, converged=converged, residuals=residuals,
            stopped_on=stopped_on,
        )


def _has_stagnated(residuals, window: int, ratio: float) -> bool:
    """True once the residual has stopped improving between consecutive windows.

    Compares the mean of the last `window` residuals against the mean of the `window`
    before it. Averaging matters: the fp32 residual jitters by a few percent around its
    floor, so comparing individual checks would either fire early on a lucky pair or
    never fire at all.
    """
    if window <= 0 or len(residuals) < 2 * window:
        return False
    values = [r for _, r in residuals]
    recent = sum(values[-window:]) / window
    earlier = sum(values[-2 * window:-window]) / window
    if earlier <= 0.0:
        return True
    return (earlier - recent) / earlier < ratio


def _np_dtype(ti, dtype):
    return np.float32 if dtype == ti.f32 else np.float64


def lid_driven_cavity_3d(
    n: int = 64,
    nz: int = 4,
    reynolds: float = 1000.0,
    lid_velocity: float = 0.1,
    collision: Literal["bgk", "trt", "regularized"] = "trt",
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
