"""Passive-scalar transport on a D3Q7 lattice: the tracer the blend time is measured on.

A scalar needs far less lattice than momentum: no pressure, no stress tensor, just
concentration and its flux, so seven velocities (rest + six axes) carry it. The lattice
sound speed of D3Q7 is cs^2 = 1/4, giving

    D = (tau - 1/2) / 4          g_i^eq = w_i c (1 + 4 e_i . u)

with weights 1/4 (rest) and 1/8 (axes).

**The diffusivity is almost entirely turbulent, and that is stated, not hidden.** The
schema's Sc = 1000 puts the molecular Batchelor scale ~32x below Kolmogorov -- no
feasible grid resolves it, ours included. The effective diffusivity is

    D_eff = nu / Sc + nu_t / Sc_t

with Sc_t = 0.7 by default, read per cell from the flow solver's eddy-viscosity field.
In the turbulent tank the second term dominates by orders of magnitude, which is the
standard closure and also the honest description of what is actually being computed:
turbulent blend time is controlled by large-scale convection plus modelled small-scale
mixing, not by molecular diffusion.

One approximation specific to the rotating tank: the scalar bounces back off *static*
solids only, and treats blade-swept cells as fluid with the local (zero) velocity. The
swept torus is one to two percent of the tank volume, so the error this admits is
bounded by that fraction -- far inside the +-30% the blend-time comparison targets --
and it spares the scalar lattice the whole fresh-node machinery.
"""

from __future__ import annotations

import numpy as np

from lms.lbm.solver3d import _taichi

# Rest + axis pairs, ordered so OPPOSITE is index arithmetic like d3q19.
EX7 = np.array([0, 1, -1, 0, 0, 0, 0], dtype=np.int32)
EY7 = np.array([0, 0, 0, 1, -1, 0, 0], dtype=np.int32)
EZ7 = np.array([0, 0, 0, 0, 0, 1, -1], dtype=np.int32)
W7 = np.array([1 / 4] + [1 / 8] * 6)
OPP7 = np.array([0, 2, 1, 4, 3, 6, 5], dtype=np.int32)
CS2_7 = 1.0 / 4.0


def diffusivity_to_tau(d: float) -> float:
    return 4.0 * d + 0.5


class ScalarD3Q7:
    """Advection-diffusion of one passive scalar, coupled one-way to a flow solver.

    `vel` and `nu_t` are the *flow solver's own Taichi fields*, captured by reference:
    the scalar always sees the current velocity and eddy viscosity with no copies and
    no host round-trip per step.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        d_molecular: float,
        solid: np.ndarray,
        vel,
        nu_t=None,
        sc_t: float = 0.7,
        dtype=None,
    ) -> None:
        ti = _taichi()
        self.ti = ti
        self.shape = shape
        self.dtype = dtype if dtype is not None else ti.lang.impl.get_runtime().default_fp
        self.d_molecular = float(d_molecular)
        self.sc_t = float(sc_t)

        self.g = ti.Vector.field(7, self.dtype, shape=shape)
        self.g_new = ti.Vector.field(7, self.dtype, shape=shape)
        self.c = ti.field(self.dtype, shape=shape)
        self.solid = ti.field(ti.i32, shape=shape)
        self.solid.from_numpy((solid != 0).astype(np.int32))

        self._vel = vel
        self._nu_t = nu_t
        self._build_kernels()

    def _build_kernels(self) -> None:
        ti = self.ti
        nx, ny, nz = self.shape
        d_mol = self.d_molecular
        inv_sc_t = 1.0 / self.sc_t
        has_nu_t = self._nu_t is not None
        vel = self._vel
        nu_t = self._nu_t
        EX, EY, EZ, W, OPP = EX7, EY7, EZ7, W7, OPP7

        @ti.func
        def geq(c, u):
            out = ti.Vector.zero(self.dtype, 7)
            for q in ti.static(range(7)):
                eu = EX[q] * u[0] + EY[q] * u[1] + EZ[q] * u[2]
                out[q] = W[q] * c * (1.0 + 4.0 * eu)
            return out

        @ti.kernel
        def set_field():
            for i, j, k in self.g:
                u = vel[i, j, k]
                self.g[i, j, k] = geq(self.c[i, j, k], u)
                self.g_new[i, j, k] = self.g[i, j, k]

        def make_step(src, dst):
            @ti.kernel
            def step_kernel():
                for i, j, k in src:
                    g = ti.Vector.zero(self.dtype, 7)
                    for q in ti.static(range(7)):
                        si = (i - EX[q] + nx) % nx
                        sj = (j - EY[q] + ny) % ny
                        sk = (k - EZ[q] + nz) % nz
                        g[q] = src[si, sj, sk][q]

                    if self.solid[i, j, k] != 0:
                        # Zero-flux wall: plain bounce-back conserves the scalar
                        # exactly, which is the property the CoV diagnostic rests on.
                        b = ti.Vector.zero(self.dtype, 7)
                        for q in ti.static(range(7)):
                            b[q] = g[OPP[q]]
                        dst[i, j, k] = b
                        self.c[i, j, k] = 0.0
                    else:
                        c = 0.0
                        for q in ti.static(range(7)):
                            c += g[q]
                        self.c[i, j, k] = c

                        # Per-cell relaxation from the local eddy viscosity: the
                        # turbulent diffusivity varies as strongly across the tank as
                        # the turbulence does, so a single global tau would mix the
                        # bulk with the jet's coefficient.
                        d_local = d_mol
                        if ti.static(has_nu_t):
                            d_local += nu_t[i, j, k] * inv_sc_t
                        omega = 1.0 / (4.0 * d_local + 0.5)

                        u = vel[i, j, k]
                        eq = geq(c, u)
                        out = ti.Vector.zero(self.dtype, 7)
                        for q in ti.static(range(7)):
                            out[q] = g[q] - omega * (g[q] - eq[q])
                        dst[i, j, k] = out

            return step_kernel

        self._set_field = set_field
        self._steps = (make_step(self.g, self.g_new), make_step(self.g_new, self.g))
        self._parity = 0

    def set_concentration(self, c: np.ndarray) -> None:
        """Initialise the scalar field at local equilibrium with the current flow."""
        self.c.from_numpy(np.ascontiguousarray(
            c, dtype=np.float32 if self.dtype == self.ti.f32 else np.float64
        ))
        self._set_field()
        self._parity = 0

    def step(self) -> None:
        self._steps[self._parity]()
        self._parity ^= 1

    def concentration(self) -> np.ndarray:
        return self.c.to_numpy()

    def total(self) -> float:
        """Total scalar content; conserved exactly by streaming and bounce-back."""
        g = self.g.to_numpy() if self._parity == 0 else self.g_new.to_numpy()
        return float(g.sum())
