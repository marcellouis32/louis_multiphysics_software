"""Parametric voxelisation of the stirred tank: vessel, baffles, Rushton impeller.

No STL anywhere in this module, deliberately. The standard tank is fully parametric --
cylinder vessel, box baffles, disc-and-blade impeller -- and every dimension comes from
the schema. STL import stays deferred until a custom-geometry case actually needs it.

Everything here is pure NumPy and CPU-side, in the same spirit as `d3q19.py`: this is
the *oracle*. Phase 2b evaluates the rotating impeller analytically inside the GPU
kernel, and at any fixed angle that in-kernel test must reproduce these masks exactly.
A geometry bug is therefore caught by an array comparison, not by staring at a wrong
power number.

Conventions, fixed here and relied on everywhere downstream:

  * Index order (x, y, z), matching `solver3d.py`. The shaft axis is z, pointing up.
  * The tank's fluid bore spans `cells_across_tank` cells, so `dx = T / n` exactly as
    `Numerics.dx()` defines it. The domain adds a one-cell solid rim on every face,
    and the axis sits at the domain centre `((nx-1)/2, (ny-1)/2)` -- a point symmetric
    under 90-degree grid rotations, which is what lets the tests demand exact equality
    between a rotated mask and a rotated *array*.
  * Halfway bounce-back places the effective wall midway between the last fluid and
    first solid node, so the staircase circle's effective radius is only known to
    within about half a cell. That uncertainty is measured, not assumed: the
    Taylor-Couette ladder in `validation/couette.py` exists to put a number on it.

The impeller uses the textbook Rushton proportions for everything the schema does not
specify: disc diameter 3/4 D, blade length D/4 (from the disc rim inward of the tip),
plate thickness D/50. Blade height comes from the schema (`blade_width_m`, D/5 in the
standard case). Hub and shaft are modest cylinders; they contribute a few percent of
the torque at most -- the blades and disc dominate -- and their proportions are stated
rather than load-bearing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lms.lbm.units import LatticeScales
from lms.schema.case import Case

# Deterministic tie-breaking for boundary-plane nodes. Node coordinates about the axis
# are half-integers, so a plate clamped to one cell thick puts its faces at exactly
# |y| = 0.5 -- and whether those nodes are solid would otherwise depend on 1e-16
# rounding in the rotation, making mask(theta) differ from a rotated mask(0) on a
# scattering of boundary cells. The epsilon makes inclusion deterministic. Consequence,
# stated rather than hidden: a clamped plate catches BOTH half-integer planes and is
# effectively two cells thick. At the milestone resolution that is ~4% of D instead of
# the textbook 2%; the resolution ladder in 2b measures what that costs.
_TIE = 1e-6

# Textbook Rushton proportions, as fractions of the impeller diameter D.
DISC_DIAMETER = 0.75
BLADE_LENGTH = 0.25
PLATE_THICKNESS = 0.02      # disc and blade thickness; at least one cell when voxelised
HUB_DIAMETER = 0.20
SHAFT_DIAMETER = 0.10


def _node_coords(shape: tuple[int, int, int]):
    """Node-centre coordinates with the origin on the tank axis at the domain centre
    of the x-y plane. z stays a plain index, since heights are measured from the
    bottom wall."""
    nx, ny, nz = shape
    i, j, k = np.mgrid[0:nx, 0:ny, 0:nz].astype(np.float64)
    return i - (nx - 1) / 2.0, j - (ny - 1) / 2.0, k


@dataclass
class Tank:
    """A voxelised tank: static solids, and the impeller as a function of angle.

    `static_solid` is the vessel, baffles, bottom and lid. The impeller is *not* baked
    into it -- it is a function of the shaft angle, because that is what rotation means.
    `omega_shaft` is radians per step; wall velocities scale with it.
    """

    shape: tuple[int, int, int]
    static_solid: np.ndarray
    radius: float                      # fluid bore radius, cells
    impeller: dict = field(default_factory=dict)
    omega_shaft: float = 0.0

    def impeller_static_mask(self) -> np.ndarray:
        """Disc, hub and shaft: the axisymmetric parts of the impeller.

        Axisymmetry is worth exploiting, not just noting. These parts rotate without
        changing shape, so they live in the solver's *static* solid field with a
        static wall velocity omega x r -- machinery verified since Phase 1a. Only the
        blades need the dynamic in-kernel treatment, which confines the fresh-node
        problem to the blade-swept band.
        """
        p = self.impeller
        x, y, z = _node_coords(self.shape)
        r = np.hypot(x, y)
        zc = p["z_centre"]
        solid = (r <= p["disc_radius"]) & (np.abs(z - zc) <= p["thickness"] / 2.0)
        solid |= (r <= p["hub_radius"]) & (np.abs(z - zc) <= p["blade_height"] / 2.0)
        solid |= (r <= p["shaft_radius"]) & (z >= zc)
        return solid

    def blade_mask(self, theta: float = 0.0) -> np.ndarray:
        """The blades alone, at shaft angle theta.

        Implemented by rotating the *query points* by -theta and testing against the
        blade at its reference position -- exactly the transformation the GPU kernel
        applies, which is what makes this the oracle for it.
        """
        p = self.impeller
        x, y, z = _node_coords(self.shape)

        n_blades = p["n_blades"]
        half_t = p["thickness"] / 2.0 + _TIE
        half_h = p["blade_height"] / 2.0
        in_height = np.abs(z - p["z_centre"]) <= half_h
        solid = np.zeros(self.shape, dtype=bool)
        for b in range(n_blades):
            angle = theta + 2.0 * np.pi * b / n_blades
            c, s = np.cos(-angle), np.sin(-angle)
            xb = x * c - y * s          # blade frame: blade lies along +x
            yb = x * s + y * c
            solid |= (
                in_height
                & (np.abs(yb) <= half_t)
                & (xb >= p["blade_inner"])
                & (xb <= p["blade_outer"])
            )
        return solid

    def impeller_mask(self, theta: float = 0.0) -> np.ndarray:
        """The whole impeller at shaft angle theta: static parts plus blades."""
        return self.impeller_static_mask() | self.blade_mask(theta)

    def rotor_params(self, boundary: str = "halfway") -> dict:
        """Blade geometry in the form the solver kernel consumes.

        `half_t` carries the same tie epsilon as the oracle, so the kernel and
        `blade_mask` evaluate the *identical* expression -- the mask-for-mask
        comparison between them is only meaningful because of that.
        """
        if boundary not in ("halfway", "bouzidi"):
            raise ValueError(f"boundary must be halfway or bouzidi, got {boundary!r}")
        p = self.impeller
        return {
            "z_centre": p["z_centre"],
            "blade_inner": p["blade_inner"],
            "blade_outer": p["blade_outer"],
            "half_t": p["thickness"] / 2.0 + _TIE,
            "half_t_true": p["thickness"] / 2.0,
            "half_h": p["blade_height"] / 2.0,
            "n_blades": p["n_blades"],
            "omega": self.omega_shaft,
            "boundary": boundary,
        }

    def blade_link_fraction(
        self, point: np.ndarray, direction: np.ndarray, theta: float
    ) -> float:
        """Where along a lattice link the blade surface sits: q in (0, 1], or nan.

        `point` is a fluid node (axis-relative x, y and plain z), `direction` a lattice
        vector whose far end lies inside a blade. Each blade is an axis-aligned box *in
        its own frame*, so the exact crossing is a slab (ray-box) test after folding
        the node into the nearest blade's sector -- the same fold `blade_mask` and the
        kernel use, so all three agree on which blade owns the neighbourhood.

        This is the geometric input Bouzidi interpolation runs on, and the reason no
        per-link storage exists: q is exact, closed-form, and recomputable per step for
        a rotating blade. The box uses the *true* half thickness, without the tie
        epsilon -- ties exist to make node membership deterministic, but the wall is
        where the wall is.
        """
        p = self.impeller
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        sector = 2.0 * np.pi / p["n_blades"]
        phi = np.arctan2(y, x)
        angle = theta + sector * np.round((phi - theta) / sector)
        c, s = np.cos(angle), np.sin(angle)

        # Rotate point and direction into the blade frame (blade along +x).
        px, py = x * c + y * s, -x * s + y * c
        ex, ey = direction[0] * c + direction[1] * s, -direction[0] * s + direction[1] * c
        ez = float(direction[2])

        lo = np.array([p["blade_inner"], -p["thickness"] / 2.0,
                       p["z_centre"] - p["blade_height"] / 2.0])
        hi = np.array([p["blade_outer"], p["thickness"] / 2.0,
                       p["z_centre"] + p["blade_height"] / 2.0])
        origin = np.array([px, py, z])
        vec = np.array([ex, ey, ez])

        t_lo, t_hi = 0.0, np.inf
        for a in range(3):
            if abs(vec[a]) < 1e-12:
                if origin[a] < lo[a] or origin[a] > hi[a]:
                    return float("nan")
                continue
            t1 = (lo[a] - origin[a]) / vec[a]
            t2 = (hi[a] - origin[a]) / vec[a]
            t_lo = max(t_lo, min(t1, t2))
            t_hi = min(t_hi, max(t1, t2))
        if t_lo > t_hi or t_lo > 1.0:
            return float("nan")
        return float(min(max(t_lo, 1e-6), 1.0))

    def impeller_wall_velocity(self, mask: np.ndarray) -> np.ndarray:
        """Per-node wall velocity u = omega x r for the given impeller mask,
        shape (nx, ny, nz, 3)."""
        x, y, _ = _node_coords(self.shape)
        vel = np.zeros(self.shape + (3,))
        vel[..., 0] = -self.omega_shaft * y
        vel[..., 1] = self.omega_shaft * x
        vel[mask == 0] = 0.0
        return np.where(mask[..., None], vel, 0.0)


def vessel_mask(shape: tuple[int, int, int], radius: float) -> np.ndarray:
    """Cylindrical shell plus flat bottom and lid.

    The lid is no-slip, an approximation stated rather than hidden: the real liquid
    surface is free, but in a baffled tank at Fr ~ 0.25 there is no vortex and the
    literature puts the effect on the power number at a few percent. The escalation, if
    the Np budget ever demands it, is a specular-reflection free-slip lid.
    """
    x, y, _ = _node_coords(shape)
    solid = np.hypot(x, y) > radius
    solid[:, :, 0] = True
    solid[:, :, -1] = True
    return solid


def baffle_mask(
    shape: tuple[int, int, int],
    radius: float,
    count: int,
    width: float,
    wall_offset: float,
    thickness: float,
) -> np.ndarray:
    """`count` flat baffles, evenly spaced, each a radial plate spanning the full
    liquid height. All lengths in cells."""
    x, y, _ = _node_coords(shape)
    solid = np.zeros(shape, dtype=bool)
    outer = radius - wall_offset
    inner = outer - width
    for b in range(count):
        angle = 2.0 * np.pi * b / count
        c, s = np.cos(-angle), np.sin(-angle)
        xb = x * c - y * s
        yb = x * s + y * c
        solid |= (np.abs(yb) <= thickness / 2.0 + _TIE) & (xb >= inner) & (xb <= outer)
    # Baffles exist only in the fluid column, not through the bottom/lid walls; the
    # vessel mask already owns those planes.
    solid[:, :, 0] = False
    solid[:, :, -1] = False
    return solid


def tank_from_case(case: Case, scales: LatticeScales) -> Tank:
    """Build the full tank a Case describes, in lattice units."""
    shape = (scales.nx, scales.ny, scales.nz)
    imp = case.primary_impeller
    dx = scales.dx_m

    radius = case.numerics.cells_across_tank / 2.0
    solid = vessel_mask(shape, radius)

    baffles = case.geometry.baffles
    if baffles.count > 0:
        solid |= baffle_mask(
            shape,
            radius,
            count=baffles.count,
            width=baffles.width_ratio * case.geometry.vessel.diameter_m / dx,
            wall_offset=baffles.wall_offset_ratio * case.geometry.vessel.diameter_m / dx,
            thickness=max(baffles.thickness_m / dx, 1.0),
        )

    d_lat = imp.diameter_m / dx
    # Plates thinner than one cell would voxelise to nothing; clamp and say so. At the
    # milestone resolution D/50 is ~0.85 cells, so the clamp is active and the blades
    # are one cell thick -- the exact situation the staircase-vs-Bouzidi decision in
    # the plan is about.
    thickness = max(PLATE_THICKNESS * d_lat, 1.0)
    tank = Tank(
        shape=shape,
        static_solid=solid,
        radius=radius,
        impeller={
            "z_centre": imp.clearance_m / dx,      # above the bottom wall plane
            "disc_radius": DISC_DIAMETER * d_lat / 2.0,
            "blade_outer": d_lat / 2.0,
            "blade_inner": d_lat / 2.0 - BLADE_LENGTH * d_lat,
            "blade_height": (imp.blade_width_m or 0.2 * imp.diameter_m) / dx,
            "hub_radius": HUB_DIAMETER * d_lat / 2.0,
            "shaft_radius": SHAFT_DIAMETER * d_lat / 2.0,
            "thickness": thickness,
            "n_blades": imp.n_blades or 6,
        },
        omega_shaft=scales.omega_shaft,
    )
    return tank
