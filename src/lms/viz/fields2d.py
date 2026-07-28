"""2D field rendering.

The flagship renderer is `plot_lic`: line integral convolution smears a noise
texture along streamlines, so flow *structure* is visible everywhere at once
rather than only where a hand-placed streamline seed happened to land. Quiver
plots and sparse streamlines both lie about recirculation zones; LIC does not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import PowerNorm
from scipy.ndimage import map_coordinates

from lms.viz.style import (
    ACCENT,
    ACCENT_WARM,
    BACKGROUND,
    DIVERGING_DARK,
    FLOW,
    MUTED,
    annotate,
    colorbar,
    house_style,
    plt,
    signed_asinh_norm,
)


def _equalize(texture: np.ndarray) -> np.ndarray:
    """Histogram-equalise to a uniform [0, 1]. LIC output is near-Gaussian and
    looks flat and grey without this; equalising is what makes the streaks read."""
    flat = texture.ravel()
    order = flat.argsort()
    ranks = np.empty(flat.size, dtype=np.float64)
    ranks[order] = np.arange(flat.size)
    return (ranks / max(flat.size - 1, 1)).reshape(texture.shape)


def line_integral_convolution(
    ux: np.ndarray,
    uy: np.ndarray,
    n_steps: int = 42,
    step_size: float = 0.7,
    seed: int = 0,
    upsample: int = 3,
) -> np.ndarray:
    """Streamline-aligned texture, normalised to [0, 1].

    The texture is generated at `upsample` times the grid resolution. Rendering
    LIC at grid resolution wastes it: the noise ends up at the same scale as the
    cells, so the streaks alias into mush. Integration happens in upsampled pixel
    space while direction is sampled from the coarse velocity field.

    Both forward and backward integration with a linear taper keeps the texture
    symmetric and free of direction-of-travel smearing.
    """
    ny, nx = ux.shape
    hy, hx = ny * upsample, nx * upsample
    noise = np.random.default_rng(seed).random((hy, hx))

    speed = np.hypot(ux, uy)
    scale = np.where(speed > 1e-12, 1.0 / np.maximum(speed, 1e-12), 0.0)
    vx, vy = ux * scale, uy * scale

    grid_y, grid_x = np.mgrid[0:hy, 0:hx].astype(np.float64)
    accum = np.zeros((hy, hx), dtype=np.float64)
    total_weight = 0.0

    for direction in (1.0, -1.0):
        px, py = grid_x.copy(), grid_y.copy()
        for k in range(n_steps):
            coarse = np.stack([(py / upsample).ravel(), (px / upsample).ravel()])
            sx = map_coordinates(vx, coarse, order=1, mode="nearest").reshape(hy, hx)
            sy = map_coordinates(vy, coarse, order=1, mode="nearest").reshape(hy, hx)
            px = np.clip(px + direction * step_size * sx, 0, hx - 1)
            py = np.clip(py + direction * step_size * sy, 0, hy - 1)

            weight = 1.0 - k / n_steps
            accum += weight * map_coordinates(
                noise, np.stack([py.ravel(), px.ravel()]), order=1, mode="nearest"
            ).reshape(hy, hx)
            total_weight += weight

    return _equalize(accum / max(total_weight, 1e-12))


def _shade(speed: np.ndarray, texture: np.ndarray, cmap, norm, depth: float = 0.5):
    """Composite the LIC texture as luminance over a colour-mapped scalar field.

    Speed is resampled up to the texture resolution. The gain is centred on 1.0
    rather than darkening, and a cubic highlight term adds specular sparkle on the
    bright streaks, which is what stops the result looking like flat grey noise.
    """
    hy, hx = texture.shape
    sy, sx = speed.shape
    if (sy, sx) != (hy, hx):
        yy, xx = np.mgrid[0:hy, 0:hx].astype(np.float64)
        coords = np.stack([(yy * (sy - 1) / (hy - 1)).ravel(), (xx * (sx - 1) / (hx - 1)).ravel()])
        speed = map_coordinates(speed, coords, order=1, mode="nearest").reshape(hy, hx)

    rgba = plt.get_cmap(cmap)(norm(speed))
    gain = (1.0 - depth) + depth * 2.0 * texture
    rgba[..., :3] *= gain[..., None]
    rgba[..., :3] += 0.22 * (texture**3)[..., None]
    return np.clip(rgba, 0.0, 1.0)


def _upsample_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    hy, hx = shape
    my, mx = mask.shape
    yy, xx = np.mgrid[0:hy, 0:hx]
    return mask[
        np.clip((yy * my) // hy, 0, my - 1),
        np.clip((xx * mx) // hx, 0, mx - 1),
    ]


def plot_lic(
    ux: np.ndarray,
    uy: np.ndarray,
    title: str = "Velocity field",
    subtitle: str = "",
    cmap=FLOW,
    save: str | Path | None = None,
    n_steps: int = 42,
    show_streamlines: bool = False,
    solid: np.ndarray | None = None,
):
    speed = np.hypot(ux, uy)
    texture = line_integral_convolution(ux, uy, n_steps=n_steps)
    # Recirculating flows span orders of magnitude in speed. A linear scale crushes
    # the slow core to black and hides exactly the structure that matters.
    norm = PowerNorm(gamma=0.55, vmin=0.0, vmax=float(np.percentile(speed, 99.5)) or 1.0)

    image = _shade(speed, texture, cmap, norm)
    if solid is not None:
        # Solid cells have no velocity, so LIC integrates pure noise there and the
        # walls come out as speckle. Paint them flat instead.
        wall = _upsample_mask(solid.astype(bool), texture.shape)
        image[wall] = [0.10, 0.13, 0.17, 1.0]

    with house_style():
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        ax.imshow(image, origin="lower", interpolation="bilinear")

        if show_streamlines:
            ny, nx = ux.shape
            ys, xs = np.arange(ny), np.arange(nx)
            ax.streamplot(
                xs, ys, ux, uy, color="white", linewidth=0.5, density=1.1, arrowsize=0.7
            )

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar(fig, mappable, ax, "speed  [lattice units]")

        ax.set_title(title)
        if subtitle:
            annotate(ax, subtitle, "lower left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if save:
            fig.savefig(save)
        return fig


def plot_vorticity(
    vorticity: np.ndarray,
    title: str = "Vorticity",
    subtitle: str = "",
    save: str | Path | None = None,
    solid: np.ndarray | None = None,
):
    mask = solid.astype(bool) if solid is not None else None
    field = np.ma.masked_where(mask, vorticity) if mask is not None else vorticity
    with house_style():
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        norm = signed_asinh_norm(np.asarray(vorticity), mask=mask)
        cmap = DIVERGING_DARK.copy()
        cmap.set_bad("#1a222c")
        im = ax.imshow(field, origin="lower", cmap=cmap, norm=norm)
        colorbar(fig, im, ax, r"$\omega_z$  [lattice units]")
        ax.set_title(title)
        if subtitle:
            annotate(ax, subtitle, "lower left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if save:
            fig.savefig(save)
        return fig


def plot_ghia_comparison(
    y: np.ndarray,
    u: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    ghia_y: np.ndarray,
    ghia_u: np.ndarray,
    ghia_x: np.ndarray,
    ghia_v: np.ndarray,
    reynolds: float,
    resolution: int,
    rms_u: float,
    rms_v: float,
    save: str | Path | None = None,
):
    with house_style():
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

        ax = axes[0]
        ax.plot(u, y, color=ACCENT, label=f"LBM  {resolution}²")
        ax.scatter(
            ghia_u, ghia_y, s=26, facecolor="none", edgecolor=ACCENT_WARM,
            linewidth=1.3, label="Ghia et al. 1982", zorder=3,
        )
        ax.axvline(0, color=MUTED, linewidth=0.7, alpha=0.5)
        ax.set_xlabel(r"$u\,/\,U_{lid}$")
        ax.set_ylabel(r"$y\,/\,L$")
        ax.set_title("u along vertical centreline")
        ax.legend(loc="upper left")
        annotate(ax, f"RMS = {rms_u:.4f}", "lower right")

        ax = axes[1]
        ax.plot(x, v, color=ACCENT, label=f"LBM  {resolution}²")
        ax.scatter(
            ghia_x, ghia_v, s=26, facecolor="none", edgecolor=ACCENT_WARM,
            linewidth=1.3, label="Ghia et al. 1982", zorder=3,
        )
        ax.axhline(0, color=MUTED, linewidth=0.7, alpha=0.5)
        ax.set_xlabel(r"$x\,/\,L$")
        ax.set_ylabel(r"$v\,/\,U_{lid}$")
        ax.set_title("v along horizontal centreline")
        ax.legend(loc="lower left")
        annotate(ax, f"RMS = {rms_v:.4f}", "upper right")

        fig.suptitle(
            f"Lid-driven cavity verification   ·   Re = {reynolds:g}",
            x=0.012, ha="left", fontsize=12.5, weight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        if save:
            fig.savefig(save)
        return fig


def plot_convergence(residuals, save: str | Path | None = None):
    steps = [s for s, _ in residuals]
    values = [r for _, r in residuals]
    with house_style():
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        ax.semilogy(steps, values, color=ACCENT)
        ax.set_xlabel("timestep")
        ax.set_ylabel("relative change in |u|")
        ax.set_title("Convergence to steady state")
        if save:
            fig.savefig(save)
        return fig


def _levels_from_norm(norm, count: int) -> np.ndarray:
    """Contour levels spaced evenly in *colour* space rather than data space, so a
    nonlinear norm (power, asinh) puts its resolution where the structure is."""
    levels = np.asarray(norm.inverse(np.linspace(0.0, 1.0, count + 1)), dtype=float)
    levels = np.unique(levels[np.isfinite(levels)])
    return levels


def plot_contours(
    field: np.ndarray,
    label: str,
    title: str = "",
    subtitle: str = "",
    cmap=FLOW,
    norm=None,
    n_filled: int = 28,
    n_lines: int = 14,
    solid: np.ndarray | None = None,
    extent: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    save: str | Path | None = None,
):
    """Filled contours with overlaid isolines - the conventional CFD deliverable.

    Complements the LIC renderer rather than replacing it: contours give readable
    quantitative levels, LIC gives continuous structure.
    """
    mask = solid.astype(bool) if solid is not None else None
    data = np.ma.masked_where(mask, field) if mask is not None else np.ma.asarray(field)
    if norm is None:
        norm = PowerNorm(gamma=1.0, vmin=float(data.min()), vmax=float(data.max()))

    filled = _levels_from_norm(norm, n_filled)
    lines = _levels_from_norm(norm, n_lines)

    ny, nx = field.shape
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[2], extent[3], ny)

    with house_style():
        fig, ax = plt.subplots(figsize=(6.6, 6.0))
        cf = ax.contourf(xs, ys, data, levels=filled, cmap=cmap, norm=norm, extend="both")
        # Kills the hairline seams contourf leaves between bands. ContourSet became a
        # Collection in matplotlib 3.8; .collections was removed in 3.10.
        if hasattr(cf, "set_edgecolor"):
            cf.set_edgecolor("face")
        else:
            for c in cf.collections:
                c.set_edgecolor("face")
        ax.contour(xs, ys, data, levels=lines, colors="white", linewidths=0.45, alpha=0.32)

        colorbar(fig, plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax, label)
        ax.set_title(title)
        if subtitle:
            annotate(ax, subtitle, "lower left")
        ax.set_xlabel("$x\\,/\\,L$")
        ax.set_ylabel("$y\\,/\\,L$")
        ax.set_aspect("equal")
        ax.grid(False)
        if save:
            fig.savefig(save)
        return fig


def plot_streamfunction(
    psi: np.ndarray,
    title: str = "Stream function",
    subtitle: str = "",
    solid: np.ndarray | None = None,
    extent: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    markers: dict[str, tuple[float, float]] | None = None,
    save: str | Path | None = None,
):
    """Iso-contours of psi are exact streamlines.

    Levels are split between the primary circulation and the much weaker corner
    vortices; a single linear set would render the corner eddies as blank space,
    which is precisely where the interesting recirculation lives.
    """
    mask = solid.astype(bool) if solid is not None else None
    data = np.ma.masked_where(mask, psi) if mask is not None else np.ma.asarray(psi)
    lo, hi = float(data.min()), float(data.max())

    strong = np.linspace(lo, 0.0, 16)[:-1]
    weak = np.sign(hi) * np.geomspace(max(abs(hi), 1e-12) * 1e-3, max(abs(hi), 1e-12), 8)
    levels = np.unique(np.concatenate([strong, [0.0], weak]))

    ny, nx = psi.shape
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[2], extent[3], ny)

    with house_style():
        fig, ax = plt.subplots(figsize=(6.6, 6.0))
        ax.contourf(xs, ys, data, levels=40, cmap=FLOW, alpha=0.85)
        ax.contour(xs, ys, data, levels=levels, colors=ACCENT, linewidths=0.7, alpha=0.9)

        if markers:
            for name, (mx, my) in markers.items():
                style = dict(marker="+", ms=13, mew=2.0)
                colour = ACCENT_WARM if "Ghia" in name else "white"
                if "Ghia" in name:
                    style = dict(marker="o", ms=11, mew=1.8, mfc="none")
                ax.plot(mx, my, color=colour, linestyle="none", label=name, **style)
            ax.legend(loc="upper left", labelcolor=MUTED)

        ax.set_title(title)
        if subtitle:
            annotate(ax, subtitle, "lower right")
        ax.set_xlabel("$x\\,/\\,L$")
        ax.set_ylabel("$y\\,/\\,L$")
        ax.set_aspect("equal")
        ax.grid(False)
        if save:
            fig.savefig(save)
        return fig


def plot_order_of_accuracy(
    ladders: dict,
    title: str = "Order of accuracy",
    subtitle: str = "",
    save: str | Path | None = None,
):
    """Log-log error against resolution, with a fitted slope per refinement ladder.

    `ladders` maps a label to (sizes, errors, fitted_order). The reference triangles
    are drawn at the fitted slope rather than at an idealised 1 or 2, because the
    honest question is what the solver actually achieved, not what it should have.
    """
    with house_style():
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        colours = [ACCENT, ACCENT_WARM]

        for (label, (sizes, errors, order)), colour in zip(ladders.items(), colours):
            sizes = np.asarray(sizes, dtype=float)
            errors = np.asarray(errors, dtype=float)
            ax.loglog(
                sizes, errors, "o-", color=colour, linewidth=1.6, markersize=6.5,
                markerfacecolor=BACKGROUND, markeredgewidth=1.6,
                label=f"{label}   (slope {order:.2f})",
            )

        # Second-order guide, anchored just under the coarsest point of the first
        # ladder so the eye can compare slopes directly instead of across the axes.
        all_sizes = np.array(
            sorted({s for sizes, _, _ in ladders.values() for s in sizes}), dtype=float
        )
        first_sizes, first_errors, _ = next(iter(ladders.values()))
        anchor = np.asarray(first_errors, dtype=float)[np.argmin(first_sizes)] * 0.45
        guide = anchor * (all_sizes[0] / all_sizes) ** 2
        ax.loglog(all_sizes, guide, "--", color=MUTED, linewidth=1.1, alpha=0.75)
        mid = len(all_sizes) // 2
        ax.annotate(
            r"slope $-2$",
            xy=(all_sizes[mid], guide[mid]),
            xytext=(6, -14),
            textcoords="offset points",
            color=MUTED,
            fontsize=9,
        )

        ax.set_xlabel("lattice resolution  $n$")
        ax.set_ylabel(r"relative $L_2$ velocity error")
        ax.set_xticks(all_sizes)
        ax.set_xticklabels([f"{int(s)}" for s in all_sizes])
        ax.minorticks_off()
        ax.legend(loc="lower left", framealpha=0.0)
        ax.set_title(title, loc="left", pad=14)
        if subtitle:
            ax.text(
                0.0, 1.015, subtitle, transform=ax.transAxes,
                fontsize=9, color=MUTED, ha="left", va="bottom",
            )
        fig.tight_layout()
        if save:
            fig.savefig(save)
        return fig
