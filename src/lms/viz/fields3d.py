"""3D field rendering, by slicing back into the 2D renderers.

Deliberately thin. `fields2d.py` and `style.py` already carry the house language --
asinh norms, the FLOW and DIVERGING_DARK maps, LIC, contour conventions -- and all of it
applies unchanged to a plane cut through a 3D field. Building a second rendering stack
for 3D would mean two sets of colour decisions drifting apart.

What 3D genuinely adds is the need to say *where* the plane is and to show more than one
at once, since a single slice through a 3D flow is a claim about the whole field that a
single slice cannot support. Hence `plot_orthogonal_slices`, which puts the three centre
planes side by side on one shared colour scale.

Volume rendering and isosurfaces are a larger conversation and would pull in pyvista;
slices first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lms.viz.fields2d import _draw_contour_layers, contour_inputs, plot_contours
from lms.viz.style import FLOW, MUTED, annotate, colorbar, house_style, plt

AXES = {"x": 0, "y": 1, "z": 2}


def slice_plane(field: np.ndarray, axis: str = "z", index: int | None = None) -> np.ndarray:
    """Take a plane through a 3D field, normal to `axis`.

    Returns it transposed into the (row, column) order matplotlib draws, so that the
    horizontal axis is the first remaining coordinate and the vertical the second --
    which is what makes a y-normal slice of a cavity look like the cavity.
    """
    if axis not in AXES:
        raise ValueError(f"axis must be one of {sorted(AXES)}, got {axis!r}")
    ax = AXES[axis]
    if index is None:
        index = field.shape[ax] // 2
    return np.take(field, index, axis=ax).T


def in_plane_components(
    ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, axis: str = "z", index: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The two velocity components lying *in* a slice plane, in draw order.

    Streamlines and LIC drawn from the wrong pair are the classic 3D slicing error: the
    picture looks plausible and describes a flow that does not exist. Taking a z-normal
    cut means (ux, uy); a x-normal cut means (uy, uz).
    """
    trio = {"x": (uy, uz), "y": (ux, uz), "z": (ux, uy)}[axis]
    return slice_plane(trio[0], axis, index), slice_plane(trio[1], axis, index)


def plot_slice(
    field: np.ndarray,
    label: str,
    axis: str = "z",
    index: int | None = None,
    title: str = "",
    subtitle: str = "",
    **kwargs,
):
    """Contour one plane through a 3D field, using the 2D renderer unchanged."""
    plane = slice_plane(field, axis, index)
    where = index if index is not None else field.shape[AXES[axis]] // 2
    note = f"{axis} = {where}" if not subtitle else f"{subtitle}   ·   {axis} = {where}"
    return plot_contours(plane, label=label, title=title, subtitle=note, **kwargs)


def plot_orthogonal_slices(
    field: np.ndarray,
    label: str,
    title: str = "",
    subtitle: str = "",
    cmap=FLOW,
    norm=None,
    n_filled: int = 28,
    n_lines: int = 14,
    solid: np.ndarray | None = None,
    indices: tuple[int | None, int | None, int | None] = (None, None, None),
    save: str | Path | None = None,
):
    """The three centre planes side by side, on one shared colour scale.

    The shared scale is the point. Three independently normalised panels invite the eye
    to compare colours that mean different things, which is precisely the mistake a
    reader of a 3D field is most likely to make.
    """
    if field.ndim != 3:
        raise ValueError(f"expected a 3D field, got shape {field.shape}")

    mask3d = solid.astype(bool) if solid is not None else None
    if norm is None:
        # Fitted to the fluid region across the whole volume, not to any one plane.
        sample = field[~mask3d] if mask3d is not None else field
        norm = contour_inputs(np.asarray(sample).reshape(1, -1))[5]

    with house_style():
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))
        for ax, axis, index in zip(axes, ("x", "y", "z"), indices):
            plane = slice_plane(field, axis, index)
            plane_solid = slice_plane(mask3d, axis, index) if mask3d is not None else None
            data, xs, ys, filled, lines, _ = contour_inputs(
                plane, norm=norm, n_filled=n_filled, n_lines=n_lines, solid=plane_solid
            )
            _draw_contour_layers(ax, xs, ys, data, filled, lines, cmap, norm)

            at = index if index is not None else field.shape[AXES[axis]] // 2
            others = [a for a in ("x", "y", "z") if a != axis]
            ax.set_title(f"{axis} = {at}", loc="left")
            ax.set_xlabel(f"${others[0]}$")
            ax.set_ylabel(f"${others[1]}$")
            ax.set_aspect("equal")
            ax.grid(False)

        # Lay the panels out *before* attaching the colorbar. A colorbar that spans
        # several axes steals space from all of them, and tight_layout cannot account
        # for it -- calling it afterwards warns and produces a skewed figure.
        fig.tight_layout(rect=(0, 0, 0.99, 0.90))
        colorbar(fig, plt.cm.ScalarMappable(norm=norm, cmap=cmap), list(axes), label)
        fig.suptitle(title, x=0.012, ha="left", fontsize=12.5, weight="bold")
        if subtitle:
            fig.text(0.012, 0.915, subtitle, fontsize=9, color=MUTED, ha="left")
        if save:
            fig.savefig(save)
        return fig


def plot_span_profiles(
    profiles: dict,
    reference: tuple[np.ndarray, np.ndarray] | None = None,
    title: str = "Mid-plane profile against span",
    subtitle: str = "",
    xlabel: str = r"$u\,/\,U_{lid}$",
    ylabel: str = r"$y\,/\,L$",
    save: str | Path | None = None,
):
    """Mid-plane centreline profiles at several span aspect ratios, over a reference.

    This is the figure that makes the end-wall effect legible: the curves should march
    toward the 2D reference as the span grows, and the gap that remains at span 1 is the
    physics, not an error.
    """
    from lms.viz.style import ACCENT_WARM

    with house_style():
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        shades = plt.get_cmap(FLOW)(np.linspace(0.35, 0.92, max(len(profiles), 1)))

        if reference is not None:
            rx, ry = reference
            ax.scatter(rx, ry, s=30, facecolor="none", edgecolor=ACCENT_WARM,
                       linewidth=1.4, label="Ghia et al. 1982 (2D)", zorder=4)

        for (name, (vals, coord)), shade in zip(sorted(profiles.items()), shades):
            ax.plot(vals, coord, color=shade, linewidth=1.7, label=name)

        ax.axvline(0, color=MUTED, linewidth=0.7, alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if subtitle:
            annotate(ax, subtitle, "lower right")
        ax.legend(loc="upper left", framealpha=0.0)
        fig.tight_layout()
        if save:
            fig.savefig(save)
        return fig


__all__ = [
    "in_plane_components",
    "plot_decay",
    "plot_energy_spectra",
    "plot_orthogonal_slices",
    "plot_slice",
    "plot_span_profiles",
    "slice_plane",
]


def plot_energy_spectra(
    curves: dict,
    title: str = "Energy spectrum",
    subtitle: str = "",
    slope_guide: float = -5.0 / 3.0,
    guide_window: tuple[int, int] | None = None,
    reference: str | None = None,
    decades: float = 8.0,
    save: str | Path | None = None,
):
    """Log-log E(k) for several runs, with a Kolmogorov guide over a stated window.

    `curves` maps a label to `(k, E)`. Three choices matter for readability:

    `decades` clips the vertical range to a stated span below the peak. A resolved run
    falls twelve decades by its Nyquist wavenumber, and letting the axis show all of it
    compresses the energy-containing range -- the part anyone is actually reading -- into
    the top fifth of the figure.

    `reference` names the curve to draw as the baseline. It gets a pale, heavier line so
    the eye can tell "the answer" from "the approximations to it" without consulting the
    legend.

    The guide is drawn only across `guide_window`, because that window is the claim being
    made. A -5/3 line stretched over the whole axis asserts a scaling range that does not
    exist at these resolutions.
    """
    from lms.viz.style import ACCENT_WARM, INK

    with house_style():
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        others = [name for name in curves if name != reference]
        shades = plt.get_cmap(FLOW)(np.linspace(0.45, 0.9, max(len(others), 1)))
        colours = dict(zip(others, shades))

        peak = 0.0
        for label, (k, e) in curves.items():
            k = np.asarray(k, dtype=float)
            e = np.asarray(e, dtype=float)
            good = (k > 0) & (e > 0)
            peak = max(peak, float(e[good].max()) if good.any() else 0.0)
            if label == reference:
                ax.loglog(k[good], e[good], "-", color=INK, linewidth=2.6, alpha=0.55,
                          label=f"{label}  (reference)", zorder=2)
            else:
                ax.loglog(k[good], e[good], "-", color=colours[label], linewidth=1.7,
                          label=label, zorder=3)

        if guide_window is not None and peak > 0:
            lo, hi = guide_window
            anchor_k, anchor_e = None, None
            for k, e in curves.values():
                k, e = np.asarray(k, float), np.asarray(e, float)
                band = (k >= lo) & (k <= hi) & (e > 0)
                if band.any():
                    anchor_k, anchor_e = k[band][0], e[band][0]
                    break
            if anchor_k is not None:
                kk = np.linspace(lo, hi, 20)
                ax.loglog(kk, anchor_e * (kk / anchor_k) ** slope_guide, "--",
                          color=ACCENT_WARM, linewidth=1.3, alpha=0.9, zorder=4,
                          label=f"$k^{{{slope_guide:.2f}}}$ over $k \\in [{lo},\\,{hi}]$")

        if peak > 0:
            ax.set_ylim(peak * 10.0**-decades, peak * 3.0)

        ax.set_xlabel("wavenumber $k$")
        ax.set_ylabel("$E(k)$")
        ax.set_title(title, loc="left", pad=16)
        if subtitle:
            ax.text(0.0, 1.015, subtitle, transform=ax.transAxes,
                    fontsize=9, color=MUTED, ha="left", va="bottom")
        # Lower left is where the spectra themselves end up; upper right is empty.
        ax.legend(loc="upper right", framealpha=0.0, fontsize=8.5)
        fig.tight_layout()
        if save:
            fig.savefig(save)
        return fig


def plot_decay(
    curves: dict,
    ylabel: str = "kinetic energy",
    title: str = "Energy decay",
    subtitle: str = "",
    save: str | Path | None = None,
):
    """Scalar histories against eddy-turnover time, one line per run."""
    with house_style():
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        shades = plt.get_cmap(FLOW)(np.linspace(0.35, 0.92, max(len(curves), 1)))
        for (label, (t, y)), shade in zip(curves.items(), shades):
            ax.semilogy(t, y, "-", color=shade, linewidth=1.7, label=label)
        ax.set_xlabel("$t\\,/\\,T$   (eddy turnovers)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if subtitle:
            annotate(ax, subtitle, "lower left")
        ax.legend(loc="upper right", framealpha=0.0, fontsize=8.5)
        fig.tight_layout()
        if save:
            fig.savefig(save)
        return fig
