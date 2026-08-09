"""Animated contour sequences.

Written for Reynolds sweeps, where each frame is a separately converged steady
solution rather than a timestep. The animation is therefore a walk through
parameter space, and the thing it shows -- the primary vortex migrating toward the
geometric centre as inertia overtakes viscosity -- is exactly what Ghia et al.
tabulate at three points.

GIF via Pillow is the only output format here on purpose: imageio and ffmpeg are not
installed, and an animation is not worth a new dependency. `PillowWriter` ships with
matplotlib.

Four things that will otherwise produce a broken or misleading animation, all handled
below:

  1. The norm is computed once from the pooled data across every frame and held
     fixed. Per-frame normalisation makes the colours flicker and, worse, lies about
     relative magnitude -- vorticity grows with Re as the boundary layers thin, and a
     rescaling colorbar would hide precisely that.
  2. `savefig.bbox` is "tight" in the house style. Tight bounding boxes are measured
     per frame, so the figure size drifts by a pixel or two and the GIF either fails
     or shivers. It is forced to None here.
  3. The figure and colorbar are built once. `contourf` cannot be updated in place,
     so each frame clears and redraws the main Axes -- but a colorbar created inside
     that loop would stack up one per frame.
  4. `dpi` is explicit. `figure.dpi` (130) and `savefig.dpi` (200) differ in the rc,
     and the writer would otherwise pick a surprising one.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.animation import PillowWriter

from lms.viz.fields2d import _draw_contour_layers, _levels_from_norm
from lms.viz.style import (
    ACCENT_WARM,
    BACKGROUND,
    FLOW,
    INK,
    annotate,
    colorbar,
    house_style,
    plt,
)


def pooled_norm(frames: Sequence[np.ndarray], builder, solid: np.ndarray | None = None):
    """Build one norm from every frame at once.

    `builder` takes the pooled sample and returns a norm, so the caller keeps control
    of the scale type (PowerNorm for speed, asinh for vorticity) while this function
    guarantees the sample it is fitted to spans the whole animation.
    """
    mask = solid.astype(bool) if solid is not None else None
    pool = np.concatenate(
        [np.asarray(f)[~mask].ravel() if mask is not None else np.asarray(f).ravel() for f in frames]
    )
    return builder(pool[np.isfinite(pool)])


def contour_animation(
    fields: Sequence[np.ndarray],
    reynolds: Sequence[float],
    label: str,
    norm,
    title: str = "",
    subtitle: str = "",
    cmap=FLOW,
    n_filled: int = 28,
    n_lines: int = 14,
    solid: np.ndarray | None = None,
    extent: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    labels: Sequence[str] | None = None,
    markers: Sequence[dict] | None = None,
    track: Sequence[tuple[float, float]] | None = None,
    converged: Sequence[bool] | None = None,
    cap_note: str = "step cap reached (not converged)",
    fps: int = 6,
    dpi: int = 92,
    figsize: tuple[float, float] = (6.6, 6.0),
    save: str | Path = "animation.gif",
) -> Path:
    """Write a GIF of filled contours, one frame per entry in `fields`.

    `norm` is required and is never recomputed: passing it in is what makes the colour
    scale comparable across frames, and it is left unmutated so a caller can reuse it
    for a second animation on the same scale.

    `markers` is an optional per-frame list of `{name: (x, y)}` overlays -- used to
    track the computed vortex centre against Ghia's tabulated position. `track` draws
    the path those centres have traced so far, which is what shows the vortex is
    moving rather than merely located. `converged` flags any frame that hit its step
    cap instead of converging, which is annotated on the frame rather than quietly
    ignored.
    """
    if len(fields) != len(reynolds):
        raise ValueError(f"{len(fields)} fields but {len(reynolds)} Reynolds numbers")

    mask = solid.astype(bool) if solid is not None else None
    filled = _levels_from_norm(norm, n_filled)
    lines = _levels_from_norm(norm, n_lines) if n_lines > 0 else None

    ny, nx = np.asarray(fields[0]).shape
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[2], extent[3], ny)

    save = Path(save)
    save.parent.mkdir(parents=True, exist_ok=True)

    # Gotcha 2: tight bbox resizes the canvas per frame; GIF frames must match.
    with house_style(), mpl.rc_context({"savefig.bbox": None}):
        fig, ax = plt.subplots(figsize=figsize)
        # Gotcha 3: colorbar built once, outside the frame loop.
        colorbar(fig, plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax, label)
        fig.subplots_adjust(left=0.10, right=0.90, bottom=0.08, top=0.92)

        writer = PillowWriter(fps=fps)
        with writer.saving(fig, str(save), dpi=dpi):  # gotcha 4: explicit dpi
            for i, (field, re) in enumerate(zip(fields, reynolds)):
                ax.clear()
                data = np.ma.masked_where(mask, field) if mask is not None else np.ma.asarray(field)
                _draw_contour_layers(ax, xs, ys, data, filled, lines, cmap, norm)

                if track is not None:
                    track_overlay(track, ax, i)
                if markers and markers[i]:
                    _draw_markers(ax, markers[i])

                ax.set_title(title)
                # `labels` exists because this animator was written for Reynolds sweeps
                # but the flow it is best suited to is genuinely time-dependent, where
                # the frame index means elapsed time rather than a parameter value.
                caption = labels[i] if labels is not None else f"Re = {re:,.0f}"
                if converged is not None and not converged[i]:
                    caption += f"   ·   {cap_note}"
                annotate(ax, caption, "lower left")
                if subtitle:
                    annotate(ax, subtitle, "lower right")
                ax.set_xlabel("$x\\,/\\,L$")
                ax.set_ylabel("$y\\,/\\,L$")
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                ax.set_aspect("equal")
                ax.grid(False)
                writer.grab_frame()

        plt.close(fig)
    return save


def _draw_markers(ax, marks: dict) -> None:
    """Overlay named points. Ghia's tabulated centres are drawn as open circles and
    the computed ones as crosses, so agreement reads as a cross inside a ring."""
    for name, (mx, my) in marks.items():
        is_reference = "Ghia" in name
        if is_reference:
            ax.plot(
                mx, my, marker="o", ms=11, mew=1.8, mfc="none",
                color=ACCENT_WARM, linestyle="none", label=name,
            )
        else:
            ax.plot(
                mx, my, marker="+", ms=13, mew=2.0,
                color="white", linestyle="none", label=name,
            )
    # Opaque backing, not the house default of no frame: these labels sit directly on
    # top of dense contour bands and are unreadable without it.
    legend = ax.legend(loc="upper left", labelcolor=INK, framealpha=0.75)
    legend.get_frame().set_facecolor(BACKGROUND)
    legend.get_frame().set_edgecolor("none")


def track_overlay(
    centres: Sequence[tuple[float, float]],
    ax,
    upto: int,
    colour: str = INK,
) -> None:
    """Draw the path the vortex centre has taken up to the current frame.

    A single marker shows where the vortex is; the trailing path shows that it is
    *moving*, which is the entire point of the sweep.
    """
    if upto < 1:
        return
    pts = np.asarray(centres[: upto + 1], dtype=float)
    ax.plot(
        pts[:, 0], pts[:, 1], "-", color=colour, linewidth=1.0, alpha=0.45,
        solid_capstyle="round", zorder=4,
    )
    ax.scatter(
        pts[:-1, 0], pts[:-1, 1], s=7, color=colour, alpha=0.35,
        edgecolor=BACKGROUND, linewidth=0.3, zorder=4,
    )


__all__ = [
    "contour_animation",
    "pooled_norm",
    "side_by_side_animation",
    "track_overlay",
]


def side_by_side_animation(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
    titles: tuple[str, str],
    label: str,
    norm,
    captions: Sequence[str] | None = None,
    dead_after: tuple[int | None, int | None] = (None, None),
    dead_note: str = "diverged",
    title: str = "",
    subtitle: str = "",
    cmap=FLOW,
    n_filled: int = 44,
    n_lines: int = 0,
    extent: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    fps: int = 8,
    dpi: int = 92,
    figsize: tuple[float, float] = (10.4, 5.4),
    save: str | Path = "compare.gif",
) -> Path:
    """Two runs of the same problem, frame-synchronised on one shared colour scale.

    Built for the comparison where one run survives and the other does not, which brings
    two obligations that a naive implementation gets wrong in a flattering direction.

    **One norm across both panels, frozen.** A diverging run reaches enormous velocities,
    so per-panel scaling would render the blow-up as calm and the healthy run as violent
    -- exactly backwards. The norm must be fitted to the *surviving* run, not pooled over
    both, or the failure's excursion sets a scale on which everything real looks flat.

    **A dead run stops, visibly.** `dead_after` gives the last valid frame index for each
    panel; past it the panel is left blank and captioned. Rendering NaN as a colour, or
    letting the animation quietly continue with stale data, turns a measured divergence
    into a visual shrug.
    """
    frames = max(len(left), len(right))
    save = Path(save)
    save.parent.mkdir(parents=True, exist_ok=True)
    filled = _levels_from_norm(norm, n_filled)
    lines = _levels_from_norm(norm, n_lines) if n_lines > 0 else None

    with house_style(), mpl.rc_context({"savefig.bbox": None}):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        colorbar(fig, plt.cm.ScalarMappable(norm=norm, cmap=cmap), list(axes), label)
        fig.subplots_adjust(left=0.06, right=0.88, bottom=0.08, top=0.86, wspace=0.18)

        writer = PillowWriter(fps=fps)
        with writer.saving(fig, str(save), dpi=dpi):
            for i in range(frames):
                for ax, series, name, last in zip(
                    axes, (left, right), titles, dead_after
                ):
                    ax.clear()
                    alive = i < len(series) and (last is None or i <= last)
                    if alive:
                        data = np.asarray(series[i])
                        ny, nx = data.shape
                        xs = np.linspace(extent[0], extent[1], nx)
                        ys = np.linspace(extent[2], extent[3], ny)
                        _draw_contour_layers(
                            ax, xs, ys, np.ma.asarray(data), filled, lines, cmap, norm
                        )
                        ax.set_title(name, loc="left")
                    else:
                        ax.set_title(f"{name} — {dead_note}", loc="left", color=ACCENT_WARM)
                        ax.text(0.5, 0.5, dead_note, transform=ax.transAxes,
                                ha="center", va="center", color=ACCENT_WARM, fontsize=13)
                    ax.set_xlim(extent[0], extent[1])
                    ax.set_ylim(extent[2], extent[3])
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_aspect("equal")
                    ax.grid(False)

                if captions is not None and i < len(captions):
                    annotate(axes[0], captions[i], "lower left")
                if subtitle:
                    annotate(axes[1], subtitle, "lower right")
                fig.suptitle(title, x=0.012, ha="left", fontsize=12.5, weight="bold")
                writer.grab_frame()

        plt.close(fig)
    return save
