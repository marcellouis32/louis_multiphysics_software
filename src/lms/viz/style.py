"""House visual style.

Two rules that are non-negotiable for anything a client sees:

  1. No rainbow/jet colormaps. They invent visual gradients where the data is
     flat and hide real ones where it isn't, and they are unreadable to the ~8%
     of men with colour vision deficiency. Everything here is perceptually
     uniform.
  2. Diverging fields (vorticity, temperature deviation, residuals) are always
     rendered on a symmetric scale about zero, so sign is readable at a glance.
"""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import AsinhNorm, LinearSegmentedColormap, Normalize

INK = "#e8edf4"
MUTED = "#8b98ab"
BACKGROUND = "#0f1216"
PANEL = "#151a21"
GRID = "#232b36"
ACCENT = "#4cc2ff"
ACCENT_WARM = "#ff9e4a"

SEQUENTIAL = "magma"
SPEED = "viridis"
DIVERGING = "RdBu_r"

# Diverging map with a *dark* neutral instead of white. On a dark canvas a
# white-centred map turns every near-zero region into a glaring blob that
# dominates the figure; anchoring zero to the background makes the eye go to
# where the field is actually signed and strong.
DIVERGING_DARK = LinearSegmentedColormap.from_list(
    "diverging_dark",
    ["#7fd4ff", "#2f9fd6", "#1b5c85", "#12212e", "#0f1216", "#2e1d15", "#8a4a1c", "#e08b3c", "#ffd28a"],
)

# Perceptually smooth dark-to-bright ramp for speed fields on dark panels.
FLOW = LinearSegmentedColormap.from_list(
    "flow",
    ["#05070c", "#122036", "#17456b", "#1f7a92", "#4fb99a", "#b3e07c", "#fdf3b0"],
)

DARK_RC = {
    "figure.facecolor": BACKGROUND,
    "figure.edgecolor": BACKGROUND,
    "savefig.facecolor": BACKGROUND,
    "savefig.edgecolor": BACKGROUND,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "SF Pro Text", "Helvetica Neue", "DejaVu Sans"],
    "image.interpolation": "bilinear",
    "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round",
}


@contextmanager
def house_style(dark: bool = True):
    rc = dict(DARK_RC)
    if not dark:
        rc.update(
            {
                "figure.facecolor": "white",
                "savefig.facecolor": "white",
                "axes.facecolor": "#fbfcfd",
                "axes.edgecolor": "#d3dae3",
                "grid.color": "#e6ebf1",
                "text.color": "#11161d",
                "axes.labelcolor": "#11161d",
                "axes.titlecolor": "#11161d",
                "xtick.color": "#5b6675",
                "ytick.color": "#5b6675",
            }
        )
    with mpl.rc_context(rc):
        yield


def symmetric_norm(data: np.ndarray, percentile: float = 99.0) -> Normalize:
    """Symmetric colour limits clipped at a percentile, so a handful of extreme
    cells near walls don't wash out the interior structure."""
    lim = float(np.percentile(np.abs(data), percentile))
    lim = lim if lim > 0 else 1.0
    return Normalize(vmin=-lim, vmax=lim)


def signed_asinh_norm(
    data: np.ndarray, mask: np.ndarray | None = None, percentile: float = 99.8
) -> AsinhNorm:
    """Symmetric asinh scale for signed fields.

    Wall-bounded vorticity spans two or three decades: the boundary layer at a
    moving wall is enormous compared with the interior. A linear scale set by the
    extremes renders the entire interior as neutral. asinh is linear near zero and
    logarithmic in the tails, so both survive in one image.

    `mask` marks cells to exclude when choosing the limits (typically solids).
    """
    values = np.abs(np.asarray(data)[~mask] if mask is not None else np.asarray(data))
    values = values[np.isfinite(values)]
    lim = float(np.percentile(values, percentile)) if values.size else 1.0
    lim = lim if lim > 0 else 1.0
    linear_width = max(float(np.percentile(values, 60.0)), lim * 1e-3)
    return AsinhNorm(linear_width=linear_width, vmin=-lim, vmax=lim)


def annotate(ax, text: str, loc: str = "lower right") -> None:
    positions = {
        "lower right": (0.98, 0.03, "right", "bottom"),
        "lower left": (0.02, 0.03, "left", "bottom"),
        "upper right": (0.98, 0.97, "right", "top"),
        "upper left": (0.02, 0.97, "left", "top"),
    }
    x, y, ha, va = positions[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8.5,
        color=INK,
        alpha=0.85,
        bbox=dict(facecolor=BACKGROUND, edgecolor="none", alpha=0.55, pad=4),
    )


def colorbar(fig, mappable, ax, label: str):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=9, color=INK)
    cb.outline.set_edgecolor(GRID)
    cb.ax.tick_params(labelsize=8, color=GRID)
    return cb


__all__ = [
    "house_style",
    "symmetric_norm",
    "signed_asinh_norm",
    "DIVERGING_DARK",
    "annotate",
    "colorbar",
    "FLOW",
    "SPEED",
    "DIVERGING",
    "SEQUENTIAL",
    "ACCENT",
    "ACCENT_WARM",
    "INK",
    "MUTED",
    "plt",
]
