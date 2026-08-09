---
name: figure-style
description: House standard for every figure, plot and animation in this repository. Load before writing any plotting code, choosing a colormap or norm, laying out axes, or building a GIF. Also load when reviewing an existing figure. Triggers on matplotlib, colormap, colorbar, norm, contour, slice, spectrum, animation, GIF, "does this figure look right".
---

# Figure style

Visuals are a deliverable here, not a debugging aid. These rules were each paid for by a
figure that was wrong, and most of them are recorded in code as well as here:
`src/lms/viz/style.py` holds the palette and norms, `fields2d.py` / `fields3d.py` the
renderers, `animate.py` the GIF machinery.

**Read this before choosing a colormap.** The two worst rendering errors in this project
were violations of rules that already existed elsewhere in the codebase — they were just
nowhere a check could reach them.

## Colour is a claim about the data

**Never rainbow or jet.** They invent gradients where the data is flat, hide real ones,
and are unreadable to roughly 8% of men. Perceptually uniform only.

**Sequential or diverging is decided by the data, not by taste.**

| the quantity | example | use |
|---|---|---|
| cannot be negative | speed, `\|omega\|`, error magnitude, enstrophy | **sequential** — `FLOW` |
| is signed | `omega_z`, residual, temperature deviation, stream function | **diverging, symmetric about zero** — `DIVERGING_DARK` |

*The cautionary case.* Vorticity **magnitude** was once rendered on `DIVERGING_DARK` with
a symmetric norm. The colorbar advertised values down to `-1e-3` that cannot exist, half
the colormap was dead, and the quiet majority of the box — most of the picture — sat on
the neutral midpoint. `style.py` already said diverging fields get a symmetric scale; the
mistake was applying that to a field that does not diverge.

**Diverging maps take a dark neutral, never white.** On a dark canvas a white-centred map
turns every near-zero region into a glaring blob that dominates the figure. `DIVERGING_DARK`
anchors zero to the background so the eye goes where the field is actually strong.

## Norms

**`signed_asinh_norm` has two knobs and they do different jobs.**

- `percentile` sets the saturation limit. Too high and a handful of boundary-layer cells
  set the scale for the whole image.
- `linear_percentile` sets the width of the linear region, and it is the one that decides
  whether a **weak but uniform** feature is visible at all.

*The cautionary case.* The lid-driven cavity core rotates as a solid body at `|omega| ~ 2`
while the lid layer reaches ~80. With a wide linear region the core — the most physically
meaningful structure in the flow — rendered as near-black background. `percentile=97,
linear_percentile=20` recovered it.

**`PowerNorm(gamma≈0.55–0.6)` for speed fields.** Recirculating flows span orders of
magnitude and a linear scale crushes the slow core to black.

**Fit the norm to the fluid region**, excluding solid nodes. Walls hold bounce-back state,
not physical values, and will drag the scale.

## Axes and annotation

**Choose the range; do not accept the default.** An energy spectrum falls twelve decades
by its Nyquist wavenumber, and showing all of it compresses the energy-containing range —
the part anyone is reading — into the top fifth of the figure. Clip to a stated number of
decades below the peak.

**Style the reference differently from what is being compared to it.** Pale and heavier,
so "the answer" and "the approximations to it" are distinguishable without consulting the
legend.

**Legends go in the empty quadrant.** Over dense contours they need opaque backing —
`framealpha=0.75` with a `BACKGROUND` facecolor — or the labels are unreadable. Check that
the legend does not collide with an `annotate` box; both defaulting to lower-left has
happened.

**State fitting windows on the figure itself.** A `k^-5/3` guide drawn across the whole
axis asserts a scaling range that does not exist. Draw the guide only over the window it
was fitted on, and put the window in the label.

## Isolines are conditional

Readable on a smooth field. Pure noise on a turbulent one, where the contour tracer finds
structure at every scale it is given. They also roughly double GIF size. Pass `n_lines=0`
to turn them off — `_draw_contour_layers` skips them when the level set is empty.

## Animation

Four things that otherwise produce a broken or misleading GIF:

1. **Pool the norm across every frame and freeze it.** Per-frame normalisation flickers
   and, worse, lies about relative magnitude. Use `pooled_norm`.
2. **Override `savefig.bbox` from `"tight"` to `None`.** Tight bounding boxes are measured
   per frame, so the figure size drifts and the GIF shivers or fails.
3. **Build the figure and colorbar once**, outside the loop; clear only the main Axes.
   `contourf` cannot be updated in place, so each frame is a redraw, but a colorbar made
   inside the loop stacks up one per frame.
4. **Pass `dpi` explicitly.** `figure.dpi` (130) and `savefig.dpi` (200) differ in the rc.

**The caption carries the quantity that is changing** — `E/E0`, `Re`, `t/T`. Without a
number the eye reads "the picture got dimmer" and cannot tell decay from a colour trick.

Target ~600 px wide and **under 5 MB**. Dropping isolines took one pair of animations from
6.9/7.8 MB to 3.6/4.3 MB.

## Honesty

These are not stylistic.

- **Every frame is a real solve.** Never interpolate between solutions to pad a frame
  count — the interpolated frames solve nothing and the animation becomes an illustration
  rather than a result.
- **Label capped or unconverged frames, proportionately.** "Not converged" stamped on a
  field within 0.2% of its converged solution is literally true and badly misleading.
  Measure the discrepancy and state it.
- **Never render NaN as a colour.** If a run diverged, stop the animation there and say
  so in the caption. A silent truncation turns a measured divergence into a visual shrug.
- **Say where a slice is.** A single plane through a 3D field is a claim about the whole
  field that one plane cannot support.

## 3D

- **Orthogonal slices share one colour scale**, fitted to the whole volume rather than to
  any single plane. Three independently normalised panels invite the reader to compare
  colours that mean different things.
- **Take the in-plane component pair.** A `z`-normal cut wants `(ux, uy)`; an `x`-normal
  cut wants `(uy, uz)`. Streamlines drawn from the wrong pair look plausible and describe
  a flow that does not exist.
- **Call `tight_layout` before attaching a colorbar that spans several axes**, or it warns
  and skews the figure.

## Reuse before adding

`src/lms/viz/` already has: `plot_contours`, `plot_lic`, `plot_streamfunction`,
`plot_order_of_accuracy`, `plot_orthogonal_slices`, `plot_energy_spectra`, `plot_decay`,
`plot_span_profiles`, `contour_animation`, `pooled_norm`. Extend these rather than starting
a parallel rendering path — two sets of colour decisions will drift apart.
