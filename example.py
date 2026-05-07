"""Demo script: visualise the sky-image generation components.

Three figure sections:
  1. Gaussian random fields with varying power-law exponents
  2. Individual object types (nebula, sharp-edged, point sources) with
     key parameter sweeps
  3. Full composite images with random parameters

Run from the workspace root:
    python IMAGE_GENERATION/example.py

Or from inside IMAGE_GENERATION/:
    python example.py
"""

from __future__ import annotations

import sys
import os

# Allow running as a script from anywhere (adds IMAGE_GENERATION parent to path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from IMAGE_GENERATION import SkyInstrument, draw_random_image_parameters, image_generator
from IMAGE_GENERATION.gaussian_random_fields import (
    powerlaw_psd,
    grf_from_psd,
    nebula,
    sharp_edges_object,
    point_sources,
)

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Matplotlib not available — text output only.")

N_PIX = 1024
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

instrument = SkyInstrument(n_pix=N_PIX, use_cupy=False)


def _show(ax, img, title="", cmap="inferno", diverging=False, norm=None):
    """Helper: display one image in an axes."""
    if diverging:
        vmax = float(np.percentile(np.abs(img), 98))
        ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    elif norm is not None:
        ax.imshow(img, origin="lower", cmap=cmap, norm=norm)
    else:
        vmax = float(np.max(img)) if img.max() > 0 else 1.0
        ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


# ======================================================================
# Section 1 — Power-law GRFs with varying exponents
# ======================================================================

ALPHA_VALUES = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
GRF_RNG = np.random.default_rng(7)

print("=== Section 1: power-law GRFs ===")
grfs = []
for alpha in ALPHA_VALUES:
    psd = powerlaw_psd(instrument, alpha)
    field = grf_from_psd(instrument, psd, rng=np.random.default_rng(7))
    grfs.append(field)
    print(f"  alpha={alpha:.1f}  std={float(np.std(field)):.3f}")

if HAS_MPL:
    fig, axes = plt.subplots(1, len(ALPHA_VALUES), figsize=(14, 3))
    for ax, field, alpha in zip(axes, grfs, ALPHA_VALUES):
        _show(ax, field, title=f"α = {alpha:.1f}", diverging=True)
    fig.suptitle("Power-law GRFs — increasing spectral exponent", fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "example_grfs.png")
    plt.savefig(out, dpi=120)
    print(f"Saved {out}\n")


# ======================================================================
# Section 2 — Individual object types with parameter sweeps
# ======================================================================

print("=== Section 2: individual object types ===")

# --- 2a. Nebula: exponent (rows) × percentile threshold (cols) ---
nebula_exponents   = [1.5, 3.0, 5.0]
nebula_percentiles = [50.0, 75.0, 95.0]

print("  Nebula (exponent × percentile):")
nebula_grid = []
for exp in nebula_exponents:
    row = []
    for pct in nebula_percentiles:
        img = nebula(instrument, exponent=exp, percentile=pct)
        row.append(img)
        print(f"    α={exp:.1f}  p={pct:.0f}%  "
              f"nonzero={float(np.mean(img > 0)):.2%}")
    nebula_grid.append(row)

# --- 2b. Sharp-edges: shape exponent (rows) × texture floor vmin_hf (cols) ---
se_exponents_lf = [1.5, 3.0, 5.0]
se_vmin_hf      = [0.0, 0.5, 0.9]
SE_PCT_LF  = 75.0
SE_EXP_HF  = 3.0

print("  Sharp-edges (α_lf × v_min_hf):")
se_grid = []
for exp_lf in se_exponents_lf:
    row = []
    for vmin in se_vmin_hf:
        img = sharp_edges_object(
            instrument,
            exponent_lf=exp_lf,
            percentile_lf=SE_PCT_LF,
            exponent_hf=SE_EXP_HF,
            vmin_hf=vmin,
        )
        row.append(img)
        print(f"    α_lf={exp_lf:.1f}  v_min={vmin:.1f}  "
              f"nonzero={float(np.mean(img > 0)):.2%}")
    se_grid.append(row)

# --- 2c. Point sources: source count (rows) × brightness exponent (cols) ---
ps_n_values  = [5, 20, 100]
ps_exponents = [1.6, 2.0, 3.0]

print("  Point sources (n × α):")
ps_grid = []
for n in ps_n_values:
    row = []
    for exp in ps_exponents:
        img = point_sources(instrument, n=n, exponent=exp)
        row.append(img)
        print(f"    n={n:3d}  α={exp:.1f}  max={float(np.max(img)):.1f}")
    ps_grid.append(row)

if HAS_MPL:
    n_rows, n_cols = 3, 3
    fig, all_axes = plt.subplots(3 * n_rows, n_cols, figsize=(9, 27))

    # Nebula block (rows 0–2)
    for r, exp in enumerate(nebula_exponents):
        for c, pct in enumerate(nebula_percentiles):
            _show(all_axes[r, c], nebula_grid[r][c],
                  title=f"nebula  α={exp:.1f}  p={pct:.0f}%")
    all_axes[0, 1].set_title(
        f"Nebula — α (rows) × percentile threshold (cols)\n"
        f"nebula  α={nebula_exponents[0]:.1f}  p={nebula_percentiles[1]:.0f}%",
        fontsize=8)

    # Sharp-edges block (rows 3–5)
    for r, exp_lf in enumerate(se_exponents_lf):
        for c, vmin in enumerate(se_vmin_hf):
            _show(all_axes[n_rows + r, c], se_grid[r][c],
                  title=f"sharp  α_lf={exp_lf:.1f}  v_min={vmin:.1f}")

    # Point-sources block (rows 6–8)
    for r, n in enumerate(ps_n_values):
        for c, exp in enumerate(ps_exponents):
            _show(all_axes[2 * n_rows + r, c], ps_grid[r][c],
                  title=f"pts  n={n}  α={exp:.1f}")

    # Section labels on left-most column
    for row_offset, label in [
        (0,          "Nebula\n(α × percentile)"),
        (n_rows,     "Sharp edges\n(α_lf × v_min_hf)"),
        (2 * n_rows, "Point sources\n(n × α)"),
    ]:
        all_axes[row_offset, 0].set_ylabel(label, fontsize=9,
                                           rotation=90, labelpad=6,
                                           va="center")

    fig.suptitle("Individual object types — parameter sweeps", fontsize=12, y=1.001)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "example_objects.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}\n")


# ======================================================================
# Section 3 — Full composite images with random parameters
# ======================================================================

SKY_CONFIG = {
    # object-type sampling weights: (nebula, point_sources, sharp_edges_object)
    "object_type_weights": (1.0, 1.0, 1.0),
    "n_objects_min": 1,
    "n_objects_max": 5,
    "nebula_exponent_min": 1.5,
    "nebula_exponent_max": 5.0,
    "nebula_percentile_min": 50.0,
    "nebula_percentile_max": 99.0,
    "point_sources_n_min": 1,
    "point_sources_n_max": 50,
    "point_sources_exponent_min": 1.5,
    "point_sources_exponent_max": 3.5,
    "sharp_edges_exponent_lf_min": 1.5,
    "sharp_edges_exponent_lf_max": 5.0,
    "sharp_edges_percentile_lf_min": 50.0,
    "sharp_edges_percentile_lf_max": 99.0,
    "sharp_edges_exponent_hf_min": 1.5,
    "sharp_edges_exponent_hf_max": 5.0,
    "sharp_edges_vmin_hf_min": 0.0,
    "sharp_edges_vmin_hf_max": 1.0,
}

N_COMPOSITES = 6
rng_comp = np.random.default_rng(42)

print("=== Section 3: composite images ===")
composites = []
for i in range(N_COMPOSITES):
    funcs, params, fluxes = draw_random_image_parameters(**SKY_CONFIG, rng=rng_comp)
    img = image_generator(instrument, funcs, params, fluxes)
    composites.append((img, [f.__name__ for f in funcs]))
    obj_names = [f.__name__ for f in funcs]
    print(f"  [{i+1}] {obj_names}  "
          f"mean={float(np.mean(img)):.4f}  max={float(np.max(img)):.2f}")

if HAS_MPL:
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, (img, names) in zip(axes.flat, composites):
        vmax = float(np.max(img))
        vmin = vmax * 1e-5
        norm = LogNorm(vmin=vmin, vmax=vmax)
        _show(ax, np.clip(img, vmin, None), title=" + ".join(names), norm=norm)
    fig.suptitle("Composite sky images — random parameters (log scale)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "example_composites.png")
    plt.savefig(out, dpi=120)
    print(f"Saved {out}\n")

print("Done.")

if HAS_MPL:
    plt.show()
