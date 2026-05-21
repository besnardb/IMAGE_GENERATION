import os
import argparse
from utils import load_config
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from random_sky_parameters import draw_random_image_parameters
from image_generator import generate_image_from_components
from gaussian_random_fields import make_grid
from utils import np_to_fits, _show

def make_compsite_image(cfg, grid):
    rng_comp = np.random.default_rng(cfg.general.n_seed)
    composites = []
    use_cupy = cfg.general.use_gpus
    for i in range(cfg.general.n_images):
        funcs, params, fluxes = draw_random_image_parameters(cfg, rng=rng_comp)
        img = generate_image_from_components(
            grid, cfg.general.n_pix, funcs, params, fluxes,
        )
        obj_names = [f.__name__ for f in funcs]

        # Transfer to CPU
        img_cpu = img.get() if use_cupy else img
        composites.append((img_cpu, obj_names))

        print(f"  [{i+1}] {obj_names}  "
            f"mean={float(np.mean(img_cpu)):.4f}  max={float(np.max(img_cpu)):.2f}", flush=True)

        out = os.path.join(cfg.general.output_dir, f"composite_{i+1}.fits")
        np_to_fits(img_cpu, out)

        if use_cupy:
            cp.get_default_memory_pool().free_all_blocks() # free GPU memory
    
    if cfg.general.generate_previews:
        generate_preview_composite_image(cfg, composites)

def generate_preview_composite_image(cfg, composites):
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, (img, names) in zip(axes.flat, composites):
        vmax = float(np.max(img))
        vmin = vmax * 1e-5
        norm = LogNorm(vmin=vmin, vmax=vmax)
        _show(ax, np.clip(img, vmin, None), title=" + ".join(names), norm=norm)
    fig.suptitle("Composite sky images — random parameters (log scale)", fontsize=12)
    plt.tight_layout()
    OUT_DIR = cfg.general.output_dir
    out = os.path.join(OUT_DIR, "preview_composites.png")
    plt.savefig(out, dpi=120)
    print(f"Saved {out}\n")

def main(cfg):
    grid = make_grid(cfg.general.n_pix, cfg.general.use_gpus, cfg.general.device_id)
    make_compsite_image(cfg, grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    use_gpus = bool(cfg.general.use_gpus)
    print(f"Running generation on GPU")

    main(cfg)