"""Self-contained sky image generation package.

Public API
----------

draw_random_image_parameters(**ranges, rng)
    Draw random object functions, parameters, and flux weights.

image_generator(grid, n_pix, function_list, params_list, flux_list)
    Combine objects into a single normalised sky image.

Individual object generators
    nebula, point_sources, sharp_edges_object
"""

from .random_sky_parameters import (
    draw_random_image_parameters,
    draw_n_objects,
    draw_random_object_function,
    draw_nebula_params,
    draw_point_sources_params,
    draw_sharp_edges_params,
    draw_uniform_fluxes,
)
from .image_generator import image_generator
from .gaussian_random_fields import (
    make_grid,
    nebula,
    point_sources,
    sharp_edges_object,
    gaussian_psd,
    exponential_psd,
    powerlaw_psd,
    grf_from_psd,
)

__all__ = [
    "draw_random_image_parameters",
    "draw_n_objects",
    "draw_random_object_function",
    "draw_nebula_params",
    "draw_point_sources_params",
    "draw_sharp_edges_params",
    "draw_uniform_fluxes",
    "image_generator",
    "make_grid",
    "nebula",
    "point_sources",
    "sharp_edges_object",
    "gaussian_psd",
    "exponential_psd",
    "powerlaw_psd",
    "grf_from_psd",
]
