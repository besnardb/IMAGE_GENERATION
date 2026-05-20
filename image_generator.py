"""Composite image generator from multiple sky-object functions.

Given lists of object-generator functions, their parameters, and per-object
fluxes, this module produces a single normalised sky image by summing the
flux-weighted contributions.
"""

from __future__ import annotations

import numpy as np
import cupy as cp

from array_backend import is_cupy_array


def generate_image_from_components(
	grid,
	n_pix: int,
	function_list,
	params_list,
	flux_list,
	*,
	max_iter: int = 10,
):
	"""Build a composite sky image from a collection of object generators.

	Each function in *function_list* is called as
	``func(grid, n_pix, *params)`` where *params* comes from the
	corresponding entry in *params_list*.  The resulting image is
	normalised by its maximum value and then multiplied by the matching
	flux, so *flux* controls the peak brightness of each component.
	Components are accumulated into a composite.

	If a call produces an image whose sum is zero or contains NaN
	values, it is retried up to *max_iter* times before being skipped.

	Parameters
	----------
	grid : ndarray
		2D frequency grid with shape ``(2*n_pix, 2*n_pix)``.
	n_pix : int
		Number of pixels in each dimension of the output image.
	function_list : list[callable]
		Object generator functions (e.g. ``nebula``, ``point_sources``,
		``sharp_edges_object``).
	params_list : list[tuple | dict | scalar]
		Parameters for each function.  Tuples are unpacked as positional
		args; dicts are unpacked as keyword args; scalars are passed as a
		single positional arg.
	flux_list : array-like
		Per-object peak flux values (same length as *function_list*).
		Point sources should use higher values than extended emission.
	max_iter : int, optional
		Maximum number of retry attempts for an object that produces an
		all-zero or NaN image.  Default is 10.

	Returns
	-------
	ndarray
		2D image of shape ``(n_pix, n_pix)``.
	"""
	xp = cp if is_cupy_array(grid) else np
	
	composite = xp.zeros((n_pix, n_pix), dtype=xp.float64)
	flux_arr = xp.asarray(flux_list, dtype=xp.float64)

	for idx, (func, params, flux) in enumerate(
		zip(function_list, params_list, flux_arr)
	):
		flux = float(flux)
		if flux == 0.0:
			continue

		# Retry loop for degenerate realisations
		for attempt in range(max_iter):
			# Call the generator
			if isinstance(params, dict):
				obj = func(grid, n_pix, **params)
			elif isinstance(params, (tuple, list)):
				obj = func(grid, n_pix, *params)
			else:
				obj = func(grid, n_pix, params)
			del func, params  # free memory if using CuPy

			# Validate
			obj_sum = float(xp.sum(obj))
			if obj_sum > 0.0 and not xp.isnan(obj_sum):
				break
			del obj_sum  # free memory if using CuPy
		else:
			# All attempts failed – skip this object
			continue

		# Normalise by max so flux controls the peak brightness
		obj_max = float(xp.max(obj))

		if obj_max > 0.0:
			composite += flux * (obj / obj_max)
		del obj, obj_max  # free memory if using CuPy

	return composite
