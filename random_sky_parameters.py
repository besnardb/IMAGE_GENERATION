"""Random parameter generators for sky objects."""

from __future__ import annotations

import numpy as np

from gaussian_random_fields import nebula, point_sources, sharp_edges_object
from random_compat import rng_random as _rng_random


def draw_n_objects(
	*,
	n_objects_min: int = 1,
	n_objects_max: int = 20,
	rng=None,
):
	"""Draw a log-uniform integer number of objects."""
	if n_objects_min < 1 or n_objects_max < 1:
		raise ValueError("n_objects_min and n_objects_max must be >= 1")
	if n_objects_max < n_objects_min:
		raise ValueError("n_objects_max must be >= n_objects_min")

	log_min = np.log(float(n_objects_min))
	log_max = np.log(float(n_objects_max))
	if rng is None:
		u = np.random.random()
	else:
		u = float(_rng_random(rng))
	val = np.exp(log_min + (log_max - log_min) * u)
	val_int = int(np.clip(np.rint(val), n_objects_min, n_objects_max))
	return val_int


def draw_random_object_function(cfg, rng=None):
	"""Draw a random sky object generator function.

	Parameters
	----------
	weights : sequence of 3 floats or None
		Unnormalised sampling weights for ``(nebula, point_sources,
		sharp_edges_object)`` in that order.  If *None*, uniform weights
		are used.
	"""
	functions = (nebula, point_sources, sharp_edges_object)
	w = np.array([
		cfg.sky.nebula.weight,
		cfg.sky.point_sources.weight,
		cfg.sky.sharp_edges.weight,
	], dtype=float)
	if w.shape != (len(functions),):
		raise ValueError(
			f"weights must have length {len(functions)}, got {len(w)}"
		)
	if np.any(w < 0):
		raise ValueError("weights must be non-negative")
	w = w / w.sum()
	cumw = np.cumsum(w)
	if rng is None:
		u = np.random.random()
	else:
		u = float(_rng_random(rng))
	idx = int(np.searchsorted(cumw, u))
	idx = min(idx, len(functions) - 1)
	return functions[idx]


def draw_nebula_params(
	*,
	exponent_min: float = 1.5,
	exponent_max: float = 5.0,
	percentile_min: float = 50.0,
	percentile_max: float = 99.0,
	rng=None,
):
	"""Draw random parameters for nebula generation."""
	if exponent_max < exponent_min:
		raise ValueError("exponent_max must be >= exponent_min")
	if percentile_max < percentile_min:
		raise ValueError("percentile_max must be >= percentile_min")

	if rng is None:
		u1 = np.random.random()
		u2 = np.random.random()
	else:
		u1 = float(_rng_random(rng))
		u2 = float(_rng_random(rng))

	exponent = exponent_min + (exponent_max - exponent_min) * u1
	percentile = percentile_min + (percentile_max - percentile_min) * u2
	return exponent, percentile


def draw_point_sources_params(
	*,
	n_min: int = 1,
	n_max: int = 1000,
	exponent_min: float = 1.5,
	exponent_max: float = 3.5,
	rng=None,
):
	"""Draw random parameters for point source generation."""
	if n_min < 1 or n_max < 1:
		raise ValueError("n_min and n_max must be >= 1")
	if n_max < n_min:
		raise ValueError("n_max must be >= n_min")
	if exponent_max < exponent_min:
		raise ValueError("exponent_max must be >= exponent_min")

	log_min = np.log(float(n_min))
	log_max = np.log(float(n_max))
	if rng is None:
		u = np.random.random()
		u_exp = np.random.random()
	else:
		u = float(_rng_random(rng))
		u_exp = float(_rng_random(rng))

	n_val = np.exp(log_min + (log_max - log_min) * u)
	n_int = int(np.clip(np.rint(n_val), n_min, n_max))
	exponent = exponent_min + (exponent_max - exponent_min) * u_exp
	return n_int, exponent


def draw_sharp_edges_params(
	*,
	exponent_lf_min: float = 1.5,
	exponent_lf_max: float = 5.0,
	percentile_lf_min: float = 50.0,
	percentile_lf_max: float = 99.0,
	exponent_hf_min: float = 1.5,
	exponent_hf_max: float = 5.0,
	vmin_hf_min: float = 0.0,
	vmin_hf_max: float = 1.0,
	rng=None,
):
	"""Draw random parameters for sharp_edges_object generation."""
	if exponent_lf_max < exponent_lf_min:
		raise ValueError("exponent_lf_max must be >= exponent_lf_min")
	if percentile_lf_max < percentile_lf_min:
		raise ValueError("percentile_lf_max must be >= percentile_lf_min")
	if exponent_hf_max < exponent_hf_min:
		raise ValueError("exponent_hf_max must be >= exponent_hf_min")
	if vmin_hf_max < vmin_hf_min:
		raise ValueError("vmin_hf_max must be >= vmin_hf_min")

	if rng is None:
		u1 = np.random.random()
		u2 = np.random.random()
		u3 = np.random.random()
		u4 = np.random.random()
	else:
		u1 = float(_rng_random(rng))
		u2 = float(_rng_random(rng))
		u3 = float(_rng_random(rng))
		u4 = float(_rng_random(rng))

	exponent_lf = exponent_lf_min + (exponent_lf_max - exponent_lf_min) * u1
	percentile_lf = percentile_lf_min + (percentile_lf_max - percentile_lf_min) * u2
	exponent_hf = exponent_hf_min + (exponent_hf_max - exponent_hf_min) * u3
	vmin_hf = vmin_hf_min + (vmin_hf_max - vmin_hf_min) * u4
	return exponent_lf, percentile_lf, exponent_hf, vmin_hf


def draw_uniform_fluxes(
	*,
	n: int,
	rng=None,
):
	"""Draw n uniform fluxes between 0 and 1."""
	if n < 1:
		raise ValueError("n must be >= 1")
	if rng is None:
		return np.random.random(n)
	return np.asarray(_rng_random(rng, size=(n,)))


def draw_log_uniform_flux(flux_min: float, flux_max: float, rng=None) -> float:
	"""Draw a single log-uniform flux in [flux_min, flux_max]."""
	if rng is None:
		u = np.random.random()
	else:
		u = float(_rng_random(rng))
	log_min = np.log(flux_min)
	log_max = np.log(flux_max)
	return float(np.exp(log_min + (log_max - log_min) * u))


def draw_random_image_parameters(
	cfg,
	*,
	rng=None,
):
	"""Draw random parameters for image_generator.

	All min/max ranges are forwarded to the individual draw functions.
	"""
	n_objects = draw_n_objects(
		n_objects_min=cfg.general.n_objects_min,
		n_objects_max=cfg.general.n_objects_max,
		rng=rng,
	)

	function_list = []
	params_list = []

	# Always include mandatory object types first.
	mandatory_map = [
		(nebula, cfg.sky.nebula),
		(point_sources, cfg.sky.point_sources),
		(sharp_edges_object, cfg.sky.sharp_edges),
	]
	mandatory_functions = [
		func for func, section in mandatory_map
		if getattr(section, 'weight', 0.0) >= 1.0
	]

	def _draw_params(func):
		if func is nebula:
			return draw_nebula_params(
				exponent_min=cfg.sky.nebula.exponent_min,
				exponent_max=cfg.sky.nebula.exponent_max,
				percentile_min=cfg.sky.nebula.percentile_min,
				percentile_max=cfg.sky.nebula.percentile_max,
				rng=rng,
			)
		elif func is point_sources:
			return draw_point_sources_params(
				n_min=cfg.sky.point_sources.n_min,
				n_max=cfg.sky.point_sources.n_max,
				exponent_min=cfg.sky.point_sources.exponent_min,
				exponent_max=cfg.sky.point_sources.exponent_max,
				rng=rng,
			)
		elif func is sharp_edges_object:
			return draw_sharp_edges_params(
				exponent_lf_min=cfg.sky.sharp_edges.exponent_lf_min,
				exponent_lf_max=cfg.sky.sharp_edges.exponent_lf_max,
				percentile_lf_min=cfg.sky.sharp_edges.percentile_lf_min,
				percentile_lf_max=cfg.sky.sharp_edges.percentile_lf_max,
				exponent_hf_min=cfg.sky.sharp_edges.exponent_hf_min,
				exponent_hf_max=cfg.sky.sharp_edges.exponent_hf_max,
				vmin_hf_min=cfg.sky.sharp_edges.vmin_hf_min,
				vmin_hf_max=cfg.sky.sharp_edges.vmin_hf_max,
				rng=rng,
			)
		else:
			raise ValueError("Unknown object function selected")

	def _draw_flux(func):
		if func is nebula:
			return draw_log_uniform_flux(
				cfg.sky.nebula.flux_min, cfg.sky.nebula.flux_max, rng=rng
			)
		elif func is point_sources:
			return draw_log_uniform_flux(
				cfg.sky.point_sources.flux_min, cfg.sky.point_sources.flux_max, rng=rng
			)
		elif func is sharp_edges_object:
			return draw_log_uniform_flux(
				cfg.sky.sharp_edges.flux_min, cfg.sky.sharp_edges.flux_max, rng=rng
			)
		else:
			raise ValueError("Unknown object function selected")

	for func in mandatory_functions:
		function_list.append(func)
		params_list.append(_draw_params(func))

	n_random = max(0, n_objects - len(mandatory_functions))
	for _ in range(n_random):
		func = draw_random_object_function(cfg, rng=rng)
		function_list.append(func)
		params_list.append(_draw_params(func))

	flux_list = [_draw_flux(func) for func in function_list]

	return function_list, params_list, flux_list
