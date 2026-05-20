"""Gaussian random field generation and astronomical object functions.

Provides power spectral density (PSD) generators, a GRF sampler, and
three sky-object constructors (nebula, point sources, sharp-edged objects)
that operate on the frequency grids stored in a SkyInstrument instance.

All PSD functions work on ``ao_instru.fr_pix_large`` (the 2N × 2N grid).
``grf_from_psd`` draws a realisation on that large grid, then centre-crops
to ``ao_instru.n_pix × ao_instru.n_pix``.
"""

from __future__ import annotations

import numpy as np
import cupy as cp

from array_backend import is_cupy_array
from array_backend import get_xp_from_array as _get_xp


def _normalize_mean(image, xp):
	"""Normalize image so its mean is 1 (fallback to ones if mean <= 0)."""
	mean_val = float(xp.mean(image))
	if mean_val > 0.0:
		return image / mean_val
	return xp.ones_like(image)

def make_grid(n_pix: int, use_cupy: bool = False, device_id: int = 0):
	xp = cp if use_cupy else np
	if use_cupy:
		device = cp.cuda.Device(device_id)
		device.use()
		print('Using device: ', cp.cuda.runtime.getDeviceProperties(device)['name'])
	n_large = n_pix * 2
	fx = xp.fft.fftfreq(n_large, d=1.0)        # shape (2N,)
	fy = xp.fft.rfftfreq(n_large, d=1.0)       # shape (N+1,)
	fx_large, fy_large = xp.meshgrid(fx, fy, indexing='ij')
	fr_pix_large = xp.sqrt(fx_large ** 2 + fy_large ** 2)

	return fr_pix_large

# ---------------------------------------------------------------------------
# PSD functions  (all work on the 2N×2N ``fr_pix_large`` grid)
# ---------------------------------------------------------------------------


def powerlaw_psd(grid, alpha: float):
	"""Power-law PSD: P(k) ∝ k^{-α}  (DC set to 0).

	Parameters
	----------
	grid : ndarray
		2D frequency grid with shape ``(2*n_pix, 2*n_pix)``.
	alpha : float
		Power-law exponent (positive → red spectrum).

	Returns
	-------
	ndarray
		2D PSD array with shape ``(2*n_pix, 2*n_pix)``.
	"""
	xp = cp if is_cupy_array(grid) else np
	k = grid
	psd = xp.zeros_like(k, dtype=xp.float64)
	mask = k > 0
	psd[mask] = k[mask] ** (-alpha)
	return psd


# ---------------------------------------------------------------------------
# GRF sampler
# ---------------------------------------------------------------------------

def grf_from_psd(n_pix, psd, *, rng=None):
	"""Draw a Gaussian random field realisation from a 2D PSD.

	The field is generated on the 2N × 2N grid (matching ``fr_pix_large``)
	and centre-cropped to ``n_pix × n_pix``.

	Parameters
	----------
	n_pix : int
		Number of pixels in each dimension of the output image.
	psd : ndarray
		2D PSD array of shape ``(2*n_pix, n_pix+1)`` (half-spectrum from
		``make_grid``).
	rng : numpy.random.Generator or None, optional
		Random number generator.  If *None*, ``numpy.random`` is used.

	Returns
	-------
	ndarray
		Real-valued 2D array of shape ``(n_pix, n_pix)``.
	"""
	xp = _get_xp(psd)
	n_large = psd.shape[0]  # = 2 * n_pix

	# Draw real white noise in pixel space — avoids a 6 GB complex array.
	if rng is None:
		noise_np = np.random.standard_normal((n_large, n_large))
	else:
		noise_np = rng.standard_normal((n_large, n_large))

	if xp is not np:
		psd.device.use()  # ensure all CuPy ops below run on the same device as psd
		noise = xp.asarray(noise_np)
		del noise_np  # free CPU buffer as soon as GPU copy is made
	else:
		noise = noise_np
		
	print("psd device ", psd.device)
	print("noise device ", noise.device)
	# Forward real FFT → half-spectrum, same shape as psd.
	noise_fft = xp.fft.rfft2(noise)           # shape: (n_large, n_large//2+1)
	del noise  # free the (2N×2N) real array — no longer needed

	# sqrt in-place on psd to avoid a separate sqrt_psd allocation.
	xp.sqrt(psd, out=psd)
	# Multiply in-place: avoids a 3.2 GB temporary while noise_fft is alive.
	noise_fft *= psd

	# Inverse real FFT → guaranteed real output, no .real() needed.
	field_large = xp.fft.irfft2(noise_fft, s=(n_large, n_large))
	del noise_fft  # free (2N×(N+1)) complex array

	# Centre-crop to n_pix × n_pix
	start = (n_large - n_pix) // 2
	field = field_large[start:start + n_pix, start:start + n_pix].copy()
	del field_large
	return field


# ---------------------------------------------------------------------------
# Astronomical object generators
# ---------------------------------------------------------------------------

def nebula(grid, n_pix: int, exponent: float, percentile: float):
	"""Generate a nebula-like object via a thresholded power-law GRF.

	Parameters
	----------
	grid : ndarray
		2D frequency grid with shape ``(2*n_pix, 2*n_pix)``.
		Instrument providing frequency grids and backend.
	n_pix : int
		Number of pixels in each dimension of the output image.
	exponent : float
		Power-law exponent for the PSD.
	percentile : float
		Percentile threshold (0–100).  Pixels below the threshold are
		zeroed, creating a sparse, cloudy structure.

	Returns
	-------
	ndarray
		2D image of shape ``(n_pix, n_pix)`` with values ≥ 0.
	"""
	xp = cp if is_cupy_array(grid) else np
	psd = powerlaw_psd(grid, exponent)
	field = grf_from_psd(n_pix, psd)
	del psd

	# Threshold at the requested percentile
	if xp is not np:
		thresh = float(xp.percentile(field, percentile))
	else:
		thresh = float(np.percentile(field, percentile))

	# In-place subtract + clip — avoids allocating a temporary (N×N) array.
	field -= thresh
	xp.clip(field, 0.0, None, out=field)
	return _normalize_mean(field, xp)


def point_sources(grid, n_pix: int, n: int, exponent: float):
	"""Generate a field of random point sources.

	Each source is a single bright pixel placed at a random position.
	Brightnesses follow a power-law distribution.

	Parameters
	----------
	grid : ndarray
		2D frequency grid with shape ``(2*n_pix, 2*n_pix)``.
	n_pix : int
		Number of pixels in each dimension of the output image.
	n : int
		Number of point sources.
	exponent : float
		Power-law exponent for the brightness distribution.
		Brightnesses are drawn as U^{-1/(exponent-1)} where U ~ Uniform(0,1).

	Returns
	-------
	ndarray
		2D image of shape ``(n_pix, n_pix)``.
	"""
	xp = cp if is_cupy_array(grid) else np

	n = int(n)
	if n < 1:
		return xp.zeros((n_pix, n_pix), dtype=xp.float64)

	# Random positions (CPU, then transfer if needed)
	ys = np.random.randint(0, n_pix, size=n)
	xs = np.random.randint(0, n_pix, size=n)

	# Power-law brightnesses
	u = np.random.random(n)
	u = np.clip(u, 1e-12, 1.0)
	alpha = max(exponent, 1.01)  # ensure exponent > 1
	brightnesses = u ** (-1.0 / (alpha - 1.0))
	brightnesses /= brightnesses.max()  # normalise to [0, 1]

	image_np = np.zeros((n_pix, n_pix), dtype=np.float64)
	np.add.at(image_np, (ys, xs), brightnesses)
	image = xp.asarray(image_np) if xp is not np else image_np

	return _normalize_mean(image, xp)


def sharp_edges_object(
	grid,
	n_pix: int,
	exponent_lf: float,
	percentile_lf: float,
	exponent_hf: float,
	vmin_hf: float,
):
	"""Generate an object with sharp edges (e.g. galaxy-like structure).

	A low-frequency GRF defines the overall shape (thresholded at
	*percentile_lf*), while a high-frequency GRF adds internal texture.

	Parameters
	----------
	grid : ndarray
		2D frequency grid with shape ``(2*n_pix, 2*n_pix)``.
	n_pix : int
		Number of pixels in each dimension of the output image.
	exponent_lf : float
		Power-law exponent for the low-frequency (shape) component.
	percentile_lf : float
		Percentile threshold for the shape mask (0–100).
	exponent_hf : float
		Power-law exponent for the high-frequency (texture) component.
	vmin_hf : float
		Minimum clip value for the texture component (in [0, 1] of its
		range), controlling how much internal contrast is retained.

	Returns
	-------
	ndarray
		2D image of shape ``(n_pix, n_pix)`` with values ≥ 0.
	"""
	xp = cp if is_cupy_array(grid) else np

	# Low-frequency shape mask
	psd_lf = powerlaw_psd(grid, exponent_lf)
	field_lf = grf_from_psd(n_pix, psd_lf)

	if xp is not np:
		thresh_lf = float(xp.percentile(field_lf, percentile_lf))
	else:
		thresh_lf = float(np.percentile(field_lf, percentile_lf))

	mask = (field_lf >= thresh_lf).astype(xp.float64)

	# High-frequency texture
	psd_hf = powerlaw_psd(grid, exponent_hf)
	field_hf = grf_from_psd(n_pix, psd_hf)

	# Normalise texture to [0, 1]
	hf_min = float(xp.min(field_hf))
	hf_max = float(xp.max(field_hf))
	if hf_max - hf_min > 0:
		field_hf = (field_hf - hf_min) / (hf_max - hf_min)
	else:
		field_hf = xp.ones_like(field_hf)

	# Map the texture to [vmin_hf, 1]
	field_hf = float(vmin_hf) + (1.0 - float(vmin_hf)) * field_hf

	image = mask * field_hf
	return _normalize_mean(image, xp)