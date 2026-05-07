"""Minimal instrument stub for sky-image generation.

The full ``AO_instrument`` class (in ``instruments/ao_instrument.py``) provides
many quantities tied to a physical telescope and HCIPy pupil model.  The sky
generation code only needs three attributes:

* ``n_pix``        — image side length in pixels
* ``xp``           — the array backend (``numpy`` or ``cupy``)
* ``fr_pix_large`` — 2N × 2N radial frequency grid (cycles / pixel)

``SkyInstrument`` computes exactly those three attributes from a single
``n_pix`` argument so that the sky-generation modules can be used without
the rest of the V1H project.
"""

from __future__ import annotations

import numpy as np


class SkyInstrument:
    """Lightweight instrument stub for sky-image generation.

    Parameters
    ----------
    n_pix : int
        Side length of the output image in pixels.  The internal GRF grid
        is twice this size (``2 * n_pix``).
    use_cupy : bool, optional
        When ``True``, arrays are allocated on the GPU via CuPy.
        Defaults to ``False`` (NumPy / CPU).
    """

    def __init__(self, n_pix: int, use_cupy: bool = False) -> None:
        if n_pix < 4:
            raise ValueError("n_pix must be at least 4")

        self.n_pix: int = n_pix

        if use_cupy:
            import cupy as cp
            self.xp = cp
        else:
            self.xp = np

        # Build the 2N × 2N radial frequency grid (cycles / pixel)
        n_large = n_pix * 2
        fx = self.xp.fft.fftfreq(n_large, d=1.0)
        fy = self.xp.fft.fftfreq(n_large, d=1.0)
        fx_large, fy_large = self.xp.meshgrid(fx, fy)
        self.fr_pix_large = self.xp.sqrt(fx_large ** 2 + fy_large ** 2)
