import json
import yaml
import types
import numpy as np
from astropy.io import fits

def load_object(dct):
    return types.SimpleNamespace(**dct)

def load_config(config_path, section=None):
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    if section is not None:
        if not hasattr(cfg, section):
            raise KeyError(f"Section '{section}' not found in config: {config_path}")
        return getattr(cfg, section)

    return cfg

def create_new_header(image_size, pixel_size_arcsec=1.5):
    hdr = fits.Header()
    naxis1 = int(image_size)
    naxis2 = int(image_size)
    cdelt = pixel_size_arcsec / 3600.0  # arcsec → degrees
    hdr["SIMPLE"] = True
    hdr["BITPIX"] = -32
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = naxis1
    hdr["NAXIS2"] = naxis2
    hdr["WCSAXES"] = 2
    hdr["CRPIX1"] = (image_size + 1) / 2.0
    hdr["CRPIX2"] = (image_size + 1) / 2.0
    hdr["CDELT1"] = -cdelt
    hdr["CDELT2"] = cdelt
    hdr["UNIT1"] = 'deg' 
    hdr["UNIT2"] = 'deg' 
    hdr["CTYPE1"] = 'RA---SIN'
    hdr["CTYPE2"] = 'DEC--SIN'
    hdr["CRVAL1"] = 0
    hdr["CRVAL2"] = 0
    hdr["LONPOLE"] = 180.0
    hdr["LATPOLE"] = 0.0
    hdr["BUNIT"] = "JY/PIXEL"
    hdr["ORIGIN"] = "Synthetic image by IMAGE_GENERATION"
    
    return hdr

def np_to_fits(img_xp, out_path, pixel_size_arcsec=1.5):
    header = create_new_header(img_xp.shape[0], pixel_size_arcsec)
    fits.PrimaryHDU(img_xp, header=header).writeto(
                    out_path, overwrite=True
                )
    
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