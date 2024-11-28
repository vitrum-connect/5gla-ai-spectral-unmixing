import cv2
import numpy as np

from app.modules.utils import save_channels


def savi(abundances, channels, n_plant, q=70):
    """
    calc savi index. calc L value for each pixel based on abundance maps
    """
    # calc L matrix (soil correction)
    abundances_plant = abundances[:, :, 0:n_plant]
    abundances_plant_sum = np.sum(abundances_plant, axis=2)

    max_abundance = np.percentile(abundances_plant_sum.flatten(), q)
    abundances_norm = abundances_plant_sum / max_abundance
    abundances_norm[abundances_norm > 1] = 1

    soil_correction = 1 - abundances_norm
    soil_correction[soil_correction == 0] = 0.001 # avoid division by 0

    nir = channels[:, :, 4]
    red = channels[:, :, 2]

    savi = (nir - red) / (nir + red + soil_correction) * (1 + soil_correction)
    savi_norm = cv2.normalize(savi, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    save_channels(savi_norm, 'savi', 'savi')
    return savi, savi_norm
