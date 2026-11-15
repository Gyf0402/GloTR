"""Use a legal pre-gap pseudo-outage to calibrate occupancy residual level."""

from __future__ import annotations

import numpy as np


def calibrate_occupancy(prediction, pseudo_error_mean, correction_mean, alpha):
    """Move only occupancy by alpha times its prior-vs-current error mismatch."""
    prediction = np.asarray(prediction)
    if prediction.ndim != 3 or prediction.shape[-1] != 2 or not 0 <= alpha <= 1:
        raise ValueError('invalid calibration input')
    if alpha == 0:
        return prediction.copy()
    shift = alpha * (float(pseudo_error_mean) - float(correction_mean))
    return np.stack((prediction[..., 0], prediction[..., 1] + shift), axis=-1)
