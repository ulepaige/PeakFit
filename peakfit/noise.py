import numpy as np
from lmfit.models import GaussianModel


def estimate_noise(data):
    """Estimate the noise level in the data."""
    std = np.std(data)
    truncated_data = data[np.abs(data) < std]
    y, x = np.histogram(truncated_data.flatten(), bins=100)
    x = (x[1:] + x[:-1]) / 2
    model = GaussianModel()
    pars = model.guess(y, x=x)
    pars["center"].set(value=0.0, vary=False)
    out = model.fit(y, pars, x=x)
    return out.best_values["sigma"]
