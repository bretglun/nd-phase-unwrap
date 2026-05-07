# LaPlacian-based phase unwrapping described in:
# Loecher M et al. JMRI 2016; 43:833--842.

import numpy as np


def laplace(img, kernel, forward=True):
    k = np.fft.fftn(img)
    if forward:
        k = np.multiply(k, kernel)
    else: # inverse transform
        kernel[0] = 1.
        k = np.divide(k, kernel)
    return np.real(np.fft.ifftn(k))


def unwrap(arr, config):
    phase = np.angle(arr)
    kernel = np.zeros_like(phase)
    for dim, N in enumerate(phase.shape):
        vec = (np.cos(np.fft.fftfreq(N) * np.pi) - 1) * max(config.pixel_spacing) / config.pixel_spacing[dim]
        reps = [n if d!=dim else 1 for d, n in enumerate(phase.shape)]
        kernel += np.tile(np.expand_dims(vec, axis=[d for d in range(phase.ndim) if d!=dim]), reps)

    lap_wrapped = laplace(phase, kernel)
    lap = np.multiply(np.cos(phase), laplace(np.sin(phase), kernel)) - np.multiply(np.sin(phase), laplace(np.cos(phase), kernel))
    ilap_diff = laplace(lap-lap_wrapped, kernel, forward=False)

    phase += 2 * np.pi * np.round(ilap_diff / (2 * np.pi)) # truncate to 2pi jumps
    return phase