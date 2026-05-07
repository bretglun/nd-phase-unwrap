from nd_phase_unwrap import io
from nd_phase_unwrap.constants import ALGORITHMS
import numpy as np
import warnings


def unwrap(arr, config_dict={}, config_file=None):
    config = io.read_config_file(config_file)
    config.update(config_dict)
    
    algorithm = config.pop('algorithm', 'IterativeGraphCuts')
    if algorithm not in ALGORITHMS:
        raise ValueError(f'Unknown algorithm "{algorithm}". Supported algorithms are {list(ALGORITHMS.keys())}.')
    
    period = config.pop('period', 2 * np.pi)
    if np.iscomplexobj(arr):
        if period != 2 * np.pi:
            warnings.warn('For complex input, the period is inherent in the complex representation (=2π). Ignoring config keyword "period".')
            period = 2 * np.pi
        dtype = np.angle(arr).dtype
    else:
        # real-valued input is assumed to be phase and converted to complex array
        dtype = arr.dtype
        arr = np.ones_like(arr) * np.exp(2j * np.pi * arr / period)
    
    if 'pixel_spacing' not in config:
        config['pixel_spacing'] = tuple(1. for _ in range(arr.ndim))
    elif len(config['pixel_spacing']) != arr.ndim:
        raise ValueError(f'Length of "pixel_spacing" must match number of dimensions in input array (got {len(config["pixel_spacing"])} and {arr.ndim}, respectively).')

    algorithm_config = ALGORITHMS[algorithm]['config_class'](**config)
    unwrapped = ALGORITHMS[algorithm]['module'].unwrap(arr, algorithm_config)
    
    return (unwrapped * period / (2 * np.pi)).astype(dtype)