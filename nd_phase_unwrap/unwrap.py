from nd_phase_unwrap import io
from nd_phase_unwrap.constants import ALGORITHMS


def unwrap(arr, config_dict={}, config_file=None):
    config = io.read_config_file(config_file)
    config.update(config_dict)
    
    if 'algorithm' not in config:
        raise ValueError(f'No algorithm specified in config. Available algorithms are: {list(ALGORITHMS.keys())}')
    algorithm = config.pop('algorithm')
    if algorithm not in ALGORITHMS:
        raise ValueError(f'Unknown algorithm "{algorithm}". Supported algorithms are {list(ALGORITHMS.keys())}.')
    algorithm_config = ALGORITHMS[algorithm]['config_class'](**config)
    return ALGORITHMS[algorithm]['module'].unwrap(arr, algorithm_config)