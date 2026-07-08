from nd_phase_unwrap.config import LaplaceConfig, IGCconfig, ROMEOconfig
from nd_phase_unwrap import laplace, iterative_graphcuts, romeo


ALGORITHMS = {
    'Laplace': {'config_class': LaplaceConfig, 'module': laplace}, 
    'IterativeGraphCuts': {'config_class': IGCconfig, 'module': iterative_graphcuts},
    'ROMEO': {'config_class': ROMEOconfig, 'module': romeo}
}