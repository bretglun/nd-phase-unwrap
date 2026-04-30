from nd_phase_unwrap.config import LaplaceConfig, IGCconfig
from nd_phase_unwrap import laplace, iterative_graphcuts


ALGORITHMS = {
    'Laplace': {'config_class': LaplaceConfig, 'module': laplace}, 
    'IterativeGraphCuts': {'config_class': IGCconfig, 'module': iterative_graphcuts}
}