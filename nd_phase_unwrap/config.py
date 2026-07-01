from dataclasses import dataclass, fields
import numpy as np


@dataclass
class BaseConfig:
    pixel_spacing: tuple[float, ...] = None # arbitrary unit, length must match arr.ndim
    cyclic: tuple[bool, ...] = None # which dimensions are cyclic?

    def __init__(self, **overrides):
        params = {f.name: f.default for f in fields(self) if f.init}
        params.update(overrides)

        for param in params:
            if not hasattr(self, param):
                raise ValueError(f'Unknown parameter "{param}" passed to {type(self).__name__} constructor')
            setattr(self, param, params[param])
    

@dataclass
class LaplaceConfig(BaseConfig):
    pass


@dataclass
class IGCconfig(BaseConfig):
    data_cost: float = 0. # weight of data cost in energy function (to avoid global 2π-drift)
    neighbourhood_radius: float = 0. # in same units as pixel_spacing, 0 means only immediate neighbours
    mask: np.ndarray = None # boolean array of same shape as arr, True for voxels to include in graph
    neighbourhood_weight: float|tuple = 1. # Lambda values for weighting spatial to temporal neighbors. Scalar inputs assumes the time dimension to be at the last place. Alternatively can a tuple with the same dims as pixel spacing be given to set the weighting for each dimension

    # weights can be either given as scalars or an 1D array of the same length as the number of dimensions in the input array
    def __post_init__(self):
        if type(self.neighbourhood_weight) == int:
            self.neighbourhood_weight = float(self.neighbourhood_weight)
        if type(self.neighbourhood_weight) == float:
            self.neighbourhood_weight = tuple(1.0 if i != len(self.pixel_spacing)-1 else self.neighbourhood_weight for i in range(len(self.pixel_spacing)))
        if len(self.neighbourhood_weight) != len(self.pixel_spacing):
            raise ValueError(f"The given neighborhood weights do not have the same amount of dimensions {len(self.neighbourhood_weight)} as the pixel spacing {len(self.pixel_spacing)}")
        