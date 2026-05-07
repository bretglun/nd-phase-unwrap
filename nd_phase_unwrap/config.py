from dataclasses import dataclass, fields


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
    data_cost: bool = False # include data cost in energy function (to avoid global 2π-drift)