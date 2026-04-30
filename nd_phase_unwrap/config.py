from dataclasses import dataclass, fields


@dataclass
class BaseConfig():

    def __init__(self, **overrides):
        params = {f.name: f.default for f in fields(self) if f.init}
        params.update(overrides)

        for param in params:
            if not hasattr(self, param):
                raise ValueError(f'Unknown parameter "{param}" passed to {type(self).__name__} constructor')
            setattr(self, param, params[param])
    

@dataclass
class LaplaceConfig(BaseConfig):

    def __init__(self, **overrides):
        super().__init__(**overrides)


@dataclass
class IGCconfig(BaseConfig):

    def __init__(self, **overrides):
        super().__init__(**overrides)