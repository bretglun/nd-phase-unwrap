
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)

# nd-phase-unwrap

**Phase unwrapping for n-dimensional arrays**


## Overview

`nd-phase-unwrap` is a Python package for phase unwrapping of n-dimensional numpy arrays with several algorithms available. It can be used in Python scripts and as a command line tool.

## Installation

```
pip install nd-phase-unwrap
```
Alternatively, clone the repository and install dependencies, for instance using [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/bretglun/nd-phase-unwrap.git
cd nd-phase-unwrap
uv sync
```

## Usage

### Command-Line Interface
```bash
# Show help
unwrap -h

# Run with config file
unwrap nd_phase_unwrap/demo/demo.npy -p nd_phase_unwrap/demo/config.yml
```

See [demo/demo.py](demo/demo.py) for a complete example.

## Configuration
Input parameters can be provided as:
* Human-readable YAML configuration file
* Python dictionaries

An example configuration file is at [demo/config.yml](demo/config.yml).
The phase to be unwrapped is either the real-valued input array or the phase of the complex-valued input array (the magnitude may be used for weighting).

## Dependencies
See [pyproject.toml](pyproject.toml) for the complete list of dependencies.

## Citation
If you use this software in your research, please cite:

> Berglund J and Skorpil M. Multi-scale graph-cut algorithm for efficient water-fat separation. *Magn Reson Med*, 78(3):941-949, 2017. [[doi: 10.1002/mrm.26479](https://doi.org/10.1002/mrm.26479)]

## License
This program is free software: you can redistribute it and/or modify it under the terms of the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0).

See [LICENSE.md](LICENSE.md) for the full license text.

## Contact
Johan Berglund, Ph.D.  
Uppsala University Hospital, Uppsala, Sweden  
📧 johan.berglund@akademiska.se

*Copyright © 2026 Johan Berglund*