
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
unwrap --help

# Unwrap test data
unwrap test_data/2wraps.npy

# Run with config file
unwrap test_data/2wraps.npy --config-file demo/config.yml
```

See [demo/demo.py](demo/demo.py) for a complete example.

## Configuration
Input parameters can be provided as:
* Python dictionaries
* Human-readable YAML configuration file (example at [demo/config.yml](demo/config.yml))

The phase to be unwrapped is either the real-valued input array or the phase of the complex-valued input array (the magnitude may be used for weighting).

## Dependencies
See [pyproject.toml](pyproject.toml) for the complete list of dependencies.

## Citation
The iterative graph cut method for phase unwrapping is described in:

> Berglund J, Liljeblad M, and Baron T. Unwrapping Phase Contrast MRI by Iterative Graph Cuts. *Magnetic Resonance in Medicine*, 92(4):1484–1495, 2024. [[doi: 10.1002/mrm.30138](https://doi.org/10.1002/mrm.30138)]

## License
This program is free software: you can redistribute it and/or modify it under the terms of the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0).

See [LICENSE.md](LICENSE.md) for the full license text.

## Contact
Johan Berglund, Ph.D.  
Uppsala University Hospital, Uppsala, Sweden  
📧 johan.berglund@akademiska.se

*Copyright © 2026 Johan Berglund*