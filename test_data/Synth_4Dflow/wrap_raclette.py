"""Add phase wraps to the CFD 4D flow simulations of the RACLETTE ground-truth files.
``https://gitlab.ethz.ch/ibt-cmr/publications/raclette/-/tree/main?ref_type=heads``

The ``.npy`` files hold velocity-encoded phase of shape
``(x, y, z, t, 3)``, where the last dimension corresponds to the 3 velocity-encoding components.
"""

import numpy as np
from pathlib import Path

RAW_FILES = ['GroundTruth001.npy']
NUM_WRAPS = [1, 2, 3]

def wrap(arr, factor):
    if np.iscomplexobj(arr):
        return np.abs(arr) * np.exp(1j * np.angle(arr) * factor)
    return np.angle(np.exp(1j * arr * factor))


def main():
    path = Path(__file__).parent

    for fname in RAW_FILES:
        truth = np.load(path / fname)
        stem = Path(fname).stem
        for num_wraps in NUM_WRAPS:
            factor = 1 + 2 * num_wraps
            wrapped = wrap(truth, factor).astype(truth.dtype)
            out = path / f'{stem}_{num_wraps}wraps.npy'
            print(f'Writing array to "{out}"')
            np.save(out, wrapped)


if __name__ == '__main__':
    main()
