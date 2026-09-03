"""Add phase wraps to the CFD 4D flow simulations of the RACLETTE ground-truth files.
``https://gitlab.ethz.ch/ibt-cmr/publications/raclette/-/tree/main?ref_type=heads``

The ``.npy`` files hold velocity-encoded phase of shape
``(x, y, z, t, 3)``, where the last dimension corresponds to the 3 velocity-encoding components.
"""

import numpy as np
from pathlib import Path
import os
from scipy.ndimage import binary_dilation

NUM_WRAPS = [1, 2]

def wrap(arr, factor):
    if np.iscomplexobj(arr):
        return np.abs(arr) * np.exp(1j * np.angle(arr) * factor)
    return np.angle(np.exp(1j * arr * factor))

def add_noise_to_phase(arr, noise_std):
    """Add Gaussian noise to the phase of a complex array."""
    if not np.iscomplexobj(arr):
        raise ValueError('Input array must be complex.')
    phase = np.angle(arr)
    noise = np.random.normal(0, noise_std, size=phase.shape)
    noisy_phase = phase + noise
    return np.abs(arr) * np.exp(1j * noisy_phase)


def main():
    path = Path(__file__).parent
    maskpath = Path(__file__).parent.parent / "Synth_4Dflow_mask"
    outpath = Path(__file__).parent.parent / "Synth_4Dflow_wrapped" # in means in-data for further computations
    os.makedirs(maskpath, exist_ok=True)
    os.makedirs(outpath, exist_ok=True)

    for fname in [file for file in os.listdir(path) if file.endswith(".npy")]:
        truth = np.load(path / fname)
        stem = Path(fname).stem

        # Extract mask of aorta from data by extracting the non zero voxels from the angle of the complex data
        mask = np.angle(truth) != 0
        mask_dilated = binary_dilation(mask, structure=np.ones((3, 3, 3, 3, 1)), iterations=2)
        np.save(maskpath / f'{stem}_mask.npy', mask)
        np.save(maskpath / f'{stem}_mask_dilated.npy', mask_dilated)

        for num_wraps in NUM_WRAPS:
            factor = 1 + 2 * num_wraps
            wrapped = wrap(truth, factor).astype(truth.dtype)
            out = outpath / f'{stem}_{num_wraps}wraps.npy'
            print(f'Writing array to "{out}"')
            np.save(out, wrapped)

            for noise_power in [1.0, 2.0]:
                VNR = 10**noise_power
                wrapped_noisy = add_noise_to_phase(wrapped, noise_std=np.pi / VNR)
                out = outpath / f'{stem}_{num_wraps}wraps_{int(noise_power)}VNR.npy'
                print(f'Writing array to "{out}"')
                np.save(out, wrapped_noisy)


if __name__ == '__main__':
    main()
