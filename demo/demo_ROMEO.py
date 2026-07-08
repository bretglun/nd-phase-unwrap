"""Unwrap all Synth_4Dflow test data with the ROMEO tool.

This mirrors ``demo_4Dflow.py`` but uses the ROMEO algorithm
(https://github.com/korbinian90/ROMEO) instead of iterative graph cuts. ROMEO
is an external executable operating on NIfTI files; make sure it is installed
and reachable (on PATH, via the ``ROMEO_BINARY`` environment variable, or via
the ``romeo_path`` config keyword below).

Each velocity-encoding component of shape ``(x, y, z, t)`` is written to a
temporary NIfTI file and unwrapped with ROMEO. Because the 4th dimension holds
timepoints (not multi-echo data), ``individual`` unwrapping is used so each
volume is unwrapped spatially.
"""

from nd_phase_unwrap import io, unwrap
from pathlib import Path
import os
import numpy as np

CONFIG = {
    'algorithm': 'ROMEO',
    'pixel_spacing': (2.0, 2.0, 2.0, 20.0),
    'cyclic': (False, False, False, True),
    'individual': True,      # timepoints in the 4th dim -> unwrap each spatially
    'use_magnitude': False,  # test data phase is stored with unit magnitude
    'echo_times': 'epi',
    'mask': None,
    'romeo_path': 'path/to/romeo',  # uncomment if ROMEO is not on PATH
}


def demo():
    data_path = Path(__file__).parent.parent / 'test_data'
    in_path = data_path / 'Synth_4Dflow'
    mask_path = data_path / 'Synth_4Dflow_mask_dilated'
    out_path = data_path / 'Synth_4Dflow_unwrapped_romeo'
    if not out_path.is_dir():
        out_path.mkdir(parents=True)

    for file in [j for j in os.listdir(in_path) if j.endswith('.npy')]:
        print(f'Processing file {file}...')
        data_file = in_path / f'{file}'
        rawdata = np.angle(io.load_numpy_data(data_file))
        basefilename = Path(file).stem
        maskdata = io.load_numpy_data(mask_path / f'{basefilename[:5]}_mask_dilated.npy')
        saverdata = np.zeros(rawdata.shape, dtype=np.float32)
        for dim in range(rawdata.shape[-1]):
            CONFIG['mask'] = maskdata[..., 0, dim]
            print(f'Unwrapping dimension {dim} of shape {file}...')
            phaseArray = unwrap.unwrap(rawdata[..., dim], CONFIG)
            print(f'Unwrapped {file} with shape {phaseArray.shape} and dtype {phaseArray.dtype}')
            saverdata[..., dim] = phaseArray
        io.save(saverdata, out_path / f'{basefilename}_unwrapped.npy')


if __name__ == '__main__':
    demo()
