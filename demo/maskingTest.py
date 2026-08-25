'''
test if masking does a difference to unwrapping performance
test 3+1D phantom with VNR 3.0

Unwrap the rawfile without mask and compare with the output from the masked run already available in IGC_unwrapped.
The comparison shall only be done inside the mask, meaning inside the volume and all the background shall not be considered.
'''

from nd_phase_unwrap import io, unwrap
from pathlib import Path
import os 
import numpy as np
from scipy.ndimage import binary_dilation

CONFIG = {
    'algorithm': 'IterativeGraphCuts',
    'pixel_spacing': (2.0, 2.0, 2.0, 40.0),
    'cyclic': (False, False, False, True),
    'data_cost': 0.001,
    'neighbourhood_radius': 1.0,
    'neighbourhood_weight': 10.0**2.5,
    'mask': None
}

def demo():
    data_path = Path(__file__).parent.parent / 'test_data'
    in_path = data_path / '3+1D_phantom'
    out_path = data_path / '4DphantomNoMask'
    if not out_path.is_dir():
        out_path.mkdir(parents=True)

    for file in [j for j in os.listdir(in_path) if (j.endswith('1_0.npy') or j.endswith('3_0.npy'))]:
        print (f'Processing file {file}...')
        data_file = in_path / file
        rawdata = io.load_numpy_data(data_file)
        saverdata = np.zeros(rawdata.shape, dtype=np.float32)
        for dim in range(rawdata.shape[-1]):
            print(f'Unwrapping dimension {dim} of shape {file}...')
            phaseArray = unwrap.unwrap(rawdata[..., dim], CONFIG)
            print(f'Unwrapped {file} with shape {phaseArray.shape} and dtype {phaseArray.dtype}')         
            saverdata[..., dim] = phaseArray
        io.save(saverdata, out_path / f'{data_file.stem}_noMask_unwrapped.npy')

# maskedrun
    data_path = Path(__file__).parent.parent / 'test_data'
    in_path = data_path / '3+1D_phantom'
    out_path = data_path / '4DphantomMask'
    if not out_path.is_dir():
        out_path.mkdir(parents=True)

    for file in [j for j in os.listdir(in_path) if (j.endswith('1_0.npy') or j.endswith('3_0.npy'))]:
        print (f'Processing file {file}...')
        data_file = in_path / file
        rawdata = io.load_numpy_data(data_file)
        saverdata = np.zeros(rawdata.shape, dtype=np.float32)
        mask = binary_dilation(np.sum(np.angle(io.load_numpy_data(in_path/"0wraps.npy")) != 0, axis=-1),np.ones((3,3,3,3)), iterations=2)
        for dim in range(rawdata.shape[-1]):
            print(f'Unwrapping dimension {dim} of shape {file}...')
            CONFIG['mask'] = mask
            phaseArray = unwrap.unwrap(rawdata[..., dim], CONFIG)
            print(f'Unwrapped {file} with shape {phaseArray.shape} and dtype {phaseArray.dtype}')         
            saverdata[..., dim] = phaseArray
        io.save(saverdata, out_path / f'{data_file.stem}_Mask_unwrapped.npy')


if __name__ == '__main__':
    demo()