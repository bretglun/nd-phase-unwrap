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
# maskedrun
    data_path = Path(__file__).parent.parent / 'test_data'
    in_path = data_path / '3+1D_phantom'
    out_path = data_path / '4DphantomLambda'
    if not out_path.is_dir():
        out_path.mkdir(parents=True)

    mask = binary_dilation(np.sum(np.angle(io.load_numpy_data(in_path/"0wraps.npy")) != 0, axis=-1),np.ones((3,3,3,3)), iterations=2)
    CONFIG['mask'] = mask

    for file in [j for j in os.listdir(in_path) if (j.endswith('1wraps.npy') or j.endswith('3wraps.npy') or j.endswith('5wraps.npy'))]:
        print (f'Processing file {file}...')
        for lbda in [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]:
            data_file = in_path / file
            rawdata = io.load_numpy_data(data_file)
            saverdata = np.zeros(rawdata.shape, dtype=np.float32)
            CONFIG['neighbourhood_weight'] = 10**lbda
            for dim in range(rawdata.shape[-1]):
                print(f'Unwrapping dimension {dim} of shape {file}...')
                phaseArray = unwrap.unwrap(rawdata[..., dim], CONFIG)
                # print(f'Unwrapped {file} with shape {phaseArray.shape} and dtype {phaseArray.dtype}')         
                saverdata[..., dim] = phaseArray
            io.save(saverdata, out_path / f'{data_file.stem}_{str(lbda).replace(".","_")}_unwrapped.npy')


if __name__ == '__main__':
    demo()