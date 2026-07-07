from nd_phase_unwrap import io, unwrap
from pathlib import Path
import os 
import numpy as np

CONFIG = {
    'algorithm': 'IterativeGraphCuts',
    'pixel_spacing': (2.0, 2.0, 2.0, 20.0),
    'cyclic': (False, False, False, True),
    'data_cost': 0.001,
    'neighbourhood_radius': 1.0,
    'neighbourhood_weight': 10.0**3,
    'mask': None
}

def demo():
    data_path = Path(__file__).parent.parent / 'test_data'
    in_path = data_path / 'Synth_4Dflow'
    mask_path = data_path / 'Synth_4Dflow_mask_dilated'
    out_path = data_path / 'Synth_4Dflow_unwrapped'
    if not out_path.is_dir():
        out_path.mkdir(parents=True)

    # for file in [f'gt00{i}' for i in range(1, 10)]:
    for file in [j for j in os.listdir(in_path) if j.endswith('.npy')]:
            print (f'Processing file {file}...')
            data_file = in_path / f'{file}'
            rawdata = np.angle(io.load_numpy_data(data_file))
            basefilename = Path(file).stem
            maskdata = io.load_numpy_data(mask_path / f'{basefilename[:5]}_mask_dilated.npy')
            saverdata = np.zeros(rawdata.shape, dtype=np.float32)
            for dim in range(rawdata.shape[-1]):
                CONFIG['mask'] = maskdata[..., dim]
                print(f'Unwrapping dimension {dim} of shape {file}...')
                phaseArray = unwrap.unwrap(rawdata[..., dim], CONFIG)
                print(f'Unwrapped {file} with shape {phaseArray.shape} and dtype {phaseArray.dtype}')         
                saverdata[..., dim] = phaseArray
            io.save(saverdata, out_path / f'{basefilename}_unwrapped.npy')

if __name__ == '__main__':
    demo()