from nd_phase_unwrap import io, unwrap
from pathlib import Path
import os 
import numpy as np

CONFIG = {
    'algorithm': 'IterativeGraphCuts',
    'pixel_spacing': (2.0, 2.0, 2.0, 20.0),
    'cyclic': (False, False, False, True),
    'data_cost': True,
    'neighbourhood_radius': 1.0
}

def demo():
    data_path = Path(__file__).parent.parent / 'test_data'
    in_path = data_path / 'Synth_4Dflow'
    out_path = data_path / 'Synth_4Dflow'
    if not out_path.is_dir():
        out_path.mkdir(parents=True)

    # for file in [j for j in os.listdir(in_path) if j.endswith('.npy')]:
    for file in ['GroundTruth001.npy']:
        print (f'Processing file {file}...')
        data_file = in_path / file
        rawdata = io.load_numpy_data(data_file)
        saverdata = np.zeros(rawdata.shape, dtype=np.float32)
        for dim in range(rawdata.shape[-1]):
            print(f'Unwrapping dimension {dim} of shape {file}...')
            phaseArray = unwrap.unwrap(rawdata[..., dim], CONFIG)
            print(f'Unwrapped {file} with shape {phaseArray.shape} and dtype {phaseArray.dtype}')         
            saverdata[..., dim] = phaseArray
        io.save(saverdata, out_path / f'unwrapped_{file}')

if __name__ == '__main__':
    demo()