from nd_phase_unwrap import io, unwrap
from pathlib import Path
import os 
import numpy as np
from scipy.ndimage import binary_dilation

CONFIG = {
    'algorithm': 'IterativeGraphCuts',
    'pixel_spacing': (2.0, 2.0, 2.0, 40.0),
    'cyclic': (False, False, False, True),
    'data_cost': False,
    'neighbourhood_radius': 1.0,
    'neighbourhood_weight': 10.0**2.5,
    'mask': None
}

def demo():
    data_path = Path(__file__).parent.parent / 'test_data'
    in_path = data_path / '3+1D_phantom'
    out_path = data_path / 'IGC_unwrapped'
    if not out_path.is_dir():
        out_path.mkdir(parents=True)

    mask = binary_dilation(np.sum(np.angle(io.load_numpy_data(in_path/"truth.npy")) != 0, axis=-1),np.ones((3,3,3,3)), iterations=2)

    for file in [j for j in os.listdir(in_path) if j.endswith('.npy')]:
        print (f'Processing file {file}...')
        data_file = in_path / file
        rawdata = io.load_numpy_data(data_file)
        CONFIG['mask'] = mask
        saverdata = np.zeros(rawdata.shape, dtype=np.float32)
        for dim in range(rawdata.shape[-1]):
            print(f'Unwrapping dimension {dim} of shape {file}...')
            phaseArray = unwrap.unwrap(rawdata[..., dim], CONFIG)
            print(f'Unwrapped {file} with shape {phaseArray.shape} and dtype {phaseArray.dtype}')         
            saverdata[..., dim] = phaseArray
        io.save(saverdata, out_path / f'{data_file.stem}_unwrapped.npy')

    # Laplace
    config = {
        'algorithm': 'Laplace'
    }
    data_path = Path(__file__).parent.parent / 'test_data'
    in_path = [data_path / '3+1D_phantom']
    out_path = data_path / 'Laplace_unwrapped'
    if not out_path.is_dir():
        out_path.mkdir(parents=True)
        
    for folder in in_path:
        for file in [i for i in os.listdir(folder) if not i.endswith('.py')]:
            print (f'Processing file {folder/file}...')
            data_file = folder / f'{file}'
            rawdata = io.load_numpy_data(data_file)
            saverdata = np.zeros(rawdata.shape, dtype=np.float32)
            print(f'Unwrapping {file}...')
            for vencdir in range(rawdata.shape[-1]):
                phaseArray = unwrap.unwrap(rawdata[...,vencdir], config)        
                saverdata[...,vencdir] = phaseArray
            io.save(saverdata, out_path / f'{Path(file).stem}_unwrapped.npy')

    # ROMEO
    config = {
        'algorithm': 'ROMEO',
        'pixel_spacing': (2.0, 2.0, 2.0, 20.0),
        'cyclic': (False, False, False, True),
        'individual': False,      # timepoints in the 4th dim -> unwrap each spatially
        'use_magnitude': True,  # test data phase is stored with unit magnitude
        'echo_times': 'epi',
        'mask': None,
        'romeo_path': 'C:\\Users\\jar048\\Desktop\\Coding\\ROMEO_native\\bin\\romeo.exe',  # uncomment if ROMEO is not on PATH
    }
    data_path = Path(__file__).parent.parent / 'test_data'
    in_path = [data_path / '3+1D_phantom']
    out_path = data_path / 'ROMEO_unwrapped'
    if not out_path.is_dir():
        out_path.mkdir(parents=True)
        
    for folder in in_path:
        for file in [i for i in os.listdir(folder) if not i.endswith('.py')]:
            print (f'Processing file {folder/file}...')
            data_file = folder / f'{file}'
            rawdata = io.load_numpy_data(data_file)
            saverdata = np.zeros(rawdata.shape, dtype=np.float32)
            print(f'Unwrapping {file}...')
            for vencdir in range(rawdata.shape[-1]):
                phaseArray = unwrap.unwrap(rawdata[...,vencdir], config)        
                saverdata[...,vencdir] = phaseArray
            io.save(saverdata, out_path / f'{Path(file).stem}_unwrapped.npy')

if __name__ == '__main__':
    demo()