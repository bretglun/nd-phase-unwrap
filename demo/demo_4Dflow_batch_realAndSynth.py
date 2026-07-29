from nd_phase_unwrap import io, unwrap
from pathlib import Path
import os 
import numpy as np


def demo():

    # IGC
    ## REAL DATA
    config = {
        'algorithm': 'IterativeGraphCuts',
        'pixel_spacing': (2.0, 2.0, 2.0, 40.0),
        'cyclic': (False, False, False, True),
        'data_cost': 0.001,
        'neighbourhood_radius': 1.0,
        'neighbourhood_weight': 10.0**3,
    }
    data_path = Path(__file__).parent.parent / 'test_data'
    in_path = data_path / 'realdata'
    out_path = data_path / 'IGC_unwrapped'
    if not out_path.is_dir():
        out_path.mkdir(parents=True)

    for file in os.listdir(in_path):
            print (f'Processing file {file}...')
            data_file = in_path / f'{file}'
            rawdata = io.load_numpy_data(data_file)
            basefilename = Path(file).stem
            saverdata = np.zeros(rawdata.shape, dtype=np.float32)
            print(f'Unwrapping {file}...')
            for vencdir in range(rawdata.shape[-1]):
                phaseArray = unwrap.unwrap(rawdata[...,vencdir], config)
                print(f'Unwrapped {file} with shape {phaseArray.shape} and dtype {phaseArray.dtype}')         
                saverdata[...,vencdir] = phaseArray
            io.save(saverdata, out_path / f'{basefilename}_unwrapped.npy')

    ## SYNTH DATA
    config = {
        'algorithm': 'IterativeGraphCuts',
        'pixel_spacing': (2.0, 2.0, 2.0, 20.0),
        'cyclic': (False, False, False, True),
        'data_cost': 0.001,
        'neighbourhood_radius': 1.0,
        'neighbourhood_weight': 10.0**3,
        'mask': None
    }
    data_path = Path(__file__).parent.parent / 'test_data'
    in_path = data_path / 'Synth_4Dflow'
    mask_path = data_path / 'Synth_4Dflow_mask_dilated'
    out_path = data_path / 'IGC_unwrapped'
    if not out_path.is_dir():
        out_path.mkdir(parents=True)

    for file in [j for j in os.listdir(in_path) if j.endswith('s.npy') or j.endswith('R.npy')]:
            print (f'Processing file {file}...')
            data_file = in_path / f'{file}'
            rawdata = io.load_numpy_data(data_file)
            basefilename = Path(file).stem
            maskdata = io.load_numpy_data(mask_path / f'{basefilename[:5]}_mask_dilated.npy')
            saverdata = np.zeros(rawdata.shape, dtype=np.float32)
            config['mask'] = maskdata[...,0]
            print(f'Unwrapping {file}...')
            for vencdir in range(rawdata.shape[-1]):
                phaseArray = unwrap.unwrap(rawdata[...,vencdir], config)
                print(f'Unwrapped {file} with shape {phaseArray.shape} and dtype {phaseArray.dtype}')         
                saverdata[...,vencdir] = phaseArray
            io.save(saverdata, out_path / f'{basefilename}_unwrapped.npy')

    # Laplace
    config = {
        'algorithm': 'Laplace'
    }
    data_path = Path(__file__).parent.parent / 'test_data'
    in_path = [data_path / 'Synth_4Dflow', data_path / 'realdata']
    out_path = data_path / 'Laplace_unwrapped'
    if not out_path.is_dir():
        out_path.mkdir(parents=True)
        
    for folder in in_path:
        for file in os.listdir(folder) if not file.endswith('.py') else None:
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
    in_path = [data_path / 'Synth_4Dflow', data_path / 'realdata']
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