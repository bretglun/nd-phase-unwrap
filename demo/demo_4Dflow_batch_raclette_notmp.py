from nd_phase_unwrap import io, unwrap
from pathlib import Path
import os 
import numpy as np
import json


def demo():

    # IGC
    data_path = Path(__file__).parent.parent / 'test_data'
    in_path = data_path / 'Synth_4Dflow_wrapped'
    meta_path = data_path / 'metadata'
    mask_path = data_path / 'Synth_4Dflow_mask'
    out_path = data_path / 'IGC_unwrapped'
    if not out_path.is_dir():
        out_path.mkdir(parents=True)

    for file in [Path(j) for j in os.listdir(in_path) if j.endswith('s.npy') or j.endswith('R.npy')]:
        basefilename = Path(file).stem
        subjectID = file.name[2:5]
        with open(meta_path / f"{subjectID}.json", "r") as jsonfile: 
            metadata = json.load(jsonfile)
        res = metadata["acquisition"]["spatial_resolution_mm"]
        res.append(metadata["acquisition"]["temporal_resolution_ms"])
        res = [round(i,1) for i in res]
        config = {
            'algorithm': 'IterativeGraphCuts',
            'pixel_spacing': (res[0], res[1], res[2], res[3]),
            'cyclic': (False, False, False, True),
            'data_cost': False,
            'neighbourhood_radius': 1.0,
            'neighbourhood_weight': 10.0**2.5,
            'mask': None
        }
        print (f'Processing file {file}...')
        data_file = in_path / f'{file}'
        rawdata = io.load_numpy_data(data_file)
        maskdata = io.load_numpy_data(mask_path / f'{basefilename[:5]}_mask_dilated.npy')
        saverdata = np.zeros(rawdata.shape, dtype=np.float32)
        config['mask'] = maskdata[...,0]
        print(f'Unwrapping {file}...')
        for vencdir in range(rawdata.shape[-1]):
            phaseArray = unwrap.unwrap(rawdata[...,vencdir], config)
            # print(f'Unwrapped {file} with shape {phaseArray.shape} and dtype {phaseArray.dtype}')         
            saverdata[...,vencdir] = phaseArray
        io.save(saverdata, out_path / f'{basefilename}_unwrapped.npy')

    # Laplace
    config = {
        'algorithm': 'Laplace'
    }
    data_path = Path(__file__).parent.parent / 'test_data'
    in_path = [data_path / 'Synth_4Dflow_wrapped']
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
    data_path = Path(__file__).parent.parent / 'test_data'
    in_path = [data_path / 'Synth_4Dflow_wrapped']
    meta_path = data_path / 'metadata'
    out_path = data_path / 'ROMEO_unwrapped'
    if not out_path.is_dir():
        out_path.mkdir(parents=True)
        
    for folder in in_path:
        for file in [i for i in os.listdir(folder) if not i.endswith('.py')]:
            subjectID = file[2:5]
            out_filepath = out_path / f'{Path(file).stem}_unwrapped.npy'
            if out_filepath.is_file():
                continue
            with open(meta_path / f"{subjectID}.json", "r") as jsonfile: 
                metadata = json.load(jsonfile)
            res = metadata["acquisition"]["spatial_resolution_mm"]
            res.append(metadata["acquisition"]["temporal_resolution_ms"])
            res = [round(i,1) for i in res]
            config = {
                'algorithm': 'ROMEO',
                'pixel_spacing': (res[0], res[1], res[2], res[3]),
                'cyclic': (False, False, False, True),
                'individual': False,      # timepoints in the 4th dim -> unwrap each spatially
                'use_magnitude': True,  # test data phase is stored with unit magnitude
                'echo_times': 'epi',
                'mask': None,
                'romeo_path': 'C:\\Users\\jar048\\Desktop\\Coding\\ROMEO_native\\bin\\romeo.exe',  # uncomment if ROMEO is not on PATH
            }
            print (f'Processing file {folder/file}...')
            data_file = folder / f'{file}'
            rawdata = io.load_numpy_data(data_file)
            saverdata = np.zeros(rawdata.shape, dtype=np.float32)
            print(f'Unwrapping {file}...')
            for vencdir in range(rawdata.shape[-1]):
                phaseArray = unwrap.unwrap(rawdata[...,vencdir], config)        
                saverdata[...,vencdir] = phaseArray
            io.save(saverdata, out_filepath)

if __name__ == '__main__':
    demo()