from nd_phase_unwrap import io, unwrap
from pathlib import Path
import os 

CONFIG = {
    'algorithm': 'IterativeGraphCuts',
    'pixel_spacing': (2.0, 2.0, 2.0, 10.0),
    'cyclic': (False, False, False, True),
    'data_cost': False,
}

def demo():
    data_path = Path(__file__).parent.parent / 'test_data'
    in_path = data_path / '3Dphantom'
    out_path = data_path / '3Dphantom_results'
    if not out_path.is_dir():
        out_path.mkdir(parents=True)

    for file in ['truth.npy'] + [j for j in os.listdir(in_path) if j.endswith('wraps.npy')]:
        print (f'Processing file {file}...')
        data_file = in_path / file
        rawdata = io.load_numpy_data(data_file)    
        phaseArray = unwrap.unwrap(rawdata, CONFIG)
        print(f'Unwrapped {file} with shape {phaseArray.shape} and dtype {phaseArray.dtype}')
        out_file = out_path / f'{file.replace(".npy", "_unwrapped.npy")}'
        io.save(phaseArray, out_file)

if __name__ == '__main__':
    demo()