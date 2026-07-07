from nd_phase_unwrap import io, unwrap
from pathlib import Path
import os 
import numpy as np

def demo():
    data_path = Path(__file__).parent.parent / 'test_data'
    in_path = data_path / 'Synth_4Dflow'
    mask_path = data_path / 'Synth_4Dflow_mask_dilated'
    out_path = data_path / 'Synth_4Dflow_unwrapped'
    for i in os.listdir(out_path):
        splitname = i.split('_')
        basename = splitname[0]
        wraps = splitname[1]
        if wraps[0] in ['0', '1', '2', '3']:
            wrapfactor = int(wraps[0]) * 2 + 1
        else:
            wrapfactor = 1
        outarray = io.load_numpy_data(out_path / i) / wrapfactor
        mask = io.load_numpy_data(mask_path / f'{basename}_mask_dilated.npy')

        groundtruth = np.angle(io.load_numpy_data(in_path / f'{basename}.npy'))

        difference = np.abs(outarray - groundtruth)
        error = np.sum(difference > 1)
        masksize = np.sum(mask)

        print(f'{error} voxels inside the mask ({100 * error / masksize:.2f}%) for {i}')

if __name__ == '__main__':
    demo()