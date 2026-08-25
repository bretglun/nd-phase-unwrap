"""Unwrap and evaluate the wrapped RACLETTE 4D flow datasets in a single, low-space pass.

Unlike ``demo_4Dflow_batch_raclette*.py`` + ``raclette_eval.py`` (which write every unwrapped
array to disk and evaluate them in a second pass), this script never persists an unwrapped
array: each wrapped file is loaded, unwrapped, scored against ground truth, and discarded
again, all inside one worker. Only the resulting metrics are written out, as a single CSV.

Unwrapping (for algorithms that accept a mask: IGC, ROMEO) is restricted to the aorta mask
dilated by 2 iterations, matching ``test_data/Synth_4Dflow/wrap_raclette.py``. Correctness is
scored against the tight (non-dilated) mask, since the dilated halo is background and would
trivially inflate the correct-voxel fraction.
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from nd_phase_unwrap import io, unwrap

# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent.parent / 'test_data'
GT_DIR = DATA_PATH / 'Synth_4Dflow'
META_DIR = DATA_PATH / 'metadata'
OUT_CSV = DATA_PATH / 'raclette_eval' / 'raclette_eval_streaming.csv'

MASK_DILATION_ITERATIONS = 2
# no dilation across the venc-direction axis, matching wrap_raclette.py
MASK_DILATION_STRUCTURE = np.ones((3, 3, 3, 3, 1))

WRAP_VNR_PAIRS = [(wraps, vnr) for wraps in range(6) for vnr in [i for i in np.linspace(1.0, 5.0, 9)]]

BASE_CONFIGS = {
    'IGC': {
        'algorithm': 'IterativeGraphCuts',
        'cyclic': (False, False, False, True),
        'data_cost': False,
        'neighbourhood_radius': 1.0,
        'neighbourhood_weight': 10.0 ** 2.0,
        'mask': None
    },
    'Laplace': {
        'algorithm': 'Laplace',
    },
    'ROMEO': {
        'algorithm': 'ROMEO',
        'cyclic': (False, False, False, True),
        'individual': False,      # timepoints in the 4th dim -> unwrap each spatially
        'use_magnitude': True,    # test data phase is stored with unit magnitude
        'echo_times': 'epi',
        'romeo_path': 'C:\\Users\\jar048\\Desktop\\Coding\\ROMEO_native\\bin\\romeo.exe',
    },
}
USES_MASK = {'IGC'}   # Laplace has no mask parameter

NUM_WORKERS = 2   # None = os.cpu_count()

TWO_PI = 2.0 * np.pi

CSV_COLUMNS = ['rawdata_name', 'VNR', 'wraps', 'algorithm',
               'total_size', 'mask_size', 'n_correct', 'n_incorrect']

def get_mask(arr):
    mask_tight = np.angle(arr) != 0
    mask_dilated = binary_dilation(mask_tight, structure=MASK_DILATION_STRUCTURE, iterations=MASK_DILATION_ITERATIONS)
    # mask_dilated = mask_dilated.any(axis=-1)
    return mask_dilated, mask_tight

def wrap(arr, factor):
    if np.iscomplexobj(arr):
        return np.abs(arr) * np.exp(1j * np.angle(arr) * factor)
    return np.angle(np.exp(1j * arr * factor))

def add_noise_to_phase(arr, noise_std):
    """Add Gaussian noise to the phase of a complex array."""
    if not np.iscomplexobj(arr):
        raise ValueError('Input array must be complex.')
    phase = np.angle(arr)
    noise = np.random.normal(0, noise_std, size=phase.shape)
    noisy_phase = phase + noise
    return np.abs(arr) * np.exp(1j * noisy_phase)


def pixel_spacing_for(subject):

    def load_metadata(subject):
        with open(META_DIR / f'{subject}.json', 'r') as jsonfile:
            return json.load(jsonfile)
        
    metadata = load_metadata(subject)
    res = metadata['acquisition']['spatial_resolution_mm'] + [metadata['acquisition']['temporal_resolution_ms']]

    return tuple(round(r, 1) for r in res)

def ndunwrap(array, config):
    rawdata = array
    saverdata = np.zeros(rawdata.shape, dtype=np.float32)
    for vencdir in range(rawdata.shape[-1]):
        direction_config = dict(config)
        if 'mask' in config:
            direction_config['mask'] = config['mask'][..., vencdir]
        saverdata[...,vencdir] = unwrap.unwrap(rawdata[...,vencdir], direction_config)
    return saverdata

def write_row(row):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT_CSV.exists()
    with open(OUT_CSV, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(CSV_COLUMNS)
        writer.writerow(row)

def eval_unwrap(unwrapped, expected, mask_tight, subject, algo, n_wraps, vnr):
    """Count correctly unwrapped voxels inside the tight mask and append one CSV row.

    Each venc direction is unwrapped up to a global 2*pi offset, so the dominant offset is
    removed per direction before a voxel is called correct (residual below one wrap).
    """
    residual = unwrapped.astype(np.float64) - expected
    for vencdir in range(residual.shape[-1]):
        inside = residual[..., vencdir][mask_tight[..., vencdir]]
        if inside.size:
            offsets, counts = np.unique(np.round(inside / TWO_PI), return_counts=True)
            residual[..., vencdir] -= TWO_PI * offsets[counts.argmax()]

    n_mask = int(mask_tight.sum())
    n_correct = int(np.count_nonzero(np.abs(residual[mask_tight]) < np.pi))
    write_row([subject, vnr, n_wraps, algo, int(unwrapped.size), n_mask, n_correct, n_mask - n_correct])

def main():
    for subject_path in [GT_DIR/file for file in os.listdir(GT_DIR)]:
        rawTruth = io.load_numpy_data(subject_path)
        subjectID = subject_path.stem[-3:]
        mask_dil, mask_tight = get_mask(rawTruth)

        # preparing data for dispatch
        for wrap_vnr_pair in WRAP_VNR_PAIRS:
            n_wraps, vnr = wrap_vnr_pair
            factor = 1 + 2 * n_wraps                      # matching wrap_raclette.py
            expected = np.angle(rawTruth) * factor
            prepared_array = wrap(rawTruth, factor)
            prepared_array = add_noise_to_phase(prepared_array, 1/(10**vnr))
            pixel_spacing = pixel_spacing_for(subjectID)
            # prepare configs
            config = dict(BASE_CONFIGS)
            config['IGC']['mask'] = mask_dil
            config['IGC']['pixel_spacing'] = pixel_spacing
            config['ROMEO']['pixel_spacing'] = pixel_spacing

            for algo, algoconfig in config.items():
                unwrapped = ndunwrap(prepared_array, algoconfig)
                eval_unwrap(unwrapped, expected, mask_tight, subjectID, algo, n_wraps, vnr)



# Recepie for this file
    # for loop for going through every file in synth data folder
    #   load in noise free rawdata as truth array
    #   Create mask (dilated, tight)

        # For loop for wrapping and noising
        #   Note vnr and wrapcount
        #   wrap data accordingly
        #   load metadata to get voxel sizes
        #   Set metadata into config (resolution for ROMEO and IGC, and dilated_mask for IGC)

            # For loop for algo
            #   Unwrap data(array, config)
            #   Func call Eval_data (arr, algo, tightmask, vnr, wraps)
            #       Note rawdata_name, VNR, wraps, algo, total_size, size of mask, n correct voxels, n incorrect voxels
            #       Save these to CSV, omit the unwrapped array for memory saving and continue
    pass


if __name__ == '__main__':
    main()
