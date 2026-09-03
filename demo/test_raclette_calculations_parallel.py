"""Unwrap and evaluate the wrapped RACLETTE 4D flow datasets in a single, low-space pass.

Unlike ``demo_4Dflow_batch_raclette*.py`` + ``raclette_eval.py`` (which write every unwrapped
array to disk and evaluate them in a second pass), this script never persists an unwrapped
array: each wrapped file is loaded, unwrapped, scored against ground truth, and discarded
again, all inside one worker. Only the resulting metrics are written out, as a single CSV.

Subjects are processed in parallel: each worker owns one subject end-to-end (load, wrap,
unwrap, score) and pushes every scored row onto a shared queue as soon as it is computed.
The parent drains that queue and is the only process that touches the CSV, so the file can
be tailed while the run is still going.

Unwrapping (for algorithms that accept a mask: IGC, ROMEO) is restricted to the aorta mask
dilated by 2 iterations, matching ``test_data/Synth_4Dflow/wrap_raclette.py``. Correctness is
scored against the tight (non-dilated) mask, since the dilated halo is background and would
trivially inflate the correct-voxel fraction.
"""

import csv
import json
import multiprocessing
import queue
from concurrent.futures import ProcessPoolExecutor
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

NUM_WORKERS = 15   # one subject per worker

MASK_DILATION_ITERATIONS = 2
# no dilation across the venc-direction axis, matching wrap_raclette.py
MASK_DILATION_STRUCTURE = np.ones((3, 3, 3, 3, 1))

WRAP_VNR_PAIRS = [(wraps, vnr) for wraps in range(1,6) for vnr in [i for i in np.linspace(0.0, 3.0, 7)]]

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
        'romeo_path': '/home/kjain/Desktop/server_unwrap/romeo_native/bin/romeo',
    },
}
USES_MASK = {'IGC'}   # Laplace has no mask parameter


TWO_PI = 2.0 * np.pi

CSV_COLUMNS = ['rawdata_name', 'VNR exponent', 'wraps', 'algorithm',
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
    rng = np.random.default_rng()
    noise = rng.normal(0, noise_std, size=arr.shape)
    noisy_phase = np.angle(arr) + noise
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
    """Append one row and close the file, so the CSV is complete on disk after every result."""
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT_CSV.exists()
    with open(OUT_CSV, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(CSV_COLUMNS)
        writer.writerow(row)

def eval_unwrap(unwrapped, expected, mask_tight, subject, algo, n_wraps, vnr):
    """Count correctly unwrapped voxels inside the tight mask and return one CSV row.
    """
    residual = unwrapped.astype(np.float64) - expected

    n_mask = int(mask_tight.sum())
    n_correct = int(np.count_nonzero(np.abs(residual[mask_tight]) < np.pi))
    return [subject, vnr, n_wraps, algo, int(unwrapped.size), n_mask, n_correct, n_mask - n_correct]

def eval_gt(unwrapped, expected, mask_tight, subject, algo, n_wraps, vnr):
    residual = unwrapped.astype(np.float64) - expected
    n_mask = int(mask_tight.sum())
    n_correct = int(np.count_nonzero(np.abs(residual[mask_tight]) < 0.01))
    return [subject, vnr, n_wraps, algo, int(unwrapped.size), n_mask, n_correct, n_mask - n_correct]

def process_subject(subject_path, row_queue):
    """Unwrap and score every (wraps, VNR, algorithm) combination of one subject.

    Runs in a worker process and puts each row on ``row_queue`` the moment it is scored;
    nothing is written to disk here, so the parent stays the only writer of the CSV.
    """
    rawTruth = io.load_numpy_data(subject_path)
    subjectID = subject_path.stem[-3:]
    mask_dil, mask_tight = get_mask(rawTruth)

    # preparing data for dispatch
    for wrap_vnr_pair in WRAP_VNR_PAIRS:
        n_wraps, vnr = wrap_vnr_pair
        factor = 1 + 2 * n_wraps                      # matching wrap_raclette.py
        expected = np.angle(rawTruth) * factor
        prepared_array = wrap(rawTruth, factor)


        prepared_array = add_noise_to_phase(prepared_array, np.pi/(10**vnr))
        pixel_spacing = pixel_spacing_for(subjectID)
        # prepare configs
        config = {name: dict(cfg) for name, cfg in BASE_CONFIGS.items()}
        config['IGC']['mask'] = mask_dil
        config['IGC']['pixel_spacing'] = pixel_spacing
        config['ROMEO']['pixel_spacing'] = pixel_spacing

        for algo, algoconfig in config.items():
            print(f'Subject {subjectID}, {algo}, VNR expo {vnr}, {n_wraps} wraps, mask% {np.mean(mask_dil) * 100}')
            unwrapped = ndunwrap(prepared_array, algoconfig)
            row_queue.put(eval_unwrap(unwrapped, expected, mask_tight, subjectID, algo, n_wraps, vnr))

    for n_wraps in range(6):
        factor = 1 + 2 * n_wraps                      # matching wrap_raclette.py
        expected = np.angle(rawTruth) * factor
        prepared_array = wrap(rawTruth, factor)
        row_queue.put(eval_gt(np.angle(prepared_array)/factor, np.angle(rawTruth), mask_tight, subjectID, 'unprocessed', n_wraps, 'no noise'))


def main():
    subject_paths = sorted(GT_DIR.glob('*.npy'))
    with multiprocessing.Manager() as manager:
        row_queue = manager.Queue()
        with ProcessPoolExecutor(max_workers=NUM_WORKERS, max_tasks_per_child=1) as executor:
            futures = [executor.submit(process_subject, path, row_queue) for path in subject_paths]

            # drain until every worker is finished and nothing is left in flight
            while not all(future.done() for future in futures) or not row_queue.empty():
                try:
                    write_row(row_queue.get(timeout=1.0))
                except queue.Empty:
                    continue

            for future in futures:
                future.result()   # surface any worker exception



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
