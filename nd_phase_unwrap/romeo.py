"""Phase unwrapping using the ROMEO tool.

ROMEO (Rapid Opensource Minimum spanning tree algOrithm,
https://github.com/korbinian90/ROMEO) is an external command-line program that
operates on NIfTI files. This module wraps the ROMEO executable: it writes the
phase (and optionally the magnitude and a mask) to temporary NIfTI files, runs
ROMEO, and reads the unwrapped phase back into a numpy array.

The ROMEO executable is not bundled with this package. Obtain it as a
standalone binary (find it at https://github.com/korbinian90/ROMEO). Point this module at the executable via the
``romeo_path`` config keyword.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from nd_phase_unwrap import io


def _build_command(executable, phase_file, output_file, magnitude_file, mask_file, config):
    cmd = [executable, '--phase', str(phase_file), '--output', str(output_file)]
    if magnitude_file is not None:
        cmd += ['--magnitude', str(magnitude_file)]
    if mask_file is not None:
        cmd += ['--mask', str(mask_file)]
    elif config.mask_strategy:
        cmd += ['--mask', config.mask_strategy]
    if config.individual:
        cmd += ['--individual-unwrapping']
    if config.echo_times is not None:
        cmd += ['--echo-times', str(config.echo_times)]
    cmd += list(config.extra_args)
    return cmd


def unwrap(arr, config):
    if np.iscomplexobj(arr):
        phase = np.angle(arr)
        magnitude = np.abs(arr)
    else:
        phase = arr
        magnitude = None

    executable = config.romeo_path

    with tempfile.TemporaryDirectory(prefix='romeo_', ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        phase_file = tmp / 'phase.nii'
        output_file = tmp / 'unwrapped.nii'
        io.save_to_nifti(phase.astype(np.float32), phase_file, voxel_size=config.pixel_spacing)

        magnitude_file = None
        if magnitude is not None and config.use_magnitude:
            magnitude_file = tmp / 'magnitude.nii'
            io.save_to_nifti(magnitude.astype(np.float32), magnitude_file, voxel_size=config.pixel_spacing)

        mask_file = None
        if config.mask is not None:
            mask_file = tmp / 'mask.nii'
            io.save_to_nifti(np.asarray(config.mask).astype(np.uint8), mask_file, voxel_size=config.pixel_spacing)

        cmd = _build_command(executable, phase_file, output_file, magnitude_file, mask_file, config)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if 'AssertionError: Unwrap-weights are all zero!' in result.stderr:
                print('ROMEO failed due to all-zero weights; returning original phase.')
                return phase  # ROMEO failed due to all-zero weights; return original phase
            raise RuntimeError(
                f'ROMEO failed (exit code {result.returncode})'
                f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n'
                f'for command: {" ".join(cmd)}'
            )


        if not output_file.is_file():
            raise FileNotFoundError(
                f'ROMEO did not produce an output file at "{output_file}". '
                f'ROMEO stdout:\n{result.stdout}'
            )

        unwrapped = io.load_nifti(output_file)

    return unwrapped.reshape(phase.shape).astype(phase.dtype)