import numpy as np
from scipy import signal
from nd_phase_unwrap import io
from pathlib import Path


def get_Gaussian_phantom(size=64, std=10):
    kernel = signal.windows.gaussian(size, std=std)
    r = np.outer(kernel, kernel)
    arr = np.concatenate((r, -r)).T
    return arr * np.pi

def get_Gaussian_phantom3D(sizexy=32, sizez=32, timepoints=32, stdxy=5, stdz=5):
    kernelxy = signal.windows.gaussian(sizexy, std=stdxy)
    kernelz = signal.windows.gaussian(sizez, std=stdz)
    # kernelz = np.ones(sizez)
    phase = np.einsum('i,j,k->ijk', kernelxy, kernelxy, kernelz)

    phase_timeres = np.zeros(phase.shape + (timepoints,), dtype=complex)
    for t in range(timepoints):
        phase_timeres[:, :, :, t] = phase * (1 - np.abs((2*t / timepoints) - 1))

    cylindermask = np.zeros((sizexy, sizexy))
    for i in range(sizexy):
        for j in range(sizexy):
            if np.sqrt((i - sizexy/2)**2 + (j - sizexy/2)**2) < sizexy/2:
                cylindermask[i, j] = 1

    arr = np.zeros_like(phase_timeres, dtype=complex)
    arr1 = cylindermask[:, :, np.newaxis, np.newaxis] * np.exp(1j * phase_timeres * np.pi)
    arr2 = cylindermask[:, :, np.newaxis, np.newaxis] * np.exp(-1j * phase_timeres * np.pi)
    arr = np.concatenate((arr1, arr2), axis=1)

    print(arr.shape)
    return arr


def wrap(arr, factor):
    return np.abs(arr) * np.exp(1j * (np.angle(arr) * factor))


def add_noise(img, VNR):    
    std = 1 / VNR
    return img +  np.pi * np.random.normal(0, std, img.shape)


path = Path(__file__).parent

truth = get_Gaussian_phantom3D(sizexy=32, sizez=64, timepoints=32, stdxy=10, stdz=10) # ground truth
io.save(truth, path / 'truth.npy')

for num_wraps in range(0,7,2):
    factor = 1 + 2 * num_wraps
    wrapped = wrap(truth, factor)
    if num_wraps != 0:
        io.save(wrapped, path / f'{num_wraps}wraps.npy')
    
    for noise_power in [1.0, 2.0, 3.0]:
        VNR = 10**noise_power
        noisy = add_noise(wrapped, VNR)
        io.save(noisy, path / f'{num_wraps}wrapsVNR1e{str(noise_power).replace(".", "_")}.npy')