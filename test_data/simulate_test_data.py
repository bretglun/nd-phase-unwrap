import numpy as np
from scipy import signal
from nd_phase_unwrap import io
from pathlib import Path


def get_Gaussian_phantom(size=64, std=10):
    kernel = signal.windows.gaussian(size, std=std)
    r = np.outer(kernel, kernel)
    arr = np.concatenate((r, -r)).T
    return arr * np.pi


def wrap(arr, factor):
    return np.angle(np.exp(1j * arr * factor)) / factor


def add_noise(img, VNR):    
    std = 1 / VNR
    return img +  np.pi * np.random.normal(0, std, img.shape)


path = Path(__file__).parent

truth = get_Gaussian_phantom(size=32, std=5) # ground truth
io.save(truth, path / 'truth.npy')

for num_wraps in range(8):
    factor = 1 + 2 * num_wraps
    wrapped = wrap(truth, factor)
    if num_wraps != 0:
        io.save(wrapped * factor, path / f'{num_wraps}wraps.npy')
    
    for noise_power in [1.0, 1.5, 2.0, 2.5, 3.0]:
        VNR = 10**noise_power
        noisy = add_noise(wrapped, VNR)
        io.save(noisy * factor, path / f'{num_wraps}wrapsVNR1e{str(noise_power).replace(".", "_")}.npy')