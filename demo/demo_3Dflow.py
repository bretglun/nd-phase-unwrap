from nd_phase_unwrap import io, unwrap
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def demo():
    data_path = Path(__file__).parent.parent / 'test_data' / '2+1D_phantom'
    data_file = data_path / '1wrapsVNR1e3_0.npy'
    wrapped = io.load_numpy_data(data_file)
    config = {
        'algorithm': 'IterativeGraphCuts',
        'period': 2 * np.pi,
        'pixel_spacing': (2.0, 2.0),
        'cyclic': (False, False)
    }
    unwrapped = unwrap.unwrap(wrapped, config)

    fig, axes = plt.subplots(2, 1, figsize=(5, 6))
    for i, (img, title) in enumerate(((wrapped, 'Wrapped'), (unwrapped, 'Unwrapped'))):
        axes[i].imshow(img)
        axes[i].set_title(title)
    plt.show()


if __name__ == '__main__':
    demo()