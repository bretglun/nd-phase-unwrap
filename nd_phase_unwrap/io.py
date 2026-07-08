from pathlib import Path
import numpy as np
import yaml


def read_config_file(config_file):
    if not config_file:
        return {}
    config_file = Path(config_file)

    if not config_file.is_absolute():
        config_file = Path(__file__).parent.parent / config_file
    
    if not config_file.exists():
        raise FileNotFoundError(f'Config file not found: {config_file}')

    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f'Invalid YAML in config file {f}: {e}') from e


def load_numpy_data(data_file):
    if not data_file:
        raise ValueError('No data file specified')
    
    if not data_file.is_file():
        raise FileNotFoundError(f'Could not find data file "{data_file}"')
    
    if data_file.suffix != '.npy':
        raise ValueError(f'Data file must be in .npy format, not "{data_file}"')
    
    return np.load(data_file)


def save(arr, filepath):
    filepath = Path(filepath)
    outdir = filepath.parent
    outdir.mkdir(parents=True, exist_ok=True)
    if filepath.suffix != '.npy':
        raise ValueError(f'Must save as .npy file, not "{filepath}"')
    print(f'Writing array to "{filepath}"')
    np.save(filepath, arr)


def _optionalImport_nibabel():
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError(
            'The "nibabel" package is required for NIfTI conversions. '
            'Install it with "pip install nibabel".'
        ) from e
    return nib


def save_to_nifti(arr, filepath, voxel_size=None):
    nib = _optionalImport_nibabel()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.suffix not in ('.nii', '.gz'):
        raise ValueError(f'Must save as .nii or .nii.gz file, not "{filepath}"')
    affine = np.fill_diagonal(np.zeros((4, 4)), voxel_size if voxel_size is not None else 1)
    img = nib.Nifti1Image(np.asarray(arr), affine)
    print(f'Writing NIfTI array to "{filepath}"')
    nib.save(img, str(filepath))


def load_nifti(filepath):
    nib = _optionalImport_nibabel()
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f'Could not find NIfTI file "{filepath}"')
    img = nib.load(str(filepath))
    return np.asarray(img.dataobj)