from nd_phase_unwrap import unwrap, io
from pathlib import Path
import typer


def cli(npy_file: Path, config_file: Path | None = None):
    arr = io.load_numpy_data(npy_file)
    result = unwrap.unwrap(arr, config_file=config_file)
    io.save(result, npy_file.parent / f'{npy_file.stem}_unwrapped.npy')


def main():
    typer.run(cli)


if __name__ == '__main__':
    main()
