# Scanning Cavity Microscope Distortion Correction

A Python library and command-line tool that measures and corrects the
geometric distortion of a raster-scanning microscope from an image of a
periodic checkerboard calibration target.

## The problem

A raster-scanning microscope moves the sample under a fixed optical mode
using piezo actuators. Two things spoil the geometry of the recorded image:
the two scan axes are not perfectly orthogonal, and the piezo response is
not perfectly linear. As a result, distances and angles in the image do not
faithfully represent the physical sample. For quantitative measurements such as
extracting a diffusion coefficient from a spreading spot, for instance
this distortion biases the result.

This package corrects the distortion by imaging a checkerboard mirror with
squares of known pitch, measuring where the bright squares actually sit,
comparing that to where they *should* sit on a perfectly regular grid, and
fitting a bivariate polynomial that maps the ideal positions to the measured
ones. The polynomial captures the distortion; inverting it corrects any
subsequent image acquired with the same scan settings.

## Installation

Requires Python 3.9 or later.

```bash
git clone https://github.com/muratbaaris/scanning-cavity-distortion-correction
cd scanning-cavity-distortion-correction
pip install -r requirements.txt
```

To run the test suite:

```bash
python -m pytest tests/
```

## Quick start

The package ships with a synthetic checkerboard, so you can verify the
installation without any experimental data:

```bash
python -m scdc demo --output-dir results
```

This builds a synthetic target, distorts it by a known shear and anisotropic
scaling, fits a calibration, applies it, and reports the lattice angle and
basis length ratio before and after the correction. The corrected image and
the fitted calibration are written to `results/`.

## Command-line interface

Three subcommands cover the whole workflow.

```bash
# Measure a calibration from an image of the target.
python -m scdc fit target.mat --output calibration.mat --degree 3

# Correct any later image acquired with the same scan settings.
python -m scdc apply sample.mat --calibration calibration.mat \
                                --output corrected.mat

# Verify the installation on a synthetic target.
python -m scdc demo --output-dir results
```

Use `python -m scdc --help` and `python -m scdc <command> --help` for the
full list of options.

## Using the library

```python
import numpy as np
from scdc import fit_calibration, apply_calibration
from scdc.io import load_image, save_image

# Fit the calibration once, on the target image
target = load_image("target.mat")
calibration = fit_calibration(target, degree=3)
print(calibration.summary())

# Apply the same calibration to every subsequent sample image
sample = load_image("sample.mat")
corrected = apply_calibration(sample, calibration)
save_image(corrected, "corrected.npy")
```

The `Calibration` object stores the polynomial coefficients and the metadata
needed to reproduce the same output geometry on later images. It can be
written to a `.mat` file with `scdc.io.save_calibration` and read back with
`scdc.io.load_calibration`.

## Repository layout

```
scdc/                the library
├── centroids.py     detection of the bright squares of the target
├── lattice.py       basis vectors and integer lattice labels
├── polynomial.py    bivariate polynomial model of the distortion
├── geometry.py      largest-inscribed-rectangle crop for warped images
├── synthetic.py     synthetic checkerboards for tests and examples
├── calibration.py   fitting and applying a full calibration
├── io.py            reading and writing calibrations and images
├── plotting.py      visualisation helpers
├── cli.py           command line entry point
└── __main__.py      allows "python -m scdc"

tests/               pytest test suite, one file per library module
docs/                explanatory documentation
examples/            worked example notebook
```

## Documentation

- [`docs/algorithm.md`](docs/algorithm.md) — a step by step explanation of
  the seven stages of the pipeline, with the reasoning behind each of them.
- Every public function carries a numpy-style docstring; use `help(function)`
  or the built-in Python help browser.
- [`examples/demo.ipynb`](examples/demo.ipynb) — a runnable notebook that
  walks through a calibration on a synthetic target.

## Testing

The test suite is built entirely on synthetic checkerboards with a known,
analytically specified distortion. Because the correct answer is known in
advance, the tests can verify that the pipeline recovers it rather than
merely that it runs. No experimental data is required to run them.

```bash
python -m pytest tests/                # run everything
python -m pytest tests/ -v             # verbose
python -m pytest tests/test_lattice.py # a single module
```

There are more than sixty tests distributed across the modules, one file per
library module.

## License

MIT — see [LICENSE](LICENSE).
