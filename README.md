# Scanning Cavity Microscope Distortion Correction
![tests](https://github.com/muratbaaris/scanning-cavity-distortion-correction/actions/workflows/tests.yml/badge.svg)

A Python library, command-line tool, and graphical interface that measures
and corrects the geometric distortion of a raster-scanning microscope from
an image of a periodic checkerboard calibration target.

## The problem

A raster-scanning microscope moves the sample under a fixed optical mode
using piezo actuators. Two things spoil the geometry of the recorded image:
the two scan axes are not perfectly orthogonal, and the piezo response is
not perfectly linear. As a result, distances and angles in the image do not
faithfully represent the physical sample. For quantitative measurements,
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
git clone https://github.com/muratbaaris/scanning-cavity-distortion-correction.git
cd scanning-cavity-distortion-correction
pip install -e .
```

The GUI additionally requires Tkinter, which is included with most Python
distributions. On Linux you may need to install it separately:

```bash
sudo apt install python3-tk    # Ubuntu / Debian
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

Expected output:

```
Synthetic demonstration
====================================================
  applied shear       : 0.25
  applied y scaling   : 1.15

  quantity                    before       after     ideal
  ------------------------------------------------------
  lattice angle [deg]          96.46       90.06     90.00
  basis length ratio          0.7822      0.9996    1.0000

  fit residual        : 0.128 px mean, 0.267 px max
  squares used        : 151 of 151
```

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

## Graphical interface

For interactive use, a Tkinter GUI allows the user to load a calibration
image (or generate a synthetic one), select a region of interest by
dragging, fit the calibration, inspect the polynomial coefficients, and
export the corrected image, all without writing any code.

```bash
python -m scdc.gui
```

The GUI delegates all computation to the `scdc` library and contains no
algorithm code itself. If you don't have a checkerboard image handy, click
the "Generate synthetic…" button to build a distorted target you can
practice on.

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
├── gui.py           Tkinter graphical interface
└── __main__.py      allows "python -m scdc"

tests/               pytest test suite, one file per library module
docs/                explanatory documentation
```

## Documentation

- [`docs/algorithm.md`](docs/algorithm.md): a step by step explanation of
  the seven stages of the pipeline, with the reasoning behind each of them.
- Every public function carries a numpy-style docstring; use `help(function)`
  or the built-in Python help browser.

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

There are 114 tests distributed across seven test files, one per library
module.

## License

GPL v3 - see [LICENSE](LICENSE).

## Generative AI Statement

In accordance with the principles of honesty, transparency, and accountability stated in the University of Bologna policy for an ethical and responsible use of generative artificial intelligence (GenAI), in particular the use of generative artificial intelligence in “Use cases of generative artificial intelligence in preparing student work to be assessed” such as the “generating or explaining a programming code”, the author declares that Claude Opus 4.7 was used to help in drafting the clean algorithm documentation, proofreading README documentation, and refactoring the monolithic analysis script where active supervision and critical thinking were employed to account for the generated results, thus complying with the university policy principles. All generated code and text were reviewed, tested, and revised by the author, and the underlying algorithm design was previously developed for the author's own lab work. The author takes full responsibility for the correctness of the final code.
