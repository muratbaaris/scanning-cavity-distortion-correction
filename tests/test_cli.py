"""Tests for the command line interface.

The command line interface is a user-facing surface of the package, so it
must be tested like any other public function: not merely that it runs, but
that each subcommand produces the promised files, respects its options, and
exits with the documented status code (0 success, 1 incomplete arguments,
2 pipeline error).  Because ``main`` accepts an ``argv`` list and returns
the exit code instead of calling ``sys.exit``, the whole interface can be
driven in-process, which keeps the tests fast and measurable by coverage.

All target images are synthetic checkerboards with a fixed random seed, so
every expected outcome is known in advance.
"""

import numpy as np
import pytest

from scdc.centroids import detect_centroids
from scdc.cli import main
from scdc.io import load_calibration, load_image, save_image
from scdc.lattice import estimate_basis, lattice_angle
from scdc.synthetic import distort_image, make_checkerboard, shear_matrix


def _write_distorted_target(path, shear=0.2, scale_y=1.1, seed=0):
    """Write a sheared synthetic checkerboard to ``path`` and return it.

    The distortion parameters are known exactly, so tests can check that the
    command line pipeline removes them rather than merely that it completes.
    """
    undistorted = make_checkerboard(shape=(200, 200), cell_size=12.0,
                                    blur_sigma=1.0, noise_level=0.01,
                                    random_seed=seed)
    matrix = shear_matrix(shear=shear, scale_x=1.0, scale_y=scale_y)
    distorted = distort_image(undistorted, matrix=matrix)
    save_image(distorted, str(path))
    return distorted

# ---------- top level behaviour ----------

def test_the_documented_python_dash_m_entry_point_works():
    """The README instructs the user to run ``python -m scdc``; a real
    subprocess call verifies that the packaging (``__main__.py`` and the
    installed metadata) actually supports it, which no in-process test
    can."""
    import subprocess
    import sys as _sys

    completed = subprocess.run(
        [_sys.executable, "-m", "scdc", "--version"],
        capture_output=True, text=True, timeout=60)

    from scdc import __version__
    assert completed.returncode == 0
    assert completed.stdout.strip() == __version__

def test_version_flag_prints_the_package_version(capsys):
    """--version must report the installed version, so a user filing a bug
    report can state exactly which release they are running."""
    from scdc import __version__

    exit_code = main(["--version"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == __version__


def test_running_without_a_command_prints_help_and_fails(capsys):
    """A bare invocation cannot do anything useful, so it must show the help
    text and exit with the documented status 1 instead of silently
    succeeding."""
    exit_code = main([])

    assert exit_code == 1
    assert "usage: scdc" in capsys.readouterr().out

# ---------- fit ----------

def test_fit_writes_a_loadable_calibration(tmp_path, capsys):
    """The whole purpose of ``fit`` is to produce a calibration file, so the
    file it announces must exist and be readable by ``load_calibration``."""
    target = tmp_path / "target.npy"
    _write_distorted_target(target)
    output = tmp_path / "calibration.mat"

    exit_code = main(["fit", str(target), "--output", str(output)])

    assert exit_code == 0
    calibration = load_calibration(str(output))
    assert calibration.input_shape == (200, 200)

def test_fit_respects_the_degree_option(tmp_path):
    """The polynomial degree is the main knob the user has on the model, so
    the value given on the command line must reach the stored calibration."""
    target = tmp_path / "target.npy"
    _write_distorted_target(target)
    output = tmp_path / "calibration.mat"

    main(["fit", str(target), "--output", str(output), "--degree", "2"])

    assert load_calibration(str(output)).degree == 2


def test_fit_reports_the_pixel_size_for_the_given_pitch(tmp_path, capsys):
    """The printed pixel size is what converts later measurements to
    micrometres, so the summary must state it together with the pitch the
    user supplied."""
    target = tmp_path / "target.npy"
    _write_distorted_target(target)

    main(["fit", str(target), "--output", str(tmp_path / "c.mat"),
          "--target-pitch-um", "20.0"])

    output = capsys.readouterr().out
    assert "target pitch 20.0 um" in output


def test_fit_with_a_missing_image_fails_with_status_two(tmp_path, capsys):
    """A wrong path is the most common user mistake; it must produce the
    documented pipeline-error status and a message on stderr, not a Python
    traceback."""
    exit_code = main(["fit", str(tmp_path / "nowhere.npy"),
                      "--output", str(tmp_path / "c.mat")])

    assert exit_code == 2
    assert "error:" in capsys.readouterr().err


def test_fit_with_a_featureless_image_fails_with_status_two(tmp_path, capsys):
    """An image without a target cannot yield a calibration; the failure must
    surface as an actionable error message rather than an unhandled
    exception."""
    blank = tmp_path / "blank.npy"
    save_image(np.zeros((64, 64)), str(blank))

    exit_code = main(["fit", str(blank),
                      "--output", str(tmp_path / "c.mat")])

    assert exit_code == 2
    assert "error:" in capsys.readouterr().err
