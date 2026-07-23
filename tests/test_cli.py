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

# ---------- apply ----------

@pytest.fixture
def fitted_calibration(tmp_path):
    """A target image and a calibration fitted to it through the CLI itself,
    shared by the ``apply`` tests."""
    target = tmp_path / "target.npy"
    _write_distorted_target(target)
    calibration = tmp_path / "calibration.mat"
    assert main(["fit", str(target), "--output", str(calibration)]) == 0
    return target, calibration


def test_apply_writes_the_corrected_image(fitted_calibration, tmp_path):
    """``apply`` promises a corrected image at the requested path with the
    output geometry recorded in the calibration."""
    target, calibration = fitted_calibration
    output = tmp_path / "corrected.npy"

    exit_code = main(["apply", str(target),
                      "--calibration", str(calibration),
                      "--output", str(output)])

    assert exit_code == 0
    corrected = load_image(str(output))
    expected_shape = tuple(load_calibration(str(calibration)).output_shape)
    assert corrected.shape == expected_shape


def test_apply_straightens_the_lattice(fitted_calibration, tmp_path):
    """Correcting the very image the calibration was fitted on must restore
    the 90 degree lattice angle; this verifies the CLI wires the pipeline
    together correctly, not merely that files appear."""
    target, calibration = fitted_calibration
    output = tmp_path / "corrected.npy"

    main(["apply", str(target), "--calibration", str(calibration),
          "--output", str(output)])

    corrected = load_image(str(output))
    angle = lattice_angle(*estimate_basis(detect_centroids(corrected)))
    assert abs(angle - 90.0) < 0.5


def test_apply_forwards_the_fill_value(fitted_calibration, tmp_path):
    """Pixels without data must receive the value chosen on the command
    line, because a silently different value would bias any later statistics
    on the corrected image.  The output crop removes the empty corners, so a
    region of missing data is created inside the image instead by punching a
    NaN hole into the input."""
    target, calibration = fitted_calibration
    holed = np.array(load_image(str(target)))
    holed[90:110, 90:110] = np.nan
    holed_path = tmp_path / "holed.npy"
    save_image(holed, str(holed_path))
    output = tmp_path / "corrected.npy"

    main(["apply", str(holed_path), "--calibration", str(calibration),
          "--output", str(output), "--fill", "-1.0"])

    corrected = load_image(str(output))
    assert np.any(corrected == -1.0) and not np.any(np.isnan(corrected))


def test_apply_with_a_missing_calibration_fails_with_status_two(
        tmp_path, capsys):
    """Pointing ``apply`` at a calibration that does not exist must fail
    with the documented status and an explanatory message."""
    target = tmp_path / "target.npy"
    _write_distorted_target(target)

    exit_code = main(["apply", str(target),
                      "--calibration", str(tmp_path / "nowhere.mat"),
                      "--output", str(tmp_path / "out.npy")])

    assert exit_code == 2
    assert "error:" in capsys.readouterr().err


def test_apply_with_a_wrong_shaped_image_fails_with_status_two(
        fitted_calibration, tmp_path, capsys):
    """A calibration is only valid for the scan geometry it was measured on,
    so applying it to an image of a different shape must be refused."""
    _, calibration = fitted_calibration
    wrong = tmp_path / "wrong.npy"
    save_image(np.zeros((50, 60)), str(wrong))

    exit_code = main(["apply", str(wrong),
                      "--calibration", str(calibration),
                      "--output", str(tmp_path / "out.npy")])

    assert exit_code == 2
    assert "error:" in capsys.readouterr().err


# ---------- demo ----------

def test_demo_writes_the_three_promised_files(tmp_path):
    """The README tells the user which files the demo produces; all three
    must exist afterwards, in the directory the user asked for."""
    exit_code = main(["demo", "--output-dir", str(tmp_path)])

    assert exit_code == 0
    for name in ("distorted.npy", "corrected.npy", "calibration.mat"):
        assert (tmp_path / name).is_file()


def test_demo_creates_a_missing_output_directory(tmp_path):
    """The demo is the very first command a new user runs, so it should
    create the output directory rather than fail on a fresh checkout."""
    output_dir = tmp_path / "not" / "yet" / "there"

    exit_code = main(["demo", "--output-dir", str(output_dir)])

    assert exit_code == 0
    assert (output_dir / "corrected.npy").is_file()


def test_demo_reports_a_corrected_lattice_angle(tmp_path, capsys):
    """The before/after table is the evidence the demo offers that the
    installation works; the corrected image it writes must indeed have a
    near-right lattice angle."""
    main(["demo", "--output-dir", str(tmp_path)])

    corrected = load_image(str(tmp_path / "corrected.npy"))
    angle = lattice_angle(*estimate_basis(detect_centroids(corrected)))
    assert abs(angle - 90.0) < 0.5


def test_demo_is_deterministic_for_a_fixed_seed(tmp_path):
    """Replicability is a core promise of the package: two runs with the same
    seed must produce bit-identical corrected images."""
    first, second = tmp_path / "first", tmp_path / "second"

    main(["demo", "--output-dir", str(first), "--seed", "7"])
    main(["demo", "--output-dir", str(second), "--seed", "7"])

    np.testing.assert_array_equal(
        load_image(str(first / "corrected.npy")),
        load_image(str(second / "corrected.npy")))
