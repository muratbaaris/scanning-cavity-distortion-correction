"""Tests for reading and writing calibrations and images.

The core property under test is a round-trip: a calibration or an image
written by this module and read back must be indistinguishable from the
original.  A round-trip failure would silently corrupt every subsequent
analysis, so the guarantees here are the ones users rely on most.
"""

import numpy as np
import pytest
import scipy.io as sio

from scdc.calibration import Calibration
from scdc.io import (
    CalibrationFileError,
    load_calibration,
    load_image,
    save_calibration,
    save_image,
)
from scdc.polynomial import number_of_terms, polynomial_term_labels
from scdc.synthetic import make_checkerboard


COEFFICIENT_TOLERANCE = 1e-12
"""Bound on the round-trip error of a coefficient.

Saving and loading is a plain copy through double precision, so anything
above this signals data being reshaped or cast along the way rather than
numerical noise.
"""


def _make_calibration(degree=2):
    """Build a small hand-crafted Calibration for isolated I/O tests.

    The fields carry distinct values so a swap or a truncation during the
    round-trip surfaces as a mismatch on the exact field that broke.
    """
    n_terms = number_of_terms(degree)
    return Calibration(
        degree=degree,
        coefficients_x=np.linspace(-1.0, 1.0, n_terms),
        coefficients_y=np.linspace(2.0, 3.0, n_terms),
        lattice_spacing_px=12.5,
        input_shape=(160, 180),
        output_shape=(150, 170),
        crop_offset=(5, 7),
        n_detected=42,
        n_used=39,
        mean_residual_px=0.123,
        max_residual_px=0.456,
        term_labels=polynomial_term_labels(degree),
    )


# ---------- calibration round-trip ----------


def test_save_calibration_returns_the_written_path(tmp_path):
    """The return value lets users chain a save with a subsequent operation.

    The docstring in `save_calibration` promises this, so a change of return
    value would silently break user code that relies on it.
    """
    path = str(tmp_path / "cal.mat")
    returned = save_calibration(_make_calibration(), path)
    assert returned == path


def test_calibration_round_trip_preserves_scalar_fields(tmp_path):
    """Every scalar recorded in the file must come back with the same value.

    Scalars go through MATLAB's ``.mat`` container as 1x1 arrays, so a bug
    in the flattening on read would turn them into arrays rather than
    numbers and break every downstream user.
    """
    original = _make_calibration()
    path = str(tmp_path / "cal.mat")
    save_calibration(original, path)
    loaded = load_calibration(path)

    assert loaded.degree == original.degree
    assert loaded.lattice_spacing_px == pytest.approx(original.lattice_spacing_px)
    assert loaded.n_detected == original.n_detected
    assert loaded.n_used == original.n_used
    assert loaded.mean_residual_px == pytest.approx(original.mean_residual_px)
    assert loaded.max_residual_px == pytest.approx(original.max_residual_px)


def test_calibration_round_trip_preserves_coefficients_exactly(tmp_path):
    """The polynomial coefficients are the model; any drift changes the map.

    A coefficient shift below one ULP is acceptable, anything larger means
    the file format lost precision and every corrected image would be off.
    """
    original = _make_calibration(degree=3)
    path = str(tmp_path / "cal.mat")
    save_calibration(original, path)
    loaded = load_calibration(path)

    assert np.allclose(loaded.coefficients_x, original.coefficients_x,
                       atol=COEFFICIENT_TOLERANCE)
    assert np.allclose(loaded.coefficients_y, original.coefficients_y,
                       atol=COEFFICIENT_TOLERANCE)


def test_calibration_round_trip_preserves_shape_tuples(tmp_path):
    """Shapes and offsets must come back as plain tuples of Python ints.

    `apply_calibration` and the plotting helpers use them for indexing and
    array allocation, both of which reject numpy scalars or arrays in some
    versions.  Restoring them as tuples of ints keeps the public interface
    identical before and after saving.
    """
    original = _make_calibration()
    path = str(tmp_path / "cal.mat")
    save_calibration(original, path)
    loaded = load_calibration(path)

    assert loaded.input_shape == original.input_shape
    assert loaded.output_shape == original.output_shape
    assert loaded.crop_offset == original.crop_offset
    assert all(isinstance(v, int) for v in loaded.input_shape)
    assert all(isinstance(v, int) for v in loaded.crop_offset)


def test_calibration_round_trip_regenerates_term_labels(tmp_path):
    """Term labels are derived from the degree, not stored, so must be rebuilt.

    A user inspecting the coefficients after loading expects the labels to
    line up with them exactly as they did before saving.
    """
    original = _make_calibration(degree=3)
    path = str(tmp_path / "cal.mat")
    save_calibration(original, path)
    loaded = load_calibration(path)

    assert loaded.term_labels == polynomial_term_labels(original.degree)
    assert len(loaded.term_labels) == len(loaded.coefficients_x)


def test_calibration_round_trip_from_a_real_fit(tmp_path):
    """A calibration produced by the real pipeline must survive a round-trip.

    The hand-crafted calibration in the other tests exercises the I/O
    format; this one checks that nothing in the fitting path produces a
    calibration the file format cannot represent.
    """
    from scdc.calibration import fit_calibration

    image = make_checkerboard(shape=(160, 160), cell_size=12.0, blur_sigma=1.0)
    original = fit_calibration(image, degree=2)

    path = str(tmp_path / "cal.mat")
    save_calibration(original, path)
    loaded = load_calibration(path)

    assert loaded.degree == original.degree
    assert np.allclose(loaded.coefficients_x, original.coefficients_x,
                       atol=COEFFICIENT_TOLERANCE)
    assert np.allclose(loaded.coefficients_y, original.coefficients_y,
                       atol=COEFFICIENT_TOLERANCE)
    assert loaded.input_shape == original.input_shape
    assert loaded.output_shape == original.output_shape

# ---------- calibration I/O errors ----------


def test_save_calibration_rejects_a_missing_directory(tmp_path):
    """Writing to a directory that does not exist must fail explicitly.

    Otherwise the underlying MATLAB writer raises an opaque OSError that
    does not tell the user which path was tried.
    """
    path = str(tmp_path / "nowhere" / "cal.mat")
    with pytest.raises(CalibrationFileError, match="does not exist"):
        save_calibration(_make_calibration(), path)


def test_load_calibration_rejects_a_missing_file(tmp_path):
    """A missing file must fail with a dedicated error, not a generic one.

    Users automating batch runs need to distinguish "file not there" from
    "file corrupt" to decide whether to retry or to abort.
    """
    with pytest.raises(CalibrationFileError, match="no calibration file"):
        load_calibration(str(tmp_path / "does_not_exist.mat"))


def test_load_calibration_rejects_an_unreadable_file(tmp_path):
    """A file that is not a valid .mat must be reported as unreadable.

    Wrapping the low-level scipy error gives the user a message that names
    the offending path, which the raw exception does not.
    """
    bad = tmp_path / "not_a_mat.mat"
    bad.write_bytes(b"this is not a MATLAB file")
    with pytest.raises(CalibrationFileError, match="could not be read"):
        load_calibration(str(bad))


def test_load_calibration_rejects_a_file_missing_required_fields(tmp_path):
    """Absent required fields must be reported by name so the user can fix them.

    A silent default would produce a "successful" load of a nonsense
    calibration and corrupt every image it was applied to.
    """
    path = str(tmp_path / "incomplete.mat")
    sio.savemat(path, {"degree": 2, "coefficients_x": np.zeros(6)})
    with pytest.raises(CalibrationFileError, match="missing the required field"):
        load_calibration(path)


def test_load_calibration_rejects_a_coefficient_count_inconsistent_with_degree(tmp_path):
    """A degree/coefficient mismatch must be caught before the model is used.

    Evaluating a polynomial with the wrong number of coefficients does not
    raise on its own, it just returns garbage; catching it at load time is
    the only place the file itself gives the game away.
    """
    path = str(tmp_path / "wrong_count.mat")
    sio.savemat(path, {
        "degree": 3,
        "coefficients_x": np.zeros(5),
        "coefficients_y": np.zeros(5),
        "lattice_spacing_px": 10.0,
        "input_shape": np.array([100, 100], dtype=np.int64),
        "output_shape": np.array([90, 90], dtype=np.int64),
        "crop_offset": np.array([5, 5], dtype=np.int64),
    })
    with pytest.raises(CalibrationFileError, match="degree 3"):
        load_calibration(path)