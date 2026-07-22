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


# ---------- image I/O ----------


def test_save_image_returns_the_written_path(tmp_path):
    """The return value supports chaining, mirroring save_calibration."""
    path = str(tmp_path / "img.npy")
    returned = save_image(np.zeros((4, 4)), path)
    assert returned == path


def test_image_round_trip_through_npy_preserves_values(tmp_path):
    """A ``.npy`` round-trip is native NumPy and must be exact.

    Any drift would mean the routine is doing something other than a plain
    copy, which is not what users expect from a ``.npy`` file.
    """
    original = np.random.default_rng(0).standard_normal((32, 48))
    path = str(tmp_path / "img.npy")
    save_image(original, path)
    loaded = load_image(path)
    assert loaded.shape == original.shape
    assert np.allclose(loaded, original)


def test_image_round_trip_through_mat_preserves_values(tmp_path):
    """A ``.mat`` round-trip goes through MATLAB's format and must still match.

    Double precision is preserved by ``.mat`` files, so the tolerance is
    machine precision, not something laxer.
    """
    original = np.random.default_rng(1).standard_normal((24, 36))
    path = str(tmp_path / "img.mat")
    save_image(original, path)
    loaded = load_image(path)
    assert loaded.shape == original.shape
    assert np.allclose(loaded, original)


def test_save_image_rejects_an_unsupported_extension(tmp_path):
    """Only ``.mat`` and ``.npy`` are supported; anything else must fail loudly.

    Silently writing a file with a wrong extension would leave the user
    with an image they cannot load back.
    """
    path = str(tmp_path / "img.png")
    with pytest.raises(CalibrationFileError, match="unsupported image format"):
        save_image(np.zeros((4, 4)), path)


def test_save_image_rejects_a_missing_directory(tmp_path):
    """A missing directory must fail with a dedicated error, as for calibrations."""
    path = str(tmp_path / "nowhere" / "img.npy")
    with pytest.raises(CalibrationFileError, match="does not exist"):
        save_image(np.zeros((4, 4)), path)


def test_load_image_rejects_an_unsupported_extension(tmp_path):
    """The extension check happens before the file is opened, catching typos early."""
    bad = tmp_path / "img.tiff"
    bad.write_bytes(b"placeholder")
    with pytest.raises(CalibrationFileError, match="unsupported image format"):
        load_image(str(bad))


def test_load_image_rejects_a_missing_file(tmp_path):
    """A missing image file must be distinguishable from a missing calibration."""
    with pytest.raises(CalibrationFileError, match="no image file"):
        load_image(str(tmp_path / "gone.npy"))


def test_load_image_rejects_a_non_two_dimensional_array(tmp_path):
    """Downstream code assumes an ``(H, W)`` image; a 3-D array must be caught here.

    A rejected file gives the user a clear message; passing a 3-D array
    through would surface later as an obscure error deep in the pipeline.
    """
    path = str(tmp_path / "cube.npy")
    np.save(path, np.zeros((3, 4, 5)))
    with pytest.raises(CalibrationFileError, match="two-dimensional"):
        load_image(path)


def test_load_image_auto_selects_the_only_two_dimensional_array_from_mat(tmp_path):
    """Users who only have one image in a .mat file should not need a key.

    Requiring one would force them to remember an internal name they never
    chose, especially for files produced by the microscope software.
    """
    path = str(tmp_path / "single.mat")
    image = np.arange(48.0).reshape(6, 8)
    sio.savemat(path, {"microscope_image": image})
    loaded = load_image(path)
    assert np.allclose(loaded, image)


def test_load_image_picks_the_named_array_from_a_multi_array_mat(tmp_path):
    """When several 2-D arrays are present, the user selects one by name.

    This is the escape hatch for the common lab situation of a .mat file
    holding raw, background, and target arrays side by side.
    """
    path = str(tmp_path / "multi.mat")
    target = np.ones((5, 7))
    background = np.zeros((5, 7))
    sio.savemat(path, {"target": target, "background": background})
    loaded = load_image(path, key="target")
    assert np.allclose(loaded, target)


def test_load_image_rejects_a_multi_array_mat_without_a_key(tmp_path):
    """Ambiguity must not be resolved silently, or the wrong array could be used.

    The error message lists the available names so the user knows what to
    pass on the next call.
    """
    path = str(tmp_path / "multi.mat")
    sio.savemat(path, {"a": np.zeros((4, 4)), "b": np.ones((4, 4))})
    with pytest.raises(CalibrationFileError, match="several"):
        load_image(path)


def test_load_image_rejects_an_unknown_key(tmp_path):
    """Asking for a variable that isn't there must fail with a helpful message.

    The error names the variables that are present so the user can correct
    the call without opening the file in another program.
    """
    path = str(tmp_path / "multi.mat")
    sio.savemat(path, {"a": np.zeros((4, 4)), "b": np.ones((4, 4))})
    with pytest.raises(CalibrationFileError, match="no two-dimensional array"):
        load_image(path, key="missing")


def test_load_image_rejects_a_mat_with_no_two_dimensional_arrays(tmp_path):
    """A .mat file holding only higher-dimensional arrays is not a valid image source.

    A stack of images (a 3-D array) is a common thing to save from the
    microscope by mistake; the loader must refuse it rather than pick an
    arbitrary slice.
    """
    path = str(tmp_path / "no_images.mat")
    sio.savemat(path, {"stack": np.zeros((3, 4, 5))})
    with pytest.raises(CalibrationFileError, match="no two-dimensional array"):
        load_image(path)
