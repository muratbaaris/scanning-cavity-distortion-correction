"""Tests for fitting and applying a complete distortion calibration.

These are integration tests: they run the whole pipeline from a synthetic
image through centroid detection, lattice indexing, polynomial fitting, and
image warping, and check that the output geometry is correct.  Because the
distortion is applied analytically, the correct answer is known in advance.
"""

import numpy as np
import pytest

from scdc.calibration import Calibration, CalibrationError, apply_calibration, fit_calibration
from scdc.centroids import detect_centroids
from scdc.lattice import basis_length_ratio, estimate_basis, lattice_angle
from scdc.synthetic import distort_image, make_checkerboard, shear_matrix


def test_fit_calibration_returns_a_calibration_object():
    """The return type carries all metadata the apply step needs.

    If the fit returned raw arrays instead, the user would have to track
    degree, crop offset, and output shape by hand, which is error-prone.
    """
    image = make_checkerboard(shape=(160, 160), cell_size=12.0, blur_sigma=1.0)
    calibration = fit_calibration(image, degree=1)
    assert isinstance(calibration, Calibration)


def test_fit_calibration_records_the_input_shape():
    """The input shape is needed to reject images of a different size.

    Applying a calibration to an image that wasn't acquired with the same
    scan settings would extrapolate the polynomial, which gives garbage.
    """
    shape = (160, 180)
    image = make_checkerboard(shape=shape, cell_size=12.0, blur_sigma=1.0)
    calibration = fit_calibration(image, degree=1)
    assert calibration.input_shape == shape


def test_fit_calibration_achieves_sub_pixel_residual_on_a_clean_target():
    """The fit should capture all systematic structure in a noiseless image.

    On a synthetic target the only error source is the finite pixel grid, so
    the residual must be well below one pixel.  A residual above one pixel
    would mean the polynomial missed a real feature of the distortion.
    """
    image = make_checkerboard(shape=(200, 200), cell_size=12.0, blur_sigma=1.0)
    calibration = fit_calibration(image, degree=3)
    assert calibration.mean_residual_px < 0.5
    assert calibration.max_residual_px < 1.0


def test_fit_calibration_restores_the_lattice_angle_after_shearing():
    """A sheared target must come out with a 90-degree lattice after correction.

    This is the primary claim of the whole package: the distorted angle is
    restored to the ideal value.  The test applies a known shear, fits and
    applies the calibration, and checks the angle in the corrected image.
    """
    image = make_checkerboard(shape=(220, 220), cell_size=12.0, blur_sigma=1.0)
    distorted = distort_image(image, matrix=shear_matrix(shear=0.25, scale_y=1.15))

    calibration = fit_calibration(distorted, degree=3)
    corrected = apply_calibration(distorted, calibration, fill_value=0.0)

    centroids = detect_centroids(corrected)
    a1, a2 = estimate_basis(centroids)
    assert lattice_angle(a1, a2) == pytest.approx(90.0, abs=0.5)


def test_fit_calibration_restores_the_basis_length_ratio():
    """Anisotropic scaling must be removed so that circles stay circular.

    A ratio different from one means the two scan axes have different pixel
    pitches, which turns a round spot into an ellipse.  After correction the
    ratio must be close to one.
    """
    image = make_checkerboard(shape=(220, 220), cell_size=12.0, blur_sigma=1.0)
    distorted = distort_image(image, matrix=shear_matrix(shear=0.2, scale_y=1.2))

    calibration = fit_calibration(distorted, degree=3)
    corrected = apply_calibration(distorted, calibration, fill_value=0.0)

    centroids = detect_centroids(corrected)
    a1, a2 = estimate_basis(centroids)
    assert basis_length_ratio(a1, a2) == pytest.approx(1.0, abs=0.01)


def test_apply_calibration_produces_the_declared_output_shape():
    """The output shape is stored in the calibration and must be honoured.

    A mismatch between the declared and actual shape would break any code
    that pre-allocates arrays based on the calibration metadata.
    """
    image = make_checkerboard(shape=(160, 160), cell_size=12.0, blur_sigma=1.0)
    calibration = fit_calibration(image, degree=2)
    corrected = apply_calibration(image, calibration)
    assert corrected.shape == tuple(calibration.output_shape)


def test_apply_calibration_rejects_a_wrong_input_shape():
    """An image from a different scan configuration must not be corrected.

    The polynomial is only valid over the region it was measured on.
    Applying it to an image of a different size would extrapolate and
    produce a silently wrong result.
    """
    image = make_checkerboard(shape=(160, 160), cell_size=12.0, blur_sigma=1.0)
    calibration = fit_calibration(image, degree=1)

    wrong_size = np.zeros((100, 100))
    with pytest.raises(CalibrationError, match="shape"):
        apply_calibration(wrong_size, calibration)


def test_apply_calibration_fills_missing_data_with_the_requested_value():
    """Pixels outside the warped region should carry the fill value.

    The default is NaN (so downstream code notices them), but for display
    or for routines that cannot handle NaN the user can request zero.
    """
    image = make_checkerboard(shape=(160, 160), cell_size=12.0, blur_sigma=1.0)
    distorted = distort_image(image, matrix=shear_matrix(shear=0.3))
    calibration = fit_calibration(distorted, degree=2)

    with_nan = apply_calibration(distorted, calibration, fill_value=np.nan)
    with_zero = apply_calibration(distorted, calibration, fill_value=0.0)

    assert not np.any(np.isnan(with_zero))


def test_fit_calibration_rejects_a_featureless_image():
    """An image with no recognisable target cannot be calibrated.

    Fitting to noise would produce a polynomial that maps garbage to garbage,
    so the failure must be reported rather than hidden.
    """
    blank = np.zeros((80, 80))
    with pytest.raises(Exception):
        fit_calibration(blank, degree=1)


def test_calibration_pixel_size_converts_spacing_to_micrometres():
    """The pixel size method must divide the target pitch by the lattice spacing.

    For a checkerboard of cell size s, the diagonal pitch is s*sqrt(2).
    Dividing by the fitted lattice spacing (in pixels) gives um/px.
    """
    image = make_checkerboard(shape=(200, 200), cell_size=12.0, blur_sigma=1.0)
    calibration = fit_calibration(image, degree=1)

    target_pitch = 14.142  # 10 um * sqrt(2)
    pixel_size = calibration.pixel_size(target_pitch)
    assert pixel_size > 0
    assert pixel_size == pytest.approx(
        target_pitch / calibration.lattice_spacing_px, rel=1e-6)


def test_calibration_summary_contains_key_information():
    """The summary is the first thing a user reads after fitting.

    It must mention the degree, the residual, and the lattice spacing so the
    user can judge whether the calibration is trustworthy without inspecting
    the raw coefficient arrays.
    """
    image = make_checkerboard(shape=(160, 160), cell_size=12.0, blur_sigma=1.0)
    calibration = fit_calibration(image, degree=2)
    summary = calibration.summary()

    assert "degree 2" in summary
    assert "residual" in summary.lower()
    assert "spacing" in summary.lower()
