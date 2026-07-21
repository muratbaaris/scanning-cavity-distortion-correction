"""Tests for the detection of bright squares in a calibration image."""

import numpy as np
import pytest

from scdc.centroids import NoCentroidsFoundError, detect_centroids
from scdc.synthetic import ideal_lattice_points, make_checkerboard

POSITION_TOLERANCE_PX = 0.5
"""How far a detected centre may sit from the true one before we call it wrong.

Half a pixel is the accuracy a centre-of-mass estimate should reach on a
blurred, well-sampled square; a larger error would mean the blob mask is
lopsided.
"""


def test_detect_centroids_returns_x_y_pairs():
    """The rest of the package works in (x, y), so the shape must be (N, 2).

    Returning (row, column) instead would silently transpose every later
    geometric calculation, so the convention is pinned down by a test.
    """
    image = make_checkerboard(shape=(120, 120), cell_size=12.0)
    centroids = detect_centroids(image)
    assert centroids.ndim == 2
    assert centroids.shape[1] == 2


def test_detect_centroids_finds_every_interior_square():
    """All squares fully inside the field of view should be detected.

    Missing squares would thin out the lattice and could break the
    region-growing labelling, so the count is checked against the known
    geometry of the synthetic target.
    """
    shape, cell = (120, 120), 12.0
    image = make_checkerboard(shape=shape, cell_size=cell)
    all_centres = ideal_lattice_points(shape, cell)

    half_square = 0.5 * cell * 0.5
    is_interior = (
        (all_centres[:, 0] - half_square > 0)
        & (all_centres[:, 1] - half_square > 0)
        & (all_centres[:, 0] + half_square < shape[1] - 1)
        & (all_centres[:, 1] + half_square < shape[0] - 1))

    detected = detect_centroids(image)
    assert len(detected) == int(is_interior.sum())


def test_detect_centroids_locates_squares_to_sub_pixel_accuracy():
    """A detected centre must coincide with the square it belongs to.

    The whole calibration rests on these positions, so an accuracy check is
    more informative than merely counting the blobs.
    """
    shape, cell = (150, 150), 15.0
    image = make_checkerboard(shape=shape, cell_size=cell, blur_sigma=1.0)
    detected = detect_centroids(image)
    expected = ideal_lattice_points(shape, cell)

    for point in detected:
        distance_to_nearest = np.min(
            np.linalg.norm(expected - point, axis=1))
        assert distance_to_nearest < POSITION_TOLERANCE_PX


def test_detect_centroids_discards_squares_cut_by_the_border():
    """Truncated squares have a biased centre and must not reach the fit.

    A square clipped by the field of view loses part of its area, which
    displaces its centre of mass towards the image interior by an amount
    that has nothing to do with the distortion.  The image below holds one
    interior blob and one blob running off the left edge, so the two modes
    differ by exactly one detection.
    """
    image = np.zeros((40, 40))
    image[10:16, 10:16] = 1.0   # fully inside the field of view
    image[25:31, 0:6] = 1.0     # cut by the left border

    kept = detect_centroids(image, drop_border_blobs=True)
    all_blobs = detect_centroids(image, drop_border_blobs=False)

    assert len(kept) == 1
    assert len(all_blobs) == 2


def test_detect_centroids_ignores_blobs_below_the_area_threshold():
    """Small bright specks are noise and must not be mistaken for squares.

    The image holds one square of 36 pixels and one speck of 4 pixels, so a
    threshold placed between the two areas must keep exactly one of them.
    """
    image = np.zeros((40, 40))
    image[10:16, 10:16] = 1.0   # 36 pixels
    image[30:32, 30:32] = 1.0   # 4 pixels

    assert len(detect_centroids(image, min_blob_area=1)) == 2
    assert len(detect_centroids(image, min_blob_area=10)) == 1


def test_detect_centroids_treats_nan_as_background():
    """Warped images carry NaN outside their valid area and must stay usable.

    Re-detecting centroids in a corrected image is how the correction is
    verified, so NaN must not propagate into the threshold computation.
    """
    image = make_checkerboard(shape=(120, 120), cell_size=12.0)
    with_nan = image.copy()
    with_nan[0:5, :] = np.nan

    centroids = detect_centroids(with_nan)
    assert np.all(np.isfinite(centroids))


def test_detect_centroids_rejects_a_three_dimensional_array():
    """A colour image or a stack is a user mistake worth reporting clearly."""
    volume = np.zeros((5, 20, 20))
    with pytest.raises(ValueError, match="two-dimensional"):
        detect_centroids(volume)


@pytest.mark.parametrize("bad_fraction", [-0.1, 0.0, 1.0, 1.5])
def test_detect_centroids_rejects_thresholds_outside_the_unit_interval(
        bad_fraction):
    """A threshold outside (0, 1) selects everything or nothing.

    Both cases produce a confusing downstream failure, so they are caught at
    the point where the invalid value is supplied.
    """
    image = make_checkerboard(shape=(60, 60), cell_size=10.0)
    with pytest.raises(ValueError, match="threshold_fraction"):
        detect_centroids(image, threshold_fraction=bad_fraction)


def test_detect_centroids_rejects_a_non_positive_minimum_area():
    """A blob cannot contain fewer than one pixel."""
    image = make_checkerboard(shape=(60, 60), cell_size=10.0)
    with pytest.raises(ValueError, match="min_blob_area"):
        detect_centroids(image, min_blob_area=0)


def test_detect_centroids_reports_a_uniform_image_explicitly():
    """A blank image is a common acquisition failure and deserves a clear error.

    Without this check the threshold would sit exactly at the single
    intensity present and the failure would surface much later.
    """
    uniform = np.full((50, 50), 0.7)
    with pytest.raises(NoCentroidsFoundError, match="uniform"):
        detect_centroids(uniform)


def test_detect_centroids_reports_an_image_without_any_target():
    """Pure noise contains no squares, and saying so beats returning junk."""
    generator = np.random.default_rng(0)
    noise = generator.normal(0.0, 1.0, size=(40, 40))
    with pytest.raises(NoCentroidsFoundError):
        detect_centroids(noise, min_blob_area=200)
