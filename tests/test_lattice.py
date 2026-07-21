"""Tests for the measurement of the lattice geometry.

The synthetic target has a known basis and the distortion applied to it is a
known matrix, so both the undistorted and the distorted basis can be
predicted analytically and compared against the measurement.
"""

import numpy as np
import pytest

from scdc.centroids import detect_centroids
from scdc.lattice import (
    LatticeIndexingError,
    assign_lattice_indices,
    basis_length_ratio,
    estimate_basis,
    lattice_angle,
)
from scdc.synthetic import distort_image, make_checkerboard, shear_matrix

BASIS_TOLERANCE_PX = 0.1
"""Agreement required between a measured and an analytically known basis.

A tenth of a pixel is well below the centroid noise of a real image but is
achievable on a noiseless synthetic target, so a violation indicates a
genuine error in the clustering rather than ordinary measurement scatter.
"""

ANGLE_TOLERANCE_DEG = 0.5
"""Agreement required on the lattice angle, in degrees."""


def make_lattice_points(cell_size=12.0, n_per_side=7):
    """Return a clean square lattice of points for tests that need no image."""
    coordinates = np.arange(n_per_side) * cell_size
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def test_estimate_basis_recovers_the_spacing_of_an_undistorted_target():
    """On a perfect checkerboard the two steps are the cell diagonals.

    Bright squares occupy every second cell, so a nearest neighbour lies one
    cell across and one cell down; both basis vectors must therefore have
    length ``cell_size * sqrt(2)``.
    """
    cell = 12.0
    image = make_checkerboard(shape=(200, 200), cell_size=cell, blur_sigma=1.0)
    a1, a2 = estimate_basis(detect_centroids(image))

    expected = cell * np.sqrt(2.0)
    assert np.linalg.norm(a1) == pytest.approx(expected, abs=BASIS_TOLERANCE_PX)
    assert np.linalg.norm(a2) == pytest.approx(expected, abs=BASIS_TOLERANCE_PX)


def test_estimate_basis_gives_perpendicular_vectors_without_distortion():
    """An undistorted square lattice must measure exactly 90 degrees.

    This is the reference against which the distorted case is judged, so it
    has to hold tightly before any correction claim can be made.
    """
    image = make_checkerboard(shape=(200, 200), cell_size=12.0, blur_sigma=1.0)
    a1, a2 = estimate_basis(detect_centroids(image))
    assert lattice_angle(a1, a2) == pytest.approx(90.0, abs=ANGLE_TOLERANCE_DEG)


def test_estimate_basis_recovers_a_known_linear_distortion():
    """The measured basis must equal the analytically transformed one.

    ``distort_image`` samples the input at ``matrix @ q``, so a feature at
    ``p`` appears at ``inv(matrix) @ p`` and the lattice vectors transform the
    same way.  Checking against that prediction verifies the whole detection
    and clustering chain rather than merely its self-consistency.
    """
    cell = 12.0
    matrix = shear_matrix(shear=0.25, scale_x=1.0, scale_y=1.15)
    inverse = np.linalg.inv(matrix)

    image = make_checkerboard(shape=(220, 220), cell_size=cell, blur_sigma=1.0)
    distorted = distort_image(image, matrix=matrix)
    measured_a1, measured_a2 = estimate_basis(detect_centroids(distorted))

    np.testing.assert_allclose(measured_a1, inverse @ np.array([cell, cell]),
                               atol=BASIS_TOLERANCE_PX)
    np.testing.assert_allclose(measured_a2, inverse @ np.array([-cell, cell]),
                               atol=BASIS_TOLERANCE_PX)


def test_estimate_basis_orders_the_vectors_by_polar_angle():
    """A deterministic order keeps later results reproducible.

    Without a fixed convention the two vectors could be returned either way
    round, and any quantity that treats them asymmetrically would change from
    run to run.
    """
    image = make_checkerboard(shape=(160, 160), cell_size=12.0, blur_sigma=1.0)
    a1, a2 = estimate_basis(detect_centroids(image))
    assert np.arctan2(a1[1], a1[0]) <= np.arctan2(a2[1], a2[0])


def test_estimate_basis_rejects_a_point_set_that_is_too_small():
    """Two points define one direction, which is not a two-dimensional lattice."""
    with pytest.raises(ValueError, match="at least 3"):
        estimate_basis(np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_estimate_basis_rejects_wrongly_shaped_input():
    """Passing triples instead of (x, y) pairs is a realistic mistake."""
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        estimate_basis(np.zeros((10, 3)))


def test_estimate_basis_reports_collinear_points():
    """Points on a line span one direction only, so no lattice exists."""
    collinear = np.column_stack([np.arange(10.0), np.arange(10.0)])
    with pytest.raises(LatticeIndexingError):
        estimate_basis(collinear)


def test_assign_lattice_indices_labels_every_point_of_a_clean_lattice():
    """A complete, undistorted lattice must be labelled without gaps.

    Any unlabelled point is a point the calibration cannot use, so full
    coverage on the easy case is the baseline that harder cases are judged
    against.
    """
    points = make_lattice_points(cell_size=12.0, n_per_side=7)
    labels, labelled = assign_lattice_indices(
        points, np.array([12.0, 0.0]), np.array([0.0, 12.0]))

    assert labelled.all()
    assert labels.shape == points.shape


def test_assign_lattice_indices_gives_every_point_a_distinct_label():
    """Two points sharing a label would map to the same ideal position.

    This is the failure that motivated region growing in the first place, so
    it is asserted explicitly rather than left to the internal check.
    """
    points = make_lattice_points(cell_size=12.0, n_per_side=9)
    labels, labelled = assign_lattice_indices(
        points, np.array([12.0, 0.0]), np.array([0.0, 12.0]))

    used = [tuple(row) for row in labels[labelled]]
    assert len(set(used)) == len(used)


def test_assign_lattice_indices_reproduces_positions_from_labels():
    """Label times basis must reconstruct each point, up to a common shift.

    This is the property the polynomial fit relies on: the label says which
    ideal position a measured point corresponds to, so the reconstruction has
    to be faithful.
    """
    cell = 12.0
    a1, a2 = np.array([cell, 0.0]), np.array([0.0, cell])
    points = make_lattice_points(cell_size=cell, n_per_side=6)
    labels, labelled = assign_lattice_indices(points, a1, a2)

    basis = np.column_stack([a1, a2])
    reconstructed = (basis @ labels[labelled].T).T
    offsets = points[labelled] - reconstructed

    np.testing.assert_allclose(offsets - offsets[0], 0.0, atol=1e-9)


def test_assign_lattice_indices_labels_a_distorted_lattice_completely():
    """Region growing must survive the drift that breaks a global formula.

    A single global fit accumulates the change in lattice spacing across the
    image until the rounding to integers picks the wrong cell.  Propagating
    label by label keeps every decision local, so a strongly sheared target
    must still be labelled in full and without duplicates.
    """
    image = make_checkerboard(shape=(220, 220), cell_size=12.0, blur_sigma=1.0)
    distorted = distort_image(
        image, matrix=shear_matrix(shear=0.3, scale_x=1.0, scale_y=1.2))

    centroids = detect_centroids(distorted)
    a1, a2 = estimate_basis(centroids)
    labels, labelled = assign_lattice_indices(centroids, a1, a2)

    assert labelled.all()
    used = [tuple(row) for row in labels[labelled]]
    assert len(set(used)) == len(used)


def test_assign_lattice_indices_places_the_seed_at_the_origin():
    """One point must carry the label (0, 0) for the labels to be anchored."""
    points = make_lattice_points(cell_size=12.0, n_per_side=5)
    labels, labelled = assign_lattice_indices(
        points, np.array([12.0, 0.0]), np.array([0.0, 12.0]))

    assert any(tuple(row) == (0, 0) for row in labels[labelled])


def test_assign_lattice_indices_rejects_parallel_basis_vectors():
    """Parallel vectors cannot decompose a displacement into two steps."""
    points = make_lattice_points()
    with pytest.raises(LatticeIndexingError, match="parallel"):
        assign_lattice_indices(points, np.array([1.0, 1.0]),
                               np.array([2.0, 2.0]))


def test_assign_lattice_indices_rejects_an_empty_point_set():
    """Labelling nothing is a programming error worth reporting."""
    with pytest.raises(ValueError, match="empty"):
        assign_lattice_indices(np.zeros((0, 2)), np.array([1.0, 0.0]),
                               np.array([0.0, 1.0]))


def test_lattice_angle_is_ninety_degrees_for_perpendicular_vectors():
    """The reference case that the whole distortion metric is built on."""
    assert lattice_angle(np.array([1.0, 0.0]),
                         np.array([0.0, 1.0])) == pytest.approx(90.0)


def test_lattice_angle_stays_within_the_domain_of_arccos():
    """Rounding must not push the normalised dot product outside [-1, 1].

    Parallel vectors give a dot product that can land a few ulp above one,
    which would make arccos return NaN and silently poison the reported
    angle, so the value is clipped and the result checked here.
    """
    vector = np.array([3.0, 4.0])
    assert lattice_angle(vector, vector) == pytest.approx(0.0, abs=1e-6)
    assert lattice_angle(vector, -vector) == pytest.approx(180.0, abs=1e-6)


def test_lattice_angle_does_not_depend_on_the_order_of_its_arguments():
    """The angle between two vectors is symmetric, and the code must be too."""
    a1, a2 = np.array([1.0, 0.2]), np.array([-0.3, 1.0])
    assert lattice_angle(a1, a2) == pytest.approx(lattice_angle(a2, a1))


def test_lattice_angle_rejects_a_zero_length_vector():
    """A zero vector has no direction, so no angle can be defined."""
    with pytest.raises(ValueError, match="non-zero"):
        lattice_angle(np.array([0.0, 0.0]), np.array([1.0, 0.0]))


def test_basis_length_ratio_is_one_for_an_isotropic_lattice():
    """Equal-length vectors are the signature of an undistorted lattice."""
    assert basis_length_ratio(np.array([3.0, 4.0]),
                              np.array([0.0, 5.0])) == pytest.approx(1.0)


def test_basis_length_ratio_detects_anisotropic_scaling():
    """The ratio must reproduce a scaling applied to one axis only.

    Anisotropic pixel pitch is the distortion component that turns a round
    spot into an elliptical one, so the metric that reports it is checked
    against a known factor.
    """
    assert basis_length_ratio(np.array([2.0, 0.0]),
                              np.array([0.0, 4.0])) == pytest.approx(0.5)


def test_basis_length_ratio_rejects_a_zero_length_denominator():
    """Dividing by the length of a zero vector is undefined."""
    with pytest.raises(ValueError, match="non-zero"):
        basis_length_ratio(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
