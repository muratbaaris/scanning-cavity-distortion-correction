"""Tests for the bivariate polynomial model of the distortion.

Polynomials of low degree can be checked against closed-form expectations,
which makes these tests independent of the rest of the pipeline.
"""

import numpy as np
import pytest

from scdc.polynomial import (
    PolynomialFitError,
    evaluate_polynomial,
    fit_polynomial,
    number_of_terms,
    polynomial_features,
    polynomial_term_labels,
)

EXACT_FIT_TOLERANCE = 1e-8
"""Residual below which a fit counts as exact.

Recovering a polynomial from noiseless samples of itself is a linear problem
solved in double precision, so anything above this bound signals an error in
the construction of the design matrix rather than numerical noise.
"""


@pytest.mark.parametrize("degree, expected", [(0, 1), (1, 3), (2, 6), (3, 10),
                                              (4, 15)])
def test_number_of_terms_follows_the_triangular_numbers(degree, expected):
    """A bivariate polynomial of degree d has (d+1)(d+2)/2 monomials.

    Every array of coefficients is sized from this function, so an error here
    would surface as a confusing shape mismatch much later.
    """
    assert number_of_terms(degree) == expected


def test_number_of_terms_rejects_a_negative_degree():
    """A negative degree has no meaning and is caught at the source."""
    with pytest.raises(ValueError, match="non-negative"):
        number_of_terms(-1)


def test_polynomial_features_has_one_column_per_monomial():
    """The design matrix width fixes the length of the coefficient vector."""
    x = np.linspace(0.0, 1.0, 7)
    y = np.linspace(1.0, 2.0, 7)
    design = polynomial_features(x, y, degree=3)
    assert design.shape == (7, number_of_terms(3))


def test_polynomial_features_starts_with_the_constant_term():
    """The first column must be all ones so the fit can carry an offset.

    A missing constant term would force every fitted surface through the
    origin, which no real distortion does.
    """
    design = polynomial_features(np.array([2.0, 5.0]), np.array([3.0, 7.0]),
                                 degree=2)
    np.testing.assert_allclose(design[:, 0], 1.0)


def test_polynomial_features_orders_columns_by_ascending_degree():
    """The column order is a promise the coefficient labels rely on.

    Within each total degree the power of x ascends, so degree two gives
    the columns 1, y, x, y^2, x*y, x^2.  Any other order would silently
    mislabel every reported coefficient.
    """
    x = np.array([2.0])
    y = np.array([3.0])
    design = polynomial_features(x, y, degree=2)
    expected = np.array([[1.0, 3.0, 2.0, 9.0, 6.0, 4.0]])
    np.testing.assert_allclose(design, expected)


def test_polynomial_features_rejects_mismatched_coordinate_lengths():
    """Unequal coordinate arrays are a mistake, not something to broadcast."""
    with pytest.raises(ValueError, match="same length"):
        polynomial_features(np.zeros(5), np.zeros(6), degree=1)


def test_polynomial_term_labels_match_the_design_matrix_columns():
    """Each label names the column at the same index, and there is one each."""
    labels = polynomial_term_labels(2)
    assert labels == ["1", "Y", "X", "Y^2", "X*Y", "X^2"]


def test_polynomial_term_labels_count_matches_number_of_terms():
    """A label list shorter than the coefficients would break the report."""
    for degree in range(5):
        assert len(polynomial_term_labels(degree)) == number_of_terms(degree)


def test_fit_polynomial_recovers_an_exact_affine_map():
    """A degree-one fit must reproduce a matrix-and-shift transformation.

    Degree one is exactly an affine transformation, so a known matrix and
    translation must come back with a residual at machine precision.  This
    pins down the claim that the polynomial model generalises the affine one.
    """
    generator = np.random.default_rng(0)
    source = generator.uniform(-50.0, 50.0, size=(60, 2))
    matrix = np.array([[1.3, -0.2], [0.4, 0.9]])
    shift = np.array([7.0, -3.0])
    target = (matrix @ source.T).T + shift

    coefficients_x, coefficients_y, residuals = fit_polynomial(
        source, target, degree=1)

    assert residuals.max() < EXACT_FIT_TOLERANCE
    # Columns are ordered 1, y, x, so index 1 holds the y coefficient.
    np.testing.assert_allclose(
        [coefficients_x[0], coefficients_x[2], coefficients_x[1]],
        [shift[0], matrix[0, 0], matrix[0, 1]], atol=1e-8)
    np.testing.assert_allclose(
        [coefficients_y[0], coefficients_y[2], coefficients_y[1]],
        [shift[1], matrix[1, 0], matrix[1, 1]], atol=1e-8)


def test_fit_polynomial_recovers_a_known_quadratic_map():
    """A degree-two fit must reproduce a quadratic warp exactly.

    Real scanner distortion is non-linear, so the ability to recover a
    curved map is what justifies using a degree above one.
    """
    generator = np.random.default_rng(1)
    source = generator.uniform(-20.0, 20.0, size=(80, 2))
    x, y = source[:, 0], source[:, 1]
    target = np.column_stack([
        1.0 + 2.0 * x + 0.5 * y + 0.01 * x ** 2,
        -3.0 + 0.1 * x + 1.5 * y + 0.02 * x * y,
    ])

    _, _, residuals = fit_polynomial(source, target, degree=2)
    assert residuals.max() < EXACT_FIT_TOLERANCE


def test_fit_polynomial_returns_one_residual_per_point_pair():
    """The residual array is used to summarise quality point by point."""
    generator = np.random.default_rng(2)
    source = generator.uniform(0.0, 10.0, size=(25, 2))
    target = source * 2.0
    _, _, residuals = fit_polynomial(source, target, degree=1)
    assert len(residuals) == len(source)


def test_fit_polynomial_rejects_an_underdetermined_problem():
    """Fewer pairs than coefficients has no unique solution.

    Letting the least-squares solver return one of the infinitely many
    answers would produce a calibration that looks fine and warps wrongly.
    """
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(PolynomialFitError, match="at least"):
        fit_polynomial(source, source, degree=3)


def test_fit_polynomial_rejects_mismatched_point_sets():
    """Source and target must pair up one to one."""
    with pytest.raises(ValueError, match="same shape"):
        fit_polynomial(np.zeros((10, 2)), np.zeros((9, 2)), degree=1)


def test_evaluate_polynomial_inverts_the_fit():
    """Evaluating the fitted coefficients must return the fitted targets.

    Fitting and evaluating are used at opposite ends of the pipeline, so
    their conventions have to agree exactly or every warp is shifted.
    """
    generator = np.random.default_rng(3)
    source = generator.uniform(-10.0, 10.0, size=(40, 2))
    target = np.column_stack([2.0 * source[:, 0] + 1.0,
                              3.0 * source[:, 1] - 2.0])

    coefficients_x, coefficients_y, _ = fit_polynomial(source, target,
                                                       degree=1)
    mapped_x, mapped_y = evaluate_polynomial(
        source[:, 0], source[:, 1], coefficients_x, coefficients_y, degree=1)

    np.testing.assert_allclose(mapped_x, target[:, 0], atol=1e-8)
    np.testing.assert_allclose(mapped_y, target[:, 1], atol=1e-8)


def test_evaluate_polynomial_preserves_the_shape_of_its_input():
    """The warp evaluates on a 2-D pixel grid and needs the shape back."""
    coefficients = np.array([0.0, 1.0, 0.0])
    grid_y, grid_x = np.mgrid[0:5, 0:7].astype(float)
    mapped_x, mapped_y = evaluate_polynomial(
        grid_x, grid_y, coefficients, coefficients, degree=1)

    assert mapped_x.shape == grid_x.shape
    assert mapped_y.shape == grid_y.shape


def test_evaluate_polynomial_rejects_a_wrong_coefficient_count():
    """A coefficient array from a different degree is a silent-error risk."""
    with pytest.raises(ValueError, match="coefficients"):
        evaluate_polynomial(np.zeros(4), np.zeros(4),
                            np.zeros(3), np.zeros(3), degree=2)
