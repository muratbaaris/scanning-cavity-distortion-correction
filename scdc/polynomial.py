"""Bivariate polynomial model of the scanner distortion.

The distortion is represented as a pair of polynomials that map a position
on the ideal, undistorted grid to the position at which the scanner actually
recorded it.  Degree one reproduces an affine transformation exactly; higher
degrees add the curvature needed to describe the non-linear response of a
piezo actuator.
"""

import numpy as np


class PolynomialFitError(ValueError):
    """Raised when a polynomial cannot be fitted to the supplied data."""


def number_of_terms(degree):
    """Return how many monomials a bivariate polynomial of ``degree`` has.

    Parameters
    ----------
    degree : int
        Total degree of the polynomial.

    Returns
    -------
    n_terms : int
        ``(degree + 1) * (degree + 2) / 2``, that is 1, 3, 6, 10, ... for
        degree 0, 1, 2, 3, ...

    Raises
    ------
    ValueError
        If ``degree`` is negative.
    """
    degree = int(degree)
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    return (degree + 1) * (degree + 2) // 2


def polynomial_features(x, y, degree):
    """Evaluate every monomial of a bivariate polynomial at given points.

    Parameters
    ----------
    x, y : array_like, shape (N,)
        Coordinates at which the monomials are evaluated.
    degree : int
        Total degree of the polynomial.

    Returns
    -------
    design : ndarray, shape (N, number_of_terms(degree))
        Design matrix whose column ``k`` holds the ``k``-th monomial
        evaluated at every point.  Columns are ordered by ascending total
        degree, and within a degree by ascending power of ``x``, so degree 2
        yields the columns ``1, y, x, y**2, x*y, x**2``.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different lengths or ``degree`` is negative.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same length, got {x.shape} and {y.shape}")
    degree = int(degree)
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")

    columns = [(x ** power_x) * (y ** (total - power_x))
               for total in range(degree + 1)
               for power_x in range(total + 1)]
    design = np.column_stack(columns)

    assert design.shape == (len(x), number_of_terms(degree)), \
        "one row per point and one column per monomial"
    return design


def polynomial_term_labels(degree):
    """Return human-readable names for the monomials of a given degree.

    The labels are used when the fitted coefficients are shown to the user,
    so that each number can be matched to the term it multiplies.

    Parameters
    ----------
    degree : int
        Total degree of the polynomial.

    Returns
    -------
    labels : list of str
        Names such as ``['1', 'Y', 'X', 'Y^2', 'X*Y', 'X^2']``, in the same
        order as the columns produced by `polynomial_features`.

    Raises
    ------
    ValueError
        If ``degree`` is negative.
    """
    degree = int(degree)
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")

    labels = []
    for total in range(degree + 1):
        for power_x in range(total + 1):
            power_y = total - power_x
            if total == 0:
                labels.append("1")
                continue
            parts = []
            if power_x == 1:
                parts.append("X")
            elif power_x > 1:
                parts.append(f"X^{power_x}")
            if power_y == 1:
                parts.append("Y")
            elif power_y > 1:
                parts.append(f"Y^{power_y}")
            labels.append("*".join(parts))

    assert len(labels) == number_of_terms(degree), "one label per monomial"
    return labels


def fit_polynomial(source, target, degree):
    """Fit the polynomial pair that maps ``source`` points onto ``target``.

    Two independent polynomials are fitted, one predicting the ``x``
    component of the target and one predicting its ``y`` component.  Each is
    solved as an overdetermined linear least-squares problem, so the many
    available point pairs average out the noise of the individual centroid
    positions and leave only the systematic distortion.

    Parameters
    ----------
    source : array_like, shape (N, 2)
        Coordinates at which the polynomials are evaluated, in ``(x, y)``
        order.  In this package these are the ideal grid positions.
    target : array_like, shape (N, 2)
        Coordinates the polynomials should reproduce, in ``(x, y)`` order.
        In this package these are the measured centroid positions.
    degree : int
        Total degree of the polynomials.

    Returns
    -------
    coefficients_x, coefficients_y : ndarray, shape (number_of_terms(degree),)
        Coefficients of the two polynomials, ordered to match the columns of
        `polynomial_features`.
    residuals : ndarray, shape (N,)
        Euclidean distance between each predicted and actual target point,
        in the same units as the inputs.

    Raises
    ------
    ValueError
        If the two point sets have different lengths or the wrong shape.
    PolynomialFitError
        If there are fewer point pairs than coefficients to determine, in
        which case the fit would be underdetermined.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(f"source must have shape (N, 2), got {source.shape}")
    if target.shape != source.shape:
        raise ValueError(
            f"source and target must have the same shape, got "
            f"{source.shape} and {target.shape}")

    n_terms = number_of_terms(degree)
    if len(source) < n_terms:
        raise PolynomialFitError(
            f"a degree-{degree} fit needs at least {n_terms} point pairs, "
            f"got {len(source)}")

    design = polynomial_features(source[:, 0], source[:, 1], degree)
    coefficients_x, *_ = np.linalg.lstsq(design, target[:, 0], rcond=None)
    coefficients_y, *_ = np.linalg.lstsq(design, target[:, 1], rcond=None)

    predicted = np.column_stack([design @ coefficients_x,
                                 design @ coefficients_y])
    residuals = np.linalg.norm(predicted - target, axis=1)

    assert len(residuals) == len(source), "one residual per point pair"
    return coefficients_x, coefficients_y, residuals


def evaluate_polynomial(x, y, coefficients_x, coefficients_y, degree):
    """Apply a fitted polynomial pair to a set of coordinates.

    Parameters
    ----------
    x, y : array_like
        Coordinates to transform.  Arrays of any shape are accepted and the
        shape is preserved in the output.
    coefficients_x, coefficients_y : array_like
        Coefficients returned by `fit_polynomial`.
    degree : int
        Total degree of the polynomials, which must match the one used for
        the fit.

    Returns
    -------
    mapped_x, mapped_y : ndarray
        Transformed coordinates, with the same shape as the inputs.

    Raises
    ------
    ValueError
        If the number of coefficients does not match ``degree`` or if ``x``
        and ``y`` have different shapes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape, got {x.shape} and {y.shape}")

    coefficients_x = np.asarray(coefficients_x, dtype=float).ravel()
    coefficients_y = np.asarray(coefficients_y, dtype=float).ravel()
    expected = number_of_terms(degree)
    if len(coefficients_x) != expected or len(coefficients_y) != expected:
        raise ValueError(
            f"a degree-{degree} polynomial needs {expected} coefficients, got "
            f"{len(coefficients_x)} and {len(coefficients_y)}")

    design = polynomial_features(x.ravel(), y.ravel(), degree)
    mapped_x = (design @ coefficients_x).reshape(x.shape)
    mapped_y = (design @ coefficients_y).reshape(y.shape)

    assert mapped_x.shape == x.shape, "the input shape is preserved"
    return mapped_x, mapped_y
