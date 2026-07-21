"""Fitting and applying a complete distortion calibration.

A calibration is the polynomial pair that maps ideal grid coordinates to the
coordinates at which the scanner recorded them, together with the metadata
needed to reproduce the same output geometry on later images.  It is a
property of the instrument and of the scan settings, not of the sample, so a
calibration measured once on a checkerboard target can be applied to every
image acquired afterwards with the same settings.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from scdc.centroids import detect_centroids
from scdc.geometry import largest_inscribed_rectangle
from scdc.lattice import assign_lattice_indices, estimate_basis
from scdc.polynomial import (
    evaluate_polynomial,
    fit_polynomial,
    number_of_terms,
    polynomial_term_labels,
)

MARGIN_PX = 10
"""Padding added around the ideal grid so no target point sits on the edge."""


class CalibrationError(ValueError):
    """Raised when a calibration cannot be fitted or applied."""


@dataclass
class Calibration:
    """A fitted distortion model together with its provenance.

    Attributes
    ----------
    degree : int
        Total degree of the two polynomials.
    coefficients_x, coefficients_y : ndarray
        Polynomial coefficients mapping ideal to measured coordinates.
    lattice_spacing_px : float
        Spacing of the corrected output lattice, in pixels.  Combined with
        the known physical pitch of the target this converts pixels to
        micrometres.
    input_shape : tuple of int
        Shape ``(H, W)`` of the image the calibration was fitted on.  An
        image passed to `apply_calibration` must have this shape, because the
        polynomial is only valid over the region it was measured on.
    output_shape : tuple of int
        Shape ``(H, W)`` of the corrected image the calibration produces.
    crop_offset : tuple of int
        Offset ``(x, y)`` of the cropped output within the full warped grid.
    n_detected, n_used : int
        Number of centroids found and number that received a lattice label.
    mean_residual_px, max_residual_px : float
        Quality of the fit, as the distance between predicted and measured
        centroid positions.
    term_labels : list of str
        Names of the monomials, aligned with the coefficient arrays.
    """

    degree: int
    coefficients_x: np.ndarray
    coefficients_y: np.ndarray
    lattice_spacing_px: float
    input_shape: tuple
    output_shape: tuple
    crop_offset: tuple
    n_detected: int = 0
    n_used: int = 0
    mean_residual_px: float = 0.0
    max_residual_px: float = 0.0
    term_labels: list = field(default_factory=list)

    def pixel_size(self, target_pitch_um):
        """Convert the fitted lattice spacing into a physical pixel size.

        Parameters
        ----------
        target_pitch_um : float
            Distance in micrometres between neighbouring bright squares of
            the calibration target.  For a checkerboard of cell size ``s``
            the bright squares are diagonal neighbours, so this is
            ``s * sqrt(2)``.

        Returns
        -------
        pixel_size_um : float
            Size of one pixel of the corrected image, in micrometres.

        Raises
        ------
        ValueError
            If ``target_pitch_um`` is not positive.
        """
        if target_pitch_um <= 0:
            raise ValueError(
                f"target_pitch_um must be positive, got {target_pitch_um}")
        return target_pitch_um / self.lattice_spacing_px

    def summary(self):
        """Return a printable multi-line description of the calibration."""
        lines = [
            f"Polynomial distortion calibration (degree {self.degree})",
            f"  centroids used      : {self.n_used} of {self.n_detected}",
            f"  residual mean / max : {self.mean_residual_px:.3f} / "
            f"{self.max_residual_px:.3f} px",
            f"  lattice spacing     : {self.lattice_spacing_px:.3f} px",
            f"  input  shape (H, W) : {self.input_shape}",
            f"  output shape (H, W) : {self.output_shape}",
            "",
            f"  {'term':<8}{'x coefficient':>18}{'y coefficient':>18}",
            "  " + "-" * 44,
        ]
        for label, cx, cy in zip(self.term_labels,
                                 self.coefficients_x, self.coefficients_y):
            lines.append(f"  {label:<8}{cx:>18.6f}{cy:>18.6f}")
        return "\n".join(lines)


def build_ideal_targets(labels, basis_matrix):
    """Place integer lattice labels on a perfectly regular grid.

    Parameters
    ----------
    labels : array_like, shape (N, 2)
        Integer ``(i, j)`` lattice labels.
    basis_matrix : array_like, shape (2, 2)
        Columns are the two ideal basis vectors, which are equal in length
        and perpendicular by construction.

    Returns
    -------
    targets : ndarray, shape (N, 2)
        Ideal ``(x, y)`` positions, shifted so that the smallest coordinate
        in each direction equals `MARGIN_PX`.

    Raises
    ------
    ValueError
        If the inputs have the wrong shape.
    """
    labels = np.asarray(labels, dtype=float)
    basis_matrix = np.asarray(basis_matrix, dtype=float)
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError(f"labels must have shape (N, 2), got {labels.shape}")
    if basis_matrix.shape != (2, 2):
        raise ValueError(
            f"basis_matrix must have shape (2, 2), got {basis_matrix.shape}")

    targets = (basis_matrix @ labels.T).T
    targets = targets - targets.min(axis=0) + MARGIN_PX

    assert targets.shape == labels.shape, "one target per label"
    return targets


def _ideal_basis_from_measured(labels, positions):
    """Derive the spacing of the ideal output lattice from measured data.

    An affine fit from labels to measured positions gives the average step
    vectors of the distorted lattice.  Their mean length divided by sqrt(2)
    is the cell spacing, because the bright squares of a checkerboard are
    diagonal neighbours.  Using this value for the output keeps the corrected
    image at roughly the same magnification as the input.
    """
    design = np.column_stack([labels, np.ones(len(labels))])
    affine, *_ = np.linalg.lstsq(design, positions, rcond=None)
    step_vectors = affine[:2].T
    mean_step = 0.5 * (np.linalg.norm(step_vectors[:, 0])
                       + np.linalg.norm(step_vectors[:, 1]))
    return mean_step / np.sqrt(2.0)


def fit_calibration(image, degree=3, threshold_fraction=0.45,
                    min_blob_area=8):
    """Measure the distortion of the scanner from a calibration image.

    Parameters
    ----------
    image : array_like, shape (H, W)
        Image of the checkerboard calibration target.
    degree : int, optional
        Total degree of the fitted polynomials.  Degree 1 is an affine
        transformation, which captures uniform shear and scaling but cannot
        bend; degree 3 additionally captures the smooth non-linearity of a
        piezo scanner and is the default.
    threshold_fraction, min_blob_area
        Passed through to `scdc.centroids.detect_centroids`.

    Returns
    -------
    calibration : Calibration
        The fitted model, ready to be applied or saved.

    Raises
    ------
    CalibrationError
        If too few centroids are found for the requested degree.
    scdc.centroids.NoCentroidsFoundError
        If the image contains no recognisable target.
    scdc.lattice.LatticeIndexingError
        If the detected points do not form a consistent lattice.

    Examples
    --------
    >>> from scdc import make_checkerboard, fit_calibration
    >>> image = make_checkerboard(shape=(120, 120), cell_size=6)
    >>> calibration = fit_calibration(image, degree=1)
    >>> calibration.mean_residual_px < 0.5
    True
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(
            f"image must be two-dimensional, got {image.ndim} dimensions")

    centroids = detect_centroids(
        image, threshold_fraction=threshold_fraction,
        min_blob_area=min_blob_area)

    n_terms = number_of_terms(degree)
    if len(centroids) < n_terms:
        raise CalibrationError(
            f"a degree-{degree} calibration needs at least {n_terms} squares, "
            f"but only {len(centroids)} were detected")

    a1, a2 = estimate_basis(centroids)
    labels, labelled = assign_lattice_indices(centroids, a1, a2)
    used_positions = centroids[labelled]
    used_labels = labels[labelled]

    if len(used_positions) < n_terms:
        raise CalibrationError(
            f"a degree-{degree} calibration needs at least {n_terms} labelled "
            f"squares, but only {len(used_positions)} were labelled")

    spacing = _ideal_basis_from_measured(used_labels, used_positions)
    ideal_basis = spacing * np.array([[1.0, -1.0], [1.0, 1.0]])
    targets = build_ideal_targets(used_labels, ideal_basis)

    coefficients_x, coefficients_y, residuals = fit_polynomial(
        targets, used_positions, degree)

    output_shape, crop_offset = _measure_output_geometry(
        image, targets, coefficients_x, coefficients_y, degree)

    return Calibration(
        degree=int(degree),
        coefficients_x=coefficients_x,
        coefficients_y=coefficients_y,
        lattice_spacing_px=float(spacing),
        input_shape=tuple(int(v) for v in image.shape),
        output_shape=output_shape,
        crop_offset=crop_offset,
        n_detected=int(len(centroids)),
        n_used=int(len(used_positions)),
        mean_residual_px=float(residuals.mean()),
        max_residual_px=float(residuals.max()),
        term_labels=polynomial_term_labels(degree),
    )


def _measure_output_geometry(image, targets, coefficients_x, coefficients_y,
                             degree):
    """Warp once to discover which part of the output grid holds real data.

    The corrected image is the largest rectangle that lies entirely inside
    the warped quadrilateral.  Its size and position are recorded in the
    calibration so that every later image is cropped identically.
    """
    full_width = int(np.ceil(targets[:, 0].max())) + MARGIN_PX
    full_height = int(np.ceil(targets[:, 1].max())) + MARGIN_PX

    sampled = _sample_through_polynomial(
        image, coefficients_x, coefficients_y, degree,
        output_shape=(full_height, full_width), crop_offset=(0, 0))

    top, bottom, left, right = largest_inscribed_rectangle(~np.isnan(sampled))
    output_shape = (bottom - top + 1, right - left + 1)
    crop_offset = (int(left), int(top))
    return output_shape, crop_offset


def _sample_through_polynomial(image, coefficients_x, coefficients_y, degree,
                               output_shape, crop_offset):
    """Fill an output grid by sampling the input at polynomial-mapped points.

    For every pixel of the output the polynomial answers the question "which
    position of the distorted input belongs here", and the input is sampled
    there by cubic interpolation.  Positions outside the input become NaN.
    """
    out_height, out_width = output_shape
    offset_x, offset_y = crop_offset

    grid_y, grid_x = np.mgrid[0:out_height, 0:out_width].astype(float)
    grid_x = grid_x + offset_x
    grid_y = grid_y + offset_y

    source_x, source_y = evaluate_polynomial(
        grid_x, grid_y, coefficients_x, coefficients_y, degree)

    return ndimage.map_coordinates(
        image, [source_y, source_x], order=3, mode="constant", cval=np.nan)


def apply_calibration(image, calibration, fill_value=np.nan):
    """Warp an image through a previously fitted calibration.

    Parameters
    ----------
    image : array_like, shape (H, W)
        Image to correct.  Its shape must equal ``calibration.input_shape``,
        because the polynomial is only valid over the region it was measured
        on; applying it elsewhere would extrapolate.
    calibration : Calibration
        Model returned by `fit_calibration` or loaded from a file.
    fill_value : float, optional
        Value written where the correction has no data.  The default of NaN
        marks such pixels explicitly; pass ``0.0`` to obtain an array that
        can be fed directly to routines that cannot handle NaN.

    Returns
    -------
    corrected : ndarray, shape ``calibration.output_shape``
        The geometrically corrected image.

    Raises
    ------
    CalibrationError
        If the shape of ``image`` does not match the shape the calibration
        was fitted on.

    Examples
    --------
    >>> import numpy as np
    >>> from scdc import make_checkerboard, fit_calibration, apply_calibration
    >>> image = make_checkerboard(shape=(120, 120), cell_size=6)
    >>> calibration = fit_calibration(image, degree=1)
    >>> corrected = apply_calibration(image, calibration)
    >>> corrected.shape == calibration.output_shape
    True
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(
            f"image must be two-dimensional, got {image.ndim} dimensions")
    if tuple(image.shape) != tuple(calibration.input_shape):
        raise CalibrationError(
            f"this calibration was fitted on an image of shape "
            f"{tuple(calibration.input_shape)} but was given one of shape "
            f"{tuple(image.shape)}; crop the image to the calibrated region "
            "or fit a new calibration")

    corrected = _sample_through_polynomial(
        image, calibration.coefficients_x, calibration.coefficients_y,
        calibration.degree, calibration.output_shape, calibration.crop_offset)

    if not np.isnan(fill_value):
        corrected = np.where(np.isnan(corrected), fill_value, corrected)

    assert corrected.shape == tuple(calibration.output_shape), \
        "the output shape is fixed by the calibration"
    return corrected
