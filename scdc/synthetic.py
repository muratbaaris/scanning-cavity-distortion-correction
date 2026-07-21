"""Synthetic calibration targets with a known, exactly specified distortion.

Real calibration images belong to the laboratory that recorded them, so the
test suite and the worked examples of this package do not depend on any.
Instead they build a checkerboard from scratch and distort it with a
transformation whose parameters are known in advance.  Because the correct
answer is known analytically, the tests can check that the pipeline recovers
it rather than merely check that it runs.
"""

import numpy as np
from scipy import ndimage


def ideal_lattice_points(shape, cell_size, fill_fraction=0.5):
    """Return the centres of the bright squares of an ideal checkerboard.

    Parameters
    ----------
    shape : tuple of int
        Shape ``(H, W)`` of the image the lattice is meant to fill.
    cell_size : float
        Edge length of one checkerboard cell, in pixels.
    fill_fraction : float, optional
        Unused for the geometry; kept so that the signature matches
        `make_checkerboard` and the two can be called with the same
        arguments.

    Returns
    -------
    centres : ndarray, shape (N, 2)
        ``(x, y)`` positions of the bright squares, which occupy every second
        cell so that their nearest neighbours are the diagonal ones.

    Raises
    ------
    ValueError
        If ``cell_size`` is not positive or ``shape`` is not a pair of
        positive integers.
    """
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive, got {cell_size}")
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError(f"shape must be positive, got {shape}")

    n_rows = int(np.ceil(height / cell_size))
    n_cols = int(np.ceil(width / cell_size))

    centres = [((col + 0.5) * cell_size, (row + 0.5) * cell_size)
               for row in range(n_rows)
               for col in range(n_cols)
               if (row + col) % 2 == 0]

    centres = np.array(centres, dtype=float)
    assert centres.ndim == 2 and centres.shape[1] == 2, "(x, y) pairs"
    return centres


def make_checkerboard(shape=(200, 200), cell_size=10.0, fill_fraction=0.5,
                      bright_level=1.0, dark_level=0.0, blur_sigma=0.8,
                      noise_level=0.0, random_seed=None):
    """Render an undistorted checkerboard calibration target.

    Parameters
    ----------
    shape : tuple of int, optional
        Shape ``(H, W)`` of the generated image.
    cell_size : float, optional
        Edge length of one checkerboard cell, in pixels.
    fill_fraction : float, optional
        Fraction of a cell occupied by the bright square, so 1.0 makes the
        squares touch and 0.5 leaves a gap as wide as the square.
    bright_level, dark_level : float, optional
        Intensities of the squares and of the background.
    blur_sigma : float, optional
        Standard deviation of a Gaussian blur applied to the rendered image,
        which imitates the finite spot size of the microscope and gives the
        centroid detection sub-pixel information to work with.
    noise_level : float, optional
        Standard deviation of additive Gaussian noise, in units of the
        contrast between bright and dark.
    random_seed : int or None, optional
        Seed for the noise.  Passing an integer makes the image reproducible,
        which is what the tests rely on.

    Returns
    -------
    image : ndarray, shape ``shape``
        The rendered target.

    Raises
    ------
    ValueError
        If ``cell_size`` is not positive or ``fill_fraction`` is outside
        (0, 1].

    Examples
    --------
    >>> image = make_checkerboard(shape=(60, 60), cell_size=10)
    >>> image.shape
    (60, 60)
    """
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive, got {cell_size}")
    if not 0.0 < fill_fraction <= 1.0:
        raise ValueError(
            f"fill_fraction must lie in (0, 1], got {fill_fraction}")

    height, width = shape
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(float)

    centres = ideal_lattice_points(shape, cell_size)
    half_square = 0.5 * cell_size * fill_fraction

    image = np.full((height, width), float(dark_level))
    for centre_x, centre_y in centres:
        inside = ((np.abs(grid_x - centre_x) <= half_square)
                  & (np.abs(grid_y - centre_y) <= half_square))
        image[inside] = bright_level

    if blur_sigma > 0:
        image = ndimage.gaussian_filter(image, blur_sigma)

    if noise_level > 0:
        generator = np.random.default_rng(random_seed)
        contrast = abs(bright_level - dark_level)
        image = image + generator.normal(
            0.0, noise_level * contrast, size=image.shape)

    return image


def distort_image(image, matrix=None, centre=None, quadratic_strength=0.0,
                  fill_value=0.0):
    """Apply a known geometric distortion to an image.

    The distortion is expressed as the *sampling* map: the value stored at
    output position ``q`` is read from input position ``matrix @ (q - centre)
    + centre``.  A bright square that sits at input position ``p`` therefore
    appears in the output at ``inv(matrix) @ (p - centre) + centre``, which
    is what `expected_centroids_after_distortion` computes.

    Parameters
    ----------
    image : array_like, shape (H, W)
        Image to distort.
    matrix : array_like, shape (2, 2), optional
        Linear part of the sampling map, acting on ``(x, y)`` coordinates.
        Defaults to the identity, which leaves the image unchanged.
    centre : tuple of float, optional
        Fixed point ``(x, y)`` of the transformation.  Defaults to the centre
        of the image, so that the distorted content stays in view.
    quadratic_strength : float, optional
        Amount of second-order distortion added on top of the linear part,
        in units of one pixel of displacement per squared image half-width.
        Zero, the default, keeps the map exactly linear and therefore exactly
        invertible.
    fill_value : float, optional
        Value used where the map reaches outside the input image.

    Returns
    -------
    distorted : ndarray, shape (H, W)
        The distorted image.

    Raises
    ------
    ValueError
        If ``matrix`` is singular or has the wrong shape.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(
            f"image must be two-dimensional, got {image.ndim} dimensions")

    if matrix is None:
        matrix = np.eye(2)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError(f"matrix must have shape (2, 2), got {matrix.shape}")
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("matrix must be invertible to define a distortion")

    height, width = image.shape
    if centre is None:
        centre = ((width - 1) / 2.0, (height - 1) / 2.0)
    centre_x, centre_y = centre

    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(float)
    shifted_x = grid_x - centre_x
    shifted_y = grid_y - centre_y

    source_x = matrix[0, 0] * shifted_x + matrix[0, 1] * shifted_y
    source_y = matrix[1, 0] * shifted_x + matrix[1, 1] * shifted_y

    if quadratic_strength != 0.0:
        half_width = max(width, height) / 2.0
        source_x = source_x + quadratic_strength * (shifted_x / half_width) ** 2 * half_width
        source_y = source_y + quadratic_strength * (shifted_y / half_width) ** 2 * half_width

    source_x = source_x + centre_x
    source_y = source_y + centre_y

    return ndimage.map_coordinates(
        image, [source_y, source_x], order=3, mode="constant", cval=fill_value)


def expected_centroids_after_distortion(centres, matrix, centre):
    """Predict where known points end up after `distort_image`.

    Parameters
    ----------
    centres : array_like, shape (N, 2)
        Positions ``(x, y)`` in the undistorted image.
    matrix : array_like, shape (2, 2)
        The same linear sampling map passed to `distort_image`.
    centre : tuple of float
        The same fixed point passed to `distort_image`.

    Returns
    -------
    moved : ndarray, shape (N, 2)
        Positions of the same features in the distorted image.

    Raises
    ------
    ValueError
        If ``matrix`` is singular.

    Notes
    -----
    Because `distort_image` reads the input at ``matrix @ q``, a feature at
    input position ``p`` is seen in the output at ``inv(matrix) @ p``.
    """
    centres = np.asarray(centres, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("matrix must be invertible")

    inverse = np.linalg.inv(matrix)
    centre = np.asarray(centre, dtype=float)
    moved = (inverse @ (centres - centre).T).T + centre

    assert moved.shape == centres.shape, "one moved point per input point"
    return moved


def shear_matrix(shear=0.0, scale_x=1.0, scale_y=1.0):
    """Build a 2x2 sampling map with a shear and anisotropic scaling.

    Parameters
    ----------
    shear : float, optional
        Off-diagonal term, which tilts one axis with respect to the other.
    scale_x, scale_y : float, optional
        Diagonal terms, which stretch the two axes independently.

    Returns
    -------
    matrix : ndarray, shape (2, 2)

    Raises
    ------
    ValueError
        If the resulting matrix is singular.
    """
    matrix = np.array([[scale_x, shear], [0.0, scale_y]], dtype=float)
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError(
            f"scale_x={scale_x} and scale_y={scale_y} give a singular matrix")
    return matrix
