"""Detection of the bright squares of a periodic calibration target.

The calibration target is a mirror patterned with a checkerboard of
reflective squares.  In a transmission image the reflective squares appear
as bright blobs on a dark background, so they can be isolated by a simple
intensity threshold.  The sub-pixel position of each blob is then taken as
its intensity-weighted centre of mass.
"""

import numpy as np
from scipy import ndimage


class NoCentroidsFoundError(ValueError):
    """Raised when thresholding an image yields no usable blobs."""


def detect_centroids(image, threshold_fraction=0.45, min_blob_area=8,
                     drop_border_blobs=True):
    """Locate the centre of every bright square in a calibration image.

    Parameters
    ----------
    image : array_like, shape (H, W)
        Greyscale image of the calibration target.  NaN entries are treated
        as zero so that images produced by a previous warp can be re-analysed.
    threshold_fraction : float, optional
        Position of the binarisation threshold between the minimum and the
        maximum intensity of the image, so ``0.0`` keeps every pixel and
        ``1.0`` keeps none.  The default of ``0.45`` sits in the middle of
        the intensity gap between the bright squares and the dark background
        of a checkerboard, which makes the result insensitive to the exact
        value.
    min_blob_area : int, optional
        Blobs made of fewer than this many pixels are discarded as noise.
    drop_border_blobs : bool, optional
        When True (the default) blobs touching the image border are
        discarded.  Such blobs are truncated by the field of view, so their
        centre of mass is biased towards the image interior and would
        corrupt a subsequent lattice fit.

    Returns
    -------
    centroids : ndarray, shape (N, 2)
        Sub-pixel ``(x, y)`` coordinates of the detected squares, i.e.
        column index first and row index second.

    Raises
    ------
    ValueError
        If ``image`` is not two-dimensional, if ``threshold_fraction`` is
        outside the open interval (0, 1), or if ``min_blob_area`` is not
        positive.
    NoCentroidsFoundError
        If no blob survives the filtering step.

    Notes
    -----
    The returned coordinates are ordered ``(x, y)`` rather than the NumPy
    ``(row, column)`` convention because every downstream geometric routine
    in this package works in ``(x, y)``.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(
            f"image must be two-dimensional, got {image.ndim} dimensions")
    if not 0.0 < threshold_fraction < 1.0:
        raise ValueError(
            f"threshold_fraction must lie in (0, 1), got {threshold_fraction}")
    if min_blob_area < 1:
        raise ValueError(
            f"min_blob_area must be at least 1, got {min_blob_area}")

    finite = np.where(np.isnan(image), 0.0, image)
    span = finite.max() - finite.min()
    if span == 0.0:
        raise NoCentroidsFoundError(
            "image is uniform, so no bright squares can be separated from it")

    threshold = finite.min() + threshold_fraction * span
    labelled, n_blobs = ndimage.label(finite > threshold)
    if n_blobs == 0:
        raise NoCentroidsFoundError(
            "thresholding produced no blobs; the target may be out of view")

    height, width = finite.shape
    kept_labels = []
    for label in range(1, n_blobs + 1):
        rows, cols = np.where(labelled == label)
        if len(rows) < min_blob_area:
            continue
        touches_border = (cols.min() == 0 or rows.min() == 0
                          or cols.max() == width - 1 or rows.max() == height - 1)
        if drop_border_blobs and touches_border:
            continue
        kept_labels.append(label)

    if not kept_labels:
        raise NoCentroidsFoundError(
            f"all {n_blobs} blobs were rejected as too small or truncated by "
            "the image border")

    kept_mask = np.isin(labelled, kept_labels)
    relabelled, n_kept = ndimage.label(kept_mask)
    # Weighting by the mask rather than by the raw intensity keeps the centre
    # estimate independent of illumination gradients across the field of view.
    centres_rowcol = np.array(
        ndimage.center_of_mass(kept_mask, relabelled, range(1, n_kept + 1)))

    centroids = centres_rowcol[:, ::-1]
    assert centroids.shape == (n_kept, 2), "one (x, y) pair per surviving blob"
    return centroids
