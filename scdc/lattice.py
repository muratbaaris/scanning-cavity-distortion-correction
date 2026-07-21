"""Measurement of the lattice geometry of a periodic calibration target.

Two quantities are extracted here.  The first is the pair of *basis vectors*
that connect a bright square to its nearest neighbours; their deviation from
being equal in length and mutually perpendicular is a direct measure of the
distortion.  The second is an integer label ``(i, j)`` for every square,
which identifies the position it occupies on the ideal grid and therefore
allows a measured position to be paired with the position it ought to have.
"""

import numpy as np
from collections import deque
from scipy.spatial import cKDTree

UNLABELLED = np.iinfo(np.int64).max
"""Sentinel stored in the label array for squares that were never reached."""


class LatticeIndexingError(RuntimeError):
    """Raised when integer lattice labels cannot be assigned consistently."""


def estimate_basis(centroids, n_neighbours=4, length_tolerance=0.35):
    """Measure the two lattice step vectors from a cloud of centroids.

    The bright squares of a checkerboard occupy every second cell, so the
    nearest neighbour of a bright square is its *diagonal* neighbour.  The
    displacement vectors between nearest neighbours therefore cluster around
    four directions, which are two directions and their opposites.  This
    function folds the opposites onto each other and averages within each of
    the two remaining clusters.

    Parameters
    ----------
    centroids : array_like, shape (N, 2)
        Detected square centres as ``(x, y)`` pairs.
    n_neighbours : int, optional
        Number of nearest neighbours examined per centroid.  Four is the
        natural choice for a square lattice; larger values pull in
        next-nearest neighbours which are then rejected by the length filter.
    length_tolerance : float, optional
        Relative half-width of the band of accepted displacement lengths
        around the median nearest-neighbour distance.  The default of 0.35
        accepts displacements between 0.65 and 1.35 times the median, which
        excludes next-nearest neighbours (a factor of about 1.41 away) while
        tolerating substantial distortion.

    Returns
    -------
    a1, a2 : ndarray, shape (2,)
        The two lattice step vectors in ``(x, y)`` pixel units, ordered so
        that ``a1`` has the smaller polar angle.

    Raises
    ------
    ValueError
        If fewer than three centroids are supplied or the array does not have
        shape ``(N, 2)``.
    LatticeIndexingError
        If the displacement vectors do not separate into two distinct
        directions, which happens when the points are collinear or random.

    Notes
    -----
    The directions are found by a two-means clustering of the folded
    displacement vectors, seeded with the two most widely separated
    candidates.  No prior knowledge of the lattice orientation or spacing is
    required, so the routine works unchanged on targets of different pitch.
    """
    centroids = np.asarray(centroids, dtype=float)
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        raise ValueError(
            f"centroids must have shape (N, 2), got {centroids.shape}")
    if len(centroids) < 3:
        raise ValueError(
            f"at least 3 centroids are needed to define a lattice, "
            f"got {len(centroids)}")

    k = min(n_neighbours + 1, len(centroids))
    _, neighbour_index = cKDTree(centroids).query(centroids, k=k)
    # Column 0 of the query result is the point itself, hence the slice.
    offsets = np.array([centroids[neighbour_index[i, j]] - centroids[i]
                        for i in range(len(centroids))
                        for j in range(1, k)])

    lengths = np.linalg.norm(offsets, axis=1)
    median_length = np.median(lengths)
    in_band = np.abs(lengths - median_length) <= length_tolerance * median_length
    offsets = offsets[in_band]
    if len(offsets) < 2:
        raise LatticeIndexingError(
            "no consistent nearest-neighbour distance could be identified")

    folded = _fold_to_half_plane(offsets)
    first, second = _split_into_two_directions(folded)
    first = _refine_direction(folded, first)
    second = _refine_direction(folded, second)

    if np.arctan2(first[1], first[0]) > np.arctan2(second[1], second[0]):
        first, second = second, first

    assert first.shape == (2,) and second.shape == (2,), "two 2-D vectors"
    return first, second


def _refine_direction(folded, centre, capture_radius=0.3):
    """Re-average a cluster after discarding vectors far from its centre.

    Centroids at the border of the image have fewer true nearest neighbours,
    so the fixed-size neighbour query returns lattice vectors from the next
    shell for them.  Those contaminate the initial cluster mean, and this
    second pass removes them by keeping only the vectors that lie within a
    fraction of the cluster length from the current centre.
    """
    distance = np.linalg.norm(folded - centre, axis=1)
    close_enough = distance <= capture_radius * np.linalg.norm(centre)
    if not close_enough.any():
        return centre
    return folded[close_enough].mean(axis=0)


def _fold_to_half_plane(offsets):
    """Map every displacement and its opposite onto the same representative.

    A lattice step and its reverse describe the same lattice direction, so
    they must not be treated as two different clusters.  Flipping every
    vector into the half plane ``y > 0`` merges each pair.
    """
    flip = (offsets[:, 1] < 0) | ((offsets[:, 1] == 0) & (offsets[:, 0] < 0))
    folded = offsets.copy()
    folded[flip] *= -1
    return folded


def _split_into_two_directions(folded, n_iterations=10):
    """Cluster folded displacements into the two lattice directions."""
    # Seed the two clusters with the pair of vectors that are furthest apart,
    # which for a lattice are guaranteed to belong to different directions.
    pairwise = np.linalg.norm(folded[:, None, :] - folded[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(pairwise), pairwise.shape)
    centre_a, centre_b = folded[i].copy(), folded[j].copy()

    if np.allclose(centre_a, centre_b):
        raise LatticeIndexingError(
            "all displacement vectors point in the same direction, so no "
            "two-dimensional lattice can be defined")

    for _ in range(n_iterations):
        to_a = np.linalg.norm(folded - centre_a, axis=1)
        to_b = np.linalg.norm(folded - centre_b, axis=1)
        belongs_to_a = to_a < to_b
        if not belongs_to_a.any() or belongs_to_a.all():
            raise LatticeIndexingError(
                "displacement vectors did not separate into two directions")
        new_a = folded[belongs_to_a].mean(axis=0)
        new_b = folded[~belongs_to_a].mean(axis=0)
        if np.allclose(new_a, centre_a) and np.allclose(new_b, centre_b):
            break
        centre_a, centre_b = new_a, new_b

    return centre_a, centre_b


def assign_lattice_indices(centroids, a1, a2, search_radius_factor=1.6,
                           residual_tolerance=0.4):
    """Label every centroid with the integer grid position it occupies.

    Labels are propagated outwards from a seed by breadth-first search: the
    label of a square is the label of an already-labelled neighbour plus the
    small integer step that separates them.  Deriving every label from a
    single distant origin instead would accumulate the drift of the lattice
    spacing across the image until the rounding to integers picks the wrong
    grid position, which shows up as two squares sharing one label.

    Parameters
    ----------
    centroids : array_like, shape (N, 2)
        Detected square centres as ``(x, y)`` pairs.
    a1, a2 : array_like, shape (2,)
        Lattice step vectors, normally obtained from `estimate_basis`.
    search_radius_factor : float, optional
        Neighbours are looked for within this multiple of the mean basis
        length.  The default of 1.6 covers the four nearest neighbours of a
        square lattice without reaching the next shell at about 2.0.
    residual_tolerance : float, optional
        A candidate step is accepted only if the difference between the
        measured displacement and the reconstructed one is below this
        multiple of the mean basis length.  This rejects neighbours that do
        not sit on the lattice, for instance spurious detections.

    Returns
    -------
    labels : ndarray, shape (N, 2)
        Integer ``(i, j)`` labels.  Entries for centroids that could not be
        reached hold the module-level sentinel `UNLABELLED`.
    labelled : ndarray of bool, shape (N,)
        True where a label was assigned.

    Raises
    ------
    ValueError
        If the inputs have inconsistent shapes or fewer than one centroid.
    LatticeIndexingError
        If ``a1`` and ``a2`` are parallel, so that a displacement cannot be
        decomposed into them, or if two centroids receive the same label.
    """
    centroids = np.asarray(centroids, dtype=float)
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        raise ValueError(
            f"centroids must have shape (N, 2), got {centroids.shape}")
    if len(centroids) == 0:
        raise ValueError("cannot assign lattice indices to an empty point set")

    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    if a1.shape != (2,) or a2.shape != (2,):
        raise ValueError("a1 and a2 must both have shape (2,)")

    basis = np.column_stack([a1, a2])
    try:
        basis_inverse = np.linalg.inv(basis)
    except np.linalg.LinAlgError:
        raise LatticeIndexingError(
            "the basis vectors are parallel, so displacements cannot be "
            "decomposed into integer lattice steps") from None

    mean_spacing = 0.5 * (np.linalg.norm(a1) + np.linalg.norm(a2))
    tree = cKDTree(centroids)

    # Seeding at the centre of the cloud minimises the longest chain of steps
    # any label has to travel, and with it the chance of a mislabelled hop.
    centre_of_cloud = centroids.mean(axis=0)
    seed = int(np.argmin(np.linalg.norm(centroids - centre_of_cloud, axis=1)))

    labels = np.full((len(centroids), 2), UNLABELLED, dtype=np.int64)
    labels[seed] = (0, 0)
    queue = deque([seed])

    while queue:
        current = queue.popleft()
        neighbours = tree.query_ball_point(
            centroids[current], r=search_radius_factor * mean_spacing)
        for neighbour in neighbours:
            if labels[neighbour, 0] != UNLABELLED:
                continue
            displacement = centroids[neighbour] - centroids[current]
            steps = basis_inverse @ displacement
            di, dj = int(round(steps[0])), int(round(steps[1]))
            if (di, dj) == (0, 0):
                continue
            reconstructed = di * a1 + dj * a2
            if np.linalg.norm(displacement - reconstructed) > \
                    residual_tolerance * mean_spacing:
                continue
            labels[neighbour] = labels[current] + np.array([di, dj])
            queue.append(neighbour)

    labelled = labels[:, 0] != UNLABELLED
    n_labelled = int(labelled.sum())
    n_unique = len({tuple(row) for row in labels[labelled]})
    if n_unique != n_labelled:
        raise LatticeIndexingError(
            f"{n_labelled - n_unique} centroids were assigned a duplicate "
            "lattice label, so the measured and ideal positions cannot be "
            "paired unambiguously")

    assert len(labels) == len(centroids), "one label per centroid"
    return labels, labelled


def lattice_angle(a1, a2):
    """Return the angle between two lattice vectors, in degrees.

    An undistorted square lattice gives 90 degrees, so the deviation from
    that value quantifies the shear component of the distortion.

    Parameters
    ----------
    a1, a2 : array_like, shape (2,)
        Lattice step vectors.

    Returns
    -------
    angle : float
        Angle between the vectors in degrees, in the range [0, 180].

    Raises
    ------
    ValueError
        If either vector has zero length.
    """
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    norm1 = np.linalg.norm(a1)
    norm2 = np.linalg.norm(a2)
    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("lattice vectors must have non-zero length")

    # Clipping absorbs the rounding error that can push a normalised dot
    # product marginally outside the domain of arccos.
    cosine = np.clip(np.dot(a1, a2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def basis_length_ratio(a1, a2):
    """Return the ratio of the lengths of two lattice vectors.

    An undistorted square lattice gives 1, so the deviation from that value
    quantifies the anisotropic scaling component of the distortion.

    Parameters
    ----------
    a1, a2 : array_like, shape (2,)
        Lattice step vectors.

    Returns
    -------
    ratio : float
        ``|a1| / |a2|``.

    Raises
    ------
    ValueError
        If ``a2`` has zero length.
    """
    norm1 = float(np.linalg.norm(np.asarray(a1, dtype=float)))
    norm2 = float(np.linalg.norm(np.asarray(a2, dtype=float)))
    if norm2 == 0.0:
        raise ValueError("the second lattice vector must have non-zero length")
    return norm1 / norm2
