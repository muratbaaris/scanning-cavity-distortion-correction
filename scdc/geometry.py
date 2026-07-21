"""Geometric helper used to trim the empty corners of a warped image.

Warping a rectangular image through a non-trivial coordinate transformation
turns it into a curved quadrilateral, so the rectangular output array has
corners for which no input data exists.  Cropping to the largest rectangle
that fits entirely inside the valid region removes them.
"""

import numpy as np


def largest_inscribed_rectangle(valid):
    """Find the largest all-True axis-aligned rectangle in a boolean mask.

    Parameters
    ----------
    valid : array_like of bool, shape (H, W)
        Mask that is True where data is available.

    Returns
    -------
    top, bottom, left, right : int
        Inclusive row and column bounds of the largest rectangle, so that
        ``image[top:bottom + 1, left:right + 1]`` contains no False entry.
        When the mask has no True entry at all, ``(0, -1, 0, -1)`` is
        returned, which slices to an empty array.

    Raises
    ------
    ValueError
        If ``valid`` is not two-dimensional.

    Notes
    -----
    The mask is scanned row by row while maintaining, for every column, the
    height of the run of True values ending at the current row.  Each row is
    then the histogram of a classic largest-rectangle-in-a-histogram problem,
    which a monotone stack solves in time linear in the width.  The whole
    routine is therefore O(H * W).
    """
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2:
        raise ValueError(
            f"valid must be two-dimensional, got {valid.ndim} dimensions")

    height, width = valid.shape
    if height == 0 or width == 0 or not valid.any():
        return 0, -1, 0, -1

    running_heights = np.zeros(width, dtype=np.int64)
    best_area = 0
    best_bounds = (0, -1, 0, -1)

    for row in range(height):
        running_heights = np.where(valid[row], running_heights + 1, 0)
        stack = []
        # The sentinel column of height zero at index `width` forces the
        # stack to unwind completely at the end of every row.
        for column in range(width + 1):
            current_height = running_heights[column] if column < width else 0
            while stack and running_heights[stack[-1]] > current_height:
                bar = stack.pop()
                bar_height = int(running_heights[bar])
                left = stack[-1] + 1 if stack else 0
                bar_width = column - left
                area = bar_height * bar_width
                if area > best_area:
                    best_area = area
                    best_bounds = (row - bar_height + 1, row,
                                   left, column - 1)
            stack.append(column)

    top, bottom, left, right = best_bounds
    assert bottom < height and right < width, "bounds stay inside the mask"
    return top, bottom, left, right
