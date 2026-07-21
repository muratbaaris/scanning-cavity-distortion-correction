"""Tests for the largest-inscribed-rectangle search.

The routine is a pure combinatorial function on a boolean mask, so every case
can be checked against a rectangle worked out by hand.
"""

import numpy as np
import pytest

from scdc.geometry import largest_inscribed_rectangle


def rectangle_area(bounds):
    """Return the area of an inclusive (top, bottom, left, right) rectangle."""
    top, bottom, left, right = bounds
    return (bottom - top + 1) * (right - left + 1)


def test_largest_inscribed_rectangle_returns_the_whole_of_a_full_mask():
    """When every pixel is valid the answer is the entire array.

    This is the case of a warp that lost nothing, and cropping anything away
    would discard good data for no reason.
    """
    mask = np.ones((6, 9), dtype=bool)
    assert largest_inscribed_rectangle(mask) == (0, 5, 0, 8)


def test_largest_inscribed_rectangle_finds_a_rectangle_of_the_correct_area():
    """The reported bounds must enclose the largest possible area.

    The mask below has one 3 by 4 block of valid pixels, so the search must
    return an area of twelve rather than any smaller sub-block.
    """
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 1:5] = True
    assert rectangle_area(largest_inscribed_rectangle(mask)) == 12


def test_largest_inscribed_rectangle_never_includes_an_invalid_pixel():
    """The returned slice must contain no False entry, whatever the mask.

    This is the property the caller depends on: a NaN left inside the crop
    would propagate into every later analysis step.
    """
    generator = np.random.default_rng(0)
    for _ in range(20):
        mask = generator.random((12, 15)) > 0.3
        top, bottom, left, right = largest_inscribed_rectangle(mask)
        if bottom < top or right < left:
            continue
        assert mask[top:bottom + 1, left:right + 1].all()


def test_largest_inscribed_rectangle_prefers_area_over_side_length():
    """A wide flat region can beat a tall narrow one, and must be chosen.

    The mask holds a 1 by 10 strip of area ten and a 4 by 2 block of area
    eight, so a search that maximised height instead of area would pick the
    wrong one.
    """
    mask = np.zeros((6, 12), dtype=bool)
    mask[0, 0:10] = True
    mask[2:6, 10:12] = True
    assert rectangle_area(largest_inscribed_rectangle(mask)) == 10


def test_largest_inscribed_rectangle_handles_a_mask_with_no_valid_pixel():
    """An empty mask must slice to an empty array rather than raise.

    A warp can legitimately produce no valid region if the calibration and
    the image do not overlap, and the caller should see an empty result.
    """
    mask = np.zeros((5, 5), dtype=bool)
    top, bottom, left, right = largest_inscribed_rectangle(mask)
    assert np.zeros((5, 5))[top:bottom + 1, left:right + 1].size == 0


def test_largest_inscribed_rectangle_handles_a_single_valid_pixel():
    """The smallest non-empty case must give a one-by-one rectangle."""
    mask = np.zeros((4, 4), dtype=bool)
    mask[2, 3] = True
    assert largest_inscribed_rectangle(mask) == (2, 2, 3, 3)


def test_largest_inscribed_rectangle_ignores_a_diagonal_of_valid_pixels():
    """Disconnected pixels cannot form a rectangle larger than one pixel.

    A diagonal is the classic case that defeats a naive row-wise scan, so it
    is worth pinning down.
    """
    mask = np.eye(6, dtype=bool)
    assert rectangle_area(largest_inscribed_rectangle(mask)) == 1


def test_largest_inscribed_rectangle_rejects_a_three_dimensional_mask():
    """A stack of masks is a caller mistake worth reporting clearly."""
    with pytest.raises(ValueError, match="two-dimensional"):
        largest_inscribed_rectangle(np.ones((3, 4, 5), dtype=bool))
