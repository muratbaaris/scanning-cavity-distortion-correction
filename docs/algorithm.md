# The Distortion Correction Algorithm

This document explains each stage of the pipeline, the reasoning behind it,
and the assumptions it makes.

## Overview

The pipeline has seven stages:

1. **Detect centroids** - locate every bright square in the calibration image
2. **Estimate basis** - measure the two lattice step vectors
3. **Assign lattice indices** - label each square with integer grid coordinates
4. **Build ideal targets** - compute where each square should be on a perfect grid
5. **Fit polynomial** - learn the mapping from ideal to measured positions
6. **Warp image** - sample the input at polynomial-mapped positions
7. **Crop** - remove the empty corners of the warped output

## Stage 1: Centroid Detection

**Module:** `scdc.centroids`

The calibration target is a checkerboard mirror with bright reflective squares
on a dark background. The image is thresholded at 45% of the intensity range
to produce a binary mask. Connected components (blobs) are labelled; those
that are too small (noise) or that touch the image border (truncated, biased
centre) are discarded. The intensity-weighted centre of mass of each surviving
blob gives its sub-pixel position.

**Output:** an array of (x, y) positions, one per bright square.

**Why intensity-weighted centre of mass?** A geometric centre treats every
pixel in the blob equally, so noise at the edges pulls it around. Weighting
by intensity concentrates the estimate on the bright core, where the signal
is strongest.

**Why drop border blobs?** A square clipped by the edge of the field of view
is missing part of its area. The centre of mass of the visible part is
displaced towards the image interior by an amount unrelated to the distortion.
Including it would corrupt the polynomial fit.

## Stage 2: Basis Estimation

**Module:** `scdc.lattice.estimate_basis`

In a checkerboard the bright squares occupy every second cell, so the nearest
bright neighbour of a bright square is its diagonal neighbour. The two
diagonal directions define the lattice basis vectors **a₁** and **a₂**. The
code finds them by collecting all displacement vectors between nearest
neighbours, folding them into the upper half-plane (so a vector and its
opposite are treated as the same direction), and clustering the result into
two groups by iterative k-means.

A refinement pass then discards vectors that lie far from the cluster centre.
This removes contamination from next-nearest neighbours, which appear when
edge centroids have fewer true nearest neighbours and the fixed-size query
pulls in vectors from the next shell.

**Output:** two vectors a₁, a₂ in pixel coordinates.

**Why this matters for distortion:** in a perfect image a₁ and a₂ would have
equal length and meet at exactly 90°. Their deviation from that, the angle
and the length ratio, directly quantifies the distortion.

## Stage 3: Lattice Index Assignment (Region Growing)

**Module:** `scdc.lattice.assign_lattice_indices`

Each bright square occupies a unique position (i, j) on the integer grid.
The polynomial fit needs to know which position, so it can pair each measured
(x, y) with the ideal position it should have.

**The naive approach fails.** One could pick an origin, and for every other
square compute (i, j) by dividing the displacement from the origin by the
basis vectors and rounding. This works near the origin. Far away, the lattice
spacing drifts (because of the very distortion we are trying to correct), and
the accumulated error eventually exceeds half a grid step. The rounding tips
the wrong way, and two different squares end up with the same label. On our
data this produced 16 duplicate labels.

**Region growing fixes this.** Starting from a seed at the centre of the
point cloud (labelled (0, 0)), the algorithm propagates outward by
breadth-first search. Each new label is a neighbour's label plus a single
small integer step. Because every decision spans only one short hop, where
the lattice is locally uniform, the rounding is always safe. Errors never
accumulate, and every square receives a unique label.

**Output:** an integer (i, j) for each centroid.

## Stage 4: Building the Ideal Target Grid

**Module:** `scdc.calibration.build_ideal_targets`

The average lattice spacing `s_out` is extracted from an affine fit of the
labels to the measured positions: it is the mean of the two column norms of
the fitted matrix, divided by √2 (because the bright squares are diagonal
neighbours, at distance cell × √2).

A perfect basis matrix **B** is then constructed:

```
B = s_out * [[1, -1],
             [1,  1]]
```

The two columns of **B** are equal in length and perpendicular by
construction. Multiplying each square's label (i, j) by **B** gives its ideal
position: where it would sit on a perfectly regular, undistorted grid at the
same magnification as the input.

**Output:** an ideal (x, y) for each labelled square.

## Stage 5: Polynomial Fit

**Module:** `scdc.polynomial`

We now have matched pairs: an ideal position and a measured position for each
square. We fit two independent polynomials (one for x, one for y) that map
ideal to measured:

```
x_measured = c₀ + c₁·X + c₂·Y + c₃·X² + c₄·XY + c₅·Y² + ...
y_measured = d₀ + d₁·X + d₂·Y + ...
```

Each square contributes one equation per coordinate. With ~350 squares and
10 unknowns (for degree 3, which has 10 monomials), the system is heavily
overdetermined. Least squares finds the coefficients that minimise the total
squared error.

**Why degree 3?** Degree 1 is an affine transformation. It captures uniform
skew and scaling but cannot bend. The real distortion bends (piezo
nonlinearity), so degree 1 leaves a residual of several pixels. Degree 3
adds curvature and brings the residual down to the centroid noise floor
(~0.7 px on real data). Degree 4 does not improve it further, so degree 3 is
the "sweet spot".

**Output:** 20 coefficients (10 per coordinate) that completely describe the
scanner's distortion.

## Stage 6: Image Warping

**Module:** `scdc.calibration.apply_calibration`

For every pixel (X, Y) of the output canvas, the polynomial computes the
corresponding position (x, y) in the distorted input. The input image is
sampled at that position by cubic interpolation (`scipy.ndimage.map_coordinates`)
and the value is written to the output pixel. Positions outside the input
become NaN.

The polynomial is a property of the scanner, not of the sample. It encodes
the geometric distortion as a function of position. Any image acquired with
the same scan settings can be corrected by the same polynomial.

**Output:** the corrected image, with NaN in the corners where the distorted
input did not reach.

## Stage 7: Cropping

**Module:** `scdc.geometry.largest_inscribed_rectangle`

The warped output has ragged NaN corners (the parallelogram-shaped region of
the input does not fill the rectangular output). The largest axis-aligned
rectangle that fits entirely inside the valid (non-NaN) region is found using
a classic histogram-based algorithm (O(H × W)) and the image is cropped to
it.

**Output:** a clean rectangular image with no NaN, slightly smaller than the
input.

## Verification

The correction is verified in three ways:

1. **Lattice angle.** Centroids are re-detected in the corrected image and
   the angle between the new basis vectors is measured. On our data:
   78.6° → 89.9° (ideal: 90°).

2. **Residual vectors.** For each square, the difference between the
   polynomial prediction and the true position is drawn as an arrow. A good
   fit shows short, randomly oriented arrows; systematic structure would
   indicate that the polynomial degree is too low.

3. **Application to science data.** The same polynomial is applied to lipid
   diffusion images. The spot axis ratio (σx/σy) improved from 0.741 to
   0.993, and the azimuthal intensity variation dropped by a factor of 4.4,
   confirming that the correction restored the circular symmetry of a
   diffusing spot.
