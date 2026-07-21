"""Distortion correction for scanning cavity microscope images.

The package measures the geometric distortion of a raster-scanning
microscope from an image of a periodic calibration target (a checkerboard
mirror) and applies the resulting correction to arbitrary images acquired
with the same scan settings.

The public API is organised in the order in which a user would call it:

    detect_centroids      locate the bright squares of the target
    estimate_basis        measure the two lattice step vectors
    assign_lattice_indices  label each square with integer grid coordinates
    fit_calibration       combine the above into a polynomial distortion model
    apply_calibration     warp an image through a fitted model

Typical use::

    from scdc import fit_calibration, apply_calibration
    calibration = fit_calibration(checkerboard_image, degree=3)
    corrected = apply_calibration(sample_image, calibration)
"""

__version__ = "1.0.0"
