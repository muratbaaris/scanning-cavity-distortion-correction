"""Command line interface to the distortion correction pipeline.

Three subcommands cover the whole workflow::

    python -m scdc fit    target.mat  --output calibration.mat
    python -m scdc apply  sample.mat  --calibration calibration.mat \\
                                      --output corrected.mat
    python -m scdc demo   --output-dir results

``fit`` measures the distortion from an image of the calibration target,
``apply`` corrects any later image with that measurement, and ``demo``
runs the whole pipeline on a synthetic target so the installation can be
verified without any experimental data.
"""

import argparse
import os
import sys

import numpy as np

from scdc.calibration import apply_calibration, fit_calibration
from scdc.centroids import detect_centroids
from scdc.io import load_calibration, load_image, save_calibration, save_image
from scdc.lattice import basis_length_ratio, estimate_basis, lattice_angle
from scdc.synthetic import distort_image, make_checkerboard, shear_matrix

DEFAULT_DEGREE = 3
DEFAULT_TARGET_PITCH_UM = 14.142
"""Diagonal spacing of a checkerboard with 10 um cells, i.e. 10 * sqrt(2)."""


def build_parser():
    """Construct the argument parser for the command line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="scdc",
        description="Measure and correct the geometric distortion of a "
                    "raster-scanning microscope.")
    parser.add_argument("--version", action="store_true",
                        help="print the package version and exit")
    subparsers = parser.add_subparsers(dest="command")

    fit_parser = subparsers.add_parser(
        "fit", help="measure a calibration from an image of the target")
    fit_parser.add_argument("image", help="image of the calibration target")
    fit_parser.add_argument("--output", "-o", default="calibration.mat",
                            help="where to write the calibration "
                                 "(default: calibration.mat)")
    fit_parser.add_argument("--degree", "-d", type=int, default=DEFAULT_DEGREE,
                            help=f"polynomial degree "
                                 f"(default: {DEFAULT_DEGREE})")
    fit_parser.add_argument("--key", default=None,
                            help="variable name inside a .mat file")
    fit_parser.add_argument("--threshold-fraction", type=float, default=0.45,
                            help="binarisation threshold, between 0 and 1 "
                                 "(default: 0.45)")
    fit_parser.add_argument("--target-pitch-um", type=float,
                            default=DEFAULT_TARGET_PITCH_UM,
                            help="distance in micrometres between "
                                 "neighbouring bright squares "
                                 f"(default: {DEFAULT_TARGET_PITCH_UM})")

    apply_parser = subparsers.add_parser(
        "apply", help="correct an image with an existing calibration")
    apply_parser.add_argument("image", help="image to correct")
    apply_parser.add_argument("--calibration", "-c", required=True,
                              help="calibration file produced by 'fit'")
    apply_parser.add_argument("--output", "-o", default="corrected.mat",
                              help="where to write the corrected image "
                                   "(default: corrected.mat)")
    apply_parser.add_argument("--key", default=None,
                              help="variable name inside a .mat file")
    apply_parser.add_argument("--fill", type=float, default=0.0,
                              help="value written where no data exists "
                                   "(default: 0.0)")

    demo_parser = subparsers.add_parser(
        "demo", help="run the pipeline on a synthetic target")
    demo_parser.add_argument("--output-dir", "-o", default=".",
                             help="directory for the generated files "
                                  "(default: the current directory)")
    demo_parser.add_argument("--degree", "-d", type=int, default=DEFAULT_DEGREE,
                             help=f"polynomial degree "
                                  f"(default: {DEFAULT_DEGREE})")
    demo_parser.add_argument("--seed", type=int, default=0,
                             help="random seed for the synthetic noise "
                                  "(default: 0)")

    return parser


def command_fit(arguments):
    """Run the ``fit`` subcommand.

    Parameters
    ----------
    arguments : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    exit_code : int
        Zero on success.
    """
    image = load_image(arguments.image, key=arguments.key)
    calibration = fit_calibration(
        image, degree=arguments.degree,
        threshold_fraction=arguments.threshold_fraction)

    print(calibration.summary())
    pixel_size = calibration.pixel_size(arguments.target_pitch_um)
    print(f"\n  pixel size          : {pixel_size:.4f} um/px "
          f"(target pitch {arguments.target_pitch_um} um)")

    save_calibration(calibration, arguments.output)
    print(f"\nCalibration written to {arguments.output}")
    return 0


def command_apply(arguments):
    """Run the ``apply`` subcommand.

    Parameters
    ----------
    arguments : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    exit_code : int
        Zero on success.
    """
    calibration = load_calibration(arguments.calibration)
    image = load_image(arguments.image, key=arguments.key)
    corrected = apply_calibration(image, calibration,
                                  fill_value=arguments.fill)

    save_image(corrected, arguments.output)
    print(f"Corrected {image.shape} -> {corrected.shape}, "
          f"written to {arguments.output}")
    return 0


def command_demo(arguments):
    """Run the ``demo`` subcommand on a synthetic calibration target.

    The demo builds a checkerboard, distorts it by a known shear and
    anisotropic scaling, measures a calibration from the distorted image and
    applies it.  The lattice angle and the ratio of the basis lengths are
    reported before and after the correction, so the improvement can be read
    off directly.

    Parameters
    ----------
    arguments : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    exit_code : int
        Zero on success.
    """
    os.makedirs(arguments.output_dir, exist_ok=True)

    undistorted = make_checkerboard(
        shape=(220, 220), cell_size=12.0, blur_sigma=1.0,
        noise_level=0.01, random_seed=arguments.seed)
    distortion = shear_matrix(shear=0.25, scale_x=1.0, scale_y=1.15)
    distorted = distort_image(undistorted, matrix=distortion)

    before = estimate_basis(detect_centroids(distorted))
    calibration = fit_calibration(distorted, degree=arguments.degree)
    corrected = apply_calibration(distorted, calibration, fill_value=0.0)
    after = estimate_basis(detect_centroids(corrected))

    print("Synthetic demonstration")
    print("=" * 52)
    print(f"  applied shear       : 0.25")
    print(f"  applied y scaling   : 1.15")
    print()
    print(f"  {'quantity':<22}{'before':>12}{'after':>12}{'ideal':>10}")
    print("  " + "-" * 54)
    print(f"  {'lattice angle [deg]':<22}{lattice_angle(*before):>12.2f}"
          f"{lattice_angle(*after):>12.2f}{90.0:>10.2f}")
    print(f"  {'basis length ratio':<22}{basis_length_ratio(*before):>12.4f}"
          f"{basis_length_ratio(*after):>12.4f}{1.0:>10.4f}")
    print()
    print(f"  fit residual        : {calibration.mean_residual_px:.3f} px mean, "
          f"{calibration.max_residual_px:.3f} px max")
    print(f"  squares used        : {calibration.n_used} of "
          f"{calibration.n_detected}")

    paths = {
        "distorted.npy": distorted,
        "corrected.npy": corrected,
    }
    for name, array in paths.items():
        save_image(array, os.path.join(arguments.output_dir, name))
    save_calibration(calibration,
                     os.path.join(arguments.output_dir, "calibration.mat"))

    print(f"\nFiles written to {os.path.abspath(arguments.output_dir)}")
    return 0


def main(argv=None):
    """Entry point of the command line interface.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument list to parse.  ``sys.argv[1:]`` is used when omitted.

    Returns
    -------
    exit_code : int
        Zero on success, one when the arguments are incomplete, two when the
        pipeline raised an error that the user can act on.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.version:
        from scdc import __version__
        print(__version__)
        return 0

    if arguments.command is None:
        parser.print_help()
        return 1

    handlers = {
        "fit": command_fit,
        "apply": command_apply,
        "demo": command_demo,
    }
    try:
        return handlers[arguments.command](arguments)
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
