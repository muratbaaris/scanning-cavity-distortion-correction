"""Reading and writing calibrations and images.

Calibrations are stored in MATLAB ``.mat`` files because the microscope
software of the laboratory this package was written for reads and writes that
format, so a calibration can be inspected with the same tools as the raw
data.  Images are read from ``.mat`` as well, and from ``.npy`` for users
working entirely in Python.
"""

import os

import numpy as np
import scipy.io as sio

from scdc.calibration import Calibration
from scdc.polynomial import number_of_terms, polynomial_term_labels

REQUIRED_KEYS = ("degree", "coefficients_x", "coefficients_y",
                 "lattice_spacing_px", "input_shape", "output_shape",
                 "crop_offset")
"""Fields a calibration file must contain to be usable."""


class CalibrationFileError(ValueError):
    """Raised when a calibration file is missing or malformed."""


def save_calibration(calibration, path):
    """Write a calibration to a MATLAB ``.mat`` file.

    Parameters
    ----------
    calibration : scdc.calibration.Calibration
        The model to store.
    path : str
        Destination file name.  The directory must already exist.

    Returns
    -------
    path : str
        The path that was written, for convenience when chaining calls.

    Raises
    ------
    CalibrationFileError
        If the destination directory does not exist.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        raise CalibrationFileError(
            f"the directory {directory} does not exist, so the calibration "
            "cannot be written there")

    sio.savemat(path, {
        "degree": int(calibration.degree),
        "coefficients_x": np.asarray(calibration.coefficients_x),
        "coefficients_y": np.asarray(calibration.coefficients_y),
        "lattice_spacing_px": float(calibration.lattice_spacing_px),
        "input_shape": np.asarray(calibration.input_shape, dtype=np.int64),
        "output_shape": np.asarray(calibration.output_shape, dtype=np.int64),
        "crop_offset": np.asarray(calibration.crop_offset, dtype=np.int64),
        "n_detected": int(calibration.n_detected),
        "n_used": int(calibration.n_used),
        "mean_residual_px": float(calibration.mean_residual_px),
        "max_residual_px": float(calibration.max_residual_px),
    })
    return path


def load_calibration(path):
    """Read a calibration written by `save_calibration`.

    Parameters
    ----------
    path : str
        File to read.

    Returns
    -------
    calibration : scdc.calibration.Calibration

    Raises
    ------
    CalibrationFileError
        If the file does not exist, cannot be parsed, is missing a required
        field, or stores a coefficient count inconsistent with its degree.
    """
    if not os.path.isfile(path):
        raise CalibrationFileError(f"no calibration file found at {path}")

    try:
        contents = sio.loadmat(path)
    except Exception as error:
        raise CalibrationFileError(
            f"{path} could not be read as a MATLAB file: {error}") from None

    missing = [key for key in REQUIRED_KEYS if key not in contents]
    if missing:
        raise CalibrationFileError(
            f"{path} is missing the required field(s) {', '.join(missing)}")

    degree = int(np.asarray(contents["degree"]).flatten()[0])
    coefficients_x = np.asarray(contents["coefficients_x"]).flatten()
    coefficients_y = np.asarray(contents["coefficients_y"]).flatten()

    expected = number_of_terms(degree)
    if len(coefficients_x) != expected or len(coefficients_y) != expected:
        raise CalibrationFileError(
            f"{path} declares degree {degree}, which needs {expected} "
            f"coefficients, but stores {len(coefficients_x)} and "
            f"{len(coefficients_y)}")

    return Calibration(
        degree=degree,
        coefficients_x=coefficients_x,
        coefficients_y=coefficients_y,
        lattice_spacing_px=float(
            np.asarray(contents["lattice_spacing_px"]).flatten()[0]),
        input_shape=tuple(
            int(v) for v in np.asarray(contents["input_shape"]).flatten()),
        output_shape=tuple(
            int(v) for v in np.asarray(contents["output_shape"]).flatten()),
        crop_offset=tuple(
            int(v) for v in np.asarray(contents["crop_offset"]).flatten()),
        n_detected=int(np.asarray(
            contents.get("n_detected", [[0]])).flatten()[0]),
        n_used=int(np.asarray(contents.get("n_used", [[0]])).flatten()[0]),
        mean_residual_px=float(np.asarray(
            contents.get("mean_residual_px", [[0.0]])).flatten()[0]),
        max_residual_px=float(np.asarray(
            contents.get("max_residual_px", [[0.0]])).flatten()[0]),
        term_labels=polynomial_term_labels(degree),
    )


def load_image(path, key=None):
    """Read a two-dimensional image from a ``.mat`` or ``.npy`` file.

    Parameters
    ----------
    path : str
        File to read.
    key : str or None, optional
        Name of the variable to take from a ``.mat`` file.  When omitted the
        file must contain exactly one two-dimensional array, which is then
        selected automatically.

    Returns
    -------
    image : ndarray, shape (H, W)

    Raises
    ------
    CalibrationFileError
        If the file does not exist, has an unsupported extension, contains no
        two-dimensional array, or contains several of them while ``key`` was
        not given.
    """
    if not os.path.isfile(path):
        raise CalibrationFileError(f"no image file found at {path}")

    extension = os.path.splitext(path)[1].lower()
    if extension == ".npy":
        image = np.load(path)
    elif extension == ".mat":
        image = _extract_image_from_mat(path, key)
    else:
        raise CalibrationFileError(
            f"unsupported image format '{extension}'; use .mat or .npy")

    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise CalibrationFileError(
            f"{path} holds a {image.ndim}-dimensional array, but an image "
            "must be two-dimensional")
    return image


def _extract_image_from_mat(path, key):
    """Pick the requested, or the only, two-dimensional array in a .mat file."""
    contents = sio.loadmat(path)
    candidates = {name: value for name, value in contents.items()
                  if not name.startswith("__")
                  and isinstance(value, np.ndarray) and value.ndim == 2}

    if key is not None:
        if key not in candidates:
            available = ", ".join(sorted(candidates)) or "none"
            raise CalibrationFileError(
                f"{path} has no two-dimensional array called '{key}'; "
                f"available: {available}")
        return candidates[key]

    if not candidates:
        raise CalibrationFileError(
            f"{path} contains no two-dimensional array")
    if len(candidates) > 1:
        available = ", ".join(sorted(candidates))
        raise CalibrationFileError(
            f"{path} contains several two-dimensional arrays ({available}); "
            "choose one with the key argument")

    return next(iter(candidates.values()))


def save_image(image, path):
    """Write an image to a ``.mat`` or ``.npy`` file.

    Parameters
    ----------
    image : array_like, shape (H, W)
        Image to store.
    path : str
        Destination file name; the extension selects the format.

    Returns
    -------
    path : str
        The path that was written.

    Raises
    ------
    CalibrationFileError
        If the extension is unsupported or the directory does not exist.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        raise CalibrationFileError(
            f"the directory {directory} does not exist")

    image = np.asarray(image, dtype=float)
    extension = os.path.splitext(path)[1].lower()
    if extension == ".npy":
        np.save(path, image)
    elif extension == ".mat":
        sio.savemat(path, {"corrected_image": image})
    else:
        raise CalibrationFileError(
            f"unsupported image format '{extension}'; use .mat or .npy")
    return path
