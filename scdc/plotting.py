"""Visualisation helpers.

Every function here takes data that is already prepared and only draws it.
No analysis or preprocessing happens in this module, which is why it carries
no unit tests: the effort of asserting on rendered pixels would outweigh the
benefit, and there is no computation to get wrong.  Any quantity shown in a
figure is computed by the analysis modules and passed in.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_image(image, ax=None, title=None, cmap="viridis"):
    """Draw a single image with pixel axes.

    Parameters
    ----------
    image : array_like, shape (H, W)
        Image to display.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw into.  A new figure is created when omitted.
    title : str or None, optional
        Title placed above the image.
    cmap : str, optional
        Matplotlib colour map name.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes that were drawn into.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.asarray(image), cmap=cmap)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    if title:
        ax.set_title(title)
    return ax


def plot_comparison(original, corrected, titles=("Original", "Corrected"),
                    cmap="viridis"):
    """Draw an image and its corrected counterpart side by side.

    Parameters
    ----------
    original, corrected : array_like, shape (H, W)
        The two images to compare.  They need not have the same shape.
    titles : tuple of str, optional
        Titles for the left and the right panel.
    cmap : str, optional
        Matplotlib colour map name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure holding both panels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    plot_image(original, ax=axes[0], title=titles[0], cmap=cmap)
    plot_image(corrected, ax=axes[1], title=titles[1], cmap=cmap)
    fig.tight_layout()
    return fig


def plot_centroids(image, centroids, ax=None, title=None, marker_colour="red"):
    """Overlay detected square centres on the image they came from.

    Parameters
    ----------
    image : array_like, shape (H, W)
        Background image.
    centroids : array_like, shape (N, 2)
        Positions to mark, as ``(x, y)`` pairs.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw into.
    title : str or None, optional
        Title placed above the image.
    marker_colour : str, optional
        Colour of the markers.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    ax = plot_image(image, ax=ax, title=title)
    centroids = np.asarray(centroids)
    ax.plot(centroids[:, 0], centroids[:, 1], "+",
            color=marker_colour, markersize=6, linestyle="none")
    return ax


def plot_basis_vectors(image, centroids, a1, a2, angle_deg, ax=None,
                       title=None, scale=3.0):
    """Draw the two lattice vectors on the image, annotated with their angle.

    Parameters
    ----------
    image : array_like, shape (H, W)
        Background image.
    centroids : array_like, shape (N, 2)
        Detected centres, used only to place the vectors at their mean.
    a1, a2 : array_like, shape (2,)
        Lattice vectors to draw.
    angle_deg : float
        Angle between the vectors, computed by
        `scdc.lattice.lattice_angle` and passed in ready to display.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw into.
    title : str or None, optional
        Title placed above the image.
    scale : float, optional
        Factor by which the drawn arrows are lengthened for visibility.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    ax = plot_image(image, ax=ax, title=title)
    anchor = np.asarray(centroids).mean(axis=0)
    for vector, colour, label in ((np.asarray(a1), "red", "a1"),
                                  (np.asarray(a2), "cyan", "a2")):
        ax.arrow(anchor[0], anchor[1], vector[0] * scale, vector[1] * scale,
                 head_width=2.5, head_length=1.5,
                 fc=colour, ec=colour, linewidth=2, zorder=5)
        ax.text(anchor[0] + vector[0] * scale, anchor[1] + vector[1] * scale,
                f" {label}", color=colour, fontsize=12, fontweight="bold")
    ax.text(anchor[0], anchor[1], f"  {angle_deg:.1f}deg",
            color="yellow", fontsize=12, fontweight="bold")
    return ax


def plot_residuals(image, centroids, residual_vectors, mean_residual,
                   max_residual, ax=None, arrow_scale=5.0):
    """Draw the fit residual at every centroid as an arrow.

    A good fit leaves short arrows pointing in random directions; systematic
    structure in the arrows indicates that the polynomial degree is too low
    to describe the distortion.

    Parameters
    ----------
    image : array_like, shape (H, W)
        Background image.
    centroids : array_like, shape (N, 2)
        Positions at which the residuals were evaluated.
    residual_vectors : array_like, shape (N, 2)
        Difference between measured and predicted positions.
    mean_residual, max_residual : float
        Summary statistics shown in the title, computed elsewhere.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw into.
    arrow_scale : float, optional
        Factor by which the arrows are lengthened for visibility.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.asarray(image), cmap="gray", alpha=0.5)
    centroids = np.asarray(centroids)
    residual_vectors = np.asarray(residual_vectors)
    ax.quiver(centroids[:, 0], centroids[:, 1],
              residual_vectors[:, 0] * arrow_scale,
              residual_vectors[:, 1] * arrow_scale,
              color="red", angles="xy", scale_units="xy", scale=1,
              width=0.002, headwidth=4)
    ax.set_title(f"Fit residuals, mean {mean_residual:.2f} px, "
                 f"max {max_residual:.2f} px (arrows x{arrow_scale:g})")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    return ax
