"""Local grid alignment — snap cell boundaries to actual edges."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from arcadia.grid import GridResult

# Minimum edge strength relative to the image max to count as a real boundary
EDGE_THRESHOLD_RATIO = 0.15

# How far (in pixels) a boundary can drift from its expected position
DEFAULT_TOLERANCE = 2


@dataclass
class AlignedGrid:
    x_boundaries: list[int]  # len = grid_width + 1, monotonically increasing
    y_boundaries: list[int]  # len = grid_height + 1, monotonically increasing
    grid_width: int
    grid_height: int
    cell_size: int           # original global period (for reference)
    confidence: float        # carried forward from GridResult


def _compute_edge_strength(gray: np.ndarray, axis: int) -> np.ndarray:
    """Compute average absolute gradient along an axis.

    For axis=1 (horizontal), computes vertical cell boundaries (column edges).
    For axis=0 (vertical), computes horizontal cell boundaries (row edges).

    Returns a 1D array where each element is the average edge strength at that position.
    """
    grad = np.abs(np.diff(gray, axis=axis))
    # Average across the orthogonal axis — this is a voting mechanism
    # that makes detection robust even when some rows/cols have no edges
    other_axis = 1 - axis
    return grad.mean(axis=other_axis)


def _find_boundaries(
    edge_strength: np.ndarray,
    cell_size: int,
    dim: int,
    tolerance: int = DEFAULT_TOLERANCE,
) -> list[int]:
    """Find actual cell boundaries by snapping expected positions to nearby edges.

    Args:
        edge_strength: 1D edge strength signal (length = dim - 1).
        cell_size: Expected cell size from global detection.
        dim: Image dimension (width or height).
        tolerance: Max drift from expected position in pixels.

    Returns:
        Monotonically increasing list of boundary positions, length = n_cells + 1.
    """
    n_cells = round(dim / cell_size)
    if n_cells < 1:
        return [0, dim]

    # Expected boundary positions (evenly spaced)
    expected = [round(i * dim / n_cells) for i in range(n_cells + 1)]

    # Threshold: ignore edges weaker than this
    max_edge = edge_strength.max() if len(edge_strength) > 0 else 0.0
    threshold = max_edge * EDGE_THRESHOLD_RATIO

    boundaries = [0]  # first boundary is always 0

    for i in range(1, n_cells):
        ideal = expected[i]

        # Search window: [ideal - tolerance, ideal + tolerance]
        # edge_strength[j] represents the gradient between pixel j and j+1
        lo = max(0, ideal - tolerance - 1)
        hi = min(len(edge_strength), ideal + tolerance)

        if hi > lo:
            window = edge_strength[lo:hi]
            if window.max() >= threshold:
                # Snap to the strongest edge in the window
                # +1 because edge_strength[j] is the boundary after pixel j
                best = lo + int(np.argmax(window)) + 1
            else:
                best = ideal
        else:
            best = ideal

        boundaries.append(best)

    boundaries.append(dim)  # last boundary is always the image edge

    # Enforce monotonicity (safety net)
    for i in range(1, len(boundaries)):
        if boundaries[i] <= boundaries[i - 1]:
            boundaries[i] = boundaries[i - 1] + 1

    return boundaries


def align_grid(image: Image.Image, grid: GridResult) -> AlignedGrid:
    """Align grid boundaries to actual edges in the image.

    Takes the global cell_size from grid detection and snaps each boundary
    to the nearest actual color transition, allowing ±2px drift.

    Args:
        image: RGB PIL Image.
        grid: GridResult from detect_grid().

    Returns:
        AlignedGrid with explicit boundary positions.
    """
    gray = np.array(image.convert("L"), dtype=np.float64)

    # Compute edge strength along each axis
    h_edge_strength = _compute_edge_strength(gray, axis=1)  # for x boundaries
    v_edge_strength = _compute_edge_strength(gray, axis=0)  # for y boundaries

    height, width = gray.shape

    x_boundaries = _find_boundaries(h_edge_strength, grid.cell_size, width)
    y_boundaries = _find_boundaries(v_edge_strength, grid.cell_size, height)

    return AlignedGrid(
        x_boundaries=x_boundaries,
        y_boundaries=y_boundaries,
        grid_width=len(x_boundaries) - 1,
        grid_height=len(y_boundaries) - 1,
        cell_size=grid.cell_size,
        confidence=grid.confidence,
    )
