"""Color quantization for pixel art grid cells."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from arcadia.alignment import AlignedGrid
from arcadia.grid import GridResult


@dataclass
class PaletteResult:
    color_grid: list[list[str]]  # grid_height x grid_width of hex color strings
    palette: list[str]           # unique colors in the quantized palette


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _dominant_color(cell: np.ndarray, margin_fraction: float = 0.15) -> tuple[int, int, int]:
    """Extract the dominant color from a cell, ignoring edge pixels.

    Uses the central 70% of the cell (margin_fraction=0.15 on each side)
    to avoid anti-aliased edge pixels.
    """
    h, w, _ = cell.shape
    y_start = max(1, int(h * margin_fraction))
    y_end = max(y_start + 1, int(h * (1.0 - margin_fraction)))
    x_start = max(1, int(w * margin_fraction))
    x_end = max(x_start + 1, int(w * (1.0 - margin_fraction)))

    center = cell[y_start:y_end, x_start:x_end]
    if center.size == 0:
        center = cell

    # Use median for robustness against anti-aliasing outliers
    median = np.median(center.reshape(-1, 3), axis=0)
    return int(round(median[0])), int(round(median[1])), int(round(median[2]))


def _merge_similar_colors(
    colors: list[tuple[int, int, int]],
    threshold: float = 30.0,
) -> dict[tuple[int, int, int], tuple[int, int, int]]:
    """Merge colors within Euclidean RGB distance threshold.

    Returns a mapping from original color to its merged representative.
    """
    unique = sorted(set(colors))
    representatives: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    # Each cluster tracks its members and running centroid
    cluster_members: list[list[tuple[int, int, int]]] = []
    cluster_centroids: list[np.ndarray] = []

    for color in unique:
        color_arr = np.array(color, dtype=np.float64)
        merged = False
        for i, centroid in enumerate(cluster_centroids):
            dist = np.sqrt(np.sum((color_arr - centroid) ** 2))
            if dist <= threshold:
                cluster_members[i].append(color)
                # Update centroid to mean of all members
                cluster_centroids[i] = np.mean(
                    [np.array(c, dtype=np.float64) for c in cluster_members[i]],
                    axis=0,
                )
                merged = True
                break
        if not merged:
            cluster_members.append([color])
            cluster_centroids.append(color_arr.copy())

    # Map each color to its cluster's final centroid
    for members, centroid in zip(cluster_members, cluster_centroids):
        rep = (int(round(centroid[0])), int(round(centroid[1])), int(round(centroid[2])))
        for c in members:
            representatives[c] = rep

    return representatives


def quantize(image: Image.Image, grid: AlignedGrid | GridResult) -> PaletteResult:
    """Quantize image colors based on the detected grid.

    Args:
        image: RGB PIL Image.
        grid: AlignedGrid (with local boundaries) or GridResult (fixed grid).

    Returns:
        PaletteResult with color grid and palette.
    """
    pixels = np.array(image)

    # Build boundary arrays — AlignedGrid has them, GridResult uses fixed intervals
    if isinstance(grid, AlignedGrid):
        x_bounds = grid.x_boundaries
        y_bounds = grid.y_boundaries
    else:
        cs = grid.cell_size
        x_bounds = [min(col * cs, pixels.shape[1]) for col in range(grid.grid_width + 1)]
        y_bounds = [min(row * cs, pixels.shape[0]) for row in range(grid.grid_height + 1)]

    # Extract dominant color for each cell
    raw_colors: list[list[tuple[int, int, int]]] = []
    for row in range(grid.grid_height):
        color_row: list[tuple[int, int, int]] = []
        for col in range(grid.grid_width):
            cell = pixels[y_bounds[row]:y_bounds[row + 1], x_bounds[col]:x_bounds[col + 1]]
            color_row.append(_dominant_color(cell))
        raw_colors.append(color_row)

    # Flatten, merge similar colors, rebuild grid
    all_colors = [c for row in raw_colors for c in row]
    merge_map = _merge_similar_colors(all_colors)

    color_grid: list[list[str]] = []
    for row in raw_colors:
        hex_row: list[str] = []
        for color in row:
            merged = merge_map[color]
            hex_row.append(_rgb_to_hex(*merged))
        color_grid.append(hex_row)

    # Build palette from merged colors
    palette_set: set[str] = set()
    for row in color_grid:
        palette_set.update(row)
    palette = sorted(palette_set)

    return PaletteResult(color_grid=color_grid, palette=palette)
